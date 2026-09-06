#include "scar/detail/copula/multivariate/conditional.hpp"

#include "scar/detail/linalg.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace scar_internal {

using scar::SCAR_INVALID_PARAMETER;
using scar::SCAR_INVALID_SIZE;
using scar::SCAR_NUMERICAL_FAILURE;
using scar::SCAR_OK;

namespace {

bool cholesky_with_jitter(
    const std::vector<double>& matrix,
    std::size_t dimension,
    std::vector<double>& lower,
    double* applied_jitter = nullptr) {

    return linalg::cholesky_symmetric_with_jitter(
        matrix.data(), dimension, lower, applied_jitter);
}

bool solve_spd(
    const std::vector<double>& lower,
    std::size_t dimension,
    const std::vector<double>& rhs,
    std::size_t columns,
    std::vector<double>& solution) {

    return linalg::solve_spd(
        lower.data(), dimension, rhs.data(), columns, solution);
}

struct ConditionalFactors {
    std::vector<double> r_gg;
    std::vector<double> r_gf;
    std::vector<double> r_fg;
    std::vector<double> r_ff;
    std::vector<double> lower_gg;
    std::vector<double> solved_cross;
    std::vector<double> schur_base;
    std::vector<double> prepared_lower_cov;
    bool use_prepared_lower_cov = false;
};

struct ConditionalWorkerWorkspace {
    ConditionalFactors row_factors;
    std::vector<double> solved_given;
    std::vector<double> covariance;
    std::vector<double> lower_cov;
    std::vector<double> given_vector;
    std::vector<double> conditional_mean;
    std::vector<double> innovation;
};

struct ConditionalBlockResult {
    bool ran = false;
    int status = SCAR_OK;
    std::int64_t failure_index = -1;
    std::uint64_t correlation_factorizations = 0;
};

bool prepare_conditional_factors(
    const double* correlation,
    std::size_t dimension,
    const std::vector<int>& given_indices,
    const std::vector<int>& free_indices,
    bool allow_prepared_lower_cov,
    const ConditionalPolicy& policy,
    ConditionalFactors& factors) {

    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = free_indices.size();
    factors.r_gg.assign(n_given * n_given, 0.0);
    factors.r_gf.assign(n_given * n_free, 0.0);
    factors.r_fg.assign(n_free * n_given, 0.0);
    factors.r_ff.assign(n_free * n_free, 0.0);
    factors.schur_base.assign(n_free * n_free, 0.0);
    for (std::size_t i = 0; i < n_given; ++i) {
        const std::size_t given =
            static_cast<std::size_t>(given_indices[i]);
        for (std::size_t j = 0; j < n_given; ++j) {
            const std::size_t other_given =
                static_cast<std::size_t>(given_indices[j]);
            factors.r_gg[i * n_given + j] =
                correlation[given * dimension + other_given];
        }
        for (std::size_t j = 0; j < n_free; ++j) {
            const std::size_t free =
                static_cast<std::size_t>(free_indices[j]);
            factors.r_gf[i * n_free + j] =
                correlation[given * dimension + free];
        }
    }
    for (std::size_t i = 0; i < n_free; ++i) {
        const std::size_t free =
            static_cast<std::size_t>(free_indices[i]);
        for (std::size_t j = 0; j < n_given; ++j) {
            const std::size_t given =
                static_cast<std::size_t>(given_indices[j]);
            factors.r_fg[i * n_given + j] =
                correlation[free * dimension + given];
        }
        for (std::size_t j = 0; j < n_free; ++j) {
            const std::size_t other_free =
                static_cast<std::size_t>(free_indices[j]);
            factors.r_ff[i * n_free + j] =
                correlation[free * dimension + other_free];
        }
    }
    if (!cholesky_with_jitter(factors.r_gg, n_given, factors.lower_gg)
        || !solve_spd(
            factors.lower_gg,
            n_given,
            factors.r_gf,
            n_free,
            factors.solved_cross)) {
        return false;
    }
    for (std::size_t i = 0; i < n_free; ++i) {
        for (std::size_t j = 0; j < n_free; ++j) {
            double schur = factors.r_ff[i * n_free + j];
            for (std::size_t k = 0; k < n_given; ++k) {
                schur -= factors.r_fg[i * n_given + k]
                    * factors.solved_cross[k * n_free + j];
            }
            factors.schur_base[i * n_free + j] = schur;
        }
    }
    factors.use_prepared_lower_cov = false;
    if (!allow_prepared_lower_cov) {
        return true;
    }
    double applied_jitter = 0.0;
    const bool factorized = cholesky_with_jitter(
        factors.schur_base,
        n_free,
        factors.prepared_lower_cov,
        &applied_jitter);
    if (policy.require_prepared_factorization && !factorized) {
        return false;
    }
    factors.use_prepared_lower_cov = factorized
        && (policy.allow_jittered_prepared_factor
            || applied_jitter == 0.0);
    return true;
}

}  // namespace

scar::ConditionalSampleResult conditional_latent(
    scar::DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    scar::DoubleView given_latent,
    scar::DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads,
    const ConditionalPolicy& policy) {

    scar::ConditionalSampleResult out;
    out.n_rows = n_rows;
    out.n_threads_requested = n_threads;
    if (dimension < 2 || n_rows <= 0 || given_indices.empty()
        || given_indices.size() >= static_cast<std::size_t>(dimension)) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    if (!valid_thread_count(n_threads)) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }
    const std::size_t d = static_cast<std::size_t>(dimension);
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    const std::size_t n_given = given_indices.size();
    const std::size_t n_free = d - n_given;
    out.n_free = static_cast<std::int64_t>(n_free);
    if (correlation_rows != 1 && correlation_rows != n_rows) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    const std::size_t corr_rows =
        static_cast<std::size_t>(correlation_rows);
    if (correlations.size() != corr_rows * d * d
        || normal_draws.size() != rows * n_free
        || (given_latent.size() != n_given
            && given_latent.size() != rows * n_given)
        || !policy.auxiliary_sizes_valid) {
        out.status = scar::Status::InvalidSize;
        return out;
    }

    std::vector<bool> is_given(d, false);
    for (int index : given_indices) {
        if (index < 0 || index >= dimension
            || is_given[static_cast<std::size_t>(index)]) {
            out.status = scar::Status::InvalidParameter;
            return out;
        }
        is_given[static_cast<std::size_t>(index)] = true;
    }
    std::vector<int> free_indices;
    free_indices.reserve(n_free);
    for (int index = 0; index < dimension; ++index) {
        if (!is_given[static_cast<std::size_t>(index)]) {
            free_indices.push_back(index);
        }
    }

    out.values.assign(rows * n_free, 0.0);
    ConditionalFactors common_factors;
    if (correlation_rows == 1) {
        if (!prepare_conditional_factors(
                correlations.data(),
                d,
                given_indices,
                free_indices,
                true,
                policy,
                common_factors)) {
            out.status = scar::Status::NumericalFailure;
            out.failure.index = 0;
            return out;
        }
        out.correlation_factorizations = 1;
    }

    constexpr std::size_t min_rows = 32;
    constexpr std::size_t min_work = 65536;
    const bool use_threads = n_threads > 1
        && grid_parallel_worthwhile(rows, d * d, min_rows, min_work);
    std::vector<ConditionalBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    parallel_for_blocks(
        0,
        n_rows,
        static_cast<std::int64_t>(min_rows),
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            ConditionalBlockResult& block_result = block_results[block];
            block_result.ran = true;
            ConditionalWorkerWorkspace workspace;
            workspace.given_vector.assign(n_given, 0.0);
            workspace.covariance.assign(n_free * n_free, 0.0);
            workspace.conditional_mean.assign(n_free, 0.0);
            workspace.innovation.assign(n_free, 0.0);
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const ConditionalFactors* factors = &common_factors;
                if (correlation_rows != 1) {
                    if (!prepare_conditional_factors(
                            correlations.data() + row * d * d,
                            d,
                            given_indices,
                            free_indices,
                            false,
                            policy,
                            workspace.row_factors)) {
                        block_result.status = SCAR_NUMERICAL_FAILURE;
                        block_result.failure_index = row_index;
                        return;
                    }
                    ++block_result.correlation_factorizations;
                    factors = &workspace.row_factors;
                }
                for (std::size_t i = 0; i < n_given; ++i) {
                    workspace.given_vector[i] = given_latent[
                        given_latent.size() == n_given
                            ? i : row * n_given + i];
                }
                if (!solve_spd(
                        factors->lower_gg,
                        n_given,
                        workspace.given_vector,
                        1,
                        workspace.solved_given)) {
                    block_result.status = SCAR_NUMERICAL_FAILURE;
                    block_result.failure_index = row_index;
                    return;
                }

                ConditionalScale scale;
                if (policy.scale != nullptr) {
                    block_result.status = policy.scale(
                        workspace.given_vector.data(),
                        workspace.solved_given.data(),
                        n_given,
                        row,
                        policy.context,
                        scale);
                    if (block_result.status != SCAR_OK) {
                        block_result.failure_index = row_index;
                        return;
                    }
                }

                if (!factors->use_prepared_lower_cov) {
                    for (std::size_t i = 0; i < n_free; ++i) {
                        for (std::size_t j = 0; j < n_free; ++j) {
                            workspace.covariance[i * n_free + j] =
                                scale.covariance
                                * factors->schur_base[i * n_free + j];
                        }
                    }
                    if (!cholesky_with_jitter(
                            workspace.covariance,
                            n_free,
                            workspace.lower_cov)) {
                        block_result.status = SCAR_NUMERICAL_FAILURE;
                        block_result.failure_index = row_index;
                        return;
                    }
                }

                const std::vector<double>& innovation_factor =
                    factors->use_prepared_lower_cov
                    ? factors->prepared_lower_cov
                    : workspace.lower_cov;
                const double prepared_factor_scale =
                    factors->use_prepared_lower_cov
                    && policy.scale != nullptr
                    ? std::sqrt(scale.covariance)
                    : 1.0;
                linalg::row_major_matvec(
                    factors->r_fg.data(),
                    n_free,
                    n_given,
                    workspace.solved_given.data(),
                    workspace.conditional_mean.data());
                linalg::lower_triangular_matvec(
                    innovation_factor.data(),
                    n_free,
                    normal_draws.data() + row * n_free,
                    workspace.innovation.data());
                for (std::size_t i = 0; i < n_free; ++i) {
                    const double value = workspace.conditional_mean[i]
                        + scale.radial * prepared_factor_scale
                            * workspace.innovation[i];
                    if (!std::isfinite(value)) {
                        block_result.status = SCAR_NUMERICAL_FAILURE;
                        block_result.failure_index = row_index;
                        return;
                    }
                    out.values[row * n_free + i] = value;
                }
            }
        });

    for (const ConditionalBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        out.correlation_factorizations += block.correlation_factorizations;
        if (block.failure_index >= 0
            && (out.failure.index < 0
                || block.failure_index < out.failure.index)) {
            out.failure.index = block.failure_index;
            out.status = scar::status_from_int(block.status);
        }
    }
    if (out.failure.index >= 0) {
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure.index + 1) * n_free;
        std::fill(
            out.values.begin() + first_uncomputed,
            out.values.end(),
            0.0);
    }
    return out;
}

}  // namespace scar_internal
