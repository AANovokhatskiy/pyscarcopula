#include "scar/copula.hpp"
#include "scar/copula/multivariate/equicorrelation/kernel.hpp"

#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/multivariate/batch.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

namespace scar_internal {

using scar::SCAR_INVALID_FAMILY;
using scar::SCAR_INVALID_PARAMETER;
using scar::SCAR_INVALID_SIZE;
using scar::SCAR_NUMERICAL_FAILURE;
using scar::SCAR_OK;

namespace {

constexpr std::int64_t kEquicorrGridMinRowsPerBlock = 64;
constexpr std::size_t kEquicorrGridMinCells = 262144;

struct EquicorrBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
};

int validate_equicorr_batch(
    const scar::CopulaSpec& spec,
    const scar::Observations& observations,
    std::int64_t row_offset) {

    if (observations.empty() || row_offset < 0) {
        return SCAR_INVALID_SIZE;
    }
    if (spec.family != scar::CopulaFamily::EquicorrGaussian
        || spec.rotation != scar::Rotation::R0
        || spec.transform != scar::Transform::GaussianTanh
        || spec.dim < 2) {
        return SCAR_INVALID_FAMILY;
    }
    return validate_multivariate_observations(spec, observations);
}

}  // namespace

scar::MultivariateRowsResult equicorr_log_pdf_and_grad_rows(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& correlations,
    std::int64_t row_offset,
    int n_threads) {

    scar::MultivariateRowsResult out;
    out.n_threads_requested = n_threads;
    out.status = scar::status_from_int(
        validate_equicorr_batch(spec, observations, row_offset));
    out.log_pdf.assign(
        observations.size(), -std::numeric_limits<double>::infinity());
    out.dlog_dr.assign(
        observations.size(), std::numeric_limits<double>::quiet_NaN());
    if (!out.is_ok()) {
        return out;
    }
    if (!valid_thread_count(n_threads)) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }
    if (correlations.size() != 1
        && correlations.size() != observations.size()) {
        out.status = scar::Status::InvalidSize;
        return out;
    }

    const bool use_threads = n_threads > 1
        && grid_parallel_worthwhile(
            observations.size(),
            static_cast<std::size_t>(spec.dim),
            static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock),
            kEquicorrGridMinCells);
    std::vector<EquicorrBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    parallel_for_blocks(
        0,
        static_cast<std::int64_t>(observations.size()),
        kEquicorrGridMinRowsPerBlock,
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            EquicorrBlockResult& block_result = block_results[block];
            block_result.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double correlation = parameter_at(correlations, row);
                double derivative = 0.0;
                const double log_pdf = equicorr_log_pdf(
                    spec,
                    observations[row].data(),
                    correlation,
                    &derivative);
                if (!std::isfinite(correlation)
                    || !std::isfinite(log_pdf)
                    || !std::isfinite(derivative)) {
                    block_result.failure_index = row_index;
                    return;
                }
                out.log_pdf[row] = log_pdf;
                out.dlog_dr[row] = derivative;
            }
        });

    for (const EquicorrBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.row_parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure.index < 0
                || block.failure_index < out.failure.index)) {
            out.failure.index = block.failure_index;
        }
    }
    if (out.failure.index >= 0) {
        out.status = scar::Status::NumericalFailure;
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure.index + 1);
        std::fill(
            out.log_pdf.begin() + first_uncomputed,
            out.log_pdf.end(),
            -std::numeric_limits<double>::infinity());
        std::fill(
            out.dlog_dr.begin() + first_uncomputed,
            out.dlog_dr.end(),
            std::numeric_limits<double>::quiet_NaN());
    }
    return out;
}

scar::MultivariateGridResult equicorr_pdf_and_grad_grid(
    const scar::CopulaSpec& spec,
    const std::vector<std::vector<double>>& observations,
    const std::vector<double>& state_grid,
    std::int64_t row_offset,
    int n_threads) {

    scar::MultivariateGridResult out;
    out.n_threads_requested = n_threads;
    initialize_multivariate_grid(
        out, observations.size(), state_grid.size());
    if (!out.is_ok()) {
        return out;
    }
    out.status = scar::status_from_int(
        validate_equicorr_batch(spec, observations, row_offset));
    if (!valid_thread_count(n_threads)) {
        out.status = scar::Status::InvalidParameter;
    }
    if (!out.is_ok() || state_grid.empty()) {
        if (scar::ok(out.status)) {
            out.status = scar::Status::InvalidSize;
        }
        return out;
    }
    if (!std::all_of(
            state_grid.begin(), state_grid.end(), [](double value) {
                return std::isfinite(value);
            })) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }

    std::vector<double> correlation_grid(state_grid.size(), 0.0);
    std::vector<double> dpsi_grid(state_grid.size(), 0.0);
    for (std::size_t column = 0; column < state_grid.size(); ++column) {
        correlation_grid[column] =
            copula_transform(spec, state_grid[column]);
        dpsi_grid[column] = copula_dtransform(spec, state_grid[column]);
    }

    const bool use_threads = n_threads > 1
        && grid_parallel_worthwhile(
            observations.size(),
            state_grid.size(),
            static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock),
            kEquicorrGridMinCells);
    std::vector<EquicorrBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    parallel_for_blocks(
        0,
        static_cast<std::int64_t>(observations.size()),
        kEquicorrGridMinRowsPerBlock,
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            EquicorrBlockResult& block_result = block_results[block];
            block_result.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const std::size_t base = row * state_grid.size();
                EquicorrStats stats;
                if (!equicorr_sufficient_statistics(
                        spec, observations[row].data(), stats)) {
                    block_result.failure_index = row_index;
                    return;
                }
                for (std::size_t column = 0;
                     column < state_grid.size();
                     ++column) {
                    double derivative = 0.0;
                    const double log_pdf = equicorr_log_pdf_from_stats(
                        spec,
                        stats,
                        correlation_grid[column],
                        &derivative);
                    const double pdf = std::exp(log_pdf);
                    out.pdf.values[base + column] = pdf;
                    out.d_pdf_dx.values[base + column] =
                        pdf * derivative * dpsi_grid[column];
                }
                for (std::size_t column = 0;
                     column < state_grid.size();
                     ++column) {
                    if (!std::isfinite(out.pdf.values[base + column])
                        || !std::isfinite(
                            out.d_pdf_dx.values[base + column])) {
                        block_result.failure_index = row_index;
                        return;
                    }
                }
            }
        });

    for (const EquicorrBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.equicorr_parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure.index < 0
                || block.failure_index < out.failure.index)) {
            out.failure.index = block.failure_index;
        }
    }
    if (out.failure.index >= 0) {
        out.status = scar::Status::NumericalFailure;
        const std::size_t first_uncomputed =
            static_cast<std::size_t>(out.failure.index + 1)
            * state_grid.size();
        std::fill(
            out.pdf.values.begin() + first_uncomputed,
            out.pdf.values.end(),
            0.0);
        std::fill(
            out.d_pdf_dx.values.begin() + first_uncomputed,
            out.d_pdf_dx.values.end(),
            0.0);
    }
    return out;
}

}  // namespace scar_internal

namespace scar {
namespace {

constexpr std::int64_t kEquicorrGridMinRowsPerBlock = 64;
constexpr std::size_t kEquicorrGridMinCells = 262144;
constexpr std::size_t kEquicorrPreparationMinCells = 4096;

struct EquicorrBlockResult {
    bool ran = false;
    std::int64_t failure_index = -1;
};

}  // namespace

EquicorrPreparationResult prepare_equicorr_sufficient_statistics(
    ObservationView observations,
    std::size_t dimension_tile,
    int n_threads) {

    EquicorrPreparationResult out;
    out.n_threads_requested = n_threads;
    if (observations.empty() || observations.values == nullptr
        || observations.dim < 2 || dimension_tile == 0) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }

    const std::size_t dimension =
        static_cast<std::size_t>(observations.dim);
    std::size_t input_values = 0;
    if (!scar_internal::checked_size_mul(
            observations.n_obs, dimension, input_values)
        || input_values
            > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    const std::size_t dimension_tiles =
        dimension / dimension_tile
        + (dimension % dimension_tile == 0 ? 0 : 1);
    std::size_t partial_values = 0;
    if (!scar_internal::checked_size_mul(
            observations.n_obs, dimension_tiles, partial_values)) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    std::size_t temporary_values = 0;
    if (!scar_internal::checked_size_mul(
            partial_values, std::size_t{2}, temporary_values)) {
        out.status = scar::Status::InvalidSize;
        return out;
    }

    out.dimension_tiles = dimension_tiles;
    out.temporary_values = temporary_values;
    out.sum_z.assign(observations.n_obs, 0.0);
    out.sum_z2.assign(observations.n_obs, 0.0);
    std::vector<double> partial_sum(partial_values, 0.0);
    std::vector<double> partial_sum2(partial_values, 0.0);
    std::atomic<std::uint64_t> clipping_events{0};
    std::atomic<std::uint64_t> nonfinite_values{0};
    std::atomic<std::int64_t> first_failure{
        std::numeric_limits<std::int64_t>::max()};

    const auto update_first_failure =
        [&first_failure](std::int64_t index) {
            std::int64_t current =
                first_failure.load(std::memory_order_relaxed);
            while (index < current
                   && !first_failure.compare_exchange_weak(
                       current, index, std::memory_order_relaxed)) {
            }
        };
    const auto neumaier_add = [](double value, double& sum, double& carry) {
        const double next = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            carry += (sum - next) + value;
        } else {
            carry += (value - next) + sum;
        }
        sum = next;
    };
    const auto evaluate_tile =
        [&](std::size_t row,
            std::size_t tile,
            std::uint64_t& block_clipping,
            std::uint64_t& block_nonfinite) {
            const std::size_t begin = tile * dimension_tile;
            const std::size_t end =
                std::min(begin + dimension_tile, dimension);
            const std::size_t row_offset = row * dimension;
            double sum = 0.0;
            double carry = 0.0;
            double sum2 = 0.0;
            double carry2 = 0.0;
            std::uint64_t local_clipping = 0;
            std::uint64_t local_nonfinite = 0;
            for (std::size_t column = begin; column < end; ++column) {
                const std::size_t index = row_offset + column;
                const double value = observations.values[index];
                if (!std::isfinite(value)) {
                    ++local_nonfinite;
                    update_first_failure(static_cast<std::int64_t>(index));
                    continue;
                }
                const double clipped =
                    scar_internal::clip_pseudo_observation(value);
                local_clipping += clipped != value ? 1U : 0U;
                const double z = math::normal_quantile_refined(clipped);
                neumaier_add(z, sum, carry);
                neumaier_add(z * z, sum2, carry2);
            }
            const std::size_t partial_index =
                row * dimension_tiles + tile;
            partial_sum[partial_index] = sum + carry;
            partial_sum2[partial_index] = sum2 + carry2;
            block_clipping += local_clipping;
            block_nonfinite += local_nonfinite;
        };
    const auto commit_counts =
        [&](std::uint64_t block_clipping,
            std::uint64_t block_nonfinite) {
            if (block_clipping != 0) {
                clipping_events.fetch_add(
                    block_clipping, std::memory_order_relaxed);
            }
            if (block_nonfinite != 0) {
                nonfinite_values.fetch_add(
                    block_nonfinite, std::memory_order_relaxed);
            }
        };

    const bool enough_work =
        input_values >= kEquicorrPreparationMinCells;
    const bool row_parallel =
        enough_work && n_threads > 1
        && observations.n_obs >= static_cast<std::size_t>(4 * n_threads);
    const bool tile_parallel =
        enough_work && n_threads > 1 && !row_parallel
        && partial_values > 1;
    if (row_parallel) {
        out.parallel_axis = 1;
        out.parallel_blocks = scar_internal::limit_worker_count(
            n_threads, observations.n_obs);
        scar_internal::parallel_for_blocks(
            0,
            static_cast<std::int64_t>(observations.n_obs),
            1,
            n_threads,
            [&](std::int64_t begin, std::int64_t end, std::size_t) {
                std::uint64_t block_clipping = 0;
                std::uint64_t block_nonfinite = 0;
                for (std::int64_t row = begin; row < end; ++row) {
                    for (std::size_t tile = 0;
                         tile < dimension_tiles;
                         ++tile) {
                        evaluate_tile(
                            static_cast<std::size_t>(row),
                            tile,
                            block_clipping,
                            block_nonfinite);
                    }
                }
                commit_counts(block_clipping, block_nonfinite);
            });
    } else if (tile_parallel) {
        out.parallel_axis = 2;
        out.parallel_blocks = scar_internal::limit_worker_count(
            n_threads, partial_values);
        scar_internal::parallel_for_blocks(
            0,
            static_cast<std::int64_t>(partial_values),
            1,
            n_threads,
            [&](std::int64_t begin, std::int64_t end, std::size_t) {
                std::uint64_t block_clipping = 0;
                std::uint64_t block_nonfinite = 0;
                for (std::int64_t index = begin; index < end; ++index) {
                    const std::size_t flat =
                        static_cast<std::size_t>(index);
                    evaluate_tile(
                        flat / dimension_tiles,
                        flat % dimension_tiles,
                        block_clipping,
                        block_nonfinite);
                }
                commit_counts(block_clipping, block_nonfinite);
            });
    } else {
        std::uint64_t block_clipping = 0;
        std::uint64_t block_nonfinite = 0;
        for (std::size_t row = 0; row < observations.n_obs; ++row) {
            for (std::size_t tile = 0;
                 tile < dimension_tiles;
                 ++tile) {
                evaluate_tile(
                    row,
                    tile,
                    block_clipping,
                    block_nonfinite);
            }
        }
        commit_counts(block_clipping, block_nonfinite);
    }

    out.clipping_events =
        clipping_events.load(std::memory_order_relaxed);
    out.nonfinite_values =
        nonfinite_values.load(std::memory_order_relaxed);
    if (out.nonfinite_values != 0) {
        out.status = scar::Status::InvalidParameter;
        out.failure.index =
            first_failure.load(std::memory_order_relaxed);
        return out;
    }

    for (std::size_t row = 0; row < observations.n_obs; ++row) {
        double sum = 0.0;
        double carry = 0.0;
        double sum2 = 0.0;
        double carry2 = 0.0;
        for (std::size_t tile = 0; tile < dimension_tiles; ++tile) {
            const std::size_t index = row * dimension_tiles + tile;
            neumaier_add(partial_sum[index], sum, carry);
            neumaier_add(partial_sum2[index], sum2, carry2);
        }
        out.sum_z[row] = sum + carry;
        out.sum_z2[row] = sum2 + carry2;
    }
    out.status = scar::Status::Ok;
    return out;
}

MultivariateRowsResult equicorr_log_pdf_and_grad_from_stats(
    const CopulaSpec& spec,
    DoubleView sum_z,
    DoubleView sum_z2,
    const std::vector<double>& correlations,
    int n_threads) {

    MultivariateRowsResult out;
    out.n_threads_requested = n_threads;
    out.log_pdf.assign(
        sum_z.size(), -std::numeric_limits<double>::infinity());
    out.dlog_dr.assign(
        sum_z.size(), std::numeric_limits<double>::quiet_NaN());
    if (spec.family != CopulaFamily::EquicorrGaussian
        || spec.rotation != Rotation::R0
        || spec.transform != Transform::GaussianTanh
        || spec.dim < 2) {
        out.status = scar::Status::InvalidFamily;
        return out;
    }
    if (sum_z.size() == 0 || sum_z.data() == nullptr
        || sum_z2.data() == nullptr || sum_z2.size() != sum_z.size()
        || (correlations.size() != 1
            && correlations.size() != sum_z.size())) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }
    for (std::size_t row = 0; row < sum_z.size(); ++row) {
        if (!std::isfinite(sum_z[row]) || !std::isfinite(sum_z2[row])
            || sum_z2[row] < 0.0) {
            out.status = scar::Status::InvalidParameter;
            out.failure.index = static_cast<std::int64_t>(row);
            return out;
        }
    }

    const bool use_threads = n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            sum_z.size(),
            std::size_t{1},
            static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock),
            kEquicorrGridMinCells);
    std::vector<EquicorrBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(sum_z.size()),
        kEquicorrGridMinRowsPerBlock,
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            auto& block_result = block_results[block];
            block_result.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const double correlation =
                    scar_internal::parameter_at(correlations, row);
                double derivative = 0.0;
                const scar_internal::EquicorrStats stats{
                    sum_z[row], sum_z2[row]};
                const double value =
                    scar_internal::equicorr_log_pdf_from_stats(
                        spec, stats, correlation, &derivative);
                if (!std::isfinite(correlation) || !std::isfinite(value)
                    || !std::isfinite(derivative)) {
                    block_result.failure_index = row_index;
                    return;
                }
                out.log_pdf[row] = value;
                out.dlog_dr[row] = derivative;
            }
        });
    for (const auto& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.row_parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure.index < 0
                || block.failure_index < out.failure.index)) {
            out.failure.index = block.failure_index;
        }
    }
    out.status = out.failure.index >= 0
        ? Status::NumericalFailure : Status::Ok;
    return out;
}

MultivariateGridResult equicorr_pdf_and_grad_grid_from_stats(
    const CopulaSpec& spec,
    DoubleView sum_z,
    DoubleView sum_z2,
    const std::vector<double>& state_grid,
    int n_threads) {

    MultivariateGridResult out;
    out.n_threads_requested = n_threads;
    scar_internal::initialize_multivariate_grid(
        out, sum_z.size(), state_grid.size());
    if (!out.is_ok()) {
        return out;
    }
    if (spec.family != CopulaFamily::EquicorrGaussian
        || spec.rotation != Rotation::R0
        || spec.transform != Transform::GaussianTanh
        || spec.dim < 2) {
        out.status = scar::Status::InvalidFamily;
        return out;
    }
    if (sum_z.size() == 0 || sum_z.data() == nullptr
        || sum_z2.data() == nullptr || sum_z2.size() != sum_z.size()
        || state_grid.empty()) {
        out.status = scar::Status::InvalidSize;
        return out;
    }
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }
    for (std::size_t row = 0; row < sum_z.size(); ++row) {
        if (!std::isfinite(sum_z[row]) || !std::isfinite(sum_z2[row])
            || sum_z2[row] < 0.0) {
            out.status = scar::Status::InvalidParameter;
            out.failure.index = static_cast<std::int64_t>(row);
            return out;
        }
    }
    if (!std::all_of(
            state_grid.begin(), state_grid.end(), [](double value) {
                return std::isfinite(value);
            })) {
        out.status = scar::Status::InvalidParameter;
        return out;
    }

    std::vector<double> correlation_grid(state_grid.size(), 0.0);
    std::vector<double> dpsi_grid(state_grid.size(), 0.0);
    for (std::size_t column = 0; column < state_grid.size(); ++column) {
        correlation_grid[column] =
            scar_internal::copula_transform(spec, state_grid[column]);
        dpsi_grid[column] =
            scar_internal::copula_dtransform(spec, state_grid[column]);
    }
    const bool use_threads = n_threads > 1
        && scar_internal::grid_parallel_worthwhile(
            sum_z.size(),
            state_grid.size(),
            static_cast<std::size_t>(kEquicorrGridMinRowsPerBlock),
            kEquicorrGridMinCells);
    std::vector<EquicorrBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(sum_z.size()),
        kEquicorrGridMinRowsPerBlock,
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            auto& block_result = block_results[block];
            block_result.ran = true;
            for (std::int64_t row_index = begin;
                 row_index < end;
                 ++row_index) {
                const std::size_t row =
                    static_cast<std::size_t>(row_index);
                const std::size_t base = row * state_grid.size();
                const scar_internal::EquicorrStats stats{
                    sum_z[row], sum_z2[row]};
                for (std::size_t column = 0;
                     column < state_grid.size();
                     ++column) {
                    double derivative = 0.0;
                    const double log_pdf =
                        scar_internal::equicorr_log_pdf_from_stats(
                            spec,
                            stats,
                            correlation_grid[column],
                            &derivative);
                    const double pdf = std::exp(log_pdf);
                    out.pdf.values[base + column] = pdf;
                    out.d_pdf_dx.values[base + column] =
                        pdf * derivative * dpsi_grid[column];
                    if (!std::isfinite(out.pdf.values[base + column])
                        || !std::isfinite(
                            out.d_pdf_dx.values[base + column])) {
                        block_result.failure_index = row_index;
                        return;
                    }
                }
            }
        });
    for (const auto& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.equicorr_parallel_blocks;
        if (block.failure_index >= 0
            && (out.failure.index < 0
                || block.failure_index < out.failure.index)) {
            out.failure.index = block.failure_index;
        }
    }
    out.status = out.failure.index >= 0
        ? Status::NumericalFailure : Status::Ok;
    return out;
}

}  // namespace scar
