#include "scar/ou.hpp"

#include "scar/copula/rotation.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/dispatch.hpp"
#include "scar/detail/copula/student.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/factor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

struct TrajectoryBlockResult {
    bool ran = false;
    std::size_t failure_flat_index =
        std::numeric_limits<std::size_t>::max();
};

bool valid_student_spec(const CopulaSpec& spec, std::size_t n_obs) {
    std::size_t square = 0;
    const bool factor_correlation =
        spec.correlation_kind == CorrelationKind::Factor
        && spec.factor_correlation != nullptr
        && spec.factor_correlation->dimension()
            == static_cast<std::size_t>(spec.dim)
        && std::isfinite(spec.factor_correlation->logdet());
    if (spec.dim < 2
        || spec.rotation != Rotation::R0
        || spec.transform != Transform::Softplus
        || !scar_internal::valid_student_dimension(spec.dim, square)
        || (!factor_correlation && spec.l_inv.size() != square)
        || !std::isfinite(
            factor_correlation
                ? spec.factor_correlation->logdet()
                : spec.log_det)) {
        return false;
    }
    if (!spec.ppf_nodes.empty() || !spec.ppf_table.empty()) {
        return spec.ppf_n_obs == static_cast<std::int64_t>(n_obs);
    }
    return true;
}

bool valid_spec(
    const CopulaSpec& spec,
    ObservationView u,
    std::size_t n_trajectories) {

    if (u.empty() || u.data() == nullptr || n_trajectories == 0) {
        return false;
    }
    if (spec.family == CopulaFamily::Student) {
        return u.dim == spec.dim && valid_student_spec(spec, u.size());
    }
    return u.dim == 2 && scar_internal::copula_is_supported(spec);
}

}  // namespace

TrajectoryLogPdfResult copula_log_pdf_trajectory_grid(
    const CopulaSpec& copula,
    ObservationView u,
    const double* latent_paths,
    std::size_t n_trajectories,
    int n_threads) {

    TrajectoryLogPdfResult out;
    out.n_threads_requested = n_threads;
    out.log_pdf.n_obs = static_cast<std::int64_t>(u.size());
    out.log_pdf.n_grid = static_cast<std::int64_t>(n_trajectories);

    std::size_t elements = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), n_trajectories, elements)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    out.log_pdf.values.assign(
        elements, -std::numeric_limits<double>::infinity());
    if (latent_paths == nullptr) {
        out.status = SCAR_NULL_POINTER;
        return out;
    }
    if (!valid_spec(copula, u, n_trajectories)) {
        out.status = SCAR_INVALID_FAMILY;
        return out;
    }
    if (!scar_internal::valid_thread_count(n_threads)
        || elements > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        out.status = SCAR_INVALID_PARAMETER;
        return out;
    }

    const std::size_t observation_stride =
        static_cast<std::size_t>(u.dim);
    constexpr std::size_t min_cells_student = 4096;
    constexpr std::size_t min_cells_other = 262144;
    constexpr std::int64_t min_cells_per_block = 1024;
    const std::size_t min_cells = copula.family == CopulaFamily::Student
        ? min_cells_student
        : min_cells_other;
    const bool use_threads = n_threads > 1 && elements >= min_cells;
    std::vector<TrajectoryBlockResult> block_results(
        static_cast<std::size_t>(use_threads ? n_threads : 1));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(elements),
        min_cells_per_block,
        use_threads ? n_threads : 1,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            TrajectoryBlockResult& block_result = block_results[block];
            block_result.ran = true;
            scar_internal::StudentWorkspace student_workspace;
            if (copula.family == CopulaFamily::Student) {
                student_workspace.reserve_x(observation_stride);
            }
            std::size_t current_t = std::numeric_limits<std::size_t>::max();
            const double* row = nullptr;
            double v1 = 0.0;
            double v2 = 0.0;
            for (std::int64_t flat_index = begin;
                 flat_index < end;
                 ++flat_index) {
                const std::size_t flat =
                    static_cast<std::size_t>(flat_index);
                const std::size_t t = flat / n_trajectories;
                if (t != current_t) {
                    current_t = t;
                    row = u.data() + t * observation_stride;
                    if (copula.family != CopulaFamily::Student) {
                        scar::copula::apply_rotation(
                            row[0],
                            row[1],
                            static_cast<int>(copula.rotation),
                            v1,
                            v2);
                    }
                }
                const double latent = latent_paths[flat];
                const double parameter =
                    scar_internal::copula_transform(copula, latent);
                double value = -std::numeric_limits<double>::infinity();
                if (std::isfinite(parameter)) {
                    value = copula.family == CopulaFamily::Student
                        ? scar_internal::student_log_pdf(
                            copula,
                            row,
                            parameter,
                            static_cast<std::int64_t>(t),
                            student_workspace)
                        : scar_internal::copula_log_pdf_unrotated(
                            copula, v1, v2, parameter);
                }
                if (!std::isfinite(value)) {
                    block_result.failure_flat_index = flat;
                    return;
                }
                out.log_pdf.values[flat] = value;
            }
        });

    std::size_t failure_flat = std::numeric_limits<std::size_t>::max();
    for (const TrajectoryBlockResult& block : block_results) {
        if (!block.ran) {
            continue;
        }
        ++out.parallel_blocks;
        failure_flat = std::min(failure_flat, block.failure_flat_index);
    }
    if (failure_flat != std::numeric_limits<std::size_t>::max()) {
        out.status = SCAR_NUMERICAL_FAILURE;
        out.failure_index = static_cast<std::int64_t>(
            failure_flat / n_trajectories);
        std::fill(
            out.log_pdf.values.begin() + failure_flat,
            out.log_pdf.values.end(),
            -std::numeric_limits<double>::infinity());
    }
    return out;
}

}  // namespace scar
