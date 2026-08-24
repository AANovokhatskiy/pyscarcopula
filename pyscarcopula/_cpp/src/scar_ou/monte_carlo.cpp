#include "scar/ou.hpp"

#include "scar/copula/prepared_dynamic_emission.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"

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

bool valid_spec(
    const PreparedDynamicEmission& emission,
    ObservationView u,
    std::size_t n_trajectories) {

    if (u.empty() || u.data() == nullptr || n_trajectories == 0) {
        return false;
    }
    return emission.is_supported()
        && ok(emission.validate_observations(u))
        && emission.observation_cache_compatible(u.size());
}

}  // namespace

TrajectoryLogPdfResult copula_log_pdf_trajectory_grid(
    const CopulaSpec& copula,
    ObservationView u,
    const double* latent_paths,
    std::size_t n_trajectories,
    int n_threads) {

    TrajectoryLogPdfResult out;
    const PreparedDynamicEmission emission =
        PreparedDynamicEmission::borrow(copula);
    out.n_threads_requested = n_threads;
    out.log_pdf.n_obs = static_cast<std::int64_t>(u.size());
    out.log_pdf.n_grid = static_cast<std::int64_t>(n_trajectories);

    std::size_t elements = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), n_trajectories, elements)) {
        out.status = Status::InvalidSize;
        return out;
    }
    out.log_pdf.values.assign(
        elements, -std::numeric_limits<double>::infinity());
    if (latent_paths == nullptr) {
        out.status = Status::NullPointer;
        return out;
    }
    if (!valid_spec(emission, u, n_trajectories)) {
        out.status = Status::InvalidFamily;
        return out;
    }
    if (!scar_internal::valid_thread_count(n_threads)
        || elements > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        out.status = Status::InvalidParameter;
        return out;
    }

    const std::size_t observation_stride =
        static_cast<std::size_t>(u.dim);
    constexpr std::size_t min_cells_student = 4096;
    constexpr std::size_t min_cells_other = 262144;
    constexpr std::int64_t min_cells_per_block = 1024;
    const std::size_t min_cells = emission.kind() == DynamicEmissionKind::Student
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
            PreparedDynamicEmissionWorkspace workspace =
                emission.make_workspace(false);
            std::size_t current_t = std::numeric_limits<std::size_t>::max();
            const double* row = nullptr;
            for (std::int64_t flat_index = begin;
                 flat_index < end;
                 ++flat_index) {
                const std::size_t flat =
                    static_cast<std::size_t>(flat_index);
                const std::size_t t = flat / n_trajectories;
                if (t != current_t) {
                    current_t = t;
                    row = u.data() + t * observation_stride;
                }
                const double latent = latent_paths[flat];
                const double parameter =
                    emission.transform_state(latent);
                double value = -std::numeric_limits<double>::infinity();
                if (std::isfinite(parameter)) {
                    const DynamicEmissionRowResult evaluation =
                        emission.evaluate_parameter(
                            row,
                            static_cast<std::int64_t>(t),
                            parameter,
                            false,
                            workspace);
                    if (evaluation.is_ok()) {
                        value = evaluation.log_pdf;
                    }
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
        out.status = Status::NumericalFailure;
        out.failure.index = static_cast<std::int64_t>(
            failure_flat / n_trajectories);
        std::fill(
            out.log_pdf.values.begin() + failure_flat,
            out.log_pdf.values.end(),
            -std::numeric_limits<double>::infinity());
    }
    return out;
}

}  // namespace scar
