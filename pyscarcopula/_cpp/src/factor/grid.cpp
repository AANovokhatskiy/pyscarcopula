#include "scar/factor.hpp"

#include "scar/detail/copula.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scar {
namespace {

constexpr std::size_t kScalarAccumulatorCount = 4;

struct WorkerResult {
    bool ran = false;
    std::int64_t failure_index = -1;
};

void validate_threads(int n_threads) {
    if (n_threads < 1 || n_threads > 256) {
        throw std::invalid_argument("n_threads must be in [1, 256]");
    }
}

bool accumulate_tile(
    const FactorCorrelationOperator& correlation,
    const double* observation,
    double df,
    double marginal_constant,
    double marginal_constant_derivative,
    std::size_t begin,
    std::size_t end,
    double* accumulator) {

    const std::size_t rank = correlation.rank();
    const auto& inverse_uniqueness =
        correlation.inverse_uniqueness();
    const auto& weighted_loadings =
        correlation.weighted_loadings();
    for (std::size_t column = begin; column < end; ++column) {
        if (!std::isfinite(observation[column])) {
            return false;
        }
        double quantile = 0.0;
        double quantile_derivative = 0.0;
        scar_internal::student_quantile_value_and_derivative(
            observation[column],
            df,
            quantile,
            quantile_derivative);
        double marginal_log = 0.0;
        double marginal_dlog = 0.0;
        if (!scar_internal::student_marginal_log_pdf_from_quantile(
                quantile,
                quantile_derivative,
                df,
                marginal_constant,
                marginal_constant_derivative,
                marginal_log,
                marginal_dlog)) {
            return false;
        }

        const double inverse = inverse_uniqueness[column];
        accumulator[0] += inverse * quantile * quantile;
        accumulator[1] +=
            2.0 * inverse * quantile * quantile_derivative;
        accumulator[2] += marginal_log;
        accumulator[3] += marginal_dlog;
        const double* weighted =
            weighted_loadings.data() + column * rank;
        for (std::size_t factor = 0; factor < rank; ++factor) {
            accumulator[
                kScalarAccumulatorCount + factor] +=
                    weighted[factor] * quantile;
            accumulator[
                kScalarAccumulatorCount + rank + factor] +=
                    weighted[factor] * quantile_derivative;
        }
    }
    return true;
}

bool finish_cell(
    const FactorCorrelationOperator& correlation,
    double df,
    const std::vector<double>& accumulator,
    std::vector<double>& solved_projection,
    double& log_pdf,
    double& dlog_ddf) {

    const std::size_t rank = correlation.rank();
    std::copy(
        accumulator.begin() + kScalarAccumulatorCount,
        accumulator.begin() + kScalarAccumulatorCount + rank,
        solved_projection.begin());
    correlation.solve_core_inplace(solved_projection.data());
    double correction = 0.0;
    double derivative_correction = 0.0;
    for (std::size_t factor = 0; factor < rank; ++factor) {
        correction +=
            accumulator[kScalarAccumulatorCount + factor]
            * solved_projection[factor];
        derivative_correction +=
            accumulator[kScalarAccumulatorCount + rank + factor]
            * solved_projection[factor];
    }
    double quadratic_form = accumulator[0] - correction;
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon()
        * (1.0 + std::abs(accumulator[0]));
    if (quadratic_form < 0.0 && quadratic_form >= -tolerance) {
        quadratic_form = 0.0;
    }
    const double quadratic_form_derivative =
        accumulator[1] - 2.0 * derivative_correction;
    return scar_internal::student_log_pdf_from_summaries(
        correlation.dimension(),
        df,
        correlation.logdet(),
        quadratic_form,
        quadratic_form_derivative,
        accumulator[2],
        accumulator[3],
        log_pdf,
        &dlog_ddf);
}

bool evaluate_cell_sequential_tiles(
    const FactorCorrelationOperator& correlation,
    const double* observation,
    double df,
    std::size_t dimension_tile,
    std::vector<double>& accumulator,
    std::vector<double>& tile_accumulator,
    std::vector<double>& solved_projection,
    double& log_pdf,
    double& dlog_ddf) {

    double marginal_constant = 0.0;
    double marginal_constant_derivative = 0.0;
    if (!scar_internal::student_marginal_log_pdf_constants(
            df,
            marginal_constant,
            marginal_constant_derivative)) {
        return false;
    }
    std::fill(accumulator.begin(), accumulator.end(), 0.0);
    const std::size_t dimension = correlation.dimension();
    for (std::size_t begin = 0;
         begin < dimension;
         begin += dimension_tile) {
        const std::size_t end =
            std::min(dimension, begin + dimension_tile);
        std::fill(
            tile_accumulator.begin(),
            tile_accumulator.end(),
            0.0);
        if (!accumulate_tile(
                correlation,
                observation,
                df,
                marginal_constant,
                marginal_constant_derivative,
                begin,
                end,
                tile_accumulator.data())) {
            return false;
        }
        for (std::size_t index = 0;
             index < accumulator.size();
             ++index) {
            accumulator[index] += tile_accumulator[index];
        }
    }
    return finish_cell(
        correlation,
        df,
        accumulator,
        solved_projection,
        log_pdf,
        dlog_ddf);
}

}  // namespace

FactorStudentGridResult factor_student_log_pdf_and_dlog_ddf_grid(
    const FactorCorrelationOperator& correlation,
    const double* observations,
    std::size_t rows,
    const double* df_grid,
    std::size_t grid_size,
    std::size_t dimension_tile,
    int n_threads) {

    validate_threads(n_threads);
    if (observations == nullptr
        || rows == 0
        || df_grid == nullptr
        || grid_size == 0
        || dimension_tile == 0) {
        throw std::invalid_argument(
            "factor Student grid inputs must be non-empty");
    }
    for (std::size_t grid = 0; grid < grid_size; ++grid) {
        if (!std::isfinite(df_grid[grid]) || df_grid[grid] <= 2.0) {
            throw std::invalid_argument(
                "Student df grid must be finite and greater than 2");
        }
    }

    const std::size_t dimension = correlation.dimension();
    const std::size_t rank = correlation.rank();
    std::size_t cells = 0;
    std::size_t input_values = 0;
    std::size_t ppf_values = 0;
    std::size_t accumulator_width = 0;
    std::size_t double_rank = 0;
    if (!scar_internal::checked_size_mul(rows, grid_size, cells)
        || !scar_internal::checked_size_mul(
            rows, dimension, input_values)
        || !scar_internal::checked_size_mul(
            input_values, grid_size, ppf_values)
        || !scar_internal::checked_size_mul(
            rank, std::size_t{2}, double_rank)
        || !scar_internal::checked_size_add(
            kScalarAccumulatorCount,
            double_rank,
            accumulator_width)
        || cells
            > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            "factor Student grid shape is not representable");
    }
    const std::size_t dimension_tiles =
        dimension / dimension_tile
        + static_cast<std::size_t>(
            dimension % dimension_tile != 0);

    FactorStudentGridResult result;
    result.rows = rows;
    result.grid_size = grid_size;
    result.dimension_tiles = dimension_tiles;
    result.n_threads_requested = n_threads;
    result.log_pdf.assign(
        cells, std::numeric_limits<double>::quiet_NaN());
    result.dlog_ddf.assign(
        cells, std::numeric_limits<double>::quiet_NaN());
    result.ppf_exact_values =
        static_cast<std::uint64_t>(ppf_values);

    const bool use_cell_parallelism =
        n_threads > 1
        && cells >= 4 * static_cast<std::size_t>(n_threads);
    const bool use_dimension_parallelism =
        n_threads > 1
        && !use_cell_parallelism
        && dimension_tiles >= static_cast<std::size_t>(n_threads);
    result.parallel_axis = use_cell_parallelism
        ? 1
        : (use_dimension_parallelism ? 2 : 0);

    std::size_t double_accumulator_width = 0;
    std::size_t worker_values = 0;
    if (!scar_internal::checked_size_mul(
            accumulator_width,
            std::size_t{2},
            double_accumulator_width)
        || !scar_internal::checked_size_add(
            double_accumulator_width,
            rank,
            worker_values)
        || !scar_internal::checked_size_mul(
            worker_values,
            sizeof(double),
            result.worker_workspace_peak_bytes)) {
        throw std::invalid_argument(
            "factor Student worker workspace is not representable");
    }

    if (use_dimension_parallelism) {
        std::size_t partial_values = 0;
        if (!scar_internal::checked_size_mul(
                dimension_tiles,
                accumulator_width,
                partial_values)) {
            throw std::invalid_argument(
                "factor Student partial workspace is not representable");
        }
        std::size_t partial_double_bytes = 0;
        if (!scar_internal::checked_size_mul(
                partial_values,
                sizeof(double),
                partial_double_bytes)
            || !scar_internal::checked_size_add(
                partial_double_bytes,
                dimension_tiles * sizeof(unsigned char),
                result.partial_workspace_peak_bytes)) {
            throw std::invalid_argument(
                "factor Student partial workspace is not representable");
        }
        result.parallel_blocks = n_threads;
        std::vector<double> partials(partial_values, 0.0);
        std::vector<double> accumulator(accumulator_width, 0.0);
        std::vector<double> solved_projection(rank, 0.0);
        std::vector<unsigned char> tile_ok(dimension_tiles, 1);
        for (std::size_t cell = 0; cell < cells; ++cell) {
            std::fill(partials.begin(), partials.end(), 0.0);
            std::fill(
                tile_ok.begin(),
                tile_ok.end(),
                static_cast<unsigned char>(1));
            const std::size_t row = cell / grid_size;
            const std::size_t grid = cell % grid_size;
            const double row_df = df_grid[grid];
            double marginal_constant = 0.0;
            double marginal_constant_derivative = 0.0;
            if (!scar_internal::student_marginal_log_pdf_constants(
                    row_df,
                    marginal_constant,
                    marginal_constant_derivative)) {
                result.failure_index =
                    static_cast<std::int64_t>(cell);
                return result;
            }
            scar_internal::parallel_for_blocks(
                0,
                static_cast<std::int64_t>(dimension_tiles),
                1,
                n_threads,
                [&](std::int64_t begin,
                    std::int64_t end,
                    std::size_t) {
                    for (std::int64_t tile_index = begin;
                         tile_index < end;
                         ++tile_index) {
                        const std::size_t tile =
                            static_cast<std::size_t>(tile_index);
                        const std::size_t dimension_begin =
                            tile * dimension_tile;
                        const std::size_t dimension_end = std::min(
                            dimension,
                            dimension_begin + dimension_tile);
                        if (!accumulate_tile(
                                correlation,
                                observations + row * dimension,
                                row_df,
                                marginal_constant,
                                marginal_constant_derivative,
                                dimension_begin,
                                dimension_end,
                                partials.data()
                                    + tile * accumulator_width)) {
                            tile_ok[tile] = 0;
                        }
                    }
                });
            std::fill(accumulator.begin(), accumulator.end(), 0.0);
            for (std::size_t tile = 0;
                 tile < dimension_tiles;
                 ++tile) {
                if (tile_ok[tile] == 0) {
                    result.failure_index =
                        static_cast<std::int64_t>(cell);
                    return result;
                }
                const double* partial =
                    partials.data() + tile * accumulator_width;
                for (std::size_t index = 0;
                     index < accumulator_width;
                     ++index) {
                    accumulator[index] += partial[index];
                }
            }
            if (!finish_cell(
                    correlation,
                    row_df,
                    accumulator,
                    solved_projection,
                    result.log_pdf[cell],
                    result.dlog_ddf[cell])) {
                result.failure_index =
                    static_cast<std::int64_t>(cell);
                return result;
            }
        }
        return result;
    }

    const int threads = use_cell_parallelism ? n_threads : 1;
    std::vector<WorkerResult> workers(
        static_cast<std::size_t>(threads));
    scar_internal::parallel_for_blocks(
        0,
        static_cast<std::int64_t>(cells),
        1,
        threads,
        [&](std::int64_t begin,
            std::int64_t end,
            std::size_t block) {
            WorkerResult& worker = workers[block];
            worker.ran = true;
            std::vector<double> accumulator(
                accumulator_width, 0.0);
            std::vector<double> tile_accumulator(
                accumulator_width, 0.0);
            std::vector<double> solved_projection(rank, 0.0);
            for (std::int64_t cell_index = begin;
                 cell_index < end;
                 ++cell_index) {
                const std::size_t cell =
                    static_cast<std::size_t>(cell_index);
                const std::size_t row = cell / grid_size;
                const std::size_t grid = cell % grid_size;
                if (!evaluate_cell_sequential_tiles(
                        correlation,
                        observations + row * dimension,
                        df_grid[grid],
                        dimension_tile,
                        accumulator,
                        tile_accumulator,
                        solved_projection,
                        result.log_pdf[cell],
                        result.dlog_ddf[cell])) {
                    worker.failure_index = cell_index;
                    return;
                }
            }
        });
    for (const WorkerResult& worker : workers) {
        if (!worker.ran) {
            continue;
        }
        ++result.parallel_blocks;
        if (worker.failure_index >= 0
            && (result.failure_index < 0
                || worker.failure_index < result.failure_index)) {
            result.failure_index = worker.failure_index;
        }
    }
    return result;
}

}  // namespace scar
