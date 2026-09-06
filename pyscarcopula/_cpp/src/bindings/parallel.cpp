#include "array.hpp"
#include "module.hpp"

#include "scar/detail/parallel.hpp"
#include "scar/detail/linalg.hpp"

#include <pybind11/stl.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

py::dict runtime_info_to_dict(
    const scar_internal::ParallelRuntimeInfo& info) {

    py::dict out;
    out["initialized"] = info.initialized;
    out["owner_pid"] = info.owner_pid;
    out["worker_count"] = info.worker_count;
    out["batches_submitted"] = info.batches_submitted;
    out["worker_start_events"] = info.worker_start_events;
    out["tasks_submitted"] = info.tasks_submitted;
    out["peak_queued_tasks"] = info.peak_queued_tasks;
    return out;
}

}  // namespace

void bind_parallel(py::module_& m) {
    m.def("_linalg_info", []() {
        py::dict out;
        out["selected_backend"] = "portable";
        out["scalar_fallback"] = true;
        out["external_dependencies"] = false;
        out["accumulators"] = 4;
        out["portable_min_elements"] =
            scar_internal::linalg::kPortableMinElements;
        return out;
    });
    m.def(
        "_linalg_matvec_probe",
        [](py::array_t<
               double,
               py::array::c_style | py::array::forcecast> matrix,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> vector,
           bool scalar,
           int repeat) {
            const py::buffer_info matrix_info = matrix.request();
            const py::buffer_info vector_info = vector.request();
            if (matrix_info.ndim != 2
                || vector_info.ndim != 1
                || matrix_info.shape[1] != vector_info.shape[0]
                || repeat < 1) {
                throw std::invalid_argument(
                    "matrix/vector shapes or repeat are invalid");
            }
            const std::size_t rows =
                static_cast<std::size_t>(matrix_info.shape[0]);
            const std::size_t columns =
                static_cast<std::size_t>(matrix_info.shape[1]);
            const double* matrix_data =
                static_cast<const double*>(matrix_info.ptr);
            const double* vector_data =
                static_cast<const double*>(vector_info.ptr);
            for (std::size_t i = 0; i < rows * columns; ++i) {
                if (!std::isfinite(matrix_data[i])) {
                    throw std::invalid_argument("matrix must be finite");
                }
            }
            for (std::size_t i = 0; i < columns; ++i) {
                if (!std::isfinite(vector_data[i])) {
                    throw std::invalid_argument("vector must be finite");
                }
            }
            std::vector<double> out(rows, 0.0);
            const auto backend = scalar
                ? scar_internal::linalg::Backend::Scalar
                : scar_internal::linalg::Backend::Portable;
            {
                py::gil_scoped_release release;
                for (int iteration = 0; iteration < repeat; ++iteration) {
                    scar_internal::linalg::row_major_matvec(
                        matrix_data,
                        rows,
                        columns,
                        vector_data,
                        out.data(),
                        backend);
                }
            }
            return vector_to_array(out);
        },
        py::arg("matrix"),
        py::arg("vector"),
        py::arg("scalar") = false,
        py::arg("repeat") = 1);
    m.def(
        "_linalg_cholesky_solve_probe",
        [](py::array_t<
               double,
               py::array::c_style | py::array::forcecast> matrix,
           py::array_t<
               double,
               py::array::c_style | py::array::forcecast> rhs,
           bool scalar) {
            const py::buffer_info matrix_info = matrix.request();
            const py::buffer_info rhs_info = rhs.request();
            if (matrix_info.ndim != 2
                || matrix_info.shape[0] != matrix_info.shape[1]
                || rhs_info.ndim != 2
                || rhs_info.shape[0] != matrix_info.shape[0]) {
                throw std::invalid_argument(
                    "matrix must be square and rhs must have shape (d, k)");
            }
            const std::size_t dimension =
                static_cast<std::size_t>(matrix_info.shape[0]);
            const std::size_t columns =
                static_cast<std::size_t>(rhs_info.shape[1]);
            const double* matrix_data =
                static_cast<const double*>(matrix_info.ptr);
            const double* rhs_data =
                static_cast<const double*>(rhs_info.ptr);
            const auto backend = scalar
                ? scar_internal::linalg::Backend::Scalar
                : scar_internal::linalg::Backend::Portable;
            std::vector<double> lower;
            std::vector<double> solution;
            bool valid = false;
            {
                py::gil_scoped_release release;
                valid = scar_internal::linalg::cholesky_symmetric_with_jitter(
                    matrix_data,
                    dimension,
                    lower,
                    nullptr,
                    backend);
                if (valid) {
                    valid = scar_internal::linalg::solve_spd(
                        lower.data(),
                        dimension,
                        rhs_data,
                        columns,
                        solution,
                        backend);
                }
            }
            if (!valid) {
                throw std::runtime_error("factorization or solve failed");
            }
            py::array_t<double> lower_out({
                static_cast<py::ssize_t>(dimension),
                static_cast<py::ssize_t>(dimension),
            });
            py::array_t<double> solution_out({
                static_cast<py::ssize_t>(dimension),
                static_cast<py::ssize_t>(columns),
            });
            std::copy(
                lower.begin(), lower.end(),
                static_cast<double*>(lower_out.request().ptr));
            std::copy(
                solution.begin(), solution.end(),
                static_cast<double*>(solution_out.request().ptr));
            return py::make_tuple(lower_out, solution_out);
        },
        py::arg("matrix"),
        py::arg("rhs"),
        py::arg("scalar") = false);
    m.def("_parallel_runtime_info", []() {
        return runtime_info_to_dict(
            scar_internal::parallel_runtime_info());
    });
    m.def("_parallel_runtime_shutdown", []() {
        scar_internal::shutdown_parallel_runtime();
        return runtime_info_to_dict(
            scar_internal::parallel_runtime_info());
    });
    // Python atexit handlers run before the C/C++ runtime starts unloading
    // the extension and its threading support.  The C++ atexit fallback in
    // runtime.cpp remains in place for non-Python callers.
    py::module_::import("atexit").attr("register")(
        m.attr("_parallel_runtime_shutdown"));

    m.def(
        "_parallel_for_blocks_probe",
        [](std::int64_t n_items,
           std::int64_t min_grain,
           int n_threads,
           std::int64_t throw_block,
           int nested_threads) {
            if (n_items < 0) {
                throw std::invalid_argument("n_items must be >= 0");
            }
            std::vector<std::int64_t> block_ids(
                static_cast<std::size_t>(n_items), -1);
            {
                py::gil_scoped_release release;
                scar_internal::parallel_for_blocks(
                    0,
                    n_items,
                    min_grain,
                    n_threads,
                    [&](std::int64_t begin,
                        std::int64_t end,
                        std::size_t block) {
                        if (static_cast<std::int64_t>(block) == throw_block) {
                            throw std::runtime_error(
                                "parallel probe requested failure");
                        }
                        const auto fill = [&](std::int64_t nested_begin,
                                              std::int64_t nested_end,
                                              std::size_t) {
                            for (std::int64_t i = nested_begin;
                                 i < nested_end;
                                 ++i) {
                                block_ids[static_cast<std::size_t>(i)] =
                                    static_cast<std::int64_t>(block);
                            }
                        };
                        if (nested_threads > 0) {
                            scar_internal::parallel_for_blocks(
                                begin, end, 1, nested_threads, fill);
                        } else {
                            fill(begin, end, 0);
                        }
                    });
            }
            py::dict out;
            out["block_ids"] = block_ids;
            out["runtime"] = runtime_info_to_dict(
                scar_internal::parallel_runtime_info());
            return out;
        },
        py::arg("n_items"),
        py::arg("min_grain"),
        py::arg("n_threads"),
        py::arg("throw_block") = -1,
        py::arg("nested_threads") = 0);
    m.def(
        "_parallel_execution_probe",
        [](std::int64_t begin,
           std::int64_t end,
           std::size_t block_count,
           int worker_count) {
            const auto plan = scar_internal::make_parallel_execution_plan(
                begin, end, block_count, worker_count);
            std::vector<std::int64_t> starts(block_count, 0);
            std::vector<std::int64_t> stops(block_count, 0);
            std::vector<std::size_t> slots(
                block_count, std::numeric_limits<std::size_t>::max());
            std::atomic<std::size_t> active{0};
            std::atomic<std::size_t> peak{0};
            std::atomic<std::size_t> completed{0};
            std::atomic<bool> caller_executed{false};
            const auto caller = std::this_thread::get_id();
            {
                py::gil_scoped_release release;
                scar_internal::execute_parallel_plan(
                    plan,
                    [&](std::int64_t block_begin,
                        std::int64_t block_end,
                        const scar_internal::ParallelBlockContext& context) {
                        const std::size_t running = active.fetch_add(1) + 1;
                        std::size_t observed = peak.load();
                        while (observed < running
                               && !peak.compare_exchange_weak(
                                   observed, running)) {
                        }
                        starts[context.block_id] = block_begin;
                        stops[context.block_id] = block_end;
                        slots[context.block_id] = context.worker_slot;
                        if (std::this_thread::get_id() == caller) {
                            caller_executed.store(true);
                        }
                        completed.fetch_add(1);
                        active.fetch_sub(1);
                    });
            }
            py::dict out;
            out["begins"] = starts;
            out["ends"] = stops;
            out["worker_slots"] = slots;
            out["planned_blocks"] = plan.block_count();
            out["completed_blocks"] = completed.load();
            out["peak_active_callbacks"] = peak.load();
            out["caller_executed"] = caller_executed.load();
            out["runtime"] = runtime_info_to_dict(
                scar_internal::parallel_runtime_info());
            return out;
        },
        py::arg("begin"),
        py::arg("end"),
        py::arg("block_count"),
        py::arg("worker_count"));
}

}  // namespace pyscarcopula::bindings
