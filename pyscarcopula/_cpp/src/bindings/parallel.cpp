#include "common.hpp"

#include "scar/detail/parallel.hpp"

#include <cstdint>
#include <stdexcept>
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
    return out;
}

}  // namespace

void bind_parallel(py::module_& m) {
    m.def("_parallel_runtime_info", []() {
        return runtime_info_to_dict(
            scar_internal::parallel_runtime_info());
    });
    m.def("_parallel_runtime_shutdown", []() {
        scar_internal::shutdown_parallel_runtime();
        return runtime_info_to_dict(
            scar_internal::parallel_runtime_info());
    });

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
}

}  // namespace pyscarcopula::bindings
