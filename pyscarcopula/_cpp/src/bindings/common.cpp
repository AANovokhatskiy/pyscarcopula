#include "module.hpp"

#include "scar/detail/safety.hpp"
#include "scar/status.hpp"

#include <exception>
#include <new>

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_common(py::module_& module) {
    module.doc() = "pybind11 bindings for the pyscarcopula SCAR C++ kernels";
#if defined(_MSC_VER)
#define PYSCA_STRINGIFY_DETAIL(value) #value
#define PYSCA_STRINGIFY(value) PYSCA_STRINGIFY_DETAIL(value)
    module.attr("__cpp_compiler__") =
        py::str("MSVC " PYSCA_STRINGIFY(_MSC_VER));
#undef PYSCA_STRINGIFY
#undef PYSCA_STRINGIFY_DETAIL
#elif defined(__clang__)
    module.attr("__cpp_compiler__") = py::str("Clang " __clang_version__);
#elif defined(__GNUC__)
    module.attr("__cpp_compiler__") = py::str("GCC " __VERSION__);
#else
    module.attr("__cpp_compiler__") = py::str("unknown");
#endif

    py::register_exception_translator([](std::exception_ptr exception) {
        if (!exception) {
            return;
        }
        try {
            std::rethrow_exception(exception);
        } catch (const std::bad_alloc&) {
            PyErr_SetString(
                PyExc_MemoryError,
                "C++ SCAR kernel allocation failed after size validation");
        }
    });

    module.attr("MAX_GRID_SIZE") = py::int_(scar_internal::kMaxGridSize);
    module.attr("MAX_DENSE_GRID_SIZE") =
        py::int_(scar_internal::kMaxDenseGridSize);
    module.attr("MAX_SPECTRAL_ORDER") =
        py::int_(scar_internal::kMaxSpectralOrder);
    module.attr("PSEUDO_OBS_EPS") = py::float_(scar_internal::kPseudoObsEps);
    module.attr("H_FUNCTION_EPS") = py::float_(scar_internal::kHEps);
    module.attr("PDF_FLOOR") = py::float_(scar_internal::kPdfEps);
    module.attr("ROSENBLATT_OUTPUT_EPS") =
        py::float_(scar_internal::kRosenblattEps);
    module.attr("HERMITE_RULE_CACHE_MAX_ENTRIES") =
        py::int_(scar_internal::kHermiteRuleCacheMaxEntries);
    module.attr("HERMITE_RULE_CACHE_MAX_BYTES") =
        py::int_(scar_internal::kHermiteRuleCacheMaxBytes);

    module.attr("SCAR_OK") = py::int_(scar::SCAR_OK);
    module.attr("SCAR_NULL_POINTER") = py::int_(scar::SCAR_NULL_POINTER);
    module.attr("SCAR_INVALID_SIZE") = py::int_(scar::SCAR_INVALID_SIZE);
    module.attr("SCAR_INVALID_FAMILY") = py::int_(scar::SCAR_INVALID_FAMILY);
    module.attr("SCAR_INVALID_ROTATION") =
        py::int_(scar::SCAR_INVALID_ROTATION);
    module.attr("SCAR_INVALID_TRANSFORM") =
        py::int_(scar::SCAR_INVALID_TRANSFORM);
    module.attr("SCAR_INVALID_PARAMETER") =
        py::int_(scar::SCAR_INVALID_PARAMETER);
    module.attr("SCAR_NUMERICAL_FAILURE") =
        py::int_(scar::SCAR_NUMERICAL_FAILURE);
}

}  // namespace pyscarcopula::bindings
