#include "common.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// PYBIND11_MODULE expands a third-party variadic macro that triggers
// GCC's zero-variadic-argument pedantic diagnostic under -Werror.
PYBIND11_MODULE(_scar_cpp, module) {
    pyscarcopula::bindings::bind_common(module);
    pyscarcopula::bindings::bind_copula(module);
    pyscarcopula::bindings::bind_multivariate(module);
    pyscarcopula::bindings::bind_scar_ou_types(module);
    pyscarcopula::bindings::bind_gas(module);
    pyscarcopula::bindings::bind_scar_ou(module);
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
