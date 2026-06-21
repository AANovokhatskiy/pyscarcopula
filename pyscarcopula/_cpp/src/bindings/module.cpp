#include "common.hpp"

PYBIND11_MODULE(_scar_cpp, module, pybind11::multiple_interpreters::not_supported()) {
    pyscarcopula::bindings::bind_common(module);
    pyscarcopula::bindings::bind_copula(module);
    pyscarcopula::bindings::bind_multivariate(module);
    pyscarcopula::bindings::bind_scar_ou_types(module);
    pyscarcopula::bindings::bind_gas(module);
    pyscarcopula::bindings::bind_scar_ou(module);
}
