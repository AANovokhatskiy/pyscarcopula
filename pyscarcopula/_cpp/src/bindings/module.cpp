#include "module.hpp"

PYBIND11_MODULE(_scar_cpp, module, pybind11::multiple_interpreters::not_supported()) {
    pyscarcopula::bindings::bind_common(module);
    pyscarcopula::bindings::bind_validation(module);
    pyscarcopula::bindings::bind_statistics(module);
    pyscarcopula::bindings::bind_parallel(module);
    pyscarcopula::bindings::bind_factor(module);
    pyscarcopula::bindings::bind_copula(module);
    pyscarcopula::bindings::bind_capability(module);
    pyscarcopula::bindings::bind_multivariate(module);
    pyscarcopula::bindings::bind_scar_ou_types(module);
    pyscarcopula::bindings::bind_jacobi(module);
    pyscarcopula::bindings::bind_jacobi_sampling(module);
    pyscarcopula::bindings::bind_rvine(module);
    pyscarcopula::bindings::bind_gas(module);
    pyscarcopula::bindings::bind_scar_ou(module);
}
