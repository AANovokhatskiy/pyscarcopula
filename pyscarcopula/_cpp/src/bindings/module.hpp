#pragma once

#include <pybind11/pybind11.h>

namespace pyscarcopula::bindings {

void bind_common(pybind11::module_& module);
void bind_validation(pybind11::module_& module);
void bind_statistics(pybind11::module_& module);
void bind_parallel(pybind11::module_& module);
void bind_copula(pybind11::module_& module);
void bind_capability(pybind11::module_& module);
void bind_factor(pybind11::module_& module);
void bind_multivariate(pybind11::module_& module);
void bind_scar_ou_types(pybind11::module_& module);
void bind_jacobi(pybind11::module_& module);
void bind_jacobi_sampling(pybind11::module_& module);
void bind_rvine(pybind11::module_& module);
void bind_gas(pybind11::module_& module);
void bind_scar_ou(pybind11::module_& module);

}  // namespace pyscarcopula::bindings
