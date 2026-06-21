#include "common.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_scar_ou_types(py::module_& m) {
    py::class_<scar::OuParams>(m, "OuParams")
        .def(py::init<>())
        .def_readwrite("kappa", &scar::OuParams::kappa)
        .def_readwrite("mu", &scar::OuParams::mu)
        .def_readwrite("nu", &scar::OuParams::nu);

    py::class_<scar::OuNumericalConfig>(m, "OuNumericalConfig")
        .def(py::init<>())
        .def_readwrite("K", &scar::OuNumericalConfig::K)
        .def_readwrite("grid_range", &scar::OuNumericalConfig::grid_range)
        .def_readwrite("adaptive", &scar::OuNumericalConfig::adaptive)
        .def_readwrite("pts_per_sigma", &scar::OuNumericalConfig::pts_per_sigma)
        .def_readwrite("max_K", &scar::OuNumericalConfig::max_K)
        .def_readwrite("r_gh", &scar::OuNumericalConfig::r_gh)
        .def_readwrite("gh_order", &scar::OuNumericalConfig::gh_order)
        .def_readwrite("auto_small_kdt", &scar::OuNumericalConfig::auto_small_kdt)
        .def_readwrite(
            "spectral_basis_order",
            &scar::OuNumericalConfig::spectral_basis_order)
        .def_readwrite(
            "spectral_quad_order",
            &scar::OuNumericalConfig::spectral_quad_order);
}

}  // namespace pyscarcopula::bindings
