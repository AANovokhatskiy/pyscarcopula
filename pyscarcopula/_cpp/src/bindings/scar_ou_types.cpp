#include "module.hpp"

#include "scar/scar_ou/types.hpp"

namespace py = pybind11;

namespace pyscarcopula::bindings {

void bind_scar_ou_types(py::module_& m) {
    py::enum_<scar::OuBackend>(
        m, "OuBackend", "Numerical SCAR-OU propagation backend.")
        .value("Spectral", scar::OuBackend::Spectral)
        .value("LocalGh", scar::OuBackend::LocalGh)
        .value("Matrix", scar::OuBackend::Matrix);

    py::enum_<scar::OuGridMethod>(
        m, "OuGridMethod", "Native matrix-transition storage.")
        .value("Auto", scar::OuGridMethod::Auto)
        .value("Dense", scar::OuGridMethod::Dense)
        .value("Sparse", scar::OuGridMethod::Sparse);

    py::class_<scar::OuParams>(
        m,
        "OuParams",
        "Parameters of an Ornstein-Uhlenbeck latent process.")
        .def(py::init<>())
        .def_readwrite(
            "kappa", &scar::OuParams::kappa, "Mean-reversion rate.")
        .def_readwrite("mu", &scar::OuParams::mu, "Long-run mean.")
        .def_readwrite("nu", &scar::OuParams::nu, "Diffusion scale.");

    py::class_<scar::OuNumericalConfig>(
        m,
        "OuNumericalConfig",
        "Grid, quadrature, and backend-dispatch settings for SCAR-OU.")
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
            &scar::OuNumericalConfig::spectral_quad_order)
        .def_readwrite("n_threads", &scar::OuNumericalConfig::n_threads)
        .def_readwrite(
            "corr_gradient_block_bytes",
            &scar::OuNumericalConfig::corr_gradient_block_bytes)
        .def_readwrite(
            "grid_method", &scar::OuNumericalConfig::grid_method);
}

}  // namespace pyscarcopula::bindings
