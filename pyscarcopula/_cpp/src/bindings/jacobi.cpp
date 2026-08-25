#include "module.hpp"

#include "array.hpp"
#include "scar/jacobi.hpp"

#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

template <typename ResultType>
py::dict status_dict(const ResultType& result) {
    py::dict output;
    output["status"] = static_cast<int>(result.status);
    output["failure_index"] = result.failure.index;
    output["failure_row"] = result.failure.row;
    output["failure_coordinate"] = result.failure.coordinate;
    output["failure_operation"] = result.failure.operation;
    return output;
}

py::dict params_to_dict(const scar::JacobiParamsResult& result) {
    py::dict output = status_dict(result);
    output["values"] = py::make_tuple(
        result.value.kappa, result.value.m, result.value.xi);
    return output;
}

py::dict raw_to_dict(const scar::JacobiRawParamsResult& result) {
    py::dict output = status_dict(result);
    output["values"] = result.value;
    return output;
}

py::dict bounds_to_dict(const scar::JacobiRawBoundsResult& result) {
    py::dict output = status_dict(result);
    output["lower"] = result.value.lower;
    output["upper"] = result.value.upper;
    return output;
}

py::dict shape_to_dict(const scar::JacobiShapeResult& result) {
    py::dict output = status_dict(result);
    output["alpha"] = result.value.alpha;
    output["beta"] = result.value.beta;
    output["dalpha"] = result.value.dalpha;
    output["dbeta"] = result.value.dbeta;
    return output;
}

py::dict scalar_to_dict(const scar::JacobiScalarResult& result) {
    py::dict output = status_dict(result);
    output["value"] = result.value;
    return output;
}

py::dict vector_to_dict(const scar::JacobiVectorResult& result) {
    py::dict output = status_dict(result);
    output["values"] = vector_to_array(result.value);
    return output;
}

py::dict memory_to_dict(const scar::JacobiMemoryResult& result) {
    py::dict output = status_dict(result);
    output["elements"] = result.value.elements;
    output["bytes"] = result.value.bytes;
    output["budget_bytes"] = result.value.budget_bytes;
    output["within_budget"] = result.value.within_budget;
    return output;
}

py::dict boundary_to_dict(const scar::JacobiBoundaryResult& result) {
    py::dict output = status_dict(result);
    output["value"] = result.value.value;
    output["intervened"] = result.value.intervened;
    return output;
}

py::dict quadrature_to_dict(const scar::JacobiQuadratureResult& result) {
    py::dict output = status_dict(result);
    output["nodes"] = vector_to_array(result.value.nodes);
    output["weights"] = vector_to_array(result.value.weights);
    return output;
}

py::dict basis_to_dict(const scar::JacobiBasisResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["basis"] = matrix_to_array(
        result.value.basis,
        static_cast<std::size_t>(result.value.quad_order),
        static_cast<std::size_t>(result.value.basis_order));
    output["basis_derivative"] = matrix_to_array(
        result.value.basis_derivative,
        static_cast<std::size_t>(result.value.quad_order),
        static_cast<std::size_t>(result.value.basis_order));
    return output;
}

py::dict fixed_rule_to_dict(const scar::JacobiFixedRuleResult& result) {
    py::dict output = status_dict(result);
    output["tau"] = vector_to_array(result.value.tau);
    output["weights"] = vector_to_array(result.value.weights);
    output["weight_derivatives"] = matrix_to_array(
        result.value.weight_derivatives,
        3,
        static_cast<std::size_t>(result.value.quad_order));
    return output;
}

}  // namespace

void bind_jacobi(py::module_& m) {
    py::enum_<scar::JacobiBoundaryPolicy>(m, "JacobiBoundaryPolicy")
        .value("Reflect", scar::JacobiBoundaryPolicy::Reflect)
        .value("Clip", scar::JacobiBoundaryPolicy::Clip);

    py::class_<scar::JacobiParams>(m, "JacobiParams")
        .def(py::init<>())
        .def_readwrite("kappa", &scar::JacobiParams::kappa)
        .def_readwrite("m", &scar::JacobiParams::m)
        .def_readwrite("xi", &scar::JacobiParams::xi);

    py::class_<scar::JacobiParameterBounds>(m, "JacobiParameterBounds")
        .def(py::init<>())
        .def_readwrite(
            "kappa_lower", &scar::JacobiParameterBounds::kappa_lower)
        .def_readwrite(
            "kappa_upper", &scar::JacobiParameterBounds::kappa_upper)
        .def_readwrite("xi_lower", &scar::JacobiParameterBounds::xi_lower)
        .def_readwrite("xi_upper", &scar::JacobiParameterBounds::xi_upper)
        .def_readwrite("tau_eps", &scar::JacobiParameterBounds::tau_eps);

    py::class_<scar::JacobiNumericalConfig>(m, "JacobiNumericalConfig")
        .def(py::init<>())
        .def_readwrite("quad_order", &scar::JacobiNumericalConfig::quad_order)
        .def_readwrite(
            "basis_order", &scar::JacobiNumericalConfig::basis_order)
        .def_readwrite("gh_order", &scar::JacobiNumericalConfig::gh_order)
        .def_readwrite("n_obs", &scar::JacobiNumericalConfig::n_obs)
        .def_readwrite("matrix", &scar::JacobiNumericalConfig::matrix)
        .def_readwrite("gradient", &scar::JacobiNumericalConfig::gradient)
        .def_readwrite(
            "memory_budget_bytes",
            &scar::JacobiNumericalConfig::memory_budget_bytes)
        .def_readwrite("tau_eps", &scar::JacobiNumericalConfig::tau_eps)
        .def_readwrite("theta_cap", &scar::JacobiNumericalConfig::theta_cap)
        .def_readwrite(
            "stationary_shape_max",
            &scar::JacobiNumericalConfig::stationary_shape_max)
        .def_readwrite(
            "lamperti_eps", &scar::JacobiNumericalConfig::lamperti_eps)
        .def_readwrite("boundary", &scar::JacobiNumericalConfig::boundary);

    m.def("jacobi_raw_to_physical", [](
            const std::array<double, 3>& raw) {
        scar::JacobiParamsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_raw_to_physical(raw); }
        return params_to_dict(result);
    });
    m.def("jacobi_physical_to_raw", [](
            const scar::JacobiParams& params, double tau_eps) {
        scar::JacobiRawParamsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_physical_to_raw(params, tau_eps); }
        return raw_to_dict(result);
    });
    m.def("jacobi_raw_bounds", [](
            const scar::JacobiParameterBounds& bounds) {
        scar::JacobiRawBoundsResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_raw_bounds(bounds); }
        return bounds_to_dict(result);
    });
    m.def("jacobi_stationary_shape", [](
            const scar::JacobiParams& params) {
        scar::JacobiShapeResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_stationary_shape(params); }
        return shape_to_dict(result);
    });
    m.def("jacobi_validate_params", [](
            const scar::JacobiParams& params, double shape_max) {
        int status = 0;
        { py::gil_scoped_release release;
          status = static_cast<int>(
              scar::validate_jacobi_params(params, shape_max)); }
        return status;
    });
    m.def("jacobi_validate_config", [](
            const scar::JacobiNumericalConfig& config) {
        int status = 0;
        { py::gil_scoped_release release;
          status = static_cast<int>(scar::validate_jacobi_config(config)); }
        return status;
    });
    m.def("jacobi_resolve_dt", [](std::int64_t n_obs) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_resolve_dt(n_obs); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_estimate_workspace", [](
            const scar::JacobiNumericalConfig& config) {
        scar::JacobiMemoryResult result;
        { py::gil_scoped_release release;
          result = scar::estimate_jacobi_workspace(config); }
        return memory_to_dict(result);
    });
    m.def("jacobi_estimate_sampling_workspace", [](
            std::int64_t n, const scar::JacobiNumericalConfig& config) {
        scar::JacobiMemoryResult result;
        { py::gil_scoped_release release;
          result = scar::estimate_jacobi_sampling_workspace(n, config); }
        return memory_to_dict(result);
    });
    m.def("jacobi_lamperti_values", [](
            Float64Array tau, double xi) {
        const std::vector<double> values = flat_vector_from_array(tau, "tau");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_lamperti_values(values, xi); }
        return vector_to_dict(result);
    });
    m.def("jacobi_inverse_lamperti_values", [](
            Float64Array values, double xi) {
        const std::vector<double> input =
            flat_vector_from_array(values, "values");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_inverse_lamperti_values(input, xi); }
        return vector_to_dict(result);
    });
    m.def("jacobi_lamperti_drift_values", [](
            const scar::JacobiParams& params,
            Float64Array tau,
            double interior_eps) {
        const std::vector<double> values = flat_vector_from_array(tau, "tau");
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_lamperti_drift_values(
              params, values, interior_eps); }
        return vector_to_dict(result);
    }, py::arg("params"), py::arg("tau"), py::arg("interior_eps") = 0.0);
    m.def("jacobi_apply_boundary", [](
            double value,
            double upper,
            scar::JacobiBoundaryPolicy policy) {
        scar::JacobiBoundaryResult result;
        { py::gil_scoped_release release;
          result = scar::apply_jacobi_boundary(value, upper, policy); }
        return boundary_to_dict(result);
    });
    m.def("jacobi_log_beta", [](double alpha, double beta) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_log_beta(alpha, beta); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_digamma", [](double value) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_digamma(value); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_trigamma", [](double value) {
        scar::JacobiScalarResult result;
        { py::gil_scoped_release release;
          result = scar::jacobi_trigamma(value); }
        return scalar_to_dict(result);
    });
    m.def("jacobi_gauss_hermite_rule", [](
            int order, std::uint64_t memory_budget_bytes) {
        scar::JacobiQuadratureResult result;
        { py::gil_scoped_release release;
          result = scar::gauss_hermite_probability_rule(
              order, memory_budget_bytes); }
        return quadrature_to_dict(result);
    }, py::arg("order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_gauss_jacobi_rule", [](
            double alpha,
            double beta,
            int order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiQuadratureResult result;
        { py::gil_scoped_release release;
          result = scar::gauss_jacobi_probability_rule(
              alpha, beta, order, memory_budget_bytes); }
        return quadrature_to_dict(result);
    }, py::arg("alpha"), py::arg("beta"), py::arg("order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_build_rule", [](
            double alpha,
            double beta,
            int quad_order,
            int basis_order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiBasisResult result;
        { py::gil_scoped_release release;
          result = scar::build_jacobi_rule(
              alpha, beta, quad_order, basis_order, memory_budget_bytes); }
        return basis_to_dict(result);
    }, py::arg("alpha"), py::arg("beta"), py::arg("quad_order"),
       py::arg("basis_order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_build_fixed_rule", [](
            const scar::JacobiParams& params,
            int quad_order,
            std::uint64_t memory_budget_bytes) {
        scar::JacobiFixedRuleResult result;
        { py::gil_scoped_release release;
          result = scar::build_fixed_jacobi_rule(
              params, quad_order, memory_budget_bytes); }
        return fixed_rule_to_dict(result);
    }, py::arg("params"), py::arg("quad_order"),
       py::arg("memory_budget_bytes") =
            scar::kDefaultJacobiMemoryBudgetBytes);
    m.def("jacobi_evaluate_polynomials", [](
            double x,
            double alpha,
            double beta,
            int order,
            bool derivative) {
        scar::JacobiVectorResult result;
        { py::gil_scoped_release release;
          result = scar::evaluate_jacobi_polynomials(
              x, alpha, beta, order, derivative); }
        return vector_to_dict(result);
    }, py::arg("x"), py::arg("alpha"), py::arg("beta"),
       py::arg("order"), py::arg("derivative") = false);
}

}  // namespace pyscarcopula::bindings
