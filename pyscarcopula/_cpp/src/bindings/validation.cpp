#include "array.hpp"
#include "module.hpp"

#include "scar/numerical_validation.hpp"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

template <typename T>
py::array_t<double> pobs_array(py::handle source, scar::RankTies ties) {
    using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
    const Array values = Array::ensure(source);
    if (!values) {
        throw py::type_error("data must be a real numeric array");
    }
    const py::buffer_info info = values.request();
    if (info.ndim != 2) {
        throw py::value_error("data must have shape (n_observations, dimension)");
    }
    std::vector<double> result;
    {
        py::gil_scoped_release release;
        result = scar::pseudo_observations(
            scar::Span<const T>{values.data(), static_cast<std::size_t>(info.size)},
            static_cast<std::size_t>(info.shape[0]),
            static_cast<std::size_t>(info.shape[1]), ties);
    }
    return matrix_to_array(result,
        static_cast<std::size_t>(info.shape[0]),
        static_cast<std::size_t>(info.shape[1]));
}

scar::DoubleView raw_flat_view(const Float64Array& values) {
    const py::buffer_info info = values.request();
    return {
        static_cast<const double*>(info.ptr),
        static_cast<std::size_t>(info.size),
    };
}

py::array_t<double> shaped_array(
    const std::vector<double>& values,
    const py::buffer_info& source) {

    py::array_t<double> output(source.shape);
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::dict numerical_validation_to_dict(
    const scar::NumericalValidationResult& value) {

    py::dict output;
    output["status"] = static_cast<int>(value.status);
    output["code"] = static_cast<int>(value.code);
    output["row"] = value.failure.row;
    output["coordinate"] = value.failure.coordinate;
    return output;
}

py::dict final_fit_to_dict(const scar::FinalFitValidation& value) {
    py::dict output;
    output["reasons"] = value.reasons;
    output["optimizer_abs_tolerance"] = value.optimizer_abs_tolerance;
    output["optimizer_rel_tolerance"] = value.optimizer_rel_tolerance;
    output["has_projected_gradient"] = value.has_projected_gradient;
    output["projected_gradient_norm"] = value.projected_gradient_norm;
    output["projected_gradient_tolerance"] =
        value.projected_gradient_tolerance;
    py::tuple flags(value.boundary_flags.size());
    for (std::size_t index = 0; index < value.boundary_flags.size(); ++index) {
        flags[index] = py::bool_(value.boundary_flags[index] != 0);
    }
    output["boundary_flags"] = std::move(flags);
    output["has_ou_diagnostics"] = value.has_ou_diagnostics;
    output["kappa_dt"] = value.kappa_dt;
    output["rho"] = value.rho;
    output["stationary_std"] = value.stationary_std;
    output["conditional_std"] = value.conditional_std;
    output["has_parameter_growth"] = value.has_parameter_growth;
    output["parameter_growth"] = value.parameter_growth;
    return output;
}

py::dict backend_agreement_to_dict(
    const scar::BackendAgreementValidation& value) {
    py::dict output;
    output["reasons"] = value.reasons;
    output["value"] = value.value;
    output["difference"] = value.difference;
    output["tolerance"] = value.tolerance;
    return output;
}

}  // namespace

void bind_validation(py::module_& m) {
    m.def("validation_pobs", [](py::object source, const std::string& ties_method) {
        if (ties_method != "ordinal" && ties_method != "legacy") {
            throw py::value_error("ties_method must be 'ordinal' or 'legacy'");
        }
        const auto ties = ties_method == "legacy" ? scar::RankTies::Legacy : scar::RankTies::Ordinal;
        const py::array values = py::array::ensure(source);
        if (!values) {
            throw py::type_error("data must be a real numeric array");
        }
        switch (values.dtype().kind()) {
        case 'i': return pobs_array<std::int64_t>(values, ties);
        case 'u': return pobs_array<std::uint64_t>(values, ties);
        case 'b':
        case 'f': return pobs_array<double>(values, ties);
        default: throw py::type_error("data must be a real numeric array");
        }
    }, py::arg("values"), py::arg("ties_method") = "ordinal");

    m.def(
        "validation_objective_is_invalid",
        &scar::objective_is_invalid,
        py::arg("value"));

    m.def(
        "validation_clip_open_unit",
        [](Float64Array values, double epsilon) {
            const py::buffer_info info = values.request();
            const scar::ClipResult outcome = scar::clip_open_unit(
                raw_flat_view(values), epsilon);
            py::dict output;
            output["values"] = shaped_array(outcome.values, info);
            output["status"] = static_cast<int>(outcome.status);
            output["code"] = static_cast<int>(outcome.code);
            output["row"] = outcome.failure.row;
            output["coordinate"] = outcome.failure.coordinate;
            return output;
        },
        py::arg("values"),
        py::arg("epsilon"));

    m.def(
        "validation_open_unit_clip_required",
        [](Float64Array values, double epsilon) {
            const scar::Result<bool> outcome = scar::open_unit_clip_required(
                raw_flat_view(values), epsilon);
            py::dict output;
            output["value"] = outcome.value;
            output["status"] = static_cast<int>(outcome.status);
            output["row"] = outcome.failure.row;
            output["coordinate"] = outcome.failure.coordinate;
            return output;
        },
        py::arg("values"),
        py::arg("epsilon"));

    m.def(
        "validation_validate_fit_data",
        [](Float64Array values) {
            const py::buffer_info info = values.request();
            if (info.ndim != 2) {
                throw py::value_error(
                    "data must have shape (n_observations, dimension)");
            }
            const scar::NumericalValidationResult outcome =
                scar::validate_fit_observations(
                    raw_flat_view(values),
                    static_cast<std::size_t>(info.shape[0]),
                    static_cast<std::size_t>(info.shape[1]));
            return numerical_validation_to_dict(outcome);
        },
        py::arg("values"));

    m.def(
        "validation_validate_equicorr_prepared",
        [](Float64Array sum_z,
           Float64Array sum_z2,
           int dimension,
           double clipping_epsilon) {
            const py::buffer_info first = sum_z.request();
            const py::buffer_info second = sum_z2.request();
            if (first.ndim != 1 || second.ndim != 1
                || first.shape != second.shape) {
                throw py::value_error(
                    "sum_z and sum_z2 must have equal one-dimensional shapes");
            }
            const scar::NumericalValidationResult outcome =
                scar::validate_equicorr_prepared_statistics(
                    raw_flat_view(sum_z),
                    raw_flat_view(sum_z2),
                    dimension,
                    clipping_epsilon);
            return numerical_validation_to_dict(outcome);
        },
        py::arg("sum_z"),
        py::arg("sum_z2"),
        py::arg("dimension"),
        py::arg("clipping_epsilon"));

    m.def(
        "validation_valid_ou_final_parameters",
        [](Float64Array values) {
            return scar::valid_ou_final_parameters(raw_flat_view(values));
        },
        py::arg("values"));

    m.def(
        "validation_validate_ou_final_fit",
        [](Float64Array final_parameters,
           Float64Array initial_parameters,
           Float64Array lower,
           Float64Array upper,
           double optimizer_value,
           double selected_value,
           Float64Array selected_gradient,
           bool selected_evaluation_succeeded,
           const std::string& selected_engine,
           const std::string& selected_error,
           std::int64_t n_obs,
           bool strict_gradient_policy,
           const std::optional<double>& explicit_gradient_tolerance,
           double optimizer_gtol,
           double rho_tolerance,
           double growth_limit) {
            return final_fit_to_dict(scar::validate_ou_final_fit(
                raw_flat_view(final_parameters),
                raw_flat_view(initial_parameters),
                raw_flat_view(lower),
                raw_flat_view(upper),
                optimizer_value,
                selected_value,
                raw_flat_view(selected_gradient),
                selected_evaluation_succeeded,
                selected_engine,
                selected_error,
                n_obs,
                strict_gradient_policy,
                explicit_gradient_tolerance.has_value(),
                explicit_gradient_tolerance.value_or(0.0),
                optimizer_gtol,
                rho_tolerance,
                growth_limit));
        },
        py::arg("final_parameters"),
        py::arg("initial_parameters"),
        py::arg("lower"),
        py::arg("upper"),
        py::arg("optimizer_value"),
        py::arg("selected_value"),
        py::arg("selected_gradient"),
        py::arg("selected_evaluation_succeeded"),
        py::arg("selected_engine"),
        py::arg("selected_error"),
        py::arg("n_obs"),
        py::arg("strict_gradient_policy"),
        py::arg("explicit_gradient_tolerance"),
        py::arg("optimizer_gtol"),
        py::arg("rho_tolerance"),
        py::arg("growth_limit"));

    m.def(
        "validation_validate_backend_agreement",
        [](bool enabled,
           bool evaluation_succeeded,
           const std::string& engine,
           const std::string& error,
           double validation_value,
           double selected_value,
           std::int64_t n_obs,
           double abs_per_observation,
           double relative_tolerance) {
            return backend_agreement_to_dict(
                scar::validate_backend_agreement(
                    enabled,
                    evaluation_succeeded,
                    engine,
                    error,
                    validation_value,
                    selected_value,
                    n_obs,
                    abs_per_observation,
                    relative_tolerance));
        },
        py::arg("enabled"),
        py::arg("evaluation_succeeded"),
        py::arg("engine"),
        py::arg("error"),
        py::arg("validation_value"),
        py::arg("selected_value"),
        py::arg("n_obs"),
        py::arg("abs_per_observation"),
        py::arg("relative_tolerance"));

    m.def(
        "validation_validate_correlation_fit_state",
        [](Float64Array raw_parameters,
           std::size_t expected_parameter_count,
           Float64Array correlation,
           std::size_t dimension,
           Float64Array inverse_factor,
           double log_determinant,
           double tolerance) {
            return scar::validate_correlation_fit_state(
                raw_flat_view(raw_parameters),
                expected_parameter_count,
                raw_flat_view(correlation),
                dimension,
                raw_flat_view(inverse_factor),
                log_determinant,
                tolerance);
        },
        py::arg("raw_parameters"),
        py::arg("expected_parameter_count"),
        py::arg("correlation"),
        py::arg("dimension"),
        py::arg("inverse_factor"),
        py::arg("log_determinant"),
        py::arg("tolerance"));
}

}  // namespace pyscarcopula::bindings
