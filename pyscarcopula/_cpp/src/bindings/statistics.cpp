#include "array.hpp"
#include "module.hpp"

#include "scar/copula/model_statistics.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

using Int64Array = py::array_t<
    std::int64_t,
    py::array::c_style | py::array::forcecast>;

scar::DoubleView raw_double_view(const Float64Array& values) {
    const py::buffer_info info = values.request();
    return {
        static_cast<const double*>(info.ptr),
        static_cast<std::size_t>(info.size),
    };
}

scar::Span<const std::int64_t> raw_int64_view(const Int64Array& values) {
    const py::buffer_info info = values.request();
    return {
        static_cast<const std::int64_t*>(info.ptr),
        static_cast<std::size_t>(info.size),
    };
}

py::dict double_result(const scar::Result<double>& value) {
    py::dict output;
    output["value"] = value.value;
    output["status"] = static_cast<int>(value.status);
    output["failure_index"] = value.failure.index;
    output["failure_coordinate"] = value.failure.coordinate;
    return output;
}

py::dict bool_result(const scar::Result<bool>& value) {
    py::dict output;
    output["value"] = value.value;
    output["status"] = static_cast<int>(value.status);
    output["failure_index"] = value.failure.index;
    output["failure_coordinate"] = value.failure.coordinate;
    return output;
}

py::dict int64_result(const scar::Result<std::int64_t>& value) {
    py::dict output;
    output["value"] = value.value;
    output["status"] = static_cast<int>(value.status);
    output["failure_index"] = value.failure.index;
    output["failure_coordinate"] = value.failure.coordinate;
    return output;
}

py::dict ranks_result(
    scar::Result<std::vector<std::int64_t>> value) {

    py::dict output;
    output["value"] = py::array_t<std::int64_t>(value.value.size());
    auto array = py::cast<py::array_t<std::int64_t>>(output["value"]);
    std::copy(value.value.begin(), value.value.end(), array.mutable_data());
    output["status"] = static_cast<int>(value.status);
    output["failure_index"] = value.failure.index;
    output["failure_coordinate"] = value.failure.coordinate;
    return output;
}

}  // namespace

void bind_statistics(py::module_& m) {
    py::enum_<scar::InformationCriterion>(m, "InformationCriterion")
        .value("AIC", scar::InformationCriterion::Aic)
        .value("BIC", scar::InformationCriterion::Bic)
        .value(
            "NEGATIVE_LOG_LIKELIHOOD",
            scar::InformationCriterion::NegativeLogLikelihood);

    m.def("statistics_sum_values", [](Float64Array values) {
        return double_result(scar::sum_values(raw_double_view(values)));
    });
    m.def("statistics_sum_int64", [](Int64Array values) {
        return int64_result(scar::sum_int64(raw_int64_view(values)));
    });
    m.def("statistics_sum_absolute", [](Float64Array values) {
        return double_result(scar::sum_absolute(raw_double_view(values)));
    });
    m.def("statistics_add_scores", [](double left, double right) {
        return double_result(scar::add_scores(left, right));
    });
    m.def(
        "statistics_information_criterion",
        [](double log_likelihood,
           std::int64_t parameter_count,
           std::int64_t observation_count,
           scar::InformationCriterion criterion) {
            return double_result(scar::information_criterion(
                log_likelihood,
                parameter_count,
                observation_count,
                criterion));
        },
        py::arg("log_likelihood"),
        py::arg("parameter_count"),
        py::arg("observation_count"),
        py::arg("criterion"));
    m.def("statistics_dense_ranks_no_ties", [](Float64Array values) {
        const py::buffer_info info = values.request();
        if (info.ndim != 1) {
            throw py::value_error("values must be one-dimensional");
        }
        return ranks_result(scar::dense_ranks_no_ties(raw_double_view(values)));
    });
    m.def("statistics_dense_rank_matrix_no_ties", [](Float64Array values) {
        const py::buffer_info info = values.request();
        if (info.ndim != 2) {
            throw py::value_error("values must be two-dimensional");
        }
        auto result = scar::dense_rank_matrix_no_ties({
            static_cast<const double*>(info.ptr),
            static_cast<std::size_t>(info.shape[0]),
            static_cast<int>(info.shape[1]),
        });
        py::dict output;
        output["value"] = py::array_t<std::int64_t>({
            static_cast<py::ssize_t>(result.value.rows),
            static_cast<py::ssize_t>(result.value.columns),
        });
        auto array = py::cast<py::array_t<std::int64_t>>(output["value"]);
        std::copy(
            result.value.values.begin(),
            result.value.values.end(),
            array.mutable_data());
        output["status"] = static_cast<int>(result.status);
        output["failure_index"] = result.failure.index;
        output["failure_coordinate"] = result.failure.coordinate;
        return output;
    });
    m.def(
        "statistics_kendall_tau_from_dense_ranks",
        [](Int64Array first, Int64Array second) {
            const py::buffer_info first_info = first.request();
            const py::buffer_info second_info = second.request();
            if (first_info.ndim != 1 || second_info.ndim != 1) {
                throw py::value_error("dense ranks must be one-dimensional");
            }
            return double_result(scar::kendall_tau_from_dense_ranks(
                raw_int64_view(first), raw_int64_view(second)));
        });
    m.def(
        "statistics_kendall_tau",
        [](Float64Array first, Float64Array second) {
            const py::buffer_info first_info = first.request();
            const py::buffer_info second_info = second.request();
            if (first_info.ndim != 1 || second_info.ndim != 1) {
                throw py::value_error("observations must be one-dimensional");
            }
            return double_result(scar::kendall_tau(
                raw_double_view(first), raw_double_view(second)));
        });
    m.def("statistics_tau_for_itau", [](double tau, bool preserve_sign) {
        return double_result(scar::tau_for_itau(tau, preserve_sign));
    });
    m.def("statistics_rotation_compatible", [](double tau, int rotation) {
        return bool_result(scar::rotation_compatible(tau, rotation));
    });
    m.def("statistics_absolute_below", [](double value, double threshold) {
        return bool_result(scar::absolute_below(value, threshold));
    });
    m.def("statistics_absolute_value", [](double value) {
        return double_result(scar::absolute_value(value));
    });
    m.def("statistics_is_finite", [](double value) {
        return bool_result(scar::is_finite_value(value));
    });
    m.def("statistics_is_nan", [](double value) {
        return bool_result(scar::is_nan_value(value));
    });
}

}  // namespace pyscarcopula::bindings
