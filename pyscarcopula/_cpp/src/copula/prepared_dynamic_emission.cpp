#include "scar/copula/prepared_dynamic_emission.hpp"

#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/copula/multivariate/correlation/parameterization.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/copula/pair/gaussian.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/dispatch.hpp"
#include "scar/detail/safety.hpp"
#include "scar/status.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace scar {

Result<CopulaSpec> prepare_shrinkage_dynamic_spec(
    const CopulaSpec& template_spec,
    DoubleView base_correlation,
    double raw_shrinkage) {

    Result<CopulaSpec> result;
    if (template_spec.family != CopulaFamily::Student
        || template_spec.dim < 2
        || !std::isfinite(raw_shrinkage)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    const std::size_t dimension =
        static_cast<std::size_t>(template_spec.dim);
    const auto correlation = make_shrinkage_correlation(
        raw_shrinkage, base_correlation, dimension);
    if (!correlation.is_ok()) {
        result.status = correlation.status;
        result.failure = correlation.failure;
        return result;
    }
    const auto prepared = prepare_dense_correlation(
        {correlation.value.data(), correlation.value.size()}, dimension);
    if (!prepared.is_ok()) {
        result.status = prepared.status;
        result.failure = prepared.failure;
        return result;
    }
    result.value = template_spec;
    result.value.correlation_kind = CorrelationKind::Shrinkage;
    // Fixed and shrinkage correlations use the same dense storage. Resetting
    // it also discards the Student PPF interpolation table, making the joint
    // objective differ from the reported/filter likelihood (and much slower).
    if (template_spec.correlation_kind == CorrelationKind::Factor) {
        result.value.reset_model_storage();
    }
    result.value.dense_inverse_cholesky() = prepared.inverse_cholesky;
    result.value.dense_log_determinant() = prepared.log_determinant;
    return result;
}

struct PreparedDynamicEmissionWorkspace::Impl {
    scar_internal::StudentWorkspace student;
};

PreparedDynamicEmissionWorkspace::PreparedDynamicEmissionWorkspace()
    : impl_(std::make_unique<Impl>()) {}

PreparedDynamicEmissionWorkspace::~PreparedDynamicEmissionWorkspace() = default;
PreparedDynamicEmissionWorkspace::PreparedDynamicEmissionWorkspace(
    PreparedDynamicEmissionWorkspace&&) noexcept = default;
PreparedDynamicEmissionWorkspace&
PreparedDynamicEmissionWorkspace::operator=(
    PreparedDynamicEmissionWorkspace&&) noexcept = default;

struct PreparedDynamicEmission::Impl {
    explicit Impl(const CopulaSpec& source, bool borrow)
        : owned_spec(borrow ? nullptr : std::make_unique<CopulaSpec>(source)),
          spec(borrow ? &source : owned_spec.get()),
          pair(*spec) {
        resolve();
    }

    void resolve() {
        pair = PreparedPairKernel(*spec);
        if (pair.is_registered()) {
            kind = DynamicEmissionKind::Pair;
            supported = pair.is_supported() && spec->dim == 2;
            student = {};
            return;
        }
        if (spec->family == CopulaFamily::Student) {
            kind = DynamicEmissionKind::Student;
            student = scar_internal::prepare_student_density(*spec);
            supported = student.valid
                && scar_internal::copula_is_supported(*spec);
            return;
        }
        if (spec->family == CopulaFamily::EquicorrGaussian) {
            kind = DynamicEmissionKind::Equicorrelation;
            supported = scar_internal::copula_is_supported(*spec);
            student = {};
            return;
        }
        kind = DynamicEmissionKind::Unsupported;
        supported = false;
        student = {};
    }

    void replace_owned(const CopulaSpec& source) {
        owned_spec = std::make_unique<CopulaSpec>(source);
        spec = owned_spec.get();
        resolve();
    }

    std::unique_ptr<CopulaSpec> owned_spec;
    const CopulaSpec* spec = nullptr;
    PreparedPairKernel pair;
    scar_internal::PreparedStudentDensity student;
    DynamicEmissionKind kind = DynamicEmissionKind::Unsupported;
    bool supported = false;
};

PreparedDynamicEmission::PreparedDynamicEmission(const CopulaSpec& spec)
    : impl_(std::make_unique<Impl>(spec, false)) {}

PreparedDynamicEmission::PreparedDynamicEmission(
    const CopulaSpec& spec,
    BorrowedSpecTag)
    : impl_(std::make_unique<Impl>(spec, true)) {}

PreparedDynamicEmission PreparedDynamicEmission::borrow(
    const CopulaSpec& spec) {
    return PreparedDynamicEmission(spec, BorrowedSpecTag{});
}

PreparedDynamicEmission::~PreparedDynamicEmission() = default;
PreparedDynamicEmission::PreparedDynamicEmission(
    PreparedDynamicEmission&&) noexcept = default;
PreparedDynamicEmission& PreparedDynamicEmission::operator=(
    PreparedDynamicEmission&&) noexcept = default;

void PreparedDynamicEmission::refresh(const CopulaSpec& spec) {
    impl_->replace_owned(spec);
}

DynamicEmissionKind PreparedDynamicEmission::kind() const noexcept {
    return impl_->kind;
}

CopulaFamily PreparedDynamicEmission::family() const noexcept {
    return impl_->spec->family;
}

CorrelationKind PreparedDynamicEmission::correlation_kind() const noexcept {
    return impl_->spec->correlation_kind;
}

int PreparedDynamicEmission::expected_dimension() const noexcept {
    return impl_->spec->model_descriptor().expected_dimension();
}

bool PreparedDynamicEmission::is_supported() const noexcept {
    return impl_->supported;
}

bool PreparedDynamicEmission::is_supported_for_ou() const noexcept {
    return impl_->supported
        && scar_internal::copula_is_supported_for_ou(*impl_->spec);
}

bool PreparedDynamicEmission::is_independent() const noexcept {
    return family() == CopulaFamily::Independent;
}

bool PreparedDynamicEmission::is_unrotated_gaussian_pair() const noexcept {
    return impl_->kind == DynamicEmissionKind::Pair
        && impl_->pair.is_unrotated_gaussian();
}

bool PreparedDynamicEmission::has_cached_observations(
    std::size_t rows) const noexcept {
    if (impl_->kind == DynamicEmissionKind::Equicorrelation) {
        return impl_->spec->equicorr_sum_scores().size() == rows
            && impl_->spec->equicorr_sum_squares().size() == rows;
    }
    if (family() == CopulaFamily::Gaussian) {
        return impl_->spec->pair_gaussian_first_scores().size() == rows
            && impl_->spec->pair_gaussian_second_scores().size() == rows;
    }
    return false;
}

bool PreparedDynamicEmission::observation_cache_compatible(
    std::size_t rows) const noexcept {

    if (impl_->kind == DynamicEmissionKind::Student) {
        const auto& nodes = impl_->spec->student_ppf_nodes();
        const auto& table = impl_->spec->student_ppf_table();
        if (rows > static_cast<std::size_t>(
                       std::numeric_limits<std::int64_t>::max())) {
            return false;
        }
        return (nodes.empty() && table.empty())
            || impl_->spec->student_ppf_observation_count()
                == static_cast<std::int64_t>(rows);
    }
    if (impl_->kind == DynamicEmissionKind::Equicorrelation) {
        const auto& sums = impl_->spec->equicorr_sum_scores();
        const auto& squares = impl_->spec->equicorr_sum_squares();
        return (sums.empty() && squares.empty())
            || (sums.size() == rows && squares.size() == rows);
    }
    if (is_unrotated_gaussian_pair()) {
        const auto& first = impl_->spec->pair_gaussian_first_scores();
        const auto& second = impl_->spec->pair_gaussian_second_scores();
        return (first.empty() && second.empty())
            || (first.size() == rows && second.size() == rows);
    }
    return true;
}

double PreparedDynamicEmission::h_from_cached_observation(
    std::size_t row,
    bool reverse,
    double parameter) const {

    if (!is_unrotated_gaussian_pair()
        || row >= impl_->spec->pair_gaussian_first_scores().size()
        || impl_->spec->pair_gaussian_first_scores().size()
            != impl_->spec->pair_gaussian_second_scores().size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double first = impl_->spec->pair_gaussian_first_scores()[row];
    const double second = impl_->spec->pair_gaussian_second_scores()[row];
    return reverse
        ? copula::pair::gaussian_h_from_quantiles(
            first, second, parameter)
        : copula::pair::gaussian_h_from_quantiles(
            second, first, parameter);
}

const CopulaSpec& PreparedDynamicEmission::compatibility_spec() const noexcept {
    return *impl_->spec;
}

Status PreparedDynamicEmission::validate_observations(
    ObservationView observations,
    bool require_nonempty) const {

    if (!impl_->supported) {
        return Status::InvalidFamily;
    }
    if (observations.dim != expected_dimension()
        || (require_nonempty && observations.empty())) {
        return Status::InvalidSize;
    }
    const bool cached = has_cached_observations(observations.size());
    if (!observations.empty() && observations.data() == nullptr && !cached) {
        return Status::NullPointer;
    }
    if (impl_->kind == DynamicEmissionKind::Equicorrelation && cached) {
        const auto& sums = impl_->spec->equicorr_sum_scores();
        const auto& squares = impl_->spec->equicorr_sum_squares();
        for (std::size_t row = 0; row < observations.size(); ++row) {
            if (!std::isfinite(sums[row])
                || !std::isfinite(squares[row])
                || squares[row] < 0.0) {
                return Status::InvalidParameter;
            }
        }
        return Status::Ok;
    }
    std::size_t values = 0;
    if (!scar_internal::checked_shape_size(
            observations.size(),
            static_cast<std::size_t>(observations.dim),
            values)) {
        return Status::InvalidSize;
    }
    for (std::size_t index = 0; index < values; ++index) {
        if (!std::isfinite(observations.data()[index])) {
            return Status::InvalidParameter;
        }
    }
    return Status::Ok;
}

PreparedDynamicEmissionWorkspace PreparedDynamicEmission::make_workspace(
    bool derivative) const {

    PreparedDynamicEmissionWorkspace workspace;
    if (impl_->kind == DynamicEmissionKind::Student) {
        const std::size_t dimension =
            static_cast<std::size_t>(expected_dimension());
        workspace.impl_->student.reserve_x(dimension);
        if (derivative) {
            workspace.impl_->student.reserve_dx_ddf(dimension);
        }
    }
    return workspace;
}

double PreparedDynamicEmission::transform_state(double state) const {
    if (!impl_->supported) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (impl_->kind == DynamicEmissionKind::Pair) {
        return impl_->pair.transform(state);
    }
    if (impl_->kind == DynamicEmissionKind::Equicorrelation) {
        return scar_internal::equicorr_transform(*impl_->spec, state);
    }
    return scar_internal::copula_transform(*impl_->spec, state);
}

double PreparedDynamicEmission::dtransform_state(double state) const {
    if (!impl_->supported) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (impl_->kind == DynamicEmissionKind::Pair) {
        return impl_->pair.dtransform(state);
    }
    if (impl_->kind == DynamicEmissionKind::Equicorrelation) {
        return scar_internal::equicorr_dtransform(*impl_->spec, state);
    }
    return scar_internal::copula_dtransform(*impl_->spec, state);
}

DynamicEmissionRowResult PreparedDynamicEmission::evaluate_parameter(
    const double* row,
    std::int64_t row_index,
    double parameter,
    bool derivative,
    PreparedDynamicEmissionWorkspace& workspace) const {

    DynamicEmissionRowResult out;
    out.parameter = parameter;
    if (!impl_->supported || !std::isfinite(parameter)) {
        out.status = impl_->supported
            ? Status::InvalidParameter : Status::InvalidFamily;
        return out;
    }
    if (impl_->kind == DynamicEmissionKind::Pair) {
        if (row == nullptr) {
            out.status = Status::NullPointer;
            return out;
        }
        out.log_pdf = impl_->pair.log_pdf(row[0], row[1], parameter);
        if (derivative) {
            out.dlog_dparameter = impl_->pair.dlog_pdf_dparameter(
                row[0], row[1], parameter);
        }
    } else if (impl_->kind == DynamicEmissionKind::Student) {
        if (row == nullptr) {
            out.status = Status::NullPointer;
            return out;
        }
        if (derivative) {
            if (!scar_internal::student_log_pdf_and_dlog_ddf(
                    impl_->student,
                    row,
                    parameter,
                    row_index,
                    out.log_pdf,
                    out.dlog_dparameter,
                    workspace.impl_->student)) {
                out.status = Status::NumericalFailure;
                out.failure.index = row_index;
                return out;
            }
        } else {
            out.log_pdf = scar_internal::student_log_pdf(
                impl_->student,
                row,
                parameter,
                row_index,
                workspace.impl_->student);
        }
    } else if (impl_->kind == DynamicEmissionKind::Equicorrelation) {
        scar_internal::EquicorrStats stats;
        const bool cached = row_index >= 0
            && static_cast<std::size_t>(row_index)
                < impl_->spec->equicorr_sum_scores().size()
            && impl_->spec->equicorr_sum_scores().size()
                == impl_->spec->equicorr_sum_squares().size();
        if (cached) {
            const std::size_t index = static_cast<std::size_t>(row_index);
            stats.sum = impl_->spec->equicorr_sum_scores()[index];
            stats.sum_squares =
                impl_->spec->equicorr_sum_squares()[index];
        } else if (!scar_internal::equicorr_sufficient_statistics(
                       *impl_->spec, row, stats)) {
            out.status = Status::NumericalFailure;
            out.failure.index = row_index;
            return out;
        }
        out.log_pdf = scar_internal::equicorr_log_pdf_from_stats(
            *impl_->spec,
            stats,
            parameter,
            derivative ? &out.dlog_dparameter : nullptr);
    } else {
        out.status = Status::InvalidFamily;
        return out;
    }
    if (!std::isfinite(out.log_pdf)
        || (derivative && !std::isfinite(out.dlog_dparameter))) {
        out.status = Status::NumericalFailure;
        out.failure.index = row_index;
    }
    return out;
}

DynamicEmissionRowResult PreparedDynamicEmission::evaluate_state(
    const double* row,
    std::int64_t row_index,
    double state,
    bool derivative,
    PreparedDynamicEmissionWorkspace& workspace) const {

    const double parameter = transform_state(state);
    DynamicEmissionRowResult out = evaluate_parameter(
        row, row_index, parameter, derivative, workspace);
    out.parameter = parameter;
    return out;
}

double PreparedDynamicEmission::log_pdf_at_state(
    const double* row,
    std::int64_t row_index,
    double state,
    PreparedDynamicEmissionWorkspace& workspace) const {

    const DynamicEmissionRowResult result = evaluate_state(
        row, row_index, state, false, workspace);
    return result.is_ok()
        ? result.log_pdf
        : std::numeric_limits<double>::quiet_NaN();
}

void PreparedDynamicEmission::prepare_grid_transform(
    const std::vector<double>& states,
    std::vector<double>& parameters,
    std::vector<double>& derivatives) const {

    scar_internal::copula_prepare_grid_transform(
        *impl_->spec, states, parameters, derivatives);
}

void PreparedDynamicEmission::fill_density_row(
    const double* observations,
    std::int64_t row_index,
    const std::vector<double>& parameters,
    double* densities,
    double* log_scale) const {

    scar_internal::copula_pdf_row_precomputed_flat(
        *impl_->spec,
        observations,
        row_index,
        parameters,
        densities,
        log_scale);
}

void PreparedDynamicEmission::fill_density_and_gradient_row(
    const double* observations,
    std::int64_t row_index,
    const std::vector<double>& parameters,
    const std::vector<double>& derivatives,
    double* densities,
    double* gradients,
    double* log_scale) const {

    scar_internal::copula_pdf_and_grad_row_precomputed_flat(
        *impl_->spec,
        observations,
        row_index,
        parameters,
        derivatives,
        densities,
        gradients,
        log_scale);
}

void PreparedDynamicEmission::fill_density_and_gradient_grid(
    const double* observations,
    std::int64_t rows,
    const std::vector<double>& parameters,
    const std::vector<double>& derivatives,
    std::vector<double>& densities,
    std::vector<double>& gradients,
    int n_threads,
    double* log_scale_sum) const {

    scar_internal::copula_pdf_and_grad_grid_precomputed(
        *impl_->spec,
        observations,
        rows,
        parameters,
        derivatives,
        densities,
        gradients,
        n_threads,
        log_scale_sum);
}

bool PreparedDynamicEmission::fill_density_and_gradient_block(
    const double* observations,
    std::int64_t first_row,
    std::int64_t rows,
    const std::vector<double>& parameters,
    const std::vector<double>& derivatives,
    std::vector<double>& densities,
    std::vector<double>& gradients,
    std::vector<double>& row_log_scales,
    int n_threads) const {

    std::size_t elements = 0;
    if (first_row < 0 || rows <= 0 || parameters.empty()
        || rows > std::numeric_limits<std::int64_t>::max() - first_row
        || parameters.size() != derivatives.size()
        || !scar_internal::checked_size_mul(
            static_cast<std::size_t>(rows), parameters.size(), elements)) {
        return false;
    }
    row_log_scales.assign(static_cast<std::size_t>(rows), 0.0);
    scar_internal::copula_pdf_and_grad_grid_precomputed(
        *impl_->spec, observations, rows, parameters, derivatives,
        densities, gradients, n_threads, nullptr, first_row,
        row_log_scales.data());
    return densities.size() == elements && gradients.size() == elements
        && std::all_of(
            row_log_scales.begin(), row_log_scales.end(),
            [](double value) { return std::isfinite(value); });
}

void PreparedDynamicEmission::fill_density_row_on_state_grid(
    const double* observations,
    std::int64_t row_index,
    const std::vector<double>& states,
    std::vector<double>& densities) const {

    scar_internal::copula_fi_row_on_grid(
        *impl_->spec, observations, row_index, states, densities);
}

double PreparedDynamicEmission::h(
    double first,
    double second,
    double parameter) const {
    return impl_->kind == DynamicEmissionKind::Pair
        ? impl_->pair.h(first, second, parameter)
        : std::numeric_limits<double>::quiet_NaN();
}

void PreparedDynamicEmission::h_pair(
    double first,
    double second,
    double parameter,
    double& first_next,
    double& second_next) const {

    if (impl_->kind == DynamicEmissionKind::Pair) {
        impl_->pair.h_pair(
            first, second, parameter, first_next, second_next);
        return;
    }
    first_next = std::numeric_limits<double>::quiet_NaN();
    second_next = std::numeric_limits<double>::quiet_NaN();
}

double PreparedDynamicEmission::inverse_h(
    double quantile,
    double given,
    double parameter) const {
    return impl_->kind == DynamicEmissionKind::Pair
        ? impl_->pair.inverse_h(quantile, given, parameter)
        : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace scar
