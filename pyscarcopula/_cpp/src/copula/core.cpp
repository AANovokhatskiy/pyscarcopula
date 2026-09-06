#include "scar/copula.hpp"

#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/detail/copula/dispatch.hpp"
#include "scar/detail/safety.hpp"
#include "scar/math/normal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace scar {

TypedModelDescriptor CopulaSpec::model_descriptor() const noexcept {
    if (family == CopulaFamily::Student) {
        if (correlation_kind == CorrelationKind::Factor) {
            return TypedModelDescriptor(
                FactorStudentDescriptor{{dim}},
                NativeModelId::StochasticStudent);
        }
        return TypedModelDescriptor(
            DenseStudentDescriptor{{dim}},
            NativeModelId::StochasticStudent,
            correlation_kind);
    }
    if (family == CopulaFamily::EquicorrGaussian) {
        return EquicorrGaussianDescriptor{{dim}};
    }
    if (family == CopulaFamily::MultivariateGaussian) {
        if (correlation_kind == CorrelationKind::Factor) {
            return FactorGaussianDescriptor{{dim}};
        }
        return DenseGaussianDescriptor{{dim}};
    }
    NativeModelId model_id = NativeModelId::Clayton;
    switch (family) {
        case CopulaFamily::Independent:
            model_id = NativeModelId::Independent;
            break;
        case CopulaFamily::Clayton:
            model_id = NativeModelId::Clayton;
            break;
        case CopulaFamily::Frank:
            model_id = NativeModelId::Frank;
            break;
        case CopulaFamily::Gumbel:
            model_id = NativeModelId::Gumbel;
            break;
        case CopulaFamily::Joe:
            model_id = NativeModelId::Joe;
            break;
        case CopulaFamily::Gaussian:
            model_id = NativeModelId::BivariateGaussian;
            break;
        default:
            break;
    }
    return TypedModelDescriptor(
        PairCopulaDescriptor{}, model_id, static_cast<int>(rotation));
}

namespace {

std::int64_t checked_size(const Observations& u) {
    return static_cast<std::int64_t>(u.size());
}

double row_value(const Observations& u, std::int64_t row, int col) {
    return u[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
}

double r_value(const std::vector<double>& r, std::int64_t row) {
    if (r.size() == 1) {
        return r[0];
    }
    return r[static_cast<std::size_t>(row)];
}

double open_sample(double value) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::runtime_error("pair sampling inverse failed to produce a finite probability");
    }
    // A mathematically interior quantile can round to an endpoint. Move only
    // that rounded endpoint by one representable step; never impose a 1e-10 floor.
    return std::clamp(value, std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0));
}

void apply_rotation(
    const CopulaSpec& spec,
    double u1,
    double u2,
    double& v1,
    double& v2) {

    scar::copula::apply_rotation(
        u1, u2, static_cast<int>(spec.rotation), v1, v2);
}

bool supports_non_pair_transform(const CopulaSpec& spec) noexcept {
    if (!scar::copula::is_valid_rotation(
            static_cast<int>(spec.rotation))) {
        return false;
    }
    if (spec.family == CopulaFamily::EquicorrGaussian) {
        return spec.dim >= 2
            && spec.rotation == Rotation::R0
            && spec.transform == Transform::GaussianTanh;
    }
    return spec.family == CopulaFamily::Student;
}

}  // namespace

bool is_supported(const CopulaSpec& spec) {
    return scar_internal::copula_is_supported(spec);
}

bool supports_transform(const CopulaSpec& spec) {
    const PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        return kernel.is_supported();
    }
    return supports_non_pair_transform(spec);
}

std::vector<double> copula_transform(
    const CopulaSpec& spec,
    const std::vector<double>& x) {

    std::vector<double> out(x.size(), 0.0);
    const PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        if (!kernel.is_supported()) {
            std::fill(out.begin(), out.end(),
                      std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        std::transform(x.begin(), x.end(), out.begin(), [&](double value) {
            return kernel.transform(value);
        });
        return out;
    }
    if (!supports_non_pair_transform(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(x.begin(), x.end(), out.begin(), [&](double value) {
        return scar_internal::copula_transform(spec, value);
    });
    return out;
}

std::vector<double> copula_inverse_transform(
    const CopulaSpec& spec,
    const std::vector<double>& r) {

    std::vector<double> out(r.size(), 0.0);
    const PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        if (!kernel.is_supported()) {
            std::fill(out.begin(), out.end(),
                      std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        std::transform(r.begin(), r.end(), out.begin(), [&](double value) {
            return kernel.inverse_transform(value);
        });
        return out;
    }
    if (!supports_non_pair_transform(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(r.begin(), r.end(), out.begin(), [&](double value) {
        return scar_internal::copula_inverse_transform(spec, value);
    });
    return out;
}

std::vector<double> copula_dtransform(
    const CopulaSpec& spec,
    const std::vector<double>& x) {

    std::vector<double> out(x.size(), 0.0);
    const PreparedPairKernel kernel(spec);
    if (kernel.is_registered()) {
        if (!kernel.is_supported()) {
            std::fill(out.begin(), out.end(),
                      std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        std::transform(x.begin(), x.end(), out.begin(), [&](double value) {
            return kernel.dtransform(value);
        });
        return out;
    }
    if (!supports_non_pair_transform(spec)) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(x.begin(), x.end(), out.begin(), [&](double value) {
        return scar_internal::copula_dtransform(spec, value);
    });
    return out;
}

std::vector<double> copula_tau_to_param(
    const CopulaSpec& spec,
    const std::vector<double>& tau) {

    std::vector<double> out(tau.size(), 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(tau.begin(), tau.end(), out.begin(), [&](double value) {
        return kernel.tau_to_parameter(value);
    });
    return out;
}

std::vector<double> copula_tau_to_param_capped(
    const CopulaSpec& spec,
    const std::vector<double>& tau,
    double theta_cap,
    bool has_theta_cap) {

    std::vector<double> out = copula_tau_to_param(spec, tau);
    if (!has_theta_cap) {
        return out;
    }
    if (!std::isfinite(theta_cap) || theta_cap <= 0.0) {
        std::fill(
            out.begin(), out.end(),
            std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    for (double& value : out) {
        value = std::min(value, theta_cap);
    }
    return out;
}

std::vector<double> copula_param_to_tau(
    const CopulaSpec& spec,
    const std::vector<double>& r) {

    std::vector<double> out(r.size(), 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }
    std::transform(r.begin(), r.end(), out.begin(), [&](double value) {
        return kernel.parameter_to_tau(value);
    });
    return out;
}

std::vector<double> copula_log_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(u);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()
        || (r.size() != 1 && r.size() != u.size())) {
        std::fill(out.begin(), out.end(), -std::numeric_limits<double>::infinity());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = kernel.log_pdf(
            row_value(u, i, 0),
            row_value(u, i, 1),
            r_value(r, i));
    }
    return out;
}

std::vector<double> copula_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    std::vector<double> out = copula_log_pdf(spec, u, r);
    for (double& value : out) {
        value = std::exp(value);
    }
    return out;
}

std::vector<double> copula_dlog_pdf_dr(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(u);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()
        || (r.size() != 1 && r.size() != u.size())) {
        std::fill(out.begin(), out.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(spec, row_value(u, i, 0), row_value(u, i, 1), v1, v2);
        out[static_cast<std::size_t>(i)] =
            kernel.dlog_pdf_dparameter_unrotated(
                v1, v2, r_value(r, i));
    }
    return out;
}

std::vector<double> copula_h(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(u);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()
        || (r.size() != 1 && r.size() != u.size())) {
        std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = kernel.h(
            row_value(u, i, 0),
            row_value(u, i, 1),
            r_value(r, i));
    }
    return out;
}

std::pair<std::vector<double>, std::vector<double>> copula_h_pair(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r) {

    const CopulaSpec transposed_spec =
        scar_internal::transposed_copula_spec(spec);
    return {
        copula_h(spec, u, r),
        [&]() {
            Observations reversed = u;
            for (std::vector<double>& row : reversed) {
                std::swap(row[0], row[1]);
            }
            return copula_h(transposed_spec, reversed, r);
        }(),
    };
}

std::vector<double> copula_h_inverse(
    const CopulaSpec& spec,
    const Observations& q_given,
    const std::vector<double>& r) {

    const std::int64_t n = checked_size(q_given);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()
        || (r.size() != 1 && r.size() != q_given.size())) {
        std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = kernel.inverse_h(
            row_value(q_given, i, 0),
            row_value(q_given, i, 1),
            r_value(r, i));
    }
    return out;
}

void copula_sample_from_uniforms_into(
    const CopulaSpec& spec, const double* uniforms, std::size_t n,
    const std::vector<double>& r, double* output) {
    if (r.size() != 1 && r.size() != n) {
        throw std::invalid_argument("pair sampling parameter has an invalid size");
    }
    const PreparedPairKernel kernel(scar_internal::transposed_copula_spec(spec));
    if (!kernel.is_supported()) {
        throw std::invalid_argument("unsupported pair-copula sampling specification");
    }
    for (std::size_t row = 0; row < n; ++row) {
        const double first = uniforms[2 * row];
        const double quantile = uniforms[2 * row + 1];
        if (!(first > 0.0 && first < 1.0)
            || !(quantile > 0.0 && quantile < 1.0)) {
            throw std::invalid_argument("pair sampling uniforms must be in (0, 1)");
        }
        const double sampled = open_sample(kernel.inverse_h(quantile, first, r.size() == 1 ? r[0] : r[row]));
        if (!(sampled > 0.0 && sampled < 1.0)) {
            throw std::invalid_argument("pair sampling transform produced a value outside (0, 1)");
        }
        output[2 * row] = first;
        output[2 * row + 1] = sampled;
    }
}

Observations copula_sample_from_uniforms(
    const CopulaSpec& spec,
    const Observations& uniforms,
    const std::vector<double>& r) {

    if (r.size() != 1 && r.size() != uniforms.size()) {
        throw std::invalid_argument(
            "pair sampling parameter must be scalar or have one value per row");
    }

    const CopulaSpec transposed_spec =
        scar_internal::transposed_copula_spec(spec);
    const PreparedPairKernel kernel(transposed_spec);
    if (!kernel.is_supported()) {
        throw std::invalid_argument(
            "pair sampling requires a supported pair-copula specification");
    }

    Observations out;
    out.reserve(uniforms.size());
    for (std::size_t row = 0; row < uniforms.size(); ++row) {
        if (uniforms[row].size() != 2) {
            throw std::invalid_argument(
                "pair sampling uniforms must have exactly two columns");
        }
        const double first = uniforms[row][0];
        const double quantile = uniforms[row][1];
        if (!std::isfinite(first)
            || !std::isfinite(quantile)
            || !(first > 0.0 && first < 1.0)
            || !(quantile > 0.0 && quantile < 1.0)) {
            throw std::invalid_argument(
                "pair sampling uniforms must be finite and in (0, 1)");
        }
        const double sampled = open_sample(kernel.inverse_h(
            quantile, first, r_value(r, static_cast<std::int64_t>(row))));
        if (!std::isfinite(sampled)
            || !(sampled > 0.0 && sampled < 1.0)) {
            throw std::invalid_argument(
                "pair sampling transform produced a value outside (0, 1)");
        }
        out.push_back({first, sampled});
    }
    return out;
}

Observations copula_conditional_sample_from_uniforms(
    const CopulaSpec& spec,
    const std::vector<double>& uniforms,
    const std::vector<double>& r,
    int given_coordinate,
    double given_value,
    const HInverseOptions* options) {

    if (options != nullptr) {
        if (!std::isfinite(options->tolerance) || options->tolerance <= 0.0) {
            throw std::invalid_argument("bisection_tol must be positive and finite");
        }
        if (options->max_iterations <= 0) {
            throw std::invalid_argument("bisection_maxiter must be positive");
        }
    }
    if (given_coordinate != 0 && given_coordinate != 1) {
        throw std::invalid_argument(
            "conditional pair sampling coordinate must be 0 or 1");
    }
    if (!std::isfinite(given_value)
        || !(given_value > 0.0 && given_value < 1.0)) {
        throw std::invalid_argument(
            "conditional pair sampling value must be in (0, 1)");
    }
    if (r.size() != 1 && r.size() != uniforms.size()) {
        throw std::invalid_argument(
            "pair sampling parameter must be scalar or have one value per row");
    }
    const CopulaSpec conditional_spec = given_coordinate == 0
        ? scar_internal::transposed_copula_spec(spec) : spec;
    const PreparedPairKernel kernel(conditional_spec);
    if (!kernel.is_supported()) {
        throw std::invalid_argument(
            "conditional pair sampling requires a supported pair copula");
    }

    Observations out;
    out.reserve(uniforms.size());
    for (std::size_t row = 0; row < uniforms.size(); ++row) {
        const double quantile = uniforms[row];
        if (!std::isfinite(quantile)
            || !(quantile >= 0.0 && quantile < 1.0)) {
            throw std::invalid_argument(
                "conditional pair sampling uniforms must be in [0, 1)");
        }
        const double sampled = kernel.inverse_h(
            quantile,
            given_value,
            r_value(r, static_cast<std::int64_t>(row)),
            options);
        if (!std::isfinite(sampled)
            || !(sampled >= 0.0 && sampled <= 1.0)) {
            throw std::invalid_argument(
                "conditional pair sampling produced a value outside [0, 1]");
        }
        out.push_back(given_coordinate == 0
            ? std::vector<double>{given_value, sampled}
            : std::vector<double>{sampled, given_value});
    }
    return out;
}

Observations copula_sample_from_rng_draws(
    const CopulaSpec& spec,
    const Observations& draws,
    const Observations& auxiliary,
    const std::vector<double>& r) {

    if (r.size() != 1 && r.size() != draws.size()) {
        throw std::invalid_argument(
            "pair sampling parameter must be scalar or have one value per row");
    }
    if (!auxiliary.empty()) {
        throw std::invalid_argument(
            "pair sampling does not accept auxiliary draws");
    }

    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()) {
        throw std::invalid_argument(
            "pair sampling requires a supported pair-copula specification");
    }

    Observations out;
    out.reserve(draws.size());
    for (std::size_t row = 0; row < draws.size(); ++row) {
        if (draws[row].size() != 2) {
            throw std::invalid_argument(
                "pair sampling RNG draws must have exactly two columns");
        }
        const double first_draw = draws[row][0];
        const double second_draw = draws[row][1];
        if (!std::isfinite(first_draw) || !std::isfinite(second_draw)) {
            throw std::invalid_argument(
                "pair sampling RNG draws must be finite");
        }
        if (spec.family != CopulaFamily::Gaussian
            && (!(first_draw > 0.0 && first_draw < 1.0)
                || !(second_draw > 0.0 && second_draw < 1.0))) {
            throw std::invalid_argument(
                "pair sampling uniform draws must be in (0, 1)");
        }
        const double parameter = r_value(
            r, static_cast<std::int64_t>(row));
        double first = 0.0;
        double second = 0.0;

        switch (spec.family) {
            case CopulaFamily::Independent:
                first = first_draw;
                second = second_draw;
                break;
            case CopulaFamily::Gaussian: {
                if (spec.rotation != Rotation::R0) {
                    throw std::invalid_argument(
                        "Gaussian pair sampling does not support rotation");
                }
                const double correlated =
                    parameter * first_draw
                    + std::sqrt(1.0 - parameter * parameter) * second_draw;
                first = math::normal_cdf(first_draw);
                second = math::normal_cdf(correlated);
                break;
            }
            case CopulaFamily::Clayton:
            case CopulaFamily::Gumbel:
            case CopulaFamily::Frank:
            case CopulaFamily::Joe: {
                const PreparedPairKernel conditional(scar_internal::transposed_copula_spec(spec));
                first = first_draw;
                second = open_sample(conditional.inverse_h(second_draw, first_draw, parameter));
                break;
            }
            default:
                throw std::invalid_argument(
                    "family-specific pair sampling is unavailable");
        }

        if (!std::isfinite(first) || !std::isfinite(second)) {
            throw std::invalid_argument(
                "pair sampling transform produced non-finite values");
        }
        out.push_back({first, second});
    }
    return out;
}

GridValues copula_pdf_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid) {

    GridValues out;
    out.n_obs = checked_size(u);
    out.n_grid = static_cast<std::int64_t>(x_grid.size());
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), x_grid.size(), value_count)) {
        return out;
    }
    out.values.assign(value_count, 0.0);

    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    std::vector<double> parameter_grid;
    std::vector<double> derivative_grid;
    kernel.prepare_parameter_grid(
        x_grid, parameter_grid, derivative_grid);
    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        const std::size_t row =
            static_cast<std::size_t>(t) * x_grid.size();
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(
            spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        for (std::int64_t j = 0; j < out.n_grid; ++j) {
            const std::size_t index = static_cast<std::size_t>(j);
            out.values[row + index] = std::exp(
                kernel.log_pdf_unrotated(v1, v2, parameter_grid[index]));
        }
    }
    return out;
}

GridValuesWithGrad copula_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid) {

    GridValuesWithGrad out;
    out.pdf.n_obs = checked_size(u);
    out.pdf.n_grid = static_cast<std::int64_t>(x_grid.size());
    out.d_pdf_dx.n_obs = out.pdf.n_obs;
    out.d_pdf_dx.n_grid = out.pdf.n_grid;
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), x_grid.size(), value_count)) {
        return out;
    }
    out.pdf.values.assign(value_count, 0.0);
    out.d_pdf_dx.values.assign(value_count, 0.0);

    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()) {
        std::fill(out.pdf.values.begin(), out.pdf.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        std::fill(out.d_pdf_dx.values.begin(), out.d_pdf_dx.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    std::vector<double> parameter_grid;
    std::vector<double> derivative_grid;
    kernel.prepare_parameter_grid(
        x_grid, parameter_grid, derivative_grid);
    for (std::int64_t t = 0; t < out.pdf.n_obs; ++t) {
        const std::size_t row =
            static_cast<std::size_t>(t)
            * static_cast<std::size_t>(out.pdf.n_grid);
        kernel.fill_grid_row_with_gradient(
            row_value(u, t, 0),
            row_value(u, t, 1),
            parameter_grid,
            derivative_grid,
            out.pdf.values.data() + row,
            out.d_pdf_dx.values.data() + row);
    }
    return out;
}

GridValues copula_pdf_parameter_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r_grid) {

    GridValues out;
    out.n_obs = checked_size(u);
    out.n_grid = static_cast<std::int64_t>(r_grid.size());
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), r_grid.size(), value_count)) {
        return out;
    }
    out.values.assign(value_count, 0.0);
    const PreparedPairKernel kernel(spec);
    if (!kernel.is_supported()) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        double v1 = 0.0;
        double v2 = 0.0;
        apply_rotation(
            spec, row_value(u, t, 0), row_value(u, t, 1), v1, v2);
        const std::size_t base =
            static_cast<std::size_t>(t) * r_grid.size();
        for (std::size_t j = 0; j < r_grid.size(); ++j) {
            out.values[base + j] = std::exp(
                kernel.log_pdf_unrotated(v1, v2, r_grid[j]));
        }
    }
    return out;
}

GridValues copula_h_parameter_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r_grid) {

    GridValues out;
    out.n_obs = checked_size(u);
    out.n_grid = static_cast<std::int64_t>(r_grid.size());
    std::size_t value_count = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), r_grid.size(), value_count)) {
        return out;
    }
    out.values.assign(value_count, 0.0);
    const CopulaSpec transposed_spec =
        scar_internal::transposed_copula_spec(spec);
    const PreparedPairKernel kernel(transposed_spec);
    if (!kernel.is_supported()) {
        std::fill(out.values.begin(), out.values.end(),
                  std::numeric_limits<double>::quiet_NaN());
        return out;
    }

    for (std::int64_t t = 0; t < out.n_obs; ++t) {
        const std::size_t base =
            static_cast<std::size_t>(t) * r_grid.size();
        for (std::size_t j = 0; j < r_grid.size(); ++j) {
            out.values[base + j] = kernel.h(
                row_value(u, t, 1),
                row_value(u, t, 0),
                r_grid[j]);
        }
    }
    return out;
}

}  // namespace scar
