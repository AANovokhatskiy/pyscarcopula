#include "scar/copula.hpp"
#include "scar/copula/capability.hpp"
#include "scar/copula/multivariate/rosenblatt.hpp"
#include "scar/copula/prepared_pair_kernel.hpp"
#include "scar/model_policy.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct PairCase {
    scar::CopulaFamily family;
    scar::NativeModelId model_id;
    scar::Rotation rotation;
    double parameter;
};

bool close(double first, double second, double tolerance = 2e-8) {
    return std::isfinite(first)
        && std::isfinite(second)
        && std::abs(first - second) <= tolerance;
}

scar::CopulaSpec make_spec(const PairCase& test) {
    scar::CopulaSpec spec = scar::default_pair_copula_spec(test.family);
    spec.rotation = test.rotation;
    return spec;
}

scar::CopulaSpec transpose_spec(scar::CopulaSpec spec) {
    if (spec.rotation == scar::Rotation::R90) {
        spec.rotation = scar::Rotation::R270;
    } else if (spec.rotation == scar::Rotation::R270) {
        spec.rotation = scar::Rotation::R90;
    }
    return spec;
}

bool throws_invalid_argument_for_bad_sampling_draw(
    const scar::CopulaSpec& spec,
    double parameter) {

    try {
        static_cast<void>(scar::copula_sample_from_uniforms(
            spec, {{0.0, 0.5}}, {parameter}));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <std::size_t Size>
bool contains_case(
    const std::array<PairCase, Size>& cases,
    scar::NativeModelId model_id,
    int rotation) {

    for (const PairCase& test : cases) {
        if (test.model_id == model_id
            && static_cast<int>(test.rotation) == rotation) {
            return true;
        }
    }
    return false;
}

}  // namespace

int run_pair_model_tests() {
    // These are the exact 15 supported static pair identities.  Keeping the
    // rotations explicit makes a registry expansion fail this direct suite
    // until the new identity receives its own Python-free case.
    constexpr std::array<PairCase, 15> cases{{
        {scar::CopulaFamily::Independent,
         scar::NativeModelId::Independent, scar::Rotation::R0, 0.0},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R0, 1.5},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R90, 1.5},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R180, 1.5},
        {scar::CopulaFamily::Clayton,
         scar::NativeModelId::Clayton, scar::Rotation::R270, 1.5},
        {scar::CopulaFamily::Frank,
         scar::NativeModelId::Frank, scar::Rotation::R0, 2.0},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R0, 1.5},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R90, 1.5},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R180, 1.5},
        {scar::CopulaFamily::Gumbel,
         scar::NativeModelId::Gumbel, scar::Rotation::R270, 1.5},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R0, 1.5},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R90, 1.5},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R180, 1.5},
        {scar::CopulaFamily::Joe,
         scar::NativeModelId::Joe, scar::Rotation::R270, 1.5},
        {scar::CopulaFamily::Gaussian,
         scar::NativeModelId::BivariateGaussian,
         scar::Rotation::R0, 0.35},
    }};
    constexpr std::array<scar::NativeOperation, 9> static_operations{{
        scar::NativeOperation::ParameterTransformBoundsInitialization,
        scar::NativeOperation::PointDensityDerivatives,
        scar::NativeOperation::RowGridDensityGradient,
        scar::NativeOperation::LikelihoodObjectiveGradient,
        scar::NativeOperation::RosenblattResidual,
        scar::NativeOperation::RadialGofSummary,
        scar::NativeOperation::UnconditionalSamplingTransform,
        scar::NativeOperation::ConditionalSamplingTransform,
        scar::NativeOperation::EdgeStructureSelectionScore,
    }};
    const scar::Observations observations{
        {0.37, 0.63},
        {0.21, 0.74},
        {0.82, 0.28},
    };
    const scar::Observations fixed_uniforms{
        {0.23, 0.61},
        {0.78, 0.34},
    };

    for (std::size_t index = 0; index < cases.size(); ++index) {
        const PairCase& test = cases[index];
        const int base = 1 + static_cast<int>(index) * 20;
        const scar::CopulaSpec spec = make_spec(test);
        const scar::PreparedPairKernel kernel(spec);
        const scar::TypedModelDescriptor descriptor = spec.model_descriptor();

        if (!kernel.is_registered()
            || !kernel.is_supported()
            || kernel.family() != test.family
            || descriptor.model_id() != test.model_id
            || descriptor.rotation() != static_cast<int>(test.rotation)) {
            return base;
        }
        for (scar::NativeOperation operation : static_operations) {
            if (!scar::query_capability(
                    descriptor, operation, scar::DynamicsKind::Mle).supported) {
                return base + 1;
            }
        }
        const scar::CapabilityInfo static_state = scar::query_capability(
            descriptor,
            scar::NativeOperation::StateFilterSmoother,
            scar::DynamicsKind::Mle);
        const scar::CapabilityInfo arbitrary_mcmc = scar::query_capability(
            descriptor,
            scar::NativeOperation::ArbitraryConditionalMcmc,
            scar::DynamicsKind::Mle);
        if (static_state.supported || static_state.reason.empty()
            || arbitrary_mcmc.supported || arbitrary_mcmc.reason.empty()) {
            return base + 2;
        }

        // parameter_domain: native public bounds, initialization and raw /
        // physical transforms.
        const scar::ParameterBoundsResult bounds =
            scar::model_public_parameter_bounds(spec);
        const scar::Result<double> initial =
            scar::default_pair_mle_parameter(spec);
        if (!bounds.is_ok()) {
            return base + 3;
        }
        double raw_parameter = 0.0;
        if (test.family == scar::CopulaFamily::Independent) {
            if (!bounds.value.lower.empty() || !bounds.value.upper.empty()
                || initial.status != scar::Status::InvalidParameter
                || kernel.transform(123.0) != 0.0
                || kernel.inverse_transform(123.0) != 0.0
                || kernel.dtransform(123.0) != 0.0) {
                return base + 3;
            }
        } else {
            raw_parameter = kernel.inverse_transform(test.parameter);
            if (!initial.is_ok()
                || bounds.value.lower.size() != 1
                || bounds.value.upper.size() != 1
                || !(test.parameter > bounds.value.lower[0])
                || !(test.parameter < bounds.value.upper[0])
                || !close(kernel.transform(raw_parameter), test.parameter,
                          2e-13)
                || !(kernel.dtransform(raw_parameter) > 0.0)) {
                return base + 3;
            }
        }

        // point_kernel and selection: density derivative, both h directions,
        // inverse-h, and Kendall/parameter mapping on one prepared dispatch.
        const double first = observations[0][0];
        const double second = observations[0][1];
        const double log_density = kernel.log_pdf(
            first, second, test.parameter);
        const double repeated_log_density = kernel.log_pdf(
            first, second, test.parameter);
        const double density = std::exp(log_density);
        const double derivative = kernel.dlog_pdf_dparameter(
            first, second, test.parameter);
        double density_from_raw = 0.0;
        double gradient_from_raw = 0.0;
        kernel.pdf_and_gradient(
            first,
            second,
            raw_parameter,
            density_from_raw,
            gradient_from_raw);
        double first_h = 0.0;
        double second_h = 0.0;
        kernel.h_pair(
            first, second, test.parameter, first_h, second_h);
        const double recovered = kernel.inverse_h(
            first_h, second, test.parameter);
        if (!std::isfinite(log_density)
            || log_density != repeated_log_density
            || !std::isfinite(derivative)
            || !close(density_from_raw, density, 2e-13)
            || !close(
                gradient_from_raw,
                density * derivative * kernel.dtransform(raw_parameter),
                2e-12)
            || !(first_h > 0.0 && first_h < 1.0)
            || !(second_h > 0.0 && second_h < 1.0)
            || !close(recovered, first, 3e-7)) {
            return base + 4;
        }
        if (test.family != scar::CopulaFamily::Independent) {
            const double tau = kernel.parameter_to_tau(test.parameter);
            if (!close(kernel.tau_to_parameter(tau), test.parameter, 1e-10)) {
                return base + 5;
            }
        }

        // row_grid_kernel: scalar, row and grid entry points must agree for
        // every exact family/rotation identity.
        const std::vector<double> parameter{test.parameter};
        const std::vector<double> row_log =
            scar::copula_log_pdf(spec, observations, parameter);
        const std::vector<double> row_pdf =
            scar::copula_pdf(spec, observations, parameter);
        const std::vector<double> row_derivative =
            scar::copula_dlog_pdf_dr(spec, observations, parameter);
        const auto row_h_pair = scar::copula_h_pair(
            spec, observations, parameter);
        if (row_log.size() != observations.size()
            || row_pdf.size() != observations.size()
            || row_derivative.size() != observations.size()
            || row_h_pair.first.size() != observations.size()
            || row_h_pair.second.size() != observations.size()) {
            return base + 6;
        }
        for (std::size_t row = 0; row < observations.size(); ++row) {
            if (!close(
                    row_log[row],
                    kernel.log_pdf(
                        observations[row][0], observations[row][1],
                        test.parameter),
                    2e-13)
                || !close(row_pdf[row], std::exp(row_log[row]), 2e-13)
                || !close(
                    row_derivative[row],
                    kernel.dlog_pdf_dparameter(
                        observations[row][0], observations[row][1],
                        test.parameter),
                    2e-13)) {
                return base + 7;
            }
        }
        const std::vector<double> raw_grid{
            raw_parameter - 0.1, raw_parameter, raw_parameter + 0.1};
        const scar::GridValues grid = scar::copula_pdf_grid(
            spec, observations, raw_grid);
        const scar::GridValuesWithGrad grid_gradient =
            scar::copula_pdf_and_grad_grid(spec, observations, raw_grid);
        if (grid.n_obs != static_cast<std::int64_t>(observations.size())
            || grid.n_grid != static_cast<std::int64_t>(raw_grid.size())
            || grid.values != grid_gradient.pdf.values
            || grid_gradient.d_pdf_dx.values.size() != grid.values.size()) {
            return base + 8;
        }
        for (std::size_t row = 0; row < observations.size(); ++row) {
            for (std::size_t column = 0; column < raw_grid.size(); ++column) {
                const std::size_t cell = row * raw_grid.size() + column;
                double expected_density = 0.0;
                double expected_gradient = 0.0;
                kernel.pdf_and_gradient(
                    observations[row][0],
                    observations[row][1],
                    raw_grid[column],
                    expected_density,
                    expected_gradient);
                if (!close(grid.values[cell], expected_density, 2e-13)
                    || !close(
                        grid_gradient.d_pdf_dx.values[cell],
                        expected_gradient,
                        2e-13)) {
                    return base + 9;
                }
            }
        }

        // objective: prepared static dispatch and requested thread count have
        // identical value/gradient semantics.
        const scar::StaticCopulaEvaluator evaluator_one(
            spec, observations, 1);
        const scar::StaticCopulaEvaluator evaluator_four(
            spec, observations, 4);
        const scar::StaticObjectiveResult objective_one =
            evaluator_one.objective(test.parameter);
        const scar::StaticObjectiveResult objective_four =
            evaluator_four.objective(test.parameter);
        double expected_nll = 0.0;
        double expected_gradient = 0.0;
        for (std::size_t row = 0; row < row_log.size(); ++row) {
            expected_nll -= row_log[row];
            expected_gradient -= row_derivative[row];
        }
        if (!objective_one.is_ok() || !objective_four.is_ok()
            || objective_one.n_threads_requested != 1
            || objective_four.n_threads_requested != 4
            || !close(
                objective_one.negative_log_likelihood, expected_nll, 2e-13)
            || !close(
                objective_one.negative_gradient,
                expected_gradient,
                2e-12)
            || objective_four.negative_log_likelihood
                != objective_one.negative_log_likelihood
            || objective_four.negative_gradient
                != objective_one.negative_gradient) {
            return base + 10;
        }

        // residual and radial reduction remain entirely inside C++.
        std::vector<double> residual_values;
        residual_values.reserve(observations.size() * 2);
        for (std::size_t row = 0; row < observations.size(); ++row) {
            residual_values.push_back(row_h_pair.first[row]);
            residual_values.push_back(row_h_pair.second[row]);
        }
        const scar::RadialSummaryResult radial = scar::radial_uniform_summary(
            scar::ObservationView{
                residual_values.data(), observations.size(), 2},
            4);
        if (!radial.is_ok()
            || radial.n_rows
                != static_cast<std::int64_t>(observations.size())
            || radial.dimension != 2
            || radial.n_threads_requested != 4
            || radial.values.size() != observations.size()) {
            return base + 11;
        }
        for (double value : radial.values) {
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                return base + 11;
            }
        }

        // sampling: caller-owned fixed draws are transformed in native code;
        // applying the matching conditional CDF recovers each draw exactly.
        const scar::Observations sampled = scar::copula_sample_from_uniforms(
            spec, fixed_uniforms, parameter);
        const scar::PreparedPairKernel transposed_kernel(
            transpose_spec(spec));
        if (sampled.size() != fixed_uniforms.size()) {
            return base + 12;
        }
        for (std::size_t row = 0; row < sampled.size(); ++row) {
            if (sampled[row].size() != 2
                || sampled[row][0] != fixed_uniforms[row][0]
                || !close(
                    transposed_kernel.h(
                        sampled[row][1], sampled[row][0], test.parameter),
                    fixed_uniforms[row][1],
                    4e-7)) {
                return base + 12;
            }
        }
        constexpr double given = 0.58;
        const std::vector<double> conditional_draws{0.17, 0.83};
        for (int given_coordinate = 0;
             given_coordinate < 2;
             ++given_coordinate) {
            const scar::Observations conditional =
                scar::copula_conditional_sample_from_uniforms(
                    spec,
                    conditional_draws,
                    parameter,
                    given_coordinate,
                    given);
            for (std::size_t row = 0; row < conditional.size(); ++row) {
                const double recovered_draw = given_coordinate == 0
                    ? transposed_kernel.h(
                        conditional[row][1], given, test.parameter)
                    : kernel.h(
                        conditional[row][0], given, test.parameter);
                if (conditional[row][static_cast<std::size_t>(given_coordinate)]
                        != given
                    || !close(
                        recovered_draw, conditional_draws[row], 4e-7)) {
                    return base + 13;
                }
            }
        }

        // failure: invalid input and native status are deterministic and do
        // not admit another numerical path.
        const scar::StaticCopulaEvaluator invalid_threads(
            spec, observations, 0);
        const scar::StaticObjectiveResult nonfinite_objective =
            evaluator_one.objective(
                std::numeric_limits<double>::quiet_NaN());
        if (invalid_threads.status() != scar::Status::InvalidParameter
            || nonfinite_objective.status != scar::Status::InvalidParameter
            || nonfinite_objective.failure.index != -1
            || !throws_invalid_argument_for_bad_sampling_draw(
                spec, test.parameter)) {
            return base + 14;
        }

        const double tails[] = {1e-8, 1.0 - 1e-8};
        for (double tail : tails) {
            if (!std::isfinite(kernel.log_pdf(tail, 0.43, test.parameter))) {
                return base + 15;
            }
            const double tail_h = kernel.h(tail, 0.43, test.parameter);
            if (!std::isfinite(tail_h) || tail_h < 0.0 || tail_h > 1.0) {
                return base + 15;
            }
        }
    }

    // Rotation rejection is an explicit frozen failure contract for the two
    // R0-only pair families.
    for (scar::CopulaFamily family : {
             scar::CopulaFamily::Frank,
             scar::CopulaFamily::Gaussian}) {
        scar::CopulaSpec invalid = scar::default_pair_copula_spec(family);
        invalid.rotation = scar::Rotation::R90;
        const scar::PreparedPairKernel kernel(invalid);
        const scar::CapabilityInfo capability = scar::query_capability(
            invalid.model_descriptor(),
            scar::NativeOperation::PointDensityDerivatives,
            scar::DynamicsKind::Mle);
        if (kernel.is_supported()
            || capability.supported
            || capability.reason.empty()) {
            return 400 + static_cast<int>(family);
        }
    }

    // Completeness is checked in the opposite direction as well: every pair
    // identity advertised by the authoritative capability registry must be
    // present in the direct-case table above.
    constexpr std::array<int, 4> rotations{{0, 90, 180, 270}};
    for (int model = static_cast<int>(scar::NativeModelId::Independent);
         model <= static_cast<int>(scar::NativeModelId::BivariateGaussian);
         ++model) {
        for (int rotation : rotations) {
            const scar::NativeModelId model_id =
                static_cast<scar::NativeModelId>(model);
            const scar::TypedModelDescriptor descriptor =
                scar::make_typed_model_descriptor(
                    model_id,
                    2,
                    scar::CorrelationKind::NotApplicable,
                    rotation);
            const bool advertised = scar::query_capability(
                descriptor,
                scar::NativeOperation::PointDensityDerivatives,
                scar::DynamicsKind::Mle).supported;
            if (advertised != contains_case(cases, model_id, rotation)) {
                return 450 + model * 4 + rotation / 90;
            }
        }
    }
    return 0;
}
