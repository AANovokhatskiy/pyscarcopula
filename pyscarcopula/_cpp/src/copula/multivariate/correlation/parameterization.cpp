#include "scar/copula/multivariate/correlation/parameterization.hpp"

#include "scar/detail/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace scar {
namespace {

constexpr double kLogitClip = 1e-12;

std::size_t correlation_parameter_count(std::size_t dimension) {
    return dimension * (dimension - 1) / 2;
}

double logistic_value(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

bool finite_view(DoubleView values) {
    return values.data() != nullptr
        && std::all_of(
            values.data(),
            values.data() + values.size(),
            [](double value) { return std::isfinite(value); });
}

bool valid_square(
    DoubleView values,
    std::size_t dimension) {
    return dimension >= 2
        && values.size() == dimension * dimension
        && finite_view(values);
}

CorrelationPreprocessingResult invalid_preprocessing(Status status) {
    CorrelationPreprocessingResult result;
    result.status = status;
    return result;
}

double kendall_tau_b(
    ObservationView observations,
    std::size_t first,
    std::size_t second) {

    long double numerator = 0.0L;
    std::uint64_t tied_first = 0;
    std::uint64_t tied_second = 0;
    const std::size_t rows = observations.n_obs;
    for (std::size_t left = 0; left < rows; ++left) {
        const double left_first = observations.row(left)[first];
        const double left_second = observations.row(left)[second];
        for (std::size_t right = left + 1; right < rows; ++right) {
            const double first_difference =
                left_first - observations.row(right)[first];
            const double second_difference =
                left_second - observations.row(right)[second];
            if (first_difference == 0.0) {
                ++tied_first;
            }
            if (second_difference == 0.0) {
                ++tied_second;
            }
            if (first_difference != 0.0 && second_difference != 0.0) {
                numerator +=
                    (first_difference > 0.0) == (second_difference > 0.0)
                    ? 1.0L
                    : -1.0L;
            }
        }
    }
    const long double pair_count =
        static_cast<long double>(rows)
        * static_cast<long double>(rows - 1) / 2.0L;
    const long double first_count =
        pair_count - static_cast<long double>(tied_first);
    const long double second_count =
        pair_count - static_cast<long double>(tied_second);
    if (!(first_count > 0.0L) || !(second_count > 0.0L)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(
        numerator / std::sqrt(first_count * second_count));
}

}  // namespace

Result<std::vector<double>> logistic_transform(DoubleView values) {
    if (!finite_view(values)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> output(values.size());
    std::transform(
        values.data(),
        values.data() + values.size(),
        output.begin(),
        logistic_value);
    return success(std::move(output));
}

Result<std::vector<double>> logit_transform(DoubleView values) {
    if (!finite_view(values)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> output(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double clipped = std::clamp(
            values[index], kLogitClip, 1.0 - kLogitClip);
        output[index] = std::log(clipped) - std::log1p(-clipped);
    }
    return success(std::move(output));
}

Result<bool> validate_correlation(
    DoubleView matrix,
    std::size_t dimension,
    double tolerance) {

    if (!valid_square(matrix, dimension)
        || !std::isfinite(tolerance)
        || !(tolerance > 0.0)) {
        return {false, Status::InvalidParameter, {}};
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        if (std::abs(matrix[row * dimension + row] - 1.0) > tolerance) {
            FailureContext failure;
            failure.coordinate = static_cast<int>(row);
            return {false, Status::InvalidParameter, failure};
        }
        for (std::size_t column = 0; column < row; ++column) {
            if (std::abs(
                    matrix[row * dimension + column]
                    - matrix[column * dimension + row]) > tolerance) {
                FailureContext failure;
                failure.coordinate = static_cast<int>(row);
                return {false, Status::InvalidParameter, failure};
            }
        }
    }
    std::vector<double> lower;
    std::size_t failed = dimension;
    if (!scar_internal::linalg::cholesky_symmetric(
            matrix.data(), dimension, lower, 0.0, &failed)) {
        FailureContext failure;
        failure.coordinate = failed < dimension
            ? static_cast<int>(failed)
            : -1;
        return {false, Status::InvalidParameter, failure};
    }
    return success(true);
}

CorrelationPreprocessingResult preprocess_correlation(
    DoubleView matrix,
    std::size_t dimension,
    double eigenvalue_floor) {

    if (!valid_square(matrix, dimension)
        || !std::isfinite(eigenvalue_floor)
        || !(eigenvalue_floor > 0.0)) {
        return invalid_preprocessing(Status::InvalidParameter);
    }
    std::vector<double> symmetric(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            symmetric[row * dimension + column] = 0.5 * (
                matrix[row * dimension + column]
                + matrix[column * dimension + row]);
        }
    }

    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    if (!scar_internal::linalg::symmetric_eigen_jacobi(
            symmetric, dimension, eigenvalues, eigenvectors)) {
        return invalid_preprocessing(Status::NumericalFailure);
    }
    CorrelationPreprocessingResult result;
    result.dimension = dimension;
    result.input_correlation.assign(
        matrix.data(), matrix.data() + matrix.size());
    result.min_eigenvalue_before = *std::min_element(
        eigenvalues.begin(), eigenvalues.end());
    for (double& value : eigenvalues) {
        value = std::max(value, eigenvalue_floor);
    }

    result.correlation.assign(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            long double value = 0.0L;
            for (std::size_t component = 0;
                 component < dimension;
                 ++component) {
                value += static_cast<long double>(
                    eigenvectors[row * dimension + component])
                    * static_cast<long double>(eigenvalues[component])
                    * static_cast<long double>(
                        eigenvectors[column * dimension + component]);
            }
            result.correlation[row * dimension + column] =
                static_cast<double>(value);
        }
    }
    std::vector<double> scale(dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        scale[index] = std::sqrt(std::max(
            result.correlation[index * dimension + index],
            eigenvalue_floor));
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            result.correlation[row * dimension + column] /=
                scale[row] * scale[column];
        }
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        result.correlation[row * dimension + row] = 1.0;
        for (std::size_t column = row + 1;
             column < dimension;
             ++column) {
            const double value = 0.5 * (
                result.correlation[row * dimension + column]
                + result.correlation[column * dimension + row]);
            result.correlation[row * dimension + column] = value;
            result.correlation[column * dimension + row] = value;
        }
    }

    std::vector<double> final_eigenvalues;
    std::vector<double> ignored_vectors;
    if (!scar_internal::linalg::symmetric_eigen_jacobi(
            result.correlation,
            dimension,
            final_eigenvalues,
            ignored_vectors)) {
        return invalid_preprocessing(Status::NumericalFailure);
    }
    result.min_eigenvalue_after = *std::min_element(
        final_eigenvalues.begin(), final_eigenvalues.end());
    result.projection_applied =
        result.min_eigenvalue_before <= eigenvalue_floor;
    const double comparison_tolerance =
        10.0 * std::numeric_limits<double>::epsilon();
    for (std::size_t index = 0;
         index < result.correlation.size();
         ++index) {
        if (std::abs(matrix[index] - result.correlation[index])
            > comparison_tolerance) {
            result.projection_applied = true;
            break;
        }
    }
    std::vector<double> lower;
    if (!scar_internal::linalg::cholesky_symmetric(
            result.correlation.data(), dimension, lower, 0.0)) {
        return invalid_preprocessing(Status::NumericalFailure);
    }
    return result;
}

DenseCorrelationPreparationResult prepare_dense_correlation(
    DoubleView matrix,
    std::size_t dimension) {

    DenseCorrelationPreparationResult result;
    result.dimension = dimension;
    if (!valid_square(matrix, dimension)) {
        result.status = Status::InvalidParameter;
        return result;
    }
    std::vector<double> lower;
    std::size_t failed = dimension;
    if (!scar_internal::linalg::cholesky_symmetric(
            matrix.data(), dimension, lower, 0.0, &failed)) {
        result.status = Status::NumericalFailure;
        result.failure.coordinate = failed < dimension
            ? static_cast<int>(failed)
            : -1;
        return result;
    }
    result.inverse_cholesky.assign(dimension * dimension, 0.0);
    for (std::size_t column = 0; column < dimension; ++column) {
        for (std::size_t row = column; row < dimension; ++row) {
            long double value = row == column ? 1.0L : 0.0L;
            for (std::size_t inner = column; inner < row; ++inner) {
                value -= static_cast<long double>(
                    lower[row * dimension + inner])
                    * result.inverse_cholesky[
                        inner * dimension + column];
            }
            const double diagonal = lower[row * dimension + row];
            result.inverse_cholesky[row * dimension + column] =
                static_cast<double>(value) / diagonal;
        }
    }
    long double log_determinant = 0.0L;
    for (std::size_t index = 0; index < dimension; ++index) {
        log_determinant += 2.0L * std::log(
            lower[index * dimension + index]);
    }
    result.log_determinant = static_cast<double>(log_determinant);
    if (!std::isfinite(result.log_determinant)) {
        result.status = Status::NumericalFailure;
        result.inverse_cholesky.clear();
    }
    return result;
}

CorrelationPreprocessingResult estimate_kendall_correlation(
    ObservationView observations,
    double eigenvalue_floor) {

    if (observations.data() == nullptr
        || observations.n_obs == 0
        || observations.dim < 2) {
        return invalid_preprocessing(Status::InvalidSize);
    }
    const std::size_t dimension =
        static_cast<std::size_t>(observations.dim);
    for (std::size_t index = 0;
         index < observations.n_obs * dimension;
         ++index) {
        if (!std::isfinite(observations.data()[index])) {
            return invalid_preprocessing(Status::InvalidParameter);
        }
    }
    std::vector<double> correlation(dimension * dimension, 0.0);
    std::vector<std::int64_t> nonfinite_pairs;
    for (std::size_t index = 0; index < dimension; ++index) {
        correlation[index * dimension + index] = 1.0;
        for (std::size_t column = index + 1;
             column < dimension;
             ++column) {
            double tau = kendall_tau_b(observations, index, column);
            if (!std::isfinite(tau)) {
                tau = 0.0;
                nonfinite_pairs.push_back(
                    static_cast<std::int64_t>(index));
                nonfinite_pairs.push_back(
                    static_cast<std::int64_t>(column));
            }
            const double value = std::sin(
                0.5 * 3.141592653589793238462643383279502884 * tau);
            correlation[index * dimension + column] = value;
            correlation[column * dimension + index] = value;
        }
    }
    CorrelationPreprocessingResult result = preprocess_correlation(
        {correlation.data(), correlation.size()},
        dimension,
        eigenvalue_floor);
    result.nonfinite_kendall_pairs = std::move(nonfinite_pairs);
    return result;
}

Result<std::vector<double>> make_shrinkage_correlation(
    double raw_parameter,
    DoubleView base,
    std::size_t dimension) {

    if (!std::isfinite(raw_parameter)
        || !valid_square(base, dimension)) {
        return {{}, Status::InvalidParameter, {}};
    }
    const double alpha = logistic_value(raw_parameter);
    std::vector<double> result(base.data(), base.data() + base.size());
    for (std::size_t row = 0; row < dimension; ++row) {
        result[row * dimension + row] = 1.0;
        for (std::size_t column = row + 1;
             column < dimension;
             ++column) {
            const double value = alpha * 0.5 * (
                base[row * dimension + column]
                + base[column * dimension + row]);
            result[row * dimension + column] = value;
            result[column * dimension + row] = value;
        }
    }
    return success(std::move(result));
}

Result<std::vector<double>> pack_cholesky_correlation(
    DoubleView correlation,
    std::size_t dimension,
    double eigenvalue_floor) {

    CorrelationPreprocessingResult prepared = preprocess_correlation(
        correlation, dimension, eigenvalue_floor);
    if (!prepared.is_ok()) {
        return {{}, prepared.status, prepared.failure};
    }
    std::vector<double> lower;
    if (!scar_internal::linalg::cholesky_symmetric(
            prepared.correlation.data(), dimension, lower, 0.0)) {
        return {{}, Status::NumericalFailure, {}};
    }
    std::vector<double> parameters;
    parameters.reserve(correlation_parameter_count(dimension));
    for (std::size_t row = 1; row < dimension; ++row) {
        const double diagonal = lower[row * dimension + row];
        if (!(diagonal > 0.0)) {
            return {{}, Status::NumericalFailure, {}};
        }
        for (std::size_t column = 0; column < row; ++column) {
            parameters.push_back(
                lower[row * dimension + column] / diagonal);
        }
    }
    return success(std::move(parameters));
}

Result<std::vector<double>> unpack_cholesky_correlation(
    DoubleView parameters,
    std::size_t dimension) {

    const std::size_t expected = correlation_parameter_count(dimension);
    if (dimension < 2
        || parameters.size() != expected
        || !finite_view(parameters)) {
        return {{}, Status::InvalidParameter, {}};
    }
    std::vector<double> lower(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        lower[index * dimension + index] = 1.0;
    }
    std::size_t position = 0;
    for (std::size_t row = 1; row < dimension; ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            lower[row * dimension + column] = parameters[position++];
        }
    }
    std::vector<double> sigma(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            long double value = 0.0L;
            const std::size_t stop = std::min(row, column);
            for (std::size_t component = 0;
                 component <= stop;
                 ++component) {
                value += static_cast<long double>(
                    lower[row * dimension + component])
                    * static_cast<long double>(
                        lower[column * dimension + component]);
            }
            sigma[row * dimension + column] = static_cast<double>(value);
            sigma[column * dimension + row] = static_cast<double>(value);
        }
    }
    std::vector<double> scales(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        scales[index] = std::sqrt(sigma[index * dimension + index]);
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            sigma[row * dimension + column] /=
                scales[row] * scales[column];
        }
        sigma[row * dimension + row] = 1.0;
    }
    return success(std::move(sigma));
}

Result<std::vector<double>> correlation_gradient_to_raw(
    DenseCorrelationMode mode,
    DoubleView parameters,
    DoubleView correlation,
    DoubleView correlation_gradient,
    DoubleView base,
    std::size_t dimension) {

    const std::size_t expected = correlation_parameter_count(dimension);
    if (!valid_square(correlation, dimension)
        || correlation_gradient.size() != expected
        || !finite_view(correlation_gradient)) {
        return {{}, Status::InvalidParameter, {}};
    }
    if (mode == DenseCorrelationMode::Shrinkage) {
        if (parameters.size() != 1
            || !finite_view(parameters)
            || !valid_square(base, dimension)) {
            return {{}, Status::InvalidParameter, {}};
        }
        const double alpha = logistic_value(parameters[0]);
        const double factor = alpha * (1.0 - alpha);
        long double gradient = 0.0L;
        std::size_t position = 0;
        for (std::size_t row = 1; row < dimension; ++row) {
            for (std::size_t column = 0; column < row; ++column) {
                gradient += static_cast<long double>(
                    correlation_gradient[position++])
                    * static_cast<long double>(factor)
                    * static_cast<long double>(
                        base[row * dimension + column]);
            }
        }
        return success(std::vector<double>{static_cast<double>(gradient)});
    }
    if (mode != DenseCorrelationMode::Cholesky
        || parameters.size() != expected
        || !finite_view(parameters)) {
        return {{}, Status::InvalidParameter, {}};
    }

    std::vector<double> lower(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        lower[index * dimension + index] = 1.0;
    }
    std::size_t position = 0;
    for (std::size_t row = 1; row < dimension; ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            lower[row * dimension + column] = parameters[position++];
        }
    }
    std::vector<double> sigma(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = 0.0;
            for (std::size_t component = 0;
                 component <= std::min(row, column);
                 ++component) {
                value += lower[row * dimension + component]
                    * lower[column * dimension + component];
            }
            sigma[row * dimension + column] = value;
            sigma[column * dimension + row] = value;
        }
    }
    std::vector<double> sigma_diagonal(dimension);
    std::vector<double> scales(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        sigma_diagonal[index] = sigma[index * dimension + index];
        scales[index] = std::sqrt(sigma_diagonal[index]);
    }
    std::vector<double> matrix_gradient(dimension * dimension, 0.0);
    position = 0;
    for (std::size_t row = 1; row < dimension; ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            const double value = 0.5 * correlation_gradient[position++];
            matrix_gradient[row * dimension + column] = value;
            matrix_gradient[column * dimension + row] = value;
        }
    }
    std::vector<double> sigma_gradient(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        double correction = 0.0;
        for (std::size_t column = 0; column < dimension; ++column) {
            correction += matrix_gradient[row * dimension + column]
                * correlation[row * dimension + column];
            sigma_gradient[row * dimension + column] =
                matrix_gradient[row * dimension + column]
                / (scales[row] * scales[column]);
        }
        sigma_gradient[row * dimension + row] -=
            correction / sigma_diagonal[row];
    }
    std::vector<double> lower_gradient(dimension * dimension, 0.0);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            double value = 0.0;
            for (std::size_t component = 0;
                 component < dimension;
                 ++component) {
                value += sigma_gradient[row * dimension + component]
                    * lower[component * dimension + column];
            }
            lower_gradient[row * dimension + column] = 2.0 * value;
        }
    }
    std::vector<double> result(expected);
    position = 0;
    for (std::size_t row = 1; row < dimension; ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            result[position++] = lower_gradient[row * dimension + column];
        }
    }
    return success(std::move(result));
}

Result<std::vector<double>> shrinkage_raw_correlation_direction(
    DoubleView parameters,
    DoubleView base,
    std::size_t dimension) {

    if (parameters.size() != 1
        || !finite_view(parameters)
        || !valid_square(base, dimension)) {
        return {{}, Status::InvalidParameter, {}};
    }
    const double alpha = logistic_value(parameters[0]);
    const double factor = alpha * (1.0 - alpha);
    std::vector<double> result(correlation_parameter_count(dimension));
    std::size_t position = 0;
    for (std::size_t row = 1; row < dimension; ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            result[position++] = factor * base[row * dimension + column];
        }
    }
    return success(std::move(result));
}

}  // namespace scar
