#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace scar_internal::linalg {

// Portable is a dependency-free, SIMD-friendly four-accumulator kernel.
// Scalar deliberately preserves the original left-to-right reduction and is
// retained as the mandatory numerical fallback.
enum class Backend {
    Scalar,
    Portable,
};

constexpr std::size_t kPortableMinElements = 32;

inline double dot_scalar(
    const double* lhs,
    const double* rhs,
    std::size_t count) noexcept {

    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline double dot_portable(
    const double* lhs,
    const double* rhs,
    std::size_t count) noexcept {

    if (count < kPortableMinElements) {
        return dot_scalar(lhs, rhs, count);
    }
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        sum0 += lhs[i] * rhs[i];
        sum1 += lhs[i + 1] * rhs[i + 1];
        sum2 += lhs[i + 2] * rhs[i + 2];
        sum3 += lhs[i + 3] * rhs[i + 3];
    }
    double sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline double dot(
    const double* lhs,
    const double* rhs,
    std::size_t count,
    Backend backend = Backend::Portable) noexcept {

    return backend == Backend::Scalar
        ? dot_scalar(lhs, rhs, count)
        : dot_portable(lhs, rhs, count);
}

inline double dot_strided(
    const double* lhs,
    std::size_t lhs_stride,
    const double* rhs,
    std::size_t rhs_stride,
    std::size_t count,
    Backend backend = Backend::Portable) noexcept {

    if (backend == Backend::Scalar || count < kPortableMinElements) {
        double sum = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            sum += lhs[i * lhs_stride] * rhs[i * rhs_stride];
        }
        return sum;
    }
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        sum0 += lhs[i * lhs_stride] * rhs[i * rhs_stride];
        sum1 += lhs[(i + 1) * lhs_stride]
            * rhs[(i + 1) * rhs_stride];
        sum2 += lhs[(i + 2) * lhs_stride]
            * rhs[(i + 2) * rhs_stride];
        sum3 += lhs[(i + 3) * lhs_stride]
            * rhs[(i + 3) * rhs_stride];
    }
    double sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < count; ++i) {
        sum += lhs[i * lhs_stride] * rhs[i * rhs_stride];
    }
    return sum;
}

inline void row_major_matvec(
    const double* matrix,
    std::size_t rows,
    std::size_t columns,
    const double* vector,
    double* out,
    Backend backend = Backend::Portable) noexcept {

    for (std::size_t row = 0; row < rows; ++row) {
        out[row] = dot(
            matrix + row * columns, vector, columns, backend);
    }
}

inline void lower_triangular_matvec(
    const double* lower,
    std::size_t dimension,
    const double* vector,
    double* out,
    Backend backend = Backend::Portable) noexcept {

    for (std::size_t row = 0; row < dimension; ++row) {
        out[row] = dot(
            lower + row * dimension,
            vector,
            row + 1,
            backend);
    }
}

inline bool cholesky_symmetric(
    const double* matrix,
    std::size_t dimension,
    std::vector<double>& lower,
    double diagonal_shift = 0.0,
    std::size_t* failure_coordinate = nullptr,
    Backend backend = Backend::Portable) {

    lower.assign(dimension * dimension, 0.0);
    if (failure_coordinate != nullptr) {
        *failure_coordinate = dimension;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = 0.5 * (
                matrix[row * dimension + column]
                + matrix[column * dimension + row]);
            if (row == column) {
                value += diagonal_shift;
            }
            value -= dot(
                lower.data() + row * dimension,
                lower.data() + column * dimension,
                column,
                backend);
            if (row == column) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    if (failure_coordinate != nullptr) {
                        *failure_coordinate = row;
                    }
                    return false;
                }
                lower[row * dimension + column] = std::sqrt(value);
            } else {
                const double diagonal =
                    lower[column * dimension + column];
                if (!(diagonal > 0.0)) {
                    if (failure_coordinate != nullptr) {
                        *failure_coordinate = column;
                    }
                    return false;
                }
                lower[row * dimension + column] = value / diagonal;
            }
        }
    }
    return true;
}

inline bool cholesky_symmetric_with_jitter(
    const double* matrix,
    std::size_t dimension,
    std::vector<double>& lower,
    double* applied_jitter = nullptr,
    Backend backend = Backend::Portable) {

    if (applied_jitter != nullptr) {
        *applied_jitter = std::numeric_limits<double>::quiet_NaN();
    }
    double jitter = 0.0;
    for (int attempt = 0; attempt < 7; ++attempt) {
        if (cholesky_symmetric(
                matrix, dimension, lower, jitter, nullptr, backend)) {
            if (applied_jitter != nullptr) {
                *applied_jitter = jitter;
            }
            return true;
        }
        jitter = jitter == 0.0 ? 1e-12 : jitter * 10.0;
    }
    return false;
}

/// Dependency-free Jacobi eigensolver for a finite symmetric dense matrix.
/// Eigenvectors are returned as row-major columns.  The caller may sort the
/// eigenpairs when a deterministic spectral order is required.
inline bool symmetric_eigen_jacobi(
    const std::vector<double>& input,
    std::size_t dimension,
    std::vector<double>& eigenvalues,
    std::vector<double>& eigenvectors) {

    if (dimension == 0
        || input.size() != dimension * dimension) {
        return false;
    }
    if (dimension == 1) {
        eigenvalues.assign(1, input.front());
        eigenvectors.assign(1, 1.0);
        return std::isfinite(input.front());
    }

    std::vector<double> matrix(input);
    eigenvectors.assign(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        eigenvectors[index * dimension + index] = 1.0;
    }
    const std::size_t maximum_iterations =
        std::max<std::size_t>(64, 64 * dimension * dimension);
    const double tolerance =
        8.0 * std::numeric_limits<double>::epsilon();

    for (std::size_t iteration = 0;
         iteration < maximum_iterations;
         ++iteration) {
        std::size_t pivot_row = 0;
        std::size_t pivot_column = 1;
        double maximum = 0.0;
        double diagonal_scale = 1.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            diagonal_scale = std::max(
                diagonal_scale,
                std::abs(matrix[row * dimension + row]));
            for (std::size_t column = row + 1;
                 column < dimension;
                 ++column) {
                const double value =
                    std::abs(matrix[row * dimension + column]);
                if (value > maximum) {
                    maximum = value;
                    pivot_row = row;
                    pivot_column = column;
                }
            }
        }
        if (maximum <= tolerance * diagonal_scale) {
            eigenvalues.resize(dimension);
            for (std::size_t index = 0; index < dimension; ++index) {
                eigenvalues[index] = matrix[index * dimension + index];
            }
            return std::all_of(
                eigenvalues.begin(),
                eigenvalues.end(),
                [](double value) { return std::isfinite(value); });
        }

        const double app = matrix[pivot_row * dimension + pivot_row];
        const double aqq = matrix[pivot_column * dimension + pivot_column];
        const double apq = matrix[pivot_row * dimension + pivot_column];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        for (std::size_t index = 0; index < dimension; ++index) {
            if (index == pivot_row || index == pivot_column) {
                continue;
            }
            const double aip = matrix[index * dimension + pivot_row];
            const double aiq = matrix[index * dimension + pivot_column];
            const double rotated_p = cosine * aip - sine * aiq;
            const double rotated_q = sine * aip + cosine * aiq;
            matrix[index * dimension + pivot_row] = rotated_p;
            matrix[pivot_row * dimension + index] = rotated_p;
            matrix[index * dimension + pivot_column] = rotated_q;
            matrix[pivot_column * dimension + index] = rotated_q;
        }
        matrix[pivot_row * dimension + pivot_row] =
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq;
        matrix[pivot_column * dimension + pivot_column] =
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq;
        matrix[pivot_row * dimension + pivot_column] = 0.0;
        matrix[pivot_column * dimension + pivot_row] = 0.0;

        for (std::size_t row = 0; row < dimension; ++row) {
            const double vip = eigenvectors[row * dimension + pivot_row];
            const double viq = eigenvectors[row * dimension + pivot_column];
            eigenvectors[row * dimension + pivot_row] =
                cosine * vip - sine * viq;
            eigenvectors[row * dimension + pivot_column] =
                sine * vip + cosine * viq;
        }
    }
    return false;
}

/// Implicit QL eigensolver for a real symmetric tridiagonal matrix.
/// ``off_diagonal[i]`` is the element between rows ``i`` and ``i + 1``;
/// the final entry is ignored.  This O(n^2) specialization is used by
/// Golub--Welsch quadrature construction.
inline bool symmetric_tridiagonal_eigen_ql(
    std::vector<double>& diagonal,
    std::vector<double>& off_diagonal,
    std::vector<double>& eigenvectors) {

    const std::size_t dimension = diagonal.size();
    if (dimension == 0 || off_diagonal.size() != dimension) {
        return false;
    }
    eigenvectors.assign(dimension * dimension, 0.0);
    for (std::size_t index = 0; index < dimension; ++index) {
        eigenvectors[index * dimension + index] = 1.0;
    }
    if (dimension == 1) {
        return std::isfinite(diagonal.front());
    }
    off_diagonal.back() = 0.0;
    const double epsilon = std::numeric_limits<double>::epsilon();

    for (std::size_t left = 0; left < dimension; ++left) {
        std::size_t iterations = 0;
        while (true) {
            std::size_t right = left;
            for (; right + 1 < dimension; ++right) {
                const double scale =
                    std::abs(diagonal[right])
                    + std::abs(diagonal[right + 1]);
                if (std::abs(off_diagonal[right])
                        <= epsilon * std::max(1.0, scale)) {
                    break;
                }
            }
            if (right == left) {
                break;
            }
            if (++iterations > 128) {
                return false;
            }

            double shift =
                (diagonal[left + 1] - diagonal[left])
                / (2.0 * off_diagonal[left]);
            double radius = std::hypot(shift, 1.0);
            shift = diagonal[right] - diagonal[left]
                + off_diagonal[left]
                / (shift + std::copysign(radius, shift));
            double sine = 1.0;
            double cosine = 1.0;
            double correction = 0.0;

            for (std::size_t reverse = right; reverse-- > left;) {
                const double sine_off = sine * off_diagonal[reverse];
                const double cosine_off = cosine * off_diagonal[reverse];
                if (std::abs(sine_off) >= std::abs(shift)) {
                    cosine = shift / sine_off;
                    radius = std::hypot(cosine, 1.0);
                    off_diagonal[reverse + 1] = sine_off * radius;
                    sine = 1.0 / radius;
                    cosine *= sine;
                } else {
                    sine = sine_off / shift;
                    radius = std::hypot(sine, 1.0);
                    off_diagonal[reverse + 1] = shift * radius;
                    cosine = 1.0 / radius;
                    sine *= cosine;
                }
                const double delta =
                    diagonal[reverse + 1] - correction;
                radius = (diagonal[reverse] - delta) * sine
                    + 2.0 * cosine * cosine_off;
                correction = sine * radius;
                diagonal[reverse + 1] = delta + correction;
                shift = cosine * radius - cosine_off;

                for (std::size_t row = 0; row < dimension; ++row) {
                    const double upper =
                        eigenvectors[row * dimension + reverse + 1];
                    const double lower =
                        eigenvectors[row * dimension + reverse];
                    eigenvectors[row * dimension + reverse + 1] =
                        sine * lower + cosine * upper;
                    eigenvectors[row * dimension + reverse] =
                        cosine * lower - sine * upper;
                }
            }
            diagonal[left] -= correction;
            off_diagonal[left] = shift;
            off_diagonal[right] = 0.0;
        }
    }
    return std::all_of(
        diagonal.begin(),
        diagonal.end(),
        [](double value) { return std::isfinite(value); });
}

inline bool solve_spd(
    const double* lower,
    std::size_t dimension,
    const double* rhs,
    std::size_t columns,
    std::vector<double>& solution,
    Backend backend = Backend::Portable) {

    solution.assign(dimension * columns, 0.0);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t i = 0; i < dimension; ++i) {
            double value = rhs[i * columns + column];
            value -= dot_strided(
                lower + i * dimension,
                1,
                solution.data() + column,
                columns,
                i,
                backend);
            const double diagonal = lower[i * dimension + i];
            if (!(diagonal > 0.0)) {
                return false;
            }
            solution[i * columns + column] = value / diagonal;
        }
        for (std::size_t reverse = dimension; reverse-- > 0;) {
            double value = solution[reverse * columns + column];
            const std::size_t count = dimension - reverse - 1;
            value -= dot_strided(
                lower + (reverse + 1) * dimension + reverse,
                dimension,
                solution.data() + (reverse + 1) * columns + column,
                columns,
                count,
                backend);
            const double diagonal = lower[reverse * dimension + reverse];
            solution[reverse * columns + column] = value / diagonal;
        }
    }
    return true;
}

}  // namespace scar_internal::linalg
