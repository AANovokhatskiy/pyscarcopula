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

inline bool cholesky_symmetric_with_jitter(
    const double* matrix,
    std::size_t dimension,
    std::vector<double>& lower,
    double* applied_jitter = nullptr,
    Backend backend = Backend::Portable) {

    lower.assign(dimension * dimension, 0.0);
    if (applied_jitter != nullptr) {
        *applied_jitter = std::numeric_limits<double>::quiet_NaN();
    }
    double jitter = 0.0;
    for (int attempt = 0; attempt < 7; ++attempt) {
        std::fill(lower.begin(), lower.end(), 0.0);
        bool valid = true;
        for (std::size_t i = 0; i < dimension && valid; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = 0.5 * (
                    matrix[i * dimension + j]
                    + matrix[j * dimension + i]);
                if (i == j) {
                    value += jitter;
                }
                value -= dot(
                    lower.data() + i * dimension,
                    lower.data() + j * dimension,
                    j,
                    backend);
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        valid = false;
                        break;
                    }
                    lower[i * dimension + j] = std::sqrt(value);
                } else {
                    const double diagonal = lower[j * dimension + j];
                    if (!(diagonal > 0.0)) {
                        valid = false;
                        break;
                    }
                    lower[i * dimension + j] = value / diagonal;
                }
            }
        }
        if (valid) {
            if (applied_jitter != nullptr) {
                *applied_jitter = jitter;
            }
            return true;
        }
        jitter = jitter == 0.0 ? 1e-12 : jitter * 10.0;
    }
    return false;
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
