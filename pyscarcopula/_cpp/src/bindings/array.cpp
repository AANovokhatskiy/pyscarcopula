#include "array.hpp"

#include "scar/core/checked_arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pyscarcopula::bindings {
namespace {

std::size_t checked_extent(py::ssize_t extent, const char* name) {
    if (extent < 0) {
        throw std::invalid_argument(
            std::string(name) + " shape is not representable");
    }
    return static_cast<std::size_t>(extent);
}

std::size_t flat_size(const py::buffer_info& info, const char* name) {
    std::size_t size = 1;
    for (const py::ssize_t extent : info.shape) {
        if (!scar::core::checked_size_mul(
                size, checked_extent(extent, name), size)) {
            throw std::invalid_argument(
                std::string(name) + " shape is not representable");
        }
    }
    return size;
}

void validate_finite_unit_interval(
    const double* data,
    std::size_t size,
    const char* name) {

    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(data[index])) {
            throw std::invalid_argument(
                std::string(name) + " must contain only finite values");
        }
        if (data[index] < 0.0 || data[index] > 1.0) {
            throw std::invalid_argument(
                std::string(name)
                + " must contain pseudo-observations in [0, 1]");
        }
    }
}

void validate_finite(
    const double* data,
    std::size_t size,
    const char* name) {

    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(data[index])) {
            throw std::invalid_argument(
                std::string(name) + " must contain only finite values");
        }
    }
}

}  // namespace

std::vector<std::vector<double>> observations_from_array(
    Float64Array values) {

    const py::buffer_info info = values.request();
    if (info.ndim != 2 || info.shape[1] < 2) {
        throw std::invalid_argument(
            "u must be a 2D float64 array with shape (n, d), d >= 2");
    }
    const std::size_t rows = checked_extent(info.shape[0], "u");
    const std::size_t columns = checked_extent(info.shape[1], "u");
    std::size_t size = 0;
    if (!scar::core::checked_size_mul(rows, columns, size)) {
        throw std::invalid_argument("u shape is not representable");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    validate_finite_unit_interval(data, size, "u");

    std::vector<std::vector<double>> output(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        output[row].assign(
            data + row * columns,
            data + (row + 1) * columns);
    }
    return output;
}

scar::ObservationView observation_view_from_array(
    int expected_dimension,
    const Float64Array& values) {

    const py::buffer_info info = values.request();
    if (expected_dimension < 0
        || info.ndim != 2
        || info.shape[1] != expected_dimension) {
        throw std::invalid_argument(
            "u must be a 2D float64 array with shape "
            "(n, expected copula dimension)");
    }
    const std::size_t rows = checked_extent(info.shape[0], "u");
    const std::size_t columns = checked_extent(info.shape[1], "u");
    std::size_t size = 0;
    if (!scar::core::checked_size_mul(rows, columns, size)) {
        throw std::invalid_argument("u shape is not representable");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    validate_finite_unit_interval(data, size, "u");
    return {data, rows, expected_dimension};
}

scar::ObservationView finite_matrix_view_from_array(
    int expected_dimension,
    const Float64Array& values,
    const char* name) {

    const py::buffer_info info = values.request();
    if (expected_dimension < 0
        || info.ndim != 2
        || info.shape[1] != expected_dimension) {
        throw std::invalid_argument(
            std::string(name) + " must be a 2D float64 array with shape "
            "(n, expected dimension)");
    }
    const std::size_t rows = checked_extent(info.shape[0], name);
    const std::size_t columns = checked_extent(info.shape[1], name);
    std::size_t size = 0;
    if (!scar::core::checked_size_mul(rows, columns, size)) {
        throw std::invalid_argument(
            std::string(name) + " shape is not representable");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    validate_finite(data, size, name);
    return {data, rows, expected_dimension};
}

std::vector<double> vector_from_array(Float64Array values) {
    const py::buffer_info info = values.request();
    const auto* data = static_cast<const double*>(info.ptr);
    if (info.ndim == 0) {
        if (!std::isfinite(data[0])) {
            throw std::invalid_argument(
                "values must contain only finite values");
        }
        return {data[0]};
    }
    if (info.ndim != 1) {
        throw std::invalid_argument(
            "values must be a scalar or a 1D float64 array");
    }
    const std::size_t size = checked_extent(info.shape[0], "values");
    std::vector<double> output(size);
    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(data[index])) {
            throw std::invalid_argument(
                "values must contain only finite values");
        }
        output[index] = data[index];
    }
    return output;
}

std::vector<double> flat_vector_from_array(
    Float64Array values,
    const char* name) {

    const py::buffer_info info = values.request();
    const auto* data = static_cast<const double*>(info.ptr);
    const std::size_t size = flat_size(info, name);
    std::vector<double> output(size);
    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(data[index])) {
            throw std::invalid_argument(
                std::string(name) + " must contain only finite values");
        }
        output[index] = data[index];
    }
    return output;
}

scar::DoubleView flat_view_from_array(
    const Float64Array& values,
    const char* name) {

    const py::buffer_info info = values.request();
    const auto* data = static_cast<const double*>(info.ptr);
    const std::size_t size = flat_size(info, name);
    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(data[index])) {
            throw std::invalid_argument(
                std::string(name) + " must contain only finite values");
        }
    }
    return {data, size};
}

std::vector<int> int_vector_from_array(IntArray values, const char* name) {
    const py::buffer_info info = values.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(
            std::string(name) + " must be a 1D integer array");
    }
    const auto* data = static_cast<const int*>(info.ptr);
    const std::size_t size = checked_extent(info.shape[0], name);
    return std::vector<int>(data, data + size);
}

py::array_t<double> vector_to_array(const std::vector<double>& values) {
    py::array_t<double> output(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::array_t<double> vector_to_array(std::vector<double>&& values) {
    auto storage = std::make_unique<std::vector<double>>(std::move(values));
    const auto count = static_cast<py::ssize_t>(storage->size());
    auto* data = storage->data();
    py::capsule owner(storage.get(), [](void* pointer) {
        delete static_cast<std::vector<double>*>(pointer);
    });
    storage.release();
    return py::array_t<double>(
        {count}, {static_cast<py::ssize_t>(sizeof(double))}, data, owner);
}

py::array_t<double> matrix_to_array(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t columns) {

    std::size_t expected = 0;
    if (!scar::core::checked_size_mul(rows, columns, expected)
        || expected != values.size()) {
        throw std::invalid_argument(
            "native matrix result has an inconsistent shape");
    }
    py::array_t<double> output({
        static_cast<py::ssize_t>(rows),
        static_cast<py::ssize_t>(columns),
    });
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::array_t<double> result_matrix_to_array(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t columns) {

    std::size_t expected = 0;
    if (!scar::core::checked_size_mul(rows, columns, expected)
        || expected != values.size()) {
        return vector_to_array(values);
    }
    return matrix_to_array(values, rows, columns);
}

py::array_t<double> result_tensor3_to_array(
    const std::vector<double>& values,
    std::size_t first,
    std::size_t second,
    std::size_t third) {

    std::size_t first_second = 0;
    std::size_t expected = 0;
    if (!scar::core::checked_size_mul(first, second, first_second)
        || !scar::core::checked_size_mul(first_second, third, expected)
        || expected != values.size()) {
        return vector_to_array(values);
    }
    py::array_t<double> output({
        static_cast<py::ssize_t>(first),
        static_cast<py::ssize_t>(second),
        static_cast<py::ssize_t>(third),
    });
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

}  // namespace pyscarcopula::bindings
