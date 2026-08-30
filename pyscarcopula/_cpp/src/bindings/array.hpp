#pragma once

#include "scar/core/span.hpp"
#include "scar/observation.hpp"

#include <pybind11/numpy.h>

#include <cstddef>
#include <vector>

namespace pyscarcopula::bindings {

using Float64Array = pybind11::array_t<
    double,
    pybind11::array::c_style | pybind11::array::forcecast>;
using IntArray = pybind11::array_t<
    int,
    pybind11::array::c_style | pybind11::array::forcecast>;

std::vector<std::vector<double>> observations_from_array(Float64Array values);
scar::ObservationView observation_view_from_array(
    int expected_dimension,
    const Float64Array& values);
scar::ObservationView finite_matrix_view_from_array(
    int expected_dimension,
    const Float64Array& values,
    const char* name);
std::vector<double> vector_from_array(Float64Array values);
std::vector<double> flat_vector_from_array(
    Float64Array values,
    const char* name);
scar::DoubleView flat_view_from_array(
    const Float64Array& values,
    const char* name);
std::vector<int> int_vector_from_array(IntArray values, const char* name);

pybind11::array_t<double> vector_to_array(
    const std::vector<double>& values);
/// Transfer an owning vector without allocating a second numerical buffer.
pybind11::array_t<double> vector_to_array(std::vector<double>&& values);
pybind11::array_t<double> matrix_to_array(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t columns);
/// Preserve a failure DTO as a flat array when no complete matrix exists.
pybind11::array_t<double> result_matrix_to_array(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t columns);
/// Preserve a failure DTO as a flat array when no complete tensor exists.
pybind11::array_t<double> result_tensor3_to_array(
    const std::vector<double>& values,
    std::size_t first,
    std::size_t second,
    std::size_t third);

}  // namespace pyscarcopula::bindings
