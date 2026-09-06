#pragma once

#include "scar/copula/multivariate/rosenblatt.hpp"

#include <cstdint>
#include <vector>

namespace scar {

using DenseStudentRosenblattResult = MultivariateRosenblattResult;

DenseStudentRosenblattResult student_rosenblatt_dense(
    DoubleView correlation,
    int dimension,
    ObservationView u,
    DoubleView df,
    int n_threads = 1);

}  // namespace scar
