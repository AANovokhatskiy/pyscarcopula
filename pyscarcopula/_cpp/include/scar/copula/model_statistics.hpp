#pragma once

#include "scar/core/matrix_view.hpp"
#include "scar/core/result.hpp"
#include "scar/core/span.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scar {

enum class InformationCriterion : int {
    Aic = 0,
    Bic = 1,
    NegativeLogLikelihood = 2,
};

struct DenseRankMatrix {
    std::vector<std::int64_t> values;
    std::size_t rows = 0;
    std::size_t columns = 0;
};

Result<double> sum_values(DoubleView values);
Result<std::int64_t> sum_int64(Span<const std::int64_t> values);
Result<double> sum_absolute(DoubleView values);
Result<double> add_scores(double left, double right);

Result<double> information_criterion(
    double log_likelihood,
    std::int64_t parameter_count,
    std::int64_t observation_count,
    InformationCriterion criterion);

Result<std::vector<std::int64_t>> dense_ranks_no_ties(DoubleView values);
Result<DenseRankMatrix> dense_rank_matrix_no_ties(DoubleMatrixView values);
Result<double> kendall_tau_from_dense_ranks(
    Span<const std::int64_t> first,
    Span<const std::int64_t> second);
Result<double> kendall_tau(DoubleView first, DoubleView second);

Result<double> tau_for_itau(double tau, bool preserve_sign);
Result<bool> rotation_compatible(double tau, int rotation);
Result<bool> absolute_below(double value, double threshold);
Result<double> absolute_value(double value);
Result<bool> is_finite_value(double value);
Result<bool> is_nan_value(double value);

}  // namespace scar
