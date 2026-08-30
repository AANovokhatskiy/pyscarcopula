#include "scar/copula/model_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace scar {
namespace {

template <typename T>
long double inversion_count(std::vector<T> values) {
    std::vector<T> scratch(values.size());
    long double inversions = 0.0L;
    for (std::size_t width = 1; width < values.size();) {
        for (std::size_t begin = 0; begin < values.size(); begin += 2 * width) {
            const std::size_t middle = std::min(begin + width, values.size());
            const std::size_t end = std::min(begin + 2 * width, values.size());
            std::size_t left = begin;
            std::size_t right = middle;
            std::size_t output = begin;
            while (left < middle && right < end) {
                if (values[left] <= values[right]) {
                    scratch[output++] = values[left++];
                } else {
                    scratch[output++] = values[right++];
                    inversions += static_cast<long double>(middle - left);
                }
            }
            while (left < middle) {
                scratch[output++] = values[left++];
            }
            while (right < end) {
                scratch[output++] = values[right++];
            }
            std::copy(
                scratch.begin() + static_cast<std::ptrdiff_t>(begin),
                scratch.begin() + static_cast<std::ptrdiff_t>(end),
                values.begin() + static_cast<std::ptrdiff_t>(begin));
        }
        if (width > values.size() / 2) {
            break;
        }
        width *= 2;
    }
    return inversions;
}

long double pair_count(std::size_t count) {
    return static_cast<long double>(count)
        * static_cast<long double>(count - 1) / 2.0L;
}

template <typename Equal>
long double tied_pair_count(
    std::size_t count,
    Equal equal_at) {

    long double output = 0.0L;
    std::size_t begin = 0;
    while (begin < count) {
        std::size_t end = begin + 1;
        while (end < count && equal_at(begin, end)) {
            ++end;
        }
        output += pair_count(end - begin);
        begin = end;
    }
    return output;
}

Result<std::vector<std::int64_t>> dense_ranks_impl(DoubleView values) {
    if (values.data() == nullptr || values.size() < 2) {
        return {{}, Status::InvalidSize, {}};
    }
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            return {{}, Status::InvalidParameter, {}};
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return values[left] < values[right];
    });
    std::vector<std::int64_t> ranks(values.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        if (index > 0 && values[order[index]] == values[order[index - 1]]) {
            return {{}, Status::InvalidParameter, {}};
        }
        ranks[order[index]] = static_cast<std::int64_t>(index + 1);
    }
    return success(std::move(ranks));
}

}  // namespace

Result<double> sum_values(DoubleView values) {
    if (values.data() == nullptr && !values.empty()) {
        return {0.0, Status::NullPointer, {}};
    }
    double output = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        output += values[index];
    }
    return success(output);
}

Result<std::int64_t> sum_int64(Span<const std::int64_t> values) {
    if (values.data() == nullptr && !values.empty()) {
        return {0, Status::NullPointer, {}};
    }
    std::int64_t output = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::int64_t value = values[index];
        if ((value > 0 && output > std::numeric_limits<std::int64_t>::max() - value)
            || (value < 0 && output < std::numeric_limits<std::int64_t>::min() - value)) {
            return {0, Status::NumericalFailure, {}};
        }
        output += value;
    }
    return success(output);
}

Result<double> sum_absolute(DoubleView values) {
    if (values.data() == nullptr && !values.empty()) {
        return {0.0, Status::NullPointer, {}};
    }
    double output = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        output += std::abs(values[index]);
    }
    return success(output);
}

Result<double> add_scores(double left, double right) {
    return success(left + right);
}

Result<double> information_criterion(
    double log_likelihood,
    std::int64_t parameter_count,
    std::int64_t observation_count,
    InformationCriterion criterion) {

    if (parameter_count < 0) {
        return {0.0, Status::InvalidParameter, {}};
    }
    const double count = static_cast<double>(parameter_count);
    switch (criterion) {
    case InformationCriterion::Aic:
        return success(-2.0 * log_likelihood + 2.0 * count);
    case InformationCriterion::Bic:
        if (observation_count <= 0) {
            return {0.0, Status::InvalidSize, {}};
        }
        return success(
            -2.0 * log_likelihood
            + count * std::log(static_cast<double>(observation_count)));
    case InformationCriterion::NegativeLogLikelihood:
        return success(-log_likelihood);
    }
    return {0.0, Status::InvalidParameter, {}};
}

Result<std::vector<std::int64_t>> dense_ranks_no_ties(DoubleView values) {
    return dense_ranks_impl(values);
}

Result<DenseRankMatrix> dense_rank_matrix_no_ties(DoubleMatrixView values) {
    if (values.values == nullptr || values.n_obs < 2 || values.dim < 1) {
        return {{}, Status::InvalidSize, {}};
    }
    DenseRankMatrix output;
    output.rows = values.n_obs;
    output.columns = static_cast<std::size_t>(values.dim);
    output.values.resize(output.rows * output.columns);
    std::vector<double> column(output.rows);
    for (std::size_t coordinate = 0; coordinate < output.columns; ++coordinate) {
        for (std::size_t row = 0; row < output.rows; ++row) {
            column[row] = values.row(row)[coordinate];
        }
        const auto ranks = dense_ranks_impl({column.data(), column.size()});
        if (!ranks.is_ok()) {
            FailureContext failure = ranks.failure;
            failure.coordinate = static_cast<int>(coordinate);
            return {{}, ranks.status, failure};
        }
        for (std::size_t row = 0; row < output.rows; ++row) {
            output.values[row * output.columns + coordinate] = ranks.value[row];
        }
    }
    return success(std::move(output));
}

Result<double> kendall_tau_from_dense_ranks(
    Span<const std::int64_t> first,
    Span<const std::int64_t> second) {

    if (first.data() == nullptr || second.data() == nullptr
        || first.size() != second.size() || first.size() < 2) {
        return {0.0, Status::InvalidSize, {}};
    }
    std::vector<std::size_t> order(first.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return first[left] < first[right];
    });
    std::vector<std::int64_t> ordered_second(order.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        if (first[order[index]] < 1 || second[order[index]] < 1
            || (index > 0 && first[order[index]] == first[order[index - 1]])) {
            return {0.0, Status::InvalidParameter, {}};
        }
        ordered_second[index] = second[order[index]];
    }
    std::vector<std::int64_t> second_check = ordered_second;
    std::sort(second_check.begin(), second_check.end());
    if (std::adjacent_find(second_check.begin(), second_check.end())
            != second_check.end()) {
        return {0.0, Status::InvalidParameter, {}};
    }
    const long double total = pair_count(order.size());
    const long double discordant = inversion_count(std::move(ordered_second));
    return success(static_cast<double>((total - 2.0L * discordant) / total));
}

Result<double> kendall_tau(DoubleView first, DoubleView second) {
    if (first.data() == nullptr || second.data() == nullptr
        || first.size() != second.size() || first.size() < 2) {
        return {0.0, Status::InvalidSize, {}};
    }
    using Pair = std::pair<double, double>;
    std::vector<Pair> pairs(first.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (!std::isfinite(first[index]) || !std::isfinite(second[index])) {
            return success(std::numeric_limits<double>::quiet_NaN());
        }
        pairs[index] = {first[index], second[index]};
    }
    std::sort(pairs.begin(), pairs.end());
    const long double ties_first = tied_pair_count(
        pairs.size(), [&](std::size_t left, std::size_t right) {
            return pairs[left].first == pairs[right].first;
        });
    const long double ties_both = tied_pair_count(
        pairs.size(), [&](std::size_t left, std::size_t right) {
            return pairs[left] == pairs[right];
        });
    std::vector<double> ordered_second(pairs.size());
    std::transform(
        pairs.begin(), pairs.end(), ordered_second.begin(),
        [](const Pair& value) { return value.second; });
    const long double discordant = inversion_count(ordered_second);
    std::sort(ordered_second.begin(), ordered_second.end());
    const long double ties_second = tied_pair_count(
        ordered_second.size(), [&](std::size_t left, std::size_t right) {
            return ordered_second[left] == ordered_second[right];
        });
    const long double total = pair_count(pairs.size());
    const long double denominator_first = total - ties_first;
    const long double denominator_second = total - ties_second;
    if (!(denominator_first > 0.0L) || !(denominator_second > 0.0L)) {
        return success(std::numeric_limits<double>::quiet_NaN());
    }
    const long double comparable =
        total - ties_first - ties_second + ties_both;
    return success(static_cast<double>(
        (comparable - 2.0L * discordant)
        / std::sqrt(denominator_first * denominator_second)));
}

Result<double> tau_for_itau(double tau, bool preserve_sign) {
    const double candidate = preserve_sign ? tau : std::abs(tau);
    if (candidate == 0.0 || candidate > 1.0 || candidate < -1.0) {
        return {0.0, Status::InvalidParameter, {}};
    }
    return success(candidate);
}

Result<bool> rotation_compatible(double tau, int rotation) {
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
        return {false, Status::InvalidRotation, {}};
    }
    if (std::abs(tau) < 0.15) {
        return success(true);
    }
    return success(
        rotation == 0 || rotation == 180 ? tau > 0.0 : tau < 0.0);
}

Result<bool> absolute_below(double value, double threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0) {
        return {false, Status::InvalidParameter, {}};
    }
    return success(std::abs(value) < threshold);
}

Result<double> absolute_value(double value) {
    return success(std::abs(value));
}

Result<bool> is_finite_value(double value) {
    return success(std::isfinite(value));
}

Result<bool> is_nan_value(double value) {
    return success(std::isnan(value));
}

}  // namespace scar
