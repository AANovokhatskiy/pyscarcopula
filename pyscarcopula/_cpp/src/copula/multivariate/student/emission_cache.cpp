#include "scar/copula/multivariate/student/emission_cache.hpp"
#include "scar/copula/multivariate/student/distribution.hpp"
#include "scar/copula/multivariate/student/quantile.hpp"
#include "scar/detail/safety.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <utility>

namespace scar_internal {
namespace {
struct Node {
    double x;
    std::vector<double> values, slopes;
};

void hermite(double left, double right, double yl, double yr,
             double sl, double sr, double x, double& value, double& slope) {
    const double h = right - left;
    const double u = (x - left) / h;
    const double b = h * sl;
    const double c = 3.0 * (yr - yl) - h * (2.0 * sl + sr);
    const double d = 2.0 * (yl - yr) + h * (sl + sr);
    value = yl + u * (b + u * (c + u * d));
    slope = (b + u * (2.0 * c + 3.0 * u * d)) / h;
}
}  // namespace

StudentEmissionCache::StudentEmissionCache(
    const PreparedStudentDensity& source, scar::ObservationView observations,
    double offset, const scar::StudentEmissionCacheConfig& config)
    : rows_(observations.n_obs), offset_(offset) {
    if (!source.valid || source.dense == nullptr || source.factor != nullptr
        || observations.values == nullptr || rows_ == 0
        || observations.dim != source.dimension || !std::isfinite(offset)
        || offset <= 2.0 || !std::isfinite(config.min_coordinate)
        || !std::isfinite(config.max_coordinate)
        || config.min_coordinate < -36.0 || config.max_coordinate > 13.0
        || config.min_coordinate >= config.max_coordinate
        || !std::isfinite(config.value_tolerance) || config.value_tolerance <= 0
        || !std::isfinite(config.score_tolerance) || config.score_tolerance <= 0
        || config.initial_intervals < 1 || config.initial_intervals > 256
        || config.max_knots < config.initial_intervals + 1
        || config.max_knots > 16385 || config.max_depth < 0
        || config.max_depth > 16) {
        throw std::invalid_argument("invalid Student emission cache configuration");
    }
    // Two final tables, plus bounded recursion/validation scratch. Account for
    // vector capacities before allocation; existing PPF/data storage is separate.
    std::size_t count = 0, bytes = 0;
    const std::size_t scratch_nodes = static_cast<std::size_t>(3 * config.max_depth + 12);
    if (!checked_size_mul(rows_, static_cast<std::size_t>(2 * config.max_knots)
                                   + 2 * scratch_nodes, count)
        || !checked_size_mul(count, sizeof(double), bytes)
        || bytes > config.max_bytes
        || config.max_knots * sizeof(double) > config.max_bytes - bytes) {
        throw std::invalid_argument("Student emission cache exceeds memory budget");
    }
    std::size_t entries = 0, dedup_bytes = 0, row_bytes = 0;
    const std::size_t bytes_per_entry = sizeof(std::pair<double, std::size_t>)
        + sizeof(std::size_t) + 3 * sizeof(double);
    const std::size_t base_bytes = bytes + config.max_knots * sizeof(double);
    if (!checked_size_mul(rows_, static_cast<std::size_t>(observations.dim), entries)
        || !checked_size_mul(entries, bytes_per_entry, dedup_bytes)
        || !checked_size_mul(static_cast<std::size_t>(observations.dim), 4 * sizeof(double), row_bytes)
        || dedup_bytes > config.max_bytes - base_bytes
        || row_bytes > config.max_bytes - base_bytes - dedup_bytes) {
        throw std::invalid_argument("Student emission cache deduplication exceeds memory budget");
    }
    diagnostics_.reserved_bytes = base_bytes + dedup_bytes + row_bytes;
    diagnostics_.observation_entries = entries;
    diagnostics_.value_tolerance = config.value_tolerance;
    diagnostics_.score_tolerance = config.score_tolerance;
    diagnostics_.min_coordinate = config.min_coordinate;
    diagnostics_.max_coordinate = config.max_coordinate;
    knots_.reserve(config.max_knots);
    values_.reserve(rows_ * config.max_knots);
    slopes_.reserve(rows_ * config.max_knots);
    // Empty cache disables both interpolation and the dynamic large-df rule.
    scar::copula::multivariate::student::PpfCache empty_ppf;
    PreparedStudentDensity model = source;
    model.ppf_cache = &empty_ppf;
    StudentWorkspace workspace;
    std::vector<std::size_t> probability_index(entries);
    std::vector<double> unique_probabilities;
    unique_probabilities.reserve(entries);
    {
        std::vector<std::pair<double, std::size_t>> ordered(entries);
        for (std::size_t index = 0; index < entries; ++index) {
            ordered[index] = {observations.values[index], index};
        }
        std::sort(ordered.begin(), ordered.end());
        for (const auto& item : ordered) {
            if (unique_probabilities.empty() || unique_probabilities.back() != item.first) {
                unique_probabilities.push_back(item.first);
            }
            probability_index[item.second] = unique_probabilities.size() - 1;
        }
    }
    diagnostics_.unique_probabilities = unique_probabilities.size();
    std::vector<double> quantiles(unique_probabilities.size());
    std::vector<double> quantile_slopes(unique_probabilities.size());
    std::vector<double> row_quantiles(observations.dim), row_slopes(observations.dim);
    auto prepare_quantiles = [&](const StudentDistributionParameters& distribution) {
        for (std::size_t index = 0; index < unique_probabilities.size(); ++index) {
            student_quantile_refined_value_and_derivative(unique_probabilities[index],
                distribution, quantiles[index], &quantile_slopes[index]);
        }
    };
    auto evaluate_row = [&](std::size_t row, double df, double& derivative) {
        for (int column = 0; column < observations.dim; ++column) {
            const auto index = probability_index[row * observations.dim + column];
            row_quantiles[column] = quantiles[index];
            row_slopes[column] = quantile_slopes[index];
        }
        return student_log_pdf_with_precomputed_quantiles(model,
            {row_quantiles.data(), row_quantiles.size()},
            {row_slopes.data(), row_slopes.size()}, df, workspace, &derivative);
    };
    endpoint_values_.resize(rows_);
    endpoint_slopes_.resize(rows_);
    const auto endpoint_distribution = student_distribution_parameters(offset);
    prepare_quantiles(endpoint_distribution);
    for (std::size_t row = 0; row < rows_; ++row) {
        endpoint_values_[row] = evaluate_row(row, offset, endpoint_slopes_[row]);
        if (!std::isfinite(endpoint_values_[row]) || !std::isfinite(endpoint_slopes_[row])) {
            throw std::runtime_error("nonfinite direct Student boundary emission");
        }
    }
    auto sample = [&](double x) {
        Node node{x, std::vector<double>(rows_), std::vector<double>(rows_)};
        const double excess = std::exp(x), df = offset + excess;
        const auto distribution = student_distribution_parameters(df);
        prepare_quantiles(distribution);
        for (std::size_t row = 0; row < rows_; ++row) {
            double derivative = 0.0;
            node.values[row] = evaluate_row(row, df, derivative);
            if (!std::isfinite(node.values[row]) || !std::isfinite(derivative)) {
                throw std::runtime_error("nonfinite direct Student cache sample");
            }
            node.slopes[row] = derivative * excess;
        }
        ++diagnostics_.sampled_coordinates;
        return node;
    };
    auto append = [&](const Node& node) {
        if (knots_.size() >= config.max_knots) {
            throw std::runtime_error("Student emission cache exhausted max_knots");
        }
        knots_.push_back(node.x);
        values_.insert(values_.end(), node.values.begin(), node.values.end());
        slopes_.insert(slopes_.end(), node.slopes.begin(), node.slopes.end());
    };
    std::function<void(const Node&, const Node&, int, const Node*)> refine;
    refine = [&](const Node& left, const Node& right, int depth, const Node* checked_midpoint) {
        double value_error = 0.0, score_error = 0.0;
        std::size_t value_error_row = 0, score_error_row = 0;
        const double midpoint_x = (left.x + right.x) * 0.5;
        Node owned_midpoint{};
        // Arithmetic reassociation can make the parent's quarter coordinate
        // differ by one ulp from this original midpoint formula. Only reuse
        // exact coordinate matches, preserving the former table bit for bit.
        if (checked_midpoint && checked_midpoint->x == midpoint_x) {
            ++diagnostics_.reused_samples;
        } else {
            owned_midpoint = sample(midpoint_x);
            checked_midpoint = &owned_midpoint;
        }
        const Node& midpoint = *checked_midpoint;
        auto check = [&](const Node& node) {
            for (std::size_t row = 0; row < rows_; ++row) {
                double value = 0.0, slope = 0.0;
                hermite(left.x, right.x, left.values[row], right.values[row],
                        left.slopes[row], right.slopes[row], node.x, value, slope);
                const double next_value_error = std::abs(value - node.values[row]);
                const double next_score_error = std::abs(slope - node.slopes[row]);
                if (next_value_error > value_error) {
                    value_error = next_value_error;
                    value_error_row = row;
                }
                if (next_score_error > score_error) {
                    score_error = next_score_error;
                    score_error_row = row;
                }
            }
        };
        check(midpoint);
        const Node left_quarter = sample(left.x * 0.75 + right.x * 0.25);
        check(left_quarter);
        const Node right_quarter = sample(left.x * 0.25 + right.x * 0.75);
        check(right_quarter);
        if (value_error <= config.value_tolerance
            && score_error <= config.score_tolerance) {
            diagnostics_.max_value_residual = std::max(
                diagnostics_.max_value_residual, value_error);
            diagnostics_.max_score_residual = std::max(
                diagnostics_.max_score_residual, score_error);
            append(right);
            return;
        }
        if (depth >= config.max_depth) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "Student emission cache exhausted max_depth=" << config.max_depth
                    << "; coordinate_interval=[" << left.x << "," << right.x << "]"
                    << "; df_interval=[" << offset + std::exp(left.x)
                    << "," << offset + std::exp(right.x) << "]"
                    << "; value_residual=" << value_error << " row=" << value_error_row
                    << "; score_residual=" << score_error << " row=" << score_error_row
                    << "; tolerances=[" << config.value_tolerance << ","
                    << config.score_tolerance << "]";
            throw std::runtime_error(message.str());
        }
        refine(left, midpoint, depth + 1, &left_quarter);
        refine(midpoint, right, depth + 1, &right_quarter);
    };
    Node left = sample(config.min_coordinate);
    append(left);
    for (std::size_t interval = 0; interval < config.initial_intervals; ++interval) {
        Node right = sample(config.min_coordinate
            + (config.max_coordinate - config.min_coordinate)
                * static_cast<double>(interval + 1) / config.initial_intervals);
        refine(left, right, 0, nullptr);
        left = std::move(right);
    }
    diagnostics_.active = true;
    diagnostics_.knots = knots_.size();
    diagnostics_.table_bytes = (values_.size() + slopes_.size() + knots_.size()
        + endpoint_values_.size() + endpoint_slopes_.size()) * sizeof(double);
}

bool StudentEmissionCache::evaluate(std::int64_t row, double df,
                                   double& value, double& derivative) const {
    if (row < 0 || static_cast<std::size_t>(row) >= rows_
        || !std::isfinite(df) || df < offset_) {
        fallbacks_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (df == offset_) {
        value = endpoint_values_[static_cast<std::size_t>(row)];
        derivative = endpoint_slopes_[static_cast<std::size_t>(row)];
        endpoint_hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    const double excess = df - offset_;
    const double x = std::log(excess);
    if (x < knots_.front() || x > knots_.back()) {
        fallbacks_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto upper = std::upper_bound(knots_.begin(), knots_.end(), x);
    const std::size_t interval = std::min(
        static_cast<std::size_t>(upper - knots_.begin() - 1), knots_.size() - 2);
    const std::size_t first = interval * rows_ + static_cast<std::size_t>(row);
    const std::size_t second = first + rows_;
    double slope = 0.0;
    hermite(knots_[interval], knots_[interval + 1], values_[first], values_[second],
            slopes_[first], slopes_[second], x, value, slope);
    derivative = slope / excess;
    hits_.fetch_add(1, std::memory_order_relaxed);
    return std::isfinite(value) && std::isfinite(derivative);
}
}  // namespace scar_internal
