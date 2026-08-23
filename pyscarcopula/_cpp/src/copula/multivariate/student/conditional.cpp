#include "scar/copula/multivariate/student/conditional.hpp"

#include "scar/detail/copula/multivariate/conditional.hpp"
#include "scar/status.hpp"

#include <cmath>

namespace {

struct StudentConditionalContext {
    scar::DoubleView degrees_of_freedom;
    scar::DoubleView chi_square_draws;
};

int student_conditional_scale(
    const double* given,
    const double* solved_given,
    std::size_t n_given,
    std::size_t row,
    const void* raw_context,
    scar_internal::ConditionalScale& scale) {

    const auto& context =
        *static_cast<const StudentConditionalContext*>(raw_context);
    const double degrees = context.degrees_of_freedom[row];
    const double chi_square = context.chi_square_draws[row];
    double delta = 0.0;
    for (std::size_t i = 0; i < n_given; ++i) {
        delta += given[i] * solved_given[i];
    }
    const double conditional_df =
        degrees + static_cast<double>(n_given);
    if (!(degrees > 2.0) || !(conditional_df > 0.0)
        || !(chi_square > 0.0) || !std::isfinite(delta)) {
        return scar::SCAR_INVALID_PARAMETER;
    }
    scale.covariance = (degrees + delta) / conditional_df;
    scale.radial = std::sqrt(conditional_df / chi_square);
    return scar::SCAR_OK;
}

}  // namespace

namespace scar {

ConditionalSampleResult multivariate_student_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& df,
    const std::vector<double>& normal_draws,
    const std::vector<double>& chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    return multivariate_student_conditional(
        {correlations.data(), correlations.size()},
        correlation_rows,
        dimension,
        given_indices,
        {given_latent.data(), given_latent.size()},
        {df.data(), df.size()},
        {normal_draws.data(), normal_draws.size()},
        {chi_square_draws.data(), chi_square_draws.size()},
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_student_conditional(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    DoubleView df,
    DoubleView normal_draws,
    DoubleView chi_square_draws,
    std::int64_t n_rows,
    int n_threads) {

    const StudentConditionalContext context{df, chi_square_draws};
    scar_internal::ConditionalPolicy policy;
    policy.auxiliary_sizes_valid =
        n_rows > 0
        && df.size() == static_cast<std::size_t>(n_rows)
        && chi_square_draws.size() == static_cast<std::size_t>(n_rows);
    policy.require_prepared_factorization = false;
    policy.allow_jittered_prepared_factor = false;
    policy.scale = &student_conditional_scale;
    policy.context = &context;
    return scar_internal::conditional_latent(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_latent,
        normal_draws,
        n_rows,
        n_threads,
        policy);
}

}  // namespace scar
