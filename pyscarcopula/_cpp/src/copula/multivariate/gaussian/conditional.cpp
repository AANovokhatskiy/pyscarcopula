#include "scar/copula/multivariate/gaussian/conditional.hpp"

#include "scar/detail/copula/multivariate/conditional.hpp"

namespace scar {

ConditionalSampleResult multivariate_gaussian_conditional(
    const std::vector<double>& correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    const std::vector<double>& given_latent,
    const std::vector<double>& normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return multivariate_gaussian_conditional(
        {correlations.data(), correlations.size()},
        correlation_rows,
        dimension,
        given_indices,
        {given_latent.data(), given_latent.size()},
        {normal_draws.data(), normal_draws.size()},
        n_rows,
        n_threads);
}

ConditionalSampleResult multivariate_gaussian_conditional(
    DoubleView correlations,
    std::int64_t correlation_rows,
    int dimension,
    const std::vector<int>& given_indices,
    DoubleView given_latent,
    DoubleView normal_draws,
    std::int64_t n_rows,
    int n_threads) {

    return scar_internal::conditional_latent(
        correlations,
        correlation_rows,
        dimension,
        given_indices,
        given_latent,
        normal_draws,
        n_rows,
        n_threads,
        scar_internal::ConditionalPolicy{});
}

}  // namespace scar
