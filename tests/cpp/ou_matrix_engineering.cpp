#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

int run_ou_matrix_engineering_tests() {
    using namespace scar_internal;
    // Exact storage equivalence at the old five-sigma support, including
    // boundary clipping and signed vectors. Independent dense reconstruction.
    OuGrid grid;
    if (!build_ou_grid(1.99, .2, .9*std::sqrt(3.98), 200,
                       257, 8., false, 4, 257, grid)) return 1;
    const int band = static_cast<int>(std::ceil(5*grid.r_kernel_grid));
    SparseTransitionMatrix compact;
    if (!build_sparse_transition_matrix(grid.z, grid.rho, grid.sigma_cond,
            grid.trap_w, grid.K, band, compact)) return 2;
    if (compact.row_start.size() != static_cast<std::size_t>(grid.K)
        || compact.data.size() <= compact.row_start.size()) return 3;
    std::vector<double> v(grid.K), actual(grid.K), transposed(grid.K);
    std::vector<double> expected(grid.K), expected_t(grid.K);
    for (int i = 0; i < grid.K; ++i) v[i] = std::sin(i*.31);
    for (int i = 0; i < grid.K; ++i) {
        const double center = grid.rho*grid.z[i];
        const double index = (center-grid.z.front())/(grid.z[1]-grid.z[0]);
        const int lo = std::max(0, static_cast<int>(std::floor(index))-band);
        const int hi = std::min(grid.K, static_cast<int>(std::ceil(index))+band+1);
        if (compact.row_start[i] != lo) return 4;
        for (int j = lo; j < hi; ++j) {
            const double q = (grid.z[j]-center)/grid.sigma_cond;
            const double p = std::exp(-.5*q*q)*grid.trap_w[j]
                /(grid.sigma_cond*std::sqrt(2*kPi));
            expected[i] += p*v[j];
            expected_t[j] += p*v[i];
        }
    }
    sparse_matvec(compact, grid.K, v, actual);
    sparse_transpose_matvec(compact, grid.K, v, transposed);
    for (int i = 0; i < grid.K; ++i) {
        if (std::abs(actual[i]-expected[i]) > 2e-14
            || std::abs(transposed[i]-expected_t[i]) > 2e-14) return 5;
    }
    // Directly sum omitted coefficients and absolute log-a derivatives,
    // independently of the continuous envelope used by the band selector.
    for (double a : {1e-6, .001, .01, .7, 100.}) {
        for (int K : {129, 513}) {
            const double h = 16./(K-1), rho = std::exp(-a);
            const double variance = -std::expm1(-2*a);
            const int radius = gaussian_transition_band(K, std::sqrt(variance)/h,
                                                        a, 8., 1000);
            if (radius < 0 || radius >= K) return 6;
            for (int i = 0; i < K; ++i) {
                const double z = -8.+i*h;
                const double center = rho*i+(1-rho)*.5*(K-1);
                double omitted = 0.;
                for (int j = 0; j < K; ++j) {
                    if (j >= std::floor(center)-radius
                        && j <= std::ceil(center)+radius) continue;
                    const double q = (-8.+j*h)-rho*z;
                    const double y = q/std::sqrt(variance);
                    const double p = h*std::exp(-.5*y*y)/std::sqrt(2*kPi*variance);
                    const double score = -a*rho/variance
                        * (rho*(1-y*y)+z*std::sqrt(variance)*y);
                    omitted += p*(1+std::abs(score));
                }
                if (omitted > kOuTransitionTailBudget/999.) return 7;
            }
        }
    }
    if (gaussian_transition_band(1, 1., .01, 8., 200) != -1
        || gaussian_transition_band(129, 1., .01, 8., 1) != -1
        || gaussian_transition_band(129, 1., std::numeric_limits<double>::quiet_NaN(),
                                    8., 200) != -1) return 8;
    return 0;
}
