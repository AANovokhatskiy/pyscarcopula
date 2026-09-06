#include "scar/ou.hpp"

#include "evaluator_internal.hpp"
#include "gradient_workspace.hpp"
#include "scar/detail/copula/common.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/detail/linalg.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/grid.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"
#include "scar/detail/scar_ou/transition.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scar {
using namespace evaluator_detail;

struct ScarOuEvaluator::Workspace : ScarOuEvaluatorWorkspace {};

ScarOuEvaluator::ScarOuEvaluator() = default;

ScarOuEvaluator::ScarOuEvaluator(
    const PreparedDynamicEmission* prepared_emission) noexcept
    : prepared_emission_(prepared_emission) {}

ScarOuEvaluator::~ScarOuEvaluator() = default;
ScarOuEvaluator::ScarOuEvaluator(ScarOuEvaluator&&) noexcept = default;
ScarOuEvaluator& ScarOuEvaluator::operator=(ScarOuEvaluator&&) noexcept =
    default;

ScarOuEvaluator::Workspace& ScarOuEvaluator::workspace() const {
    if (!workspace_) {
        workspace_ = std::make_unique<Workspace>();
    }
    return *workspace_;
}

namespace {

enum class CorrGradientMode {
    None,
    Full,
    Directional,
};

using GridGradientOperators = ScarOuGridGradientOperators;
using GridGradientWorkspace = ScarOuGridGradientWorkspace;
using SpectralGradientWorkspace = ScarOuSpectralGradientWorkspace;

const PreparedDynamicEmission& select_emission(
    const CopulaSpec& copula,
    const PreparedDynamicEmission* prepared,
    std::unique_ptr<PreparedDynamicEmission>& owner) {

    if (prepared != nullptr) {
        return *prepared;
    }
    owner = std::make_unique<PreparedDynamicEmission>(
        PreparedDynamicEmission::borrow(copula));
    return *owner;
}

bool prepare_gaussian_spectral_terms(
    const CopulaSpec& copula,
    std::int64_t n_obs,
    const std::vector<double>& r_grid,
    SpectralGradientWorkspace& workspace) {

    const std::size_t n_obs_size = static_cast<std::size_t>(n_obs);
    if (copula.family != CopulaFamily::Gaussian
        || copula.pair_gaussian_first_scores().size() != n_obs_size
        || copula.pair_gaussian_second_scores().size() != n_obs_size) {
        return false;
    }

    const std::size_t grid_size = r_grid.size();
    workspace.gaussian_r2.resize(grid_size);
    workspace.gaussian_omr2.resize(grid_size);
    workspace.gaussian_log_norm.resize(grid_size);
    workspace.gaussian_dlog_det.resize(grid_size);
    workspace.gaussian_omr2_squared.resize(grid_size);
    for (std::size_t j = 0; j < grid_size; ++j) {
        const double r = r_grid[j];
        const double r2 = r * r;
        const double omr2 = 1.0 - r2;
        workspace.gaussian_r2[j] = r2;
        workspace.gaussian_omr2[j] = omr2;
        workspace.gaussian_log_norm[j] = -0.5 * std::log(omr2);
        workspace.gaussian_dlog_det[j] = r / omr2;
        workspace.gaussian_omr2_squared[j] = omr2 * omr2;
    }
    return true;
}

void gaussian_spectral_pdf_and_grad_row(
    const CopulaSpec& copula,
    std::int64_t row,
    const std::vector<double>& r_grid,
    const std::vector<double>& dpsi_grid,
    const SpectralGradientWorkspace& workspace,
    double* fi_row,
    double* dfi_dx_row) {

    const std::size_t row_index = static_cast<std::size_t>(row);
    const double z1 = copula.pair_gaussian_first_scores()[row_index];
    const double z2 = copula.pair_gaussian_second_scores()[row_index];
    const double sum_squares = z1 * z1 + z2 * z2;
    const double cross_product = z1 * z2;
    for (std::size_t j = 0; j < r_grid.size(); ++j) {
        const double r = r_grid[j];
        const double r2 = workspace.gaussian_r2[j];
        const double omr2 = workspace.gaussian_omr2[j];
        const double numerator =
            r2 * sum_squares - 2.0 * r * cross_product;
        const double log_pdf =
            workspace.gaussian_log_norm[j] - 0.5 * numerator / omr2;
        const double pdf = std::exp(log_pdf);
        fi_row[j] = pdf;

        const double derivative_numerator =
            (2.0 * r * sum_squares - 2.0 * cross_product) * omr2
            + 2.0 * r * numerator;
        const double derivative_quadratic = derivative_numerator
            / workspace.gaussian_omr2_squared[j];
        const double derivative_log_pdf =
            workspace.gaussian_dlog_det[j]
            - 0.5 * derivative_quadratic;
        dfi_dx_row[j] =
            pdf * derivative_log_pdf * dpsi_grid[j];
    }
}

void dense_grid_matvec(
    const std::vector<double>& matrix,
    int K,
    const std::vector<double>& v,
    std::vector<double>& out) {

    scar_internal::linalg::row_major_matvec(
        matrix.data(),
        static_cast<std::size_t>(K),
        static_cast<std::size_t>(K),
        v.data(),
        out.data());
}

void local_grid_matvec(
    const GridGradientOperators& op,
    const std::vector<double>& values,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < op.K; ++row) {
        double acc = 0.0;
        const std::size_t offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(op.width);
        for (int j = 0; j < op.width; ++j) {
            const std::size_t idx = offset + static_cast<std::size_t>(j);
            acc += values[idx] * v[static_cast<std::size_t>(op.cols[idx])];
        }
        out[static_cast<std::size_t>(row)] = acc;
    }
}

void sparse_grid_matvec(
    const GridGradientOperators& op,
    const std::vector<double>& values,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < op.K; ++row) {
        double acc = 0.0;
        const int begin = op.indptr[static_cast<std::size_t>(row)];
        const int end = op.indptr[static_cast<std::size_t>(row) + 1];
        for (int offset = begin; offset < end; ++offset) {
            const std::size_t idx = static_cast<std::size_t>(offset);
            acc += values[idx]
                * v[static_cast<std::size_t>(op.cols[idx])];
        }
        out[static_cast<std::size_t>(row)] = acc;
    }
}

void operator_matvec(
    const GridGradientOperators& op,
    bool gradient,
    const std::vector<double>& v,
    std::vector<double>& out) {

    if (op.local) {
        local_grid_matvec(op, gradient ? op.grad_vals : op.vals, v, out);
    } else if (op.sparse) {
        sparse_grid_matvec(op, gradient ? op.grad_vals : op.vals, v, out);
    } else {
        dense_grid_matvec(gradient ? op.dense_grad : op.dense, op.K, v, out);
    }
}

void operator_transpose_matvec(
    const GridGradientOperators& op,
    const std::vector<double>& v,
    std::vector<double>& out) {

    std::fill(out.begin(), out.end(), 0.0);
    if (op.local || op.sparse) {
        for (int row = 0; row < op.K; ++row) {
            const double source = v[static_cast<std::size_t>(row)];
            const int begin = op.sparse
                ? op.indptr[static_cast<std::size_t>(row)]
                : row * op.width;
            const int end = op.sparse
                ? op.indptr[static_cast<std::size_t>(row) + 1]
                : begin + op.width;
            for (int offset = begin; offset < end; ++offset) {
                const std::size_t idx = static_cast<std::size_t>(offset);
                out[static_cast<std::size_t>(op.cols[idx])] +=
                    op.vals[idx] * source;
            }
        }
        return;
    }

    for (int row = 0; row < op.K; ++row) {
        const double source = v[static_cast<std::size_t>(row)];
        const std::size_t offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(op.K);
        for (int col = 0; col < op.K; ++col) {
            out[static_cast<std::size_t>(col)] +=
                op.dense[offset + static_cast<std::size_t>(col)] * source;
        }
    }
}

constexpr std::size_t kGradientEmissionBlockElements = 1U << 20;

bool fill_gradient_emission_block(
    const PreparedDynamicEmission& emission,
    const double* observations,
    std::int64_t first_row,
    std::int64_t end_row,
    const std::vector<double>& parameters,
    const std::vector<double>& derivatives,
    int n_threads,
    std::vector<double>& densities,
    std::vector<double>& gradients,
    std::vector<double>& log_scales) {

    return emission.fill_density_and_gradient_block(
        observations, first_row, end_row - first_row,
        parameters, derivatives, densities, gradients, log_scales, n_threads);
}

bool build_dense_grid_gradient_operator(
    const std::vector<double>& xi,
    const std::vector<double>& base_w,
    double rho,
    GridGradientOperators& op) {

    const int K = static_cast<int>(xi.size());
    const double omr2 = 1.0 - rho * rho;
    if (K < 2 || omr2 <= 0.0) {
        return false;
    }
    const double coeff = 1.0 / (std::sqrt(omr2) * std::sqrt(2.0 * scar_internal::kPi));
    op = GridGradientOperators{};
    op.K = K;
    op.local = false;
    std::size_t K_size = 0;
    std::size_t matrix_size = 0;
    if (!scar_internal::checked_positive_int_size(
            K, scar_internal::kMaxDenseGridSize, K_size)
        || !scar_internal::checked_size_mul(K_size, K_size, matrix_size)) {
        return false;
    }
    op.dense.assign(matrix_size, 0.0);
    op.dense_grad.assign(matrix_size, 0.0);

    for (int row = 0; row < K; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * K_size;
        for (int col = 0; col < K; ++col) {
            const double q = xi[static_cast<std::size_t>(col)]
                - rho * xi[static_cast<std::size_t>(row)];
            const double tw = coeff
                * std::exp(-0.5 * q * q / omr2)
                * base_w[static_cast<std::size_t>(col)];
            const double dlog = rho / omr2
                + q * xi[static_cast<std::size_t>(row)] / omr2
                - rho * q * q / (omr2 * omr2);
            const std::size_t idx =
                row_offset + static_cast<std::size_t>(col);
            op.dense[idx] = tw;
            op.dense_grad[idx] = dlog * tw;
        }
    }
    return true;
}

bool build_sparse_grid_gradient_operator(
    const std::vector<double>& xi,
    const std::vector<double>& base_w,
    double rho,
    int band,
    GridGradientOperators& op) {

    const int K = static_cast<int>(xi.size());
    const double omr2 = 1.0 - rho * rho;
    if (K < 2 || omr2 <= 0.0 || base_w.size() != xi.size()) {
        return false;
    }
    const double dxi = xi[1] - xi[0];
    if (!std::isfinite(dxi)
        || dxi <= 0.0
        || band < 0) {
        return false;
    }
    op = GridGradientOperators{};
    op.K = K;
    op.sparse = true;
    op.indptr.assign(static_cast<std::size_t>(K) + 1, 0);

    std::size_t nnz = 0;
    const double midpoint = 0.5 * static_cast<double>(K - 1);
    for (int row = 0; row < K; ++row) {
        const double i_center =
            rho * static_cast<double>(row)
            + (1.0 - rho) * midpoint;
        const double lo_value = std::floor(i_center) - band;
        const double hi_value = std::ceil(i_center) + band + 1.0;
        const int lo = lo_value <= 0.0
            ? 0
            : (lo_value >= static_cast<double>(K)
                ? K
                : static_cast<int>(lo_value));
        const int hi = hi_value <= 0.0
            ? 0
            : (hi_value >= static_cast<double>(K)
                ? K
                : static_cast<int>(hi_value));
        const std::size_t width = hi > lo
            ? static_cast<std::size_t>(hi - lo)
            : 0;
        if (!scar_internal::checked_size_add(nnz, width, nnz)
            || nnz > static_cast<std::size_t>(INT_MAX)) {
            return false;
        }
        op.indptr[static_cast<std::size_t>(row) + 1] =
            static_cast<int>(nnz);
    }

    op.cols.resize(nnz);
    op.vals.resize(nnz);
    op.grad_vals.resize(nnz);
    const double coeff =
        1.0 / (std::sqrt(omr2) * std::sqrt(2.0 * scar_internal::kPi));
    for (int row = 0; row < K; ++row) {
        const double xi_row = xi[static_cast<std::size_t>(row)];
        const double center = rho * xi_row;
        const double i_center =
            rho * static_cast<double>(row)
            + (1.0 - rho) * midpoint;
        const double lo_value = std::floor(i_center) - band;
        const int lo = lo_value <= 0.0
            ? 0
            : (lo_value >= static_cast<double>(K)
                ? K
                : static_cast<int>(lo_value));
        const int begin = op.indptr[static_cast<std::size_t>(row)];
        const int end = op.indptr[static_cast<std::size_t>(row) + 1];
        for (int offset = begin; offset < end; ++offset) {
            const int col = lo + offset - begin;
            const double q =
                xi[static_cast<std::size_t>(col)] - center;
            const double tw = coeff
                * std::exp(-0.5 * q * q / omr2)
                * base_w[static_cast<std::size_t>(col)];
            const double dlog = rho / omr2
                + q * xi_row / omr2
                - rho * q * q / (omr2 * omr2);
            const std::size_t idx = static_cast<std::size_t>(offset);
            op.cols[idx] = col;
            op.vals[idx] = tw;
            op.grad_vals[idx] = dlog * tw;
        }
    }
    return true;
}

bool build_local_grid_gradient_operator(
    const std::vector<double>& xi,
    double rho,
    int gh_order,
    GridGradientOperators& op) {

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    if (!scar_internal::physicists_hermite_normal_rule(
            gh_order, gh_nodes, gh_weights)) {
        return false;
    }
    const int K = static_cast<int>(xi.size());
    const int q_count = static_cast<int>(gh_nodes.size());
    if (q_count <= 0 || q_count > INT_MAX / 2) {
        return false;
    }
    const int width = q_count * 2;
    const double s2 = 1.0 - rho * rho;
    if (K < 2 || width <= 0 || s2 <= 0.0) {
        return false;
    }
    const double s = std::sqrt(s2);
    const double xi0 = xi.front();
    const double xi_last = xi.back();
    const double dxi = xi[1] - xi[0];

    op = GridGradientOperators{};
    op.K = K;
    op.width = width;
    op.local = true;
    std::size_t K_size = 0;
    std::size_t width_size = 0;
    std::size_t operator_size = 0;
    if (!scar_internal::checked_positive_int_size(
            K, scar_internal::kMaxGridSize, K_size)
        || !scar_internal::checked_positive_int_size(
            width,
            2 * scar_internal::kMaxSpectralOrder,
            width_size)
        || !scar_internal::checked_size_mul(
            K_size, width_size, operator_size)) {
        return false;
    }
    op.cols.assign(operator_size, 0);
    op.vals.assign(operator_size, 0.0);
    op.grad_vals.assign(operator_size, 0.0);

    for (int row = 0; row < K; ++row) {
        const double center = rho * xi[static_cast<std::size_t>(row)];
        const double dcenter_drho = xi[static_cast<std::size_t>(row)];
        for (int q = 0; q < q_count; ++q) {
            const double node = gh_nodes[static_cast<std::size_t>(q)];
            const double weight = gh_weights[static_cast<std::size_t>(q)];
            const double offset = std::sqrt(2.0) * s * node;
            const double doffset_drho = -std::sqrt(2.0) * rho / s * node;
            const double y = center + offset;
            const std::size_t pos =
                static_cast<std::size_t>(row) * width_size
                + 2 * static_cast<std::size_t>(q);

            if (y <= xi0) {
                op.cols[pos] = 0;
                op.cols[pos + 1] = 0;
                op.vals[pos] = weight;
                continue;
            }
            if (y >= xi_last) {
                op.cols[pos] = K - 1;
                op.cols[pos + 1] = K - 1;
                op.vals[pos] = weight;
                continue;
            }

            int left = static_cast<int>(std::floor((y - xi0) / dxi));
            if (left >= K - 1) {
                op.cols[pos] = K - 1;
                op.cols[pos + 1] = K - 1;
                op.vals[pos] = weight;
                continue;
            }

            const double lam = (y - xi[static_cast<std::size_t>(left)]) / dxi;
            const double dlam_drho = (dcenter_drho + doffset_drho) / dxi;
            op.cols[pos] = left;
            op.cols[pos + 1] = left + 1;
            op.vals[pos] = weight * (1.0 - lam);
            op.vals[pos + 1] = weight * lam;
            op.grad_vals[pos] = -weight * dlam_drho;
            op.grad_vals[pos + 1] = weight * dlam_drho;
        }
    }
    return true;
}

GradLogLikResult grid_neg_loglik_with_grad(
    const OuParams& params,
    const CopulaSpec& requested_copula,
    ObservationView u,
    const OuNumericalConfig& config,
    OuBackend backend,
    const PreparedDynamicEmission* prepared_emission,
    CorrGradientMode corr_gradient_mode = CorrGradientMode::None,
    const std::vector<double>* corr_direction = nullptr,
    GridGradientWorkspace* workspace = nullptr) {

    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        select_emission(
            requested_copula, prepared_emission, emission_owner);
    const CopulaSpec& copula = emission.compatibility_spec();
    if (!supported_ou_copula(emission)) {
        return invalid_grad(SCAR_INVALID_TRANSFORM, backend);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, backend);
    }
    if (n_obs < 2 || config.K < 2 || config.grid_range <= 0.0
        || config.pts_per_sigma <= 0) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (!valid_grid_config(config, backend)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (backend == OuBackend::LocalGh && config.gh_order <= 0) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    if (backend == OuBackend::LocalGh
        && static_cast<std::size_t>(config.gh_order)
            > scar_internal::kMaxSpectralOrder) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    const bool correlation_gradient =
        corr_gradient_mode != CorrGradientMode::None;
    if (correlation_gradient && copula.family != CopulaFamily::Student) {
        return invalid_grad(SCAR_INVALID_FAMILY, backend);
    }
    if (correlation_gradient
        && copula.correlation_kind == CorrelationKind::Factor) {
        // Factor correlation is a two-stage OU mode.  Its loading score is
        // intentionally not part of the dense off-diagonal correlation
        // coordinate system used by these entry points.
        return invalid_grad(SCAR_INVALID_TRANSFORM, backend);
    }

    const double dt = 1.0 / static_cast<double>(n_obs - 1);
    const double rho = std::exp(-params.kappa * dt);
    const double sigma = std::sqrt(0.5 * params.nu * params.nu / params.kappa);
    const double conditional_variance =
        -std::expm1(-2.0 * params.kappa * dt);
    const double sigma_cond = sigma * std::sqrt(conditional_variance);
    if (!std::isfinite(sigma) || !std::isfinite(sigma_cond)
        || sigma <= 0.0 || sigma_cond <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
    }

    int K_adaptive = config.K;
    double K_requested = static_cast<double>(config.K);
    if (config.adaptive) {
        const double dz_target =
            sigma_cond / static_cast<double>(config.pts_per_sigma);
        const double K_min_value =
            std::ceil(2.0 * config.grid_range * sigma / dz_target) + 1.0;
        if (!std::isfinite(K_min_value) || K_min_value < 2.0) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
        }
        K_requested = std::max(K_requested, K_min_value);
        if (K_min_value > static_cast<double>(scar_internal::kMaxGridSize)
            || K_min_value > static_cast<double>(INT_MAX)) {
            if (config.max_K <= 0) {
                return invalid_grad(SCAR_INVALID_SIZE, backend);
            }
            K_adaptive =
                config.max_K == INT_MAX ? INT_MAX : config.max_K + 1;
        } else {
            const int K_min = static_cast<int>(K_min_value);
            K_adaptive = std::max(config.K, K_min);
        }
    }
    int K_eff = K_adaptive;
    if (config.max_K > 0) {
        K_eff = std::min(K_adaptive, config.max_K);
        K_eff = std::max(K_eff, std::min(config.K, config.max_K));
    }
    if (K_eff < 2) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    const std::size_t K_eff_size = static_cast<std::size_t>(K_eff);
    if (K_eff_size > scar_internal::kMaxGridSize
        || (backend == OuBackend::Matrix
            && config.grid_method == OuGridMethod::Dense
            && K_eff_size > scar_internal::kMaxDenseGridSize)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }

    const double dxi =
        2.0 * config.grid_range / static_cast<double>(K_eff - 1);
    const double r_kernel_grid = std::sqrt(conditional_variance) / dxi;
    const bool adaptive_was_capped = K_eff < K_adaptive;
    if (backend == OuBackend::Matrix
        && (adaptive_was_capped || r_kernel_grid <= config.r_gh)) {
        // Explicit matrix mode remains a valid caller-selected backend.
    }

    GridGradientWorkspace local_workspace;
    GridGradientWorkspace& ws =
        workspace == nullptr ? local_workspace : *workspace;

    std::vector<double>& xi = ws.xi;
    std::vector<double>& base_w = ws.base_w;
    std::vector<double>& pw_const = ws.pw_const;
    std::vector<double>& x_grid = ws.x_grid;
    xi.assign(static_cast<std::size_t>(K_eff), 0.0);
    base_w.assign(static_cast<std::size_t>(K_eff), dxi);
    pw_const.assign(static_cast<std::size_t>(K_eff), 0.0);
    x_grid.assign(static_cast<std::size_t>(K_eff), 0.0);
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        xi[idx] = -config.grid_range + dxi * static_cast<double>(j);
        x_grid[idx] = params.mu + sigma * xi[idx];
    }
    base_w.front() *= 0.5;
    base_w.back() *= 0.5;
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        pw_const[idx] = std::exp(-0.5 * xi[idx] * xi[idx])
            / std::sqrt(2.0 * scar_internal::kPi)
            * base_w[idx];
    }

    GridGradientOperators& op = ws.op;
    bool built = false;
    if (backend == OuBackend::LocalGh) {
        built = build_local_grid_gradient_operator(
            xi, rho, config.gh_order, op);
    } else {
        const double band_value =
            std::ceil(scar_internal::kOuTransitionTailSigma * r_kernel_grid);
        if (!std::isfinite(band_value)
            || band_value < 0.0
            || band_value > static_cast<double>(INT_MAX)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        const int band = static_cast<int>(band_value);
        const bool sparse =
            config.grid_method == OuGridMethod::Sparse
            || (config.grid_method == OuGridMethod::Auto
                && (K_eff_size > scar_internal::kMaxDenseGridSize
                    || band_value < static_cast<double>(K_eff / 4)));
        built = sparse
            ? build_sparse_grid_gradient_operator(
                xi, base_w, rho, band, op)
            : build_dense_grid_gradient_operator(xi, base_w, rho, op);
    }
    if (!built) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
    }

    const double* observation_values = observation_data(emission, u);
    const std::size_t K_size = static_cast<std::size_t>(K_eff);
    std::int64_t emission_block_rows =
        static_cast<std::int64_t>(std::max<std::size_t>(
            1, kGradientEmissionBlockElements / K_size));
    if (correlation_gradient) {
        // Three blocks coexist: fi, dfi_dx and alpha_history. Larger blocks
        // avoid recomputing emissions and forward messages at checkpoints.
        // This changes storage only, never quadrature or transition support.
        const std::uint64_t rows = config.corr_gradient_block_bytes
            / (3U * sizeof(double) * K_size);
        if (rows == 0) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        emission_block_rows = static_cast<std::int64_t>(
            std::min<std::uint64_t>(rows, static_cast<std::uint64_t>(n_obs)));
    }
    std::vector<double>& fi = ws.fi;
    std::vector<double>& dfi_dx = ws.dfi_dx;
    std::vector<double>& r_grid = ws.r_grid;
    std::vector<double>& dpsi_grid = ws.dpsi_grid;
    r_grid.clear();
    dpsi_grid.clear();
    emission.prepare_grid_transform(
        x_grid, r_grid, dpsi_grid);
    std::vector<double>& beta = ws.beta;
    std::vector<double>& emission_log_scales = ws.emission_log_scales;
    std::vector<double>& target = ws.target;
    std::vector<double>& next = ws.next;
    const double drho_dkappa = -dt * rho;
    const double dsigma_dkappa = -0.5 * sigma / params.kappa;
    const double dsigma_dnu = sigma / params.nu;
    std::size_t triple_K = 0;
    if (!scar_internal::checked_size_mul(3, K_size, triple_K)) {
        return invalid_grad(SCAR_INVALID_SIZE, backend);
    }
    std::vector<double>& dx_dalpha = ws.dx_dalpha;
    dx_dalpha.assign(triple_K, 0.0);
    for (int j = 0; j < K_eff; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j);
        dx_dalpha[idx] = dsigma_dkappa * xi[idx];
        dx_dalpha[static_cast<std::size_t>(K_eff) + idx] = 1.0;
        dx_dalpha[2 * K_size + idx] =
            dsigma_dnu * xi[idx];
    }

    std::vector<double>& d_beta = ws.d_beta;
    std::vector<double>& new_d_beta = ws.new_d_beta;
    std::vector<double>& d_target = ws.d_target;
    std::vector<double>& contrib = ws.contrib;
    std::vector<double>& transition_grad = ws.transition_grad;
    beta.assign(K_size, 1.0);
    target.assign(K_size, 0.0);
    next.assign(K_size, 0.0);
    d_beta.assign(triple_K, 0.0);
    new_d_beta.assign(triple_K, 0.0);
    d_target.assign(K_size, 0.0);
    contrib.assign(K_size, 0.0);
    transition_grad.assign(K_size, 0.0);

    std::vector<double> corr_grad;
    const bool directional =
        corr_gradient_mode == CorrGradientMode::Directional;
    if (correlation_gradient) {
        if (!scar_internal::student_precision_matrix(copula, ws.precision)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        std::size_t n_corr = 0;
        if (!scar_internal::valid_student_correlation_count(copula.dim, n_corr)
            || (directional
                && (corr_direction == nullptr
                    || corr_direction->size() != n_corr))) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        const std::size_t score_width = directional ? 1 : n_corr;
        std::size_t score_elements = 0;
        if (!scar_internal::checked_size_mul(
                K_size, score_width, score_elements)) {
            return invalid_grad(SCAR_INVALID_SIZE, backend);
        }
        corr_grad.assign(score_width, 0.0);
        ws.scores.assign(score_elements, 0.0);
        ws.alpha_source.assign(K_size, 0.0);
        ws.alpha_next.assign(K_size, 0.0);

        // Balanced checkpointing retains one K-vector per split, not a
        // T-by-K history or one transition sensitivity per correlation.
        std::size_t levels = 0;
        for (std::int64_t span = n_obs; span > emission_block_rows;
             span = span / 2 + span % 2) {
            ++levels;
        }
        ws.alpha_checkpoints.resize(levels);
        for (auto& checkpoint : ws.alpha_checkpoints) {
            checkpoint.resize(K_size);
        }
    }

    auto fill_block = [&](std::int64_t first, std::int64_t end) {
        return fill_gradient_emission_block(
            emission, observation_values, first, end, r_grid, dpsi_grid,
            config.n_threads, fi, dfi_dx, emission_log_scales);
    };
    auto forward_step = [&](const double* density,
                            std::vector<double>& alpha) {
        for (std::size_t j = 0; j < K_size; ++j) {
            ws.alpha_source[j] = alpha[j] * density[j];
        }
        operator_transpose_matvec(op, ws.alpha_source, ws.alpha_next);
        double scale = 0.0;
        for (double value : ws.alpha_next) {
            scale = std::max(scale, std::abs(value));
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return false;
        }
        for (double& value : ws.alpha_next) {
            value /= scale;
        }
        alpha.swap(ws.alpha_next);
        return true;
    };

    // The backward state and its three OU sensitivities roll across leaves.
    // Row scales are counted only here, never during checkpoint recomputation.
    double emission_log_scale = 0.0;
    double cumul_logc = 0.0;
    auto backward_block = [&](std::int64_t first, std::int64_t end,
                              const std::vector<double>* prior) {
        if (!fill_block(first, end)) {
            return false;
        }
        if (prior != nullptr) {
            ws.alpha = *prior;
            ws.alpha_history.resize(fi.size());
            for (std::int64_t t = first; t < end; ++t) {
                const std::size_t offset =
                    static_cast<std::size_t>(t - first) * K_size;
                std::copy(ws.alpha.begin(), ws.alpha.end(),
                          ws.alpha_history.data() + offset);
                if (t + 1 < end
                    && !forward_step(fi.data() + offset, ws.alpha)) {
                    return false;
                }
            }
        }
        for (double row_scale : emission_log_scales) {
            emission_log_scale += row_scale;
        }
        for (std::int64_t row = end - 1; row >= first; --row) {
            const std::size_t row_offset =
                static_cast<std::size_t>(row - first) * K_size;
            const double* fi_row = fi.data() + row_offset;
            const double* dfi_dx_row = dfi_dx.data() + row_offset;
            if (prior != nullptr) {
                const double* observation = observation_values
                    + static_cast<std::size_t>(row)
                        * static_cast<std::size_t>(copula.dim);
                const bool scored = directional
                    ? scar_internal::student_corr_directional_score_row(
                        copula, observation, row, r_grid, ws.precision,
                        *corr_direction, ws.scores.data())
                    : scar_internal::student_corr_score_row(
                        copula, observation, row, r_grid, ws.precision,
                        ws.scores.data());
                if (!scored) {
                    return false;
                }
                const double* alpha = ws.alpha_history.data() + row_offset;
                double total = 0.0;
                for (std::size_t j = 0; j < K_size; ++j) {
                    total += alpha[j] * fi_row[j] * beta[j];
                }
                if (!std::isfinite(total) || total <= 0.0) {
                    return false;
                }
                for (std::size_t j = 0; j < K_size; ++j) {
                    const double posterior =
                        alpha[j] * fi_row[j] * beta[j] / total;
                    const std::size_t offset = j * corr_grad.size();
                    for (std::size_t p = 0; p < corr_grad.size(); ++p) {
                        corr_grad[p] += posterior * ws.scores[offset + p];
                    }
                }
            }
            if (row == 0) {
                // f_0 is integrated against the stationary prior below.
                continue;
            }
            for (std::size_t j = 0; j < K_size; ++j) {
                target[j] = fi_row[j] * beta[j];
            }
            operator_matvec(op, false, target, next);
            double scale = 0.0;
            for (double value : next) {
                scale = std::max(scale, std::abs(value));
            }
            if (!std::isfinite(scale) || scale <= 0.0) {
                return false;
            }
            operator_matvec(op, true, target, transition_grad);
            const double inv_scale = 1.0 / scale;
            for (int p = 0; p < 3; ++p) {
                const std::size_t p_offset =
                    static_cast<std::size_t>(p) * K_size;
                for (std::size_t j = 0; j < K_size; ++j) {
                    const double dfi = dfi_dx_row[j] * dx_dalpha[p_offset + j];
                    d_target[j] = dfi * beta[j] + fi_row[j] * d_beta[p_offset + j];
                }
                operator_matvec(op, false, d_target, contrib);
                if (p == 0) {
                    for (std::size_t j = 0; j < K_size; ++j) {
                        contrib[j] += transition_grad[j] * drho_dkappa;
                    }
                }
                for (std::size_t j = 0; j < K_size; ++j) {
                    new_d_beta[p_offset + j] = contrib[j] * inv_scale;
                }
            }
            for (std::size_t j = 0; j < K_size; ++j) {
                // Preserve legacy division: reciprocal multiplication changes
                // the analytical gradient's rounding and optimizer trajectory.
                beta[j] = next[j] / scale;
            }
            d_beta.swap(new_d_beta);
            cumul_logc += std::log(scale);
        }
        return true;
    };

    if (correlation_gradient) {
        // Recompute the midpoint's predictive distribution, visit the right
        // half first, then the left. beta remains the correct future message
        // at every leaf. Both halves shrink; depth is at most 63 for int64 T.
        auto visit = [&](auto&& self, std::int64_t first, std::int64_t end,
                         std::size_t depth,
                         const std::vector<double>& prior) -> bool {
            if (end - first <= emission_block_rows) {
                return backward_block(first, end, &prior);
            }
            // Balance by emission blocks, keeping full blocks on the right.
            // Just above a block boundary only the short left remainder
            // needs recomputing, rather than half of the entire history.
            const std::int64_t blocks =
                (end - first - 1) / emission_block_rows + 1;
            const std::int64_t middle =
                end - (blocks / 2) * emission_block_rows;
            auto& midpoint = ws.alpha_checkpoints[depth];
            midpoint = prior;
            for (std::int64_t begin = first; begin < middle;) {
                const std::int64_t block_end = begin
                    + std::min(emission_block_rows, middle - begin);
                if (!fill_block(begin, block_end)) {
                    return false;
                }
                for (std::int64_t t = begin; t < block_end; ++t) {
                    const std::size_t offset =
                        static_cast<std::size_t>(t - begin) * K_size;
                    if (!forward_step(fi.data() + offset, midpoint)) {
                        return false;
                    }
                }
                begin = block_end;
            }
            return self(self, middle, end, depth + 1, midpoint)
                && self(self, first, middle, depth + 1, prior);
        };
        if (!visit(visit, 0, n_obs, 0, pw_const)) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
        }
    } else {
        for (std::int64_t end = n_obs; end > 0;) {
            const std::int64_t first =
                std::max<std::int64_t>(0, end - emission_block_rows);
            if (!backward_block(first, end, nullptr)) {
                return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
            }
            end = first;
        }
    }

    double Z0 = 0.0;
    for (std::size_t j = 0; j < K_size; ++j) {
        Z0 += fi[j] * pw_const[j] * beta[j];
    }
    if (!std::isfinite(Z0) || Z0 <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
    }
    double grad[3] = {0.0, 0.0, 0.0};
    for (int p = 0; p < 3; ++p) {
        const std::size_t offset = static_cast<std::size_t>(p) * K_size;
        double numerator = 0.0;
        for (std::size_t j = 0; j < K_size; ++j) {
            const double dfi = dfi_dx[j] * dx_dalpha[offset + j];
            numerator += (dfi * beta[j] + fi[j] * d_beta[offset + j])
                * pw_const[j];
        }
        grad[p] = numerator / Z0;
    }
    for (double value : corr_grad) {
        if (!std::isfinite(value)) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, backend);
        }
    }

    GradLogLikResult out;
    out.neg_log_likelihood =
        -(std::log(Z0) + cumul_logc + emission_log_scale);
    out.neg_gradient = {-grad[0], -grad[1], -grad[2]};
    out.K_requested = K_requested;
    out.K_effective = K_eff;
    out.grid_was_capped = adaptive_was_capped;
    out.neg_corr_gradient.resize(corr_grad.size());
    for (std::size_t i = 0; i < corr_grad.size(); ++i) {
        out.neg_corr_gradient[i] = -corr_grad[i];
    }
    out.backend = backend;
    out.status = Status::Ok;
    return out;
}

GradLogLikResult spectral_neg_loglik_with_grad(
    const OuParams& params,
    const CopulaSpec& requested_copula,
    ObservationView u,
    const OuNumericalConfig& raw_config,
    const PreparedDynamicEmission* prepared_emission,
    CorrGradientMode corr_gradient_mode,
    const std::vector<double>* corr_direction = nullptr,
    SpectralGradientWorkspace* workspace = nullptr) {

    const OuNumericalConfig config = with_default_quad_order(raw_config);
    const std::int64_t n_obs = static_cast<std::int64_t>(u.size());
    const bool correlation_gradient =
        corr_gradient_mode != CorrGradientMode::None;
    std::unique_ptr<PreparedDynamicEmission> emission_owner;
    const PreparedDynamicEmission& emission =
        select_emission(
            requested_copula, prepared_emission, emission_owner);
    const CopulaSpec& copula = emission.compatibility_spec();
    if (!supported_ou_copula(emission)) {
        return invalid_grad(SCAR_INVALID_TRANSFORM, OuBackend::Spectral);
    }
    if (correlation_gradient && copula.family != CopulaFamily::Student) {
        return invalid_grad(SCAR_INVALID_FAMILY, OuBackend::Spectral);
    }
    if (correlation_gradient
        && copula.correlation_kind == CorrelationKind::Factor) {
        return invalid_grad(SCAR_INVALID_TRANSFORM, OuBackend::Spectral);
    }
    if (!valid_ou_params(params) || !finite_config_doubles(config)) {
        return invalid_grad(SCAR_INVALID_PARAMETER, OuBackend::Spectral);
    }
    std::size_t spectral_elements = 0;
    if (n_obs < 2
        || !scar_internal::valid_spectral_dimensions(
            config.spectral_quad_order,
            config.spectral_basis_order,
            spectral_elements)) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }

    const double sigma = params.nu / std::sqrt(2.0 * params.kappa);
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    const int quad_order = config.spectral_quad_order;
    const int basis_order = config.spectral_basis_order;
    SpectralGradientWorkspace local_workspace;
    SpectralGradientWorkspace& ws =
        workspace == nullptr ? local_workspace : *workspace;
    std::vector<double>& z = ws.z;
    std::vector<double>& weights = ws.weights;
    std::vector<double>& basis = ws.basis;
    std::vector<double>& weighted_basis = ws.weighted_basis;
    if (ws.cached_quad_order != quad_order
        || ws.cached_basis_order != basis_order
        || basis.size() != spectral_elements
        || weighted_basis.size() != spectral_elements) {
        if (!scar_internal::standard_normal_hermite_rule_with_weighted_basis(
                quad_order,
                basis_order,
                z,
                weights,
                basis,
                weighted_basis)) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
        }
        ws.cached_quad_order = quad_order;
        ws.cached_basis_order = basis_order;
    }

    const double* observation_values = observation_data(emission, u);
    const double dt = n_obs > 1 ? 1.0 / static_cast<double>(n_obs - 1) : 1.0;
    const double rho = std::exp(-params.kappa * dt);

    std::vector<double>& powers = ws.powers;
    std::vector<double>& dpowers_dkappa = ws.dpowers_dkappa;
    powers.assign(static_cast<std::size_t>(basis_order), 1.0);
    dpowers_dkappa.assign(static_cast<std::size_t>(basis_order), 0.0);
    for (int n = 1; n < basis_order; ++n) {
        const std::size_t idx = static_cast<std::size_t>(n);
        powers[idx] = powers[idx - 1] * rho;
        dpowers_dkappa[idx] = -dt * static_cast<double>(n) * powers[idx];
    }

    std::size_t triple_quad = 0;
    std::size_t triple_basis = 0;
    if (!scar_internal::checked_size_mul(
            3, static_cast<std::size_t>(quad_order), triple_quad)
        || !scar_internal::checked_size_mul(
            3, static_cast<std::size_t>(basis_order), triple_basis)) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    std::vector<double>& x_grid = ws.x_grid;
    std::vector<double>& dx_dalpha = ws.dx_dalpha;
    x_grid.assign(static_cast<std::size_t>(quad_order), 0.0);
    dx_dalpha.assign(triple_quad, 0.0);
    for (int q = 0; q < quad_order; ++q) {
        const std::size_t idx = static_cast<std::size_t>(q);
        x_grid[idx] = params.mu + sigma * z[idx];
        dx_dalpha[idx] = -0.5 * sigma / params.kappa * z[idx];
        dx_dalpha[static_cast<std::size_t>(quad_order) + idx] = 1.0;
        dx_dalpha[
            2 * static_cast<std::size_t>(quad_order) + idx] =
            sigma / params.nu * z[idx];
    }
    std::vector<double>& r_grid = ws.r_grid;
    std::vector<double>& dpsi_grid = ws.dpsi_grid;
    r_grid.clear();
    dpsi_grid.clear();
    emission.prepare_grid_transform(x_grid, r_grid, dpsi_grid);
    const bool use_gaussian_spectral_terms =
        prepare_gaussian_spectral_terms(copula, n_obs, r_grid, ws);

    std::size_t n_corr = 0;
    std::vector<double>& precision = ws.precision;
    precision.clear();
    if (correlation_gradient
        && (!scar_internal::valid_student_correlation_count(
                copula.dim, n_corr)
            || !scar_internal::student_precision_matrix(
                copula, precision))) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const bool directional =
        corr_gradient_mode == CorrGradientMode::Directional;
    if (directional
        && (corr_direction == nullptr || corr_direction->size() != n_corr)) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }
    const std::size_t corr_param_count =
        directional ? 1 : n_corr;
    std::size_t corr_basis_elements = 0;
    std::size_t score_elements = 0;
    if (correlation_gradient
        && (!scar_internal::checked_size_mul(
                corr_param_count,
                static_cast<std::size_t>(basis_order),
                corr_basis_elements)
            || !scar_internal::checked_size_mul(
                static_cast<std::size_t>(quad_order),
                corr_param_count,
                score_elements))) {
        return invalid_grad(SCAR_INVALID_SIZE, OuBackend::Spectral);
    }

    std::vector<double>& coeff = ws.coeff;
    std::vector<double>& dcoeff = ws.dcoeff;
    std::vector<double>& projected = ws.projected;
    std::vector<double>& dprojected = ws.dprojected;
    std::vector<double>& raw = ws.raw;
    std::vector<double>& draw = ws.draw;
    std::vector<double>& fi_row = ws.fi_row;
    std::vector<double>& dfi_dx_row = ws.dfi_dx_row;
    std::vector<double>& corr_coeff = ws.corr_coeff;
    std::vector<double>& corr_projected = ws.corr_projected;
    std::vector<double>& corr_raw = ws.corr_raw;
    std::vector<double>& corr_value_projected = ws.corr_value_projected;
    std::vector<double>& scores = ws.scores;
    std::vector<double>& corr_dlog_scale = ws.corr_dlog_scale;
    coeff.assign(static_cast<std::size_t>(basis_order), 0.0);
    dcoeff.assign(triple_basis, 0.0);
    projected.assign(static_cast<std::size_t>(basis_order), 0.0);
    dprojected.assign(triple_basis, 0.0);
    raw.assign(static_cast<std::size_t>(basis_order), 0.0);
    draw.assign(triple_basis, 0.0);
    fi_row.assign(static_cast<std::size_t>(quad_order), 0.0);
    dfi_dx_row.assign(static_cast<std::size_t>(quad_order), 0.0);
    corr_coeff.assign(corr_basis_elements, 0.0);
    corr_projected.assign(corr_basis_elements, 0.0);
    corr_raw.assign(corr_basis_elements, 0.0);
    corr_value_projected.assign(
        static_cast<std::size_t>(basis_order), 0.0);
    scores.assign(score_elements, 0.0);
    corr_dlog_scale.assign(corr_param_count, 0.0);

    coeff[0] = 1.0;
    double log_scale = 0.0;
    double dlog_scale[3] = {0.0, 0.0, 0.0};

    for (std::int64_t t = n_obs - 1; t >= 1; --t) {
        double emission_log_scale = 0.0;
        if (use_gaussian_spectral_terms) {
            gaussian_spectral_pdf_and_grad_row(
                copula,
                t,
                r_grid,
                dpsi_grid,
                ws,
                fi_row.data(),
                dfi_dx_row.data());
        } else {
            emission.fill_density_and_gradient_row(
                observation_values,
                t,
                r_grid,
                dpsi_grid,
                fi_row.data(),
                dfi_dx_row.data(),
                &emission_log_scale);
        }
        log_scale += emission_log_scale;

        scar_internal::project_multiply_with_grad(
            coeff,
            dcoeff,
            fi_row,
            dfi_dx_row,
            dx_dalpha,
            basis,
            weighted_basis,
            quad_order,
            basis_order,
            projected,
            dprojected);
        if (correlation_gradient) {
            const double* row =
                observation_values
                + static_cast<std::size_t>(t)
                    * static_cast<std::size_t>(copula.dim);
            if (directional) {
                if (!scar_internal::student_corr_directional_score_row(
                        copula,
                        row,
                        t,
                        r_grid,
                        precision,
                        *corr_direction,
                        scores.data())) {
                    return invalid_grad(
                        SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
                }
            } else {
                if (!scar_internal::student_corr_score_row(
                        copula,
                        row,
                        t,
                        r_grid,
                        precision,
                        scores.data())) {
                    return invalid_grad(
                        SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
                }
            }
            scar_internal::project_multiply_with_score_grad(
                coeff,
                corr_coeff,
                fi_row,
                scores,
                basis,
                weighted_basis,
                quad_order,
                basis_order,
                static_cast<int>(corr_param_count),
                corr_value_projected,
                corr_projected);
        }

        double scale = 0.0;
        int scale_idx = 0;
        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            raw[idx] = powers[idx] * projected[idx];
            for (int p = 0; p < 3; ++p) {
                draw[
                    static_cast<std::size_t>(p)
                        * static_cast<std::size_t>(basis_order)
                    + static_cast<std::size_t>(n)] =
                    powers[idx]
                    * dprojected[
                        static_cast<std::size_t>(p)
                            * static_cast<std::size_t>(basis_order)
                        + static_cast<std::size_t>(n)];
            }
            draw[idx] += dpowers_dkappa[idx] * projected[idx];
            if (std::abs(raw[idx]) > scale) {
                scale = std::abs(raw[idx]);
                scale_idx = n;
            }
        }
        if (correlation_gradient) {
            for (std::size_t p = 0; p < corr_param_count; ++p) {
                const std::size_t param_base =
                    p * static_cast<std::size_t>(basis_order);
                for (int n = 0; n < basis_order; ++n) {
                    const std::size_t idx = static_cast<std::size_t>(n);
                    corr_raw[param_base + idx] =
                        powers[idx] * corr_projected[param_base + idx];
                }
            }
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
        }

        const double sign = raw[static_cast<std::size_t>(scale_idx)] >= 0.0
            ? 1.0
            : -1.0;
        double dscale[3] = {0.0, 0.0, 0.0};
        for (int p = 0; p < 3; ++p) {
            dscale[p] = sign
                * draw[
                    static_cast<std::size_t>(p)
                        * static_cast<std::size_t>(basis_order)
                    + static_cast<std::size_t>(scale_idx)];
        }

        for (int n = 0; n < basis_order; ++n) {
            const std::size_t idx = static_cast<std::size_t>(n);
            coeff[idx] = raw[idx] / scale;
            for (int p = 0; p < 3; ++p) {
                const std::size_t didx =
                    static_cast<std::size_t>(p)
                        * static_cast<std::size_t>(basis_order)
                    + static_cast<std::size_t>(n);
                dcoeff[didx] = (draw[didx] * scale - raw[idx] * dscale[p])
                    / (scale * scale);
            }
        }
        log_scale += std::log(scale);
        for (int p = 0; p < 3; ++p) {
            dlog_scale[p] += dscale[p] / scale;
        }
        if (correlation_gradient) {
            for (std::size_t p = 0; p < corr_param_count; ++p) {
                const std::size_t param_base =
                    p * static_cast<std::size_t>(basis_order);
                const double corr_scale_derivative =
                    sign * corr_raw[
                        param_base
                        + static_cast<std::size_t>(scale_idx)];
                for (int n = 0; n < basis_order; ++n) {
                    const std::size_t idx = static_cast<std::size_t>(n);
                    corr_coeff[param_base + idx] = (
                        corr_raw[param_base + idx] * scale
                        - raw[idx] * corr_scale_derivative
                    ) / (scale * scale);
                }
                corr_dlog_scale[p] += corr_scale_derivative / scale;
            }
        }
    }

    double emission_log_scale = 0.0;
    if (use_gaussian_spectral_terms) {
        gaussian_spectral_pdf_and_grad_row(
            copula,
            0,
            r_grid,
            dpsi_grid,
            ws,
            fi_row.data(),
            dfi_dx_row.data());
    } else {
        emission.fill_density_and_gradient_row(
            observation_values,
            0,
            r_grid,
            dpsi_grid,
            fi_row.data(),
            dfi_dx_row.data(),
            &emission_log_scale);
    }
    log_scale += emission_log_scale;
    scar_internal::project_multiply_with_grad(
        coeff,
        dcoeff,
        fi_row,
        dfi_dx_row,
        dx_dalpha,
        basis,
        weighted_basis,
        quad_order,
        basis_order,
        projected,
        dprojected);
    if (correlation_gradient) {
        if (directional) {
            if (!scar_internal::student_corr_directional_score_row(
                    copula,
                    observation_values,
                    0,
                    r_grid,
                    precision,
                    *corr_direction,
                    scores.data())) {
                return invalid_grad(
                    SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
            }
        } else {
            if (!scar_internal::student_corr_score_row(
                    copula,
                    observation_values,
                    0,
                    r_grid,
                    precision,
                    scores.data())) {
                return invalid_grad(
                    SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
            }
        }
        scar_internal::project_multiply_with_score_grad(
            coeff,
            corr_coeff,
            fi_row,
            scores,
            basis,
            weighted_basis,
            quad_order,
            basis_order,
            static_cast<int>(corr_param_count),
            corr_value_projected,
            corr_projected);
    }

    const double likelihood_scaled = projected[0];
    if (!std::isfinite(likelihood_scaled) || likelihood_scaled <= 0.0) {
        return invalid_grad(SCAR_NUMERICAL_FAILURE, OuBackend::Spectral);
    }

    GradLogLikResult out;
    out.neg_log_likelihood = -(std::log(likelihood_scaled) + log_scale);
    out.neg_gradient.assign(3, 0.0);
    for (int p = 0; p < 3; ++p) {
        const double grad =
            dprojected[
                static_cast<std::size_t>(p)
                    * static_cast<std::size_t>(basis_order)]
            / likelihood_scaled
            + dlog_scale[p];
        out.neg_gradient[static_cast<std::size_t>(p)] = -grad;
    }
    out.neg_corr_gradient.assign(corr_param_count, 0.0);
    for (std::size_t p = 0; p < corr_param_count; ++p) {
        const double grad =
            corr_projected[
                p * static_cast<std::size_t>(basis_order)]
            / likelihood_scaled
            + corr_dlog_scale[p];
        out.neg_corr_gradient[p] = -grad;
    }
    out.backend = OuBackend::Spectral;
    out.status = Status::Ok;
    return out;
}

}  // namespace

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return spectral_neg_loglik_with_grad(
        params, copula, u, config, prepared_emission_,
        CorrGradientMode::None, nullptr,
        &workspace().spectral_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return spectral_neg_loglik_with_grad(
        params, copula, u, config, prepared_emission_,
        CorrGradientMode::Full, nullptr,
        &workspace().spectral_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_directional_spectral(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    const std::vector<double>& corr_direction) const {

    return spectral_neg_loglik_with_grad(
        params, copula, u, config, prepared_emission_,
        CorrGradientMode::Directional,
        &corr_direction, &workspace().spectral_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::LocalGh, prepared_emission_,
        CorrGradientMode::None, nullptr, &workspace().grid_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::Matrix, prepared_emission_,
        CorrGradientMode::None, nullptr, &workspace().grid_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::LocalGh, prepared_emission_,
        CorrGradientMode::Full, nullptr, &workspace().grid_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::Matrix, prepared_emission_,
        CorrGradientMode::Full, nullptr, &workspace().grid_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_directional_local_gh(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    const std::vector<double>& corr_direction) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::LocalGh, prepared_emission_,
        CorrGradientMode::Directional, &corr_direction,
        &workspace().grid_gradient);
}

GradLogLikResult ScarOuEvaluator::neg_loglik_with_grad_and_corr_directional_matrix(
    const OuParams& params,
    const CopulaSpec& copula,
    ObservationView u,
    const OuNumericalConfig& config,
    const std::vector<double>& corr_direction) const {

    return grid_neg_loglik_with_grad(
        params, copula, u, config, OuBackend::Matrix, prepared_emission_,
        CorrGradientMode::Directional, &corr_direction,
        &workspace().grid_gradient);
}

}  // namespace scar
