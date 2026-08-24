#pragma once

#include <vector>

namespace scar {

struct ScarOuGridGradientOperators {
    int K = 0;
    int width = 0;
    bool local = false;
    bool sparse = false;
    std::vector<double> dense;
    std::vector<double> dense_grad;
    std::vector<int> cols;
    std::vector<int> indptr;
    std::vector<double> vals;
    std::vector<double> grad_vals;
};

struct ScarOuGridGradientWorkspace {
    ScarOuGridGradientOperators op;
    std::vector<double> xi;
    std::vector<double> base_w;
    std::vector<double> pw_const;
    std::vector<double> x_grid;
    std::vector<double> fi;
    std::vector<double> dfi_dx;
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    std::vector<double> beta;
    std::vector<double> c_vals;
    std::vector<double> target;
    std::vector<double> next;
    std::vector<double> dx_dalpha;
    std::vector<double> d_beta;
    std::vector<double> new_d_beta;
    std::vector<double> d_target;
    std::vector<double> contrib;
    std::vector<double> transition_grad;
    std::vector<double> precision;
    std::vector<double> scores;
    std::vector<double> alpha;
    std::vector<double> alpha_source;
    std::vector<double> alpha_next;
};

struct ScarOuSpectralGradientWorkspace {
    int cached_quad_order = 0;
    int cached_basis_order = 0;
    std::vector<double> z;
    std::vector<double> weights;
    std::vector<double> basis;
    std::vector<double> weighted_basis;
    std::vector<double> powers;
    std::vector<double> dpowers_dkappa;
    std::vector<double> x_grid;
    std::vector<double> dx_dalpha;
    std::vector<double> r_grid;
    std::vector<double> dpsi_grid;
    std::vector<double> gaussian_r2;
    std::vector<double> gaussian_omr2;
    std::vector<double> gaussian_log_norm;
    std::vector<double> gaussian_dlog_det;
    std::vector<double> gaussian_omr2_squared;
    std::vector<double> coeff;
    std::vector<double> dcoeff;
    std::vector<double> projected;
    std::vector<double> dprojected;
    std::vector<double> raw;
    std::vector<double> draw;
    std::vector<double> fi_row;
    std::vector<double> dfi_dx_row;
    std::vector<double> precision;
    std::vector<double> scores;
    std::vector<double> corr_coeff;
    std::vector<double> corr_projected;
    std::vector<double> corr_raw;
    std::vector<double> corr_value_projected;
    std::vector<double> corr_dlog_scale;
};

struct ScarOuEvaluatorWorkspace {
    ScarOuGridGradientWorkspace grid_gradient;
    ScarOuSpectralGradientWorkspace spectral_gradient;
};

}  // namespace scar
