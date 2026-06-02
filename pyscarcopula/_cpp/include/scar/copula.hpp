#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace scar {

enum class CopulaFamily : int {
    Independent = 0,
    Clayton = 1,
    Gumbel = 2,
    Frank = 3,
    Joe = 4,
    Gaussian = 5,
};

enum class Rotation : int {
    R0 = 0,
    R90 = 90,
    R180 = 180,
    R270 = 270,
};

enum class Transform : int {
    Softplus = 1,
    XTanh = 2,
    GaussianTanh = 3,
};

struct CopulaSpec {
    CopulaFamily family = CopulaFamily::Clayton;
    Rotation rotation = Rotation::R0;
    Transform transform = Transform::Softplus;
    double offset = 0.0001;
};

using Observations = std::vector<std::vector<double>>;

struct GridValues {
    std::vector<double> values;
    std::int64_t n_obs = 0;
    std::int64_t n_grid = 0;
};

struct GridValuesWithGrad {
    GridValues pdf;
    GridValues d_pdf_dx;
};

bool is_supported(const CopulaSpec& spec);

std::vector<double> copula_log_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_pdf(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_h(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::pair<std::vector<double>, std::vector<double>> copula_h_pair(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& r);

std::vector<double> copula_h_inverse(
    const CopulaSpec& spec,
    const Observations& q_given,
    const std::vector<double>& r);

GridValues copula_pdf_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid);

GridValuesWithGrad copula_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& u,
    const std::vector<double>& x_grid);

}  // namespace scar
