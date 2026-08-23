#pragma once

#include "scar/copula.hpp"

namespace scar_internal {

scar::CopulaSpec transposed_copula_spec(const scar::CopulaSpec& spec);
double log1mexp(double x);
double logsumexp(double a, double b);

struct EquicorrStats {
    double sum = 0.0;
    double sum_squares = 0.0;
};

bool equicorr_sufficient_statistics(
    const scar::CopulaSpec& spec,
    const double* row,
    EquicorrStats& stats);
double equicorr_transform(const scar::CopulaSpec& spec, double x);
double equicorr_inverse_transform(const scar::CopulaSpec& spec, double rho);
double equicorr_dtransform(const scar::CopulaSpec& spec, double x);
double equicorr_log_pdf(
    const scar::CopulaSpec& spec,
    const double* row,
    double rho,
    double* dlog_drho);
double equicorr_log_pdf_from_stats(
    const scar::CopulaSpec& spec,
    const EquicorrStats& stats,
    double rho,
    double* dlog_drho);

double copula_transform(const scar::CopulaSpec& spec, double x);
double copula_inverse_transform(const scar::CopulaSpec& spec, double r);
double copula_dtransform(const scar::CopulaSpec& spec, double x);
double copula_tau_to_param(const scar::CopulaSpec& spec, double tau);
double copula_param_to_tau(const scar::CopulaSpec& spec, double r);

}  // namespace scar_internal
