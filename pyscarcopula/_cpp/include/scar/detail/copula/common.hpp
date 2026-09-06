#pragma once

#include "scar/copula/spec.hpp"

namespace scar_internal {

scar::CopulaSpec transposed_copula_spec(const scar::CopulaSpec& spec);
double log1mexp(double x);
double logsumexp(double a, double b);

double copula_transform(const scar::CopulaSpec& spec, double x);
double copula_inverse_transform(const scar::CopulaSpec& spec, double r);
double copula_dtransform(const scar::CopulaSpec& spec, double x);
double copula_tau_to_param(const scar::CopulaSpec& spec, double tau);
double copula_param_to_tau(const scar::CopulaSpec& spec, double r);

}  // namespace scar_internal
