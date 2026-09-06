#include "scar/copula/prepared_pair_kernel.hpp"

namespace scar_internal {

double copula_tau_to_param(const scar::CopulaSpec& spec, double tau) {
    return scar::PreparedPairKernel(spec).tau_to_parameter(tau);
}

double copula_param_to_tau(const scar::CopulaSpec& spec, double parameter) {
    return scar::PreparedPairKernel(spec).parameter_to_tau(parameter);
}

}  // namespace scar_internal
