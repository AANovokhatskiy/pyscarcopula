#pragma once

#include "scar/copula.hpp"
#include "scar/gas.hpp"
#include "scar/ou.hpp"
#include "scar/detail/copula.hpp"
#include "scar/detail/safety.hpp"
#include "scar/detail/scar_ou/quadrature.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

namespace pyscarcopula::bindings {

scar::Observations observations_from_array(
    pybind11::array_t<
        double,
        pybind11::array::c_style | pybind11::array::forcecast> u);
scar::ObservationView observation_view_from_array(
    const scar::CopulaSpec& copula,
    pybind11::array_t<
        double,
        pybind11::array::c_style | pybind11::array::forcecast> u);
std::vector<double> vector_from_array(
    pybind11::array_t<
        double,
        pybind11::array::c_style | pybind11::array::forcecast> values);
std::vector<double> flat_vector_from_array(
    pybind11::array_t<
        double,
        pybind11::array::c_style | pybind11::array::forcecast> values,
    const char* name);
std::vector<int> int_vector_from_array(
    pybind11::array_t<
        int,
        pybind11::array::c_style | pybind11::array::forcecast> values,
    const char* name);
pybind11::array_t<double> vector_to_array(
    const std::vector<double>& values);
pybind11::array_t<double> grid_values_to_array(
    const scar::GridValues& values);
void set_student_ppf_cache(
    scar::CopulaSpec& spec,
    pybind11::array_t<
        double,
        pybind11::array::c_style | pybind11::array::forcecast> nodes,
    pybind11::array_t<
        double,
        pybind11::array::c_style | pybind11::array::forcecast> table);
pybind11::list backend_chain_to_list(
    const std::vector<scar::OuBackend>& chain);
pybind11::dict loglik_result_to_dict(const scar::LogLikResult& result);
pybind11::dict grad_loglik_result_to_dict(
    const scar::GradLogLikResult& result);
pybind11::dict vector_result_to_dict(
    const std::vector<double>& values,
    int status,
    int backend);
pybind11::dict state_distribution_to_dict(
    const scar::StateDistribution& result);
pybind11::dict gas_loglik_result_to_dict(
    const scar::GasLogLikResult& result);
pybind11::dict static_objective_result_to_dict(
    const scar::StaticObjectiveResult& result);
pybind11::dict multivariate_rows_result_to_dict(
    const scar::MultivariateRowsResult& result);
pybind11::dict multivariate_grid_result_to_dict(
    const scar::MultivariateGridResult& result);
pybind11::dict conditional_sample_result_to_dict(
    const scar::ConditionalSampleResult& result);
pybind11::dict trajectory_log_pdf_result_to_dict(
    const scar::TrajectoryLogPdfResult& result);
pybind11::dict gas_filter_result_to_dict(
    const scar::GasFilterResult& result);
pybind11::dict gas_update_result_to_dict(
    const scar::GasUpdateResult& result);
pybind11::dict gas_state_result_to_dict(
    const scar::GasStateResult& result);
pybind11::dict gas_predict_result_to_dict(
    const scar::GasPredictResult& result);
pybind11::dict gas_path_result_to_dict(
    const scar::GasPathResult& result);
pybind11::dict hermite_rule_cache_info_to_dict(
    const scar_internal::HermiteRuleCacheInfo& info);

void bind_common(pybind11::module_& module);
void bind_copula(pybind11::module_& module);
void bind_multivariate(pybind11::module_& module);
void bind_scar_ou_types(pybind11::module_& module);
void bind_gas(pybind11::module_& module);
void bind_scar_ou(pybind11::module_& module);

}  // namespace pyscarcopula::bindings
