#include "scar/copula.hpp"

#include "scar/copula/multivariate/equicorrelation/kernel.hpp"
#include "scar/copula/multivariate/student/density.hpp"
#include "scar/detail/copula/multivariate/batch.hpp"
#include "scar/detail/parallel.hpp"
#include "scar/detail/safety.hpp"
#include "scar/status.hpp"

#include <limits>

namespace scar {

MultivariateRowsResult multivariate_log_pdf_and_grad(
    const CopulaSpec& spec,
    const Observations& observations,
    const std::vector<double>& parameters,
    std::int64_t row_offset,
    int n_threads) {

    if (spec.family == CopulaFamily::Student) {
        return scar_internal::student_log_pdf_and_grad_rows(
            spec, observations, parameters, row_offset, n_threads);
    }
    if (spec.family == CopulaFamily::EquicorrGaussian) {
        return scar_internal::equicorr_log_pdf_and_grad_rows(
            spec, observations, parameters, row_offset, n_threads);
    }

    MultivariateRowsResult out;
    out.n_threads_requested = n_threads;
    out.status = observations.empty() || row_offset < 0
        ? Status::InvalidSize : Status::InvalidFamily;
    out.log_pdf.assign(
        observations.size(), -std::numeric_limits<double>::infinity());
    out.dlog_dr.assign(
        observations.size(), std::numeric_limits<double>::quiet_NaN());
    return out;
}

MultivariateGridResult multivariate_pdf_and_grad_grid(
    const CopulaSpec& spec,
    const Observations& observations,
    const std::vector<double>& state_grid,
    std::int64_t row_offset,
    int n_threads) {

    if (spec.family == CopulaFamily::Student) {
        return scar_internal::student_pdf_and_grad_grid(
            spec, observations, state_grid, row_offset, n_threads);
    }
    if (spec.family == CopulaFamily::EquicorrGaussian) {
        return scar_internal::equicorr_pdf_and_grad_grid(
            spec, observations, state_grid, row_offset, n_threads);
    }

    MultivariateGridResult out;
    out.n_threads_requested = n_threads;
    scar_internal::initialize_multivariate_grid(
        out, observations.size(), state_grid.size());
    if (out.is_ok()) {
        out.status = observations.empty() || row_offset < 0
            ? Status::InvalidSize : Status::InvalidFamily;
    }
    if (!scar_internal::valid_thread_count(n_threads)) {
        out.status = Status::InvalidParameter;
    }
    return out;
}

}  // namespace scar
