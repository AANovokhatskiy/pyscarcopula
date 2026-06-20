#include "scar/ou.hpp"

#include "scar_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scar {
namespace {

bool valid_student_spec(const CopulaSpec& spec, std::size_t n_obs) {
    std::size_t square = 0;
    if (spec.dim < 2
        || spec.rotation != Rotation::R0
        || spec.transform != Transform::Softplus
        || !scar_internal::valid_student_dimension(spec.dim, square)
        || spec.l_inv.size() != square
        || !std::isfinite(spec.log_det)) {
        return false;
    }
    if (!spec.ppf_nodes.empty() || !spec.ppf_table.empty()) {
        return spec.ppf_n_obs == static_cast<std::int64_t>(n_obs);
    }
    return true;
}

bool valid_spec(
    const CopulaSpec& spec,
    ObservationView u,
    std::size_t n_trajectories) {

    if (u.empty() || u.data() == nullptr || n_trajectories == 0) {
        return false;
    }
    if (spec.family == CopulaFamily::Student) {
        return u.dim == spec.dim && valid_student_spec(spec, u.size());
    }
    return u.dim == 2 && scar_internal::copula_is_supported(spec);
}

}  // namespace

TrajectoryLogPdfResult copula_log_pdf_trajectory_grid(
    const CopulaSpec& copula,
    ObservationView u,
    const double* latent_paths,
    std::size_t n_trajectories) {

    TrajectoryLogPdfResult out;
    out.log_pdf.n_obs = static_cast<std::int64_t>(u.size());
    out.log_pdf.n_grid = static_cast<std::int64_t>(n_trajectories);

    std::size_t elements = 0;
    if (!scar_internal::checked_size_mul(
            u.size(), n_trajectories, elements)) {
        out.status = SCAR_INVALID_SIZE;
        return out;
    }
    out.log_pdf.values.assign(
        elements, -std::numeric_limits<double>::infinity());
    if (latent_paths == nullptr) {
        out.status = SCAR_NULL_POINTER;
        return out;
    }
    if (!valid_spec(copula, u, n_trajectories)) {
        out.status = SCAR_INVALID_FAMILY;
        return out;
    }

    const std::size_t observation_stride =
        static_cast<std::size_t>(u.dim);
    for (std::size_t t = 0; t < u.size(); ++t) {
        const double* row = u.data() + t * observation_stride;
        double v1 = 0.0;
        double v2 = 0.0;
        if (copula.family != CopulaFamily::Student) {
            scar_internal::apply_rotation(
                row[0],
                row[1],
                static_cast<int>(copula.rotation),
                v1,
                v2);
        }

        const std::size_t base = t * n_trajectories;
        for (std::size_t j = 0; j < n_trajectories; ++j) {
            const double latent = latent_paths[base + j];
            const double parameter =
                scar_internal::copula_transform(copula, latent);
            double value = -std::numeric_limits<double>::infinity();
            if (std::isfinite(parameter)) {
                value = copula.family == CopulaFamily::Student
                    ? scar_internal::student_log_pdf(
                        copula,
                        row,
                        parameter,
                        static_cast<std::int64_t>(t))
                    : scar_internal::copula_log_pdf_unrotated(
                        copula, v1, v2, parameter);
            }
            if (!std::isfinite(value)) {
                out.status = SCAR_NUMERICAL_FAILURE;
                out.failure_index = static_cast<std::int64_t>(t);
                return out;
            }
            out.log_pdf.values[base + j] = value;
        }
    }
    return out;
}

}  // namespace scar
