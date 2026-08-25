#pragma once

namespace scar_internal {

double student_log_gamma(double value);
double student_digamma_positive(double value);
double student_pdf_value(double value, double df);
double student_survival_positive_value(double value, double df);
void student_survival_positive_df_value_and_derivative(
    double value,
    double df,
    double& survival,
    double& derivative);
double student_cdf_value(double value, double df);
double student_cdf_refined_value(double value, double df);

}  // namespace scar_internal
