#pragma once

#include <cstdint>
#include <vector>

namespace scar {

/// Flattened row-major values with explicit two-dimensional shape.
struct GridValues {
    std::vector<double> values;
    std::int64_t n_obs = 0;
    std::int64_t n_grid = 0;
};

/// Copula density and its derivative on the same grid.
struct GridValuesWithGrad {
    GridValues pdf;
    GridValues d_pdf_dx;
};

}  // namespace scar
