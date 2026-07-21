#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace scar_internal {

struct ParallelRuntimeInfo {
    bool initialized = false;
    std::uint64_t owner_pid = 0;
    std::size_t worker_count = 0;
    std::uint64_t batches_submitted = 0;
};

/// Return whether a row-major grid is large enough to amortize dispatch.
inline bool grid_parallel_worthwhile(
    std::size_t rows,
    std::size_t columns,
    std::size_t min_rows,
    std::size_t min_cells) noexcept {

    if (columns == 0 || rows <= min_rows) {
        return false;
    }
    const std::size_t rows_for_cells =
        min_cells / columns + (min_cells % columns == 0 ? 0 : 1);
    return rows >= rows_for_cells;
}

using ParallelBlockFunction =
    std::function<void(std::int64_t, std::int64_t, std::size_t)>;

/// Execute stable contiguous blocks over [begin, end).
///
/// n_threads == 1 and small ranges execute directly in the calling thread and
/// do not inspect or initialize process-global runtime state.
void parallel_for_blocks(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t min_grain,
    int n_threads,
    const ParallelBlockFunction& function);

/// Read-only test/diagnostic snapshot. Calling this does not create the pool.
ParallelRuntimeInfo parallel_runtime_info();

/// Stop and release the pool owned by the current process. Idempotent and
/// intended for module/process teardown after all submitted work has ended.
void shutdown_parallel_runtime() noexcept;

}  // namespace scar_internal
