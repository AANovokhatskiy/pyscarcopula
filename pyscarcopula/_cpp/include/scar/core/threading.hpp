#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace scar::core {

inline constexpr int kMaxThreadCount = 256;

inline bool valid_thread_count(int n_threads) noexcept {
    return n_threads >= 1 && n_threads <= kMaxThreadCount;
}

inline void validate_thread_count(int n_threads) {
    if (!valid_thread_count(n_threads)) {
        throw std::invalid_argument("n_threads must be in [1, 256]");
    }
}

/// Limit a valid requested worker count by the number of independent items.
inline int limit_worker_count(
    int n_threads,
    std::size_t work_items) noexcept {

    if (n_threads <= 1 || work_items == 0) {
        return 1;
    }
    return static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(n_threads), work_items));
}

/// Preserve an all-workers-or-sequential policy with a caller-owned grain.
inline int worker_count_for_items(
    int n_threads,
    std::size_t work_items,
    std::size_t min_items_per_worker) noexcept {

    if (n_threads <= 1
        || min_items_per_worker == 0
        || work_items / min_items_per_worker
            < static_cast<std::size_t>(n_threads)) {
        return 1;
    }
    return limit_worker_count(n_threads, work_items);
}

}  // namespace scar::core

namespace scar_internal {

using scar::core::kMaxThreadCount;
using scar::core::limit_worker_count;
using scar::core::valid_thread_count;
using scar::core::validate_thread_count;
using scar::core::worker_count_for_items;

}  // namespace scar_internal
