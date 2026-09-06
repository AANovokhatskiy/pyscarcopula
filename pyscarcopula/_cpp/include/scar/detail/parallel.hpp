#pragma once

#include "scar/core/threading.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <utility>
#include <vector>

namespace scar_internal {

struct ParallelRuntimeInfo {
    bool initialized = false;
    std::uint64_t owner_pid = 0;
    std::size_t worker_count = 0;
    std::uint64_t batches_submitted = 0;
    std::uint64_t worker_start_events = 0;
    std::uint64_t tasks_submitted = 0;
    std::size_t peak_queued_tasks = 0;
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

struct ParallelRange {
    std::int64_t begin = 0;
    std::int64_t end = 0;
};

struct ParallelBlockContext {
    std::size_t block_id = 0;
    std::size_t worker_slot = 0;
};

using PreparedParallelBlockFunction = std::function<void(
    std::int64_t,
    std::int64_t,
    const ParallelBlockContext&)>;

enum class ParallelFailureOrigin { EnvironmentApply, Callback };

struct ParallelFailureHandler {
    void* state = nullptr;
    void (*record)(void*, std::size_t, ParallelFailureOrigin,
                   std::exception_ptr) noexcept = nullptr;
};

/// Immutable logical portions and the worker budget for one call.
class ParallelExecutionPlan {
public:
    ParallelExecutionPlan(const ParallelExecutionPlan&) = default;
    ParallelExecutionPlan(ParallelExecutionPlan&& other) noexcept;
    ParallelExecutionPlan& operator=(const ParallelExecutionPlan& other);
    ParallelExecutionPlan& operator=(ParallelExecutionPlan&& other) noexcept;

    const ParallelRange& range() const noexcept { return range_; }
    const std::vector<std::int64_t>& bounds() const noexcept {
        return bounds_;
    }
    std::size_t block_count() const noexcept { return block_count_; }
    std::size_t worker_count() const noexcept { return worker_count_; }
    bool empty() const noexcept { return block_count_ == 0; }

private:
    void reset_to_empty() noexcept;
    void swap(ParallelExecutionPlan& other) noexcept;

    ParallelExecutionPlan(
        ParallelRange range,
        std::vector<std::int64_t> bounds,
        std::size_t block_count,
        std::size_t worker_count)
        : range_(range),
          bounds_(std::move(bounds)),
          block_count_(block_count),
          worker_count_(worker_count) {}

    ParallelRange range_{};
    std::vector<std::int64_t> bounds_;
    std::size_t block_count_ = 0;
    std::size_t worker_count_ = 0;

    friend ParallelExecutionPlan make_parallel_execution_plan(
        std::int64_t,
        std::int64_t,
        std::size_t,
        int);
    friend ParallelExecutionPlan make_parallel_execution_plan_from_bounds(
        const std::vector<std::int64_t>&,
        int);
    friend ParallelExecutionPlan make_parallel_execution_plan_from_bounds(
        std::vector<std::int64_t>&&,
        int);
};

/// Preserve both failures when restoring a direct caller's environment fails.
class ParallelEnvironmentRestoreError : public std::exception {
public:
    ParallelEnvironmentRestoreError(
        std::exception_ptr primary,
        std::exception_ptr restore) noexcept
        : primary_(std::move(primary)), restore_(std::move(restore)) {}

    const char* what() const noexcept override {
        return "parallel execution failed and its floating-point environment "
               "could not be restored";
    }

    std::exception_ptr primary_exception() const noexcept { return primary_; }
    std::exception_ptr restore_exception() const noexcept { return restore_; }
    [[noreturn]] void rethrow_primary() const {
        std::rethrow_exception(primary_);
    }

private:
    std::exception_ptr primary_;
    std::exception_ptr restore_;
};

/// Build quotient/remainder logical portions over [begin, end).
ParallelExecutionPlan make_parallel_execution_plan(
    std::int64_t begin,
    std::int64_t end,
    std::size_t block_count,
    int worker_count);

/// Build a plan from strictly increasing boundaries. One boundary is empty.
ParallelExecutionPlan make_parallel_execution_plan_from_bounds(
    const std::vector<std::int64_t>& bounds,
    int worker_count);
ParallelExecutionPlan make_parallel_execution_plan_from_bounds(
    std::vector<std::int64_t>&& bounds,
    int worker_count);

/// Number of scratch slots used by this plan in the current calling thread.
std::size_t parallel_execution_slots(const ParallelExecutionPlan& plan) noexcept;

/// Execute a validated plan. Logical blocks retain their IDs as W changes.
void execute_parallel_plan(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function);

/// Record portion failures and drain before returning an entry-restore failure.
/// The non-null recorder may run concurrently for distinct logical blocks.
/// Its state and the callback must remain valid until this function returns.
/// Recorded failures are not rethrown; preparation/publication failures are.
[[nodiscard]] std::exception_ptr execute_parallel_plan_deferred(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function,
    const ParallelFailureHandler& handler);

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

#ifdef SCAR_PARALLEL_TESTING
// Compiled only into the standalone C++ test executable, never the extension.
namespace parallel_testing {
enum class Fault { none, batch_allocation, worker_creation, queue_insertion };
void fail_after(Fault fault, std::size_t successful_attempts);
void clear_fault() noexcept;
using EnvironmentFailureHook = bool (*)(std::size_t, bool) noexcept;
// True simulates failure. Restore uses block_count as its logical block ID.
void set_environment_failure_hook(EnvironmentFailureHook hook) noexcept;
using NotificationHook = void (*)() noexcept;
void set_completion_notification_hook(NotificationHook hook) noexcept;
void set_before_batch_wait_hook(NotificationHook hook) noexcept;
void reset_completion_notification_count() noexcept;
std::uint64_t completion_notification_count() noexcept;
void reset_ready_notification_counts() noexcept;
std::uint64_t ready_notify_one_count() noexcept;
std::uint64_t ready_notify_all_count() noexcept;
}  // namespace parallel_testing
#endif

}  // namespace scar_internal
