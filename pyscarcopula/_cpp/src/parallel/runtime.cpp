#include "scar/detail/parallel.hpp"

#include "scar/core/checked_arithmetic.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cfenv>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace scar_internal {
namespace {

thread_local bool in_runtime_worker = false;

#ifdef SCAR_PARALLEL_TESTING
thread_local parallel_testing::Fault pending_fault = parallel_testing::Fault::none;
thread_local std::size_t fault_countdown = 0;
thread_local parallel_testing::EnvironmentFailureHook
    environment_failure_hook = nullptr;
std::atomic<parallel_testing::NotificationHook>
    completion_notification_hook{nullptr};
std::atomic<parallel_testing::NotificationHook>
    before_batch_wait_hook{nullptr};
std::atomic<std::uint64_t> completion_notifications{0};
std::atomic<std::uint64_t> ready_notify_one_notifications{0};
std::atomic<std::uint64_t> ready_notify_all_notifications{0};

void inject_failure(parallel_testing::Fault fault) {
    if (pending_fault != fault) {
        return;
    }
    if (fault_countdown != 0) {
        --fault_countdown;
        return;
    }
    pending_fault = parallel_testing::Fault::none;
    throw std::bad_alloc();
}

void inject_environment_failure(
    parallel_testing::EnvironmentFailureHook hook,
    std::size_t block,
    bool restoring,
    const char* message) {

    if (hook != nullptr && hook(block, restoring)) {
        throw std::runtime_error(message);
    }
}
#endif

std::uint64_t current_pid() noexcept {
#ifdef _WIN32
    return static_cast<std::uint64_t>(::_getpid());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

struct Batch {
    explicit Batch(std::size_t count)
        : remaining(count)
#ifdef SCAR_PARALLEL_TESTING
          , environment_hook(environment_failure_hook)
#endif
    {
        if (std::fegetenv(&floating_point_environment) != 0) {
            throw std::runtime_error(
                "failed to capture caller floating-point environment");
        }
    }

    std::mutex mutex;
    std::condition_variable completed;
    std::size_t remaining;
    std::exception_ptr exception;
    std::fenv_t floating_point_environment{};
#ifdef SCAR_PARALLEL_TESTING
    const parallel_testing::EnvironmentFailureHook environment_hook;
#endif

    void record_exception(std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!exception) {
            exception = std::move(failure);
        }
    }

    void runner_finished() {
        bool finished = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            --remaining;
            finished = remaining == 0;
#ifdef SCAR_PARALLEL_TESTING
            if (finished) {
                completion_notifications.fetch_add(
                    1, std::memory_order_relaxed);
            }
#endif
        }
        if (finished) {
            completed.notify_one();
#ifdef SCAR_PARALLEL_TESTING
            if (const auto hook = completion_notification_hook.load(
                    std::memory_order_acquire)) {
                hook();
            }
#endif
        }
    }
};

using RunnerFunction = std::function<void(Batch&)>;

class ThreadPool {
public:
    explicit ThreadPool(std::uint64_t owner_pid) : owner_pid_(owner_pid) {}

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        shutdown();
    }

    std::uint64_t owner_pid() const noexcept {
        return owner_pid_;
    }

    void run(std::vector<RunnerFunction> runners, std::size_t workers) {
        ensure_workers(workers);
#ifdef SCAR_PARALLEL_TESTING
        inject_failure(parallel_testing::Fault::batch_allocation);
#endif
        const std::size_t runner_count = runners.size();
        auto batch = std::make_shared<Batch>(runner_count);
        bool wake_all = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t original_size = queue_.size();
            try {
                for (auto& runner : runners) {
#ifdef SCAR_PARALLEL_TESTING
                    inject_failure(parallel_testing::Fault::queue_insertion);
#endif
                    queue_.emplace_back(
                        [batch, runner = std::move(runner)]() mutable {
                            try {
                                runner(*batch);
                            } catch (...) {
                                batch->record_exception(
                                    std::current_exception());
                            }
                            // Release caller captures before reporting completion.
                            runner = nullptr;
                            batch->runner_finished();
                        });
                    // The queued entry is now the only callable owner.
                    runner = nullptr;
                }
            } catch (...) {
                // Workers cannot observe this batch while mutex_ is held.
                // Remove only our unpublished suffix before caller buffers die.
                while (queue_.size() > original_size) {
                    queue_.pop_back();
                }
                throw;
            }
            runners.clear();
            ++batches_submitted_;
            tasks_submitted_ += static_cast<std::uint64_t>(runner_count);
            peak_queued_tasks_ = std::max(
                peak_queued_tasks_, queue_.size());
            const std::size_t wake_all_threshold = workers_.size() / 3
                + (workers_.size() % 3 == 0 ? 0 : 1);
            wake_all = runner_count >= wake_all_threshold;
        }
        if (wake_all) {
#ifdef SCAR_PARALLEL_TESTING
            ready_notify_all_notifications.fetch_add(
                1, std::memory_order_relaxed);
#endif
            ready_.notify_all();
        } else {
            for (std::size_t index = 0; index < runner_count; ++index) {
#ifdef SCAR_PARALLEL_TESTING
                ready_notify_one_notifications.fetch_add(
                    1, std::memory_order_relaxed);
#endif
                ready_.notify_one();
            }
        }

#ifdef SCAR_PARALLEL_TESTING
        if (const auto hook = before_batch_wait_hook.load(
                std::memory_order_acquire)) {
            hook();
        }
#endif
        std::unique_lock<std::mutex> batch_lock(batch->mutex);
        batch->completed.wait(
            batch_lock, [&batch]() { return batch->remaining == 0; });
        if (batch->exception) {
            std::rethrow_exception(batch->exception);
        }
    }

    ParallelRuntimeInfo info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            true,
            owner_pid_,
            workers_.size(),
            batches_submitted_,
            worker_start_events_,
            tasks_submitted_,
            peak_queued_tasks_,
        };
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

private:
    void ensure_workers(std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            throw std::runtime_error("parallel runtime is shutting down");
        }
        while (workers_.size() < count) {
#ifdef SCAR_PARALLEL_TESTING
            inject_failure(parallel_testing::Fault::worker_creation);
#endif
            workers_.emplace_back([this]() { worker_loop(); });
            ++worker_start_events_;
        }
    }

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this]() {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty()) {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            in_runtime_worker = true;
            task();
            in_runtime_worker = false;
        }
    }

    const std::uint64_t owner_pid_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
    std::uint64_t batches_submitted_ = 0;
    std::uint64_t worker_start_events_ = 0;
    std::uint64_t tasks_submitted_ = 0;
    std::size_t peak_queued_tasks_ = 0;
};

std::atomic<ThreadPool*> runtime{nullptr};
std::once_flag cleanup_registration;

void cleanup_runtime() noexcept {
    shutdown_parallel_runtime();
}

ThreadPool& process_runtime() {
    const std::uint64_t pid = current_pid();
    while (true) {
        ThreadPool* existing = runtime.load(std::memory_order_acquire);
        if (existing != nullptr && existing->owner_pid() == pid) {
            return *existing;
        }

        auto candidate = std::make_unique<ThreadPool>(pid);
        ThreadPool* expected = existing;
        if (runtime.compare_exchange_strong(
                expected,
                candidate.get(),
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            ThreadPool* published = candidate.release();
            std::call_once(cleanup_registration, []() {
                std::atexit(cleanup_runtime);
            });
            // A pool inherited across fork is deliberately leaked in the
            // child: destroying std::thread objects for vanished threads is
            // not defined. The parent still owns and cleans up its instance.
            return *published;
        }
    }
}

std::int64_t checked_range_length(
    std::int64_t begin,
    std::int64_t end) {

    if (end < begin) {
        throw std::invalid_argument("parallel range end must be >= begin");
    }
    const std::uint64_t length = static_cast<std::uint64_t>(end)
        - static_cast<std::uint64_t>(begin);
    if (length > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "parallel range length is not representable");
    }
    return static_cast<std::int64_t>(length);
}

struct ValidatedBounds {
    ParallelRange range;
    std::size_t block_count;
    std::size_t worker_count;
};

ValidatedBounds validate_execution_bounds(
    const std::vector<std::int64_t>& bounds,
    int worker_count) {

    validate_thread_count(worker_count);
    if (bounds.empty()) {
        throw std::invalid_argument(
            "parallel boundaries must contain at least one value");
    }
    std::size_t boundary_bytes = 0;
    if (!checked_size_mul(
            bounds.size(), sizeof(std::int64_t), boundary_bytes)) {
        throw std::overflow_error(
            "parallel boundary storage size is not representable");
    }
    const std::size_t block_count = bounds.size() - 1;
    for (std::size_t index = 1; index < bounds.size(); ++index) {
        if (bounds[index] <= bounds[index - 1]) {
            throw std::invalid_argument(
                "parallel boundaries must be strictly increasing");
        }
    }
    const ParallelRange range{bounds.front(), bounds.back()};
    const auto length = checked_range_length(range.begin, range.end);
    if (block_count > static_cast<std::uint64_t>(length)) {
        throw std::invalid_argument(
            "parallel block count exceeds the range length");
    }
    if (boundary_bytes == 0) {
        throw std::logic_error("parallel boundary validation failed");
    }
    return {
        range,
        block_count,
        block_count == 0 ? 0 : static_cast<std::size_t>(worker_count),
    };
}

void apply_floating_point_environment(
    const std::fenv_t& environment,
    const char* message) {

    if (std::fesetenv(&environment) != 0) {
        throw std::runtime_error(message);
    }
}

void keep_first_exception(std::exception_ptr& first) noexcept {
    if (!first) {
        first = std::current_exception();
    }
}

void execute_plan_direct(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function) {

#ifdef SCAR_PARALLEL_TESTING
    const auto captured_hook = environment_failure_hook;
#endif
    std::fenv_t entry_environment{};
    if (std::fegetenv(&entry_environment) != 0) {
        throw std::runtime_error(
            "failed to capture entry floating-point environment");
    }
    std::exception_ptr first_exception;
    for (std::size_t block = 0; block < plan.block_count(); ++block) {
        try {
#ifdef SCAR_PARALLEL_TESTING
            inject_environment_failure(
                captured_hook,
                block,
                false,
                "failed to apply caller floating-point environment");
#endif
            apply_floating_point_environment(
                entry_environment,
                "failed to apply caller floating-point environment");
            const ParallelBlockContext context{block, 0};
            function(
                plan.bounds()[block],
                plan.bounds()[block + 1],
                context);
        } catch (...) {
            keep_first_exception(first_exception);
        }
    }
    std::exception_ptr restore_exception;
    try {
#ifdef SCAR_PARALLEL_TESTING
        inject_environment_failure(
            captured_hook,
            plan.block_count(),
            true,
            "failed to restore entry floating-point environment");
#endif
        apply_floating_point_environment(
            entry_environment,
            "failed to restore entry floating-point environment");
    } catch (...) {
        restore_exception = std::current_exception();
    }
    if (first_exception && restore_exception) {
        throw ParallelEnvironmentRestoreError(
            first_exception, restore_exception);
    }
    if (restore_exception) {
        std::rethrow_exception(restore_exception);
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

void execute_deferred_portion(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function,
    const ParallelFailureHandler& handler,
    const std::fenv_t& environment,
    std::size_t block,
    std::size_t slot
#ifdef SCAR_PARALLEL_TESTING
    , parallel_testing::EnvironmentFailureHook hook
#endif
) noexcept {

    ParallelFailureOrigin origin = ParallelFailureOrigin::EnvironmentApply;
    try {
#ifdef SCAR_PARALLEL_TESTING
        inject_environment_failure(
            hook, block, false,
            "failed to apply caller floating-point environment");
#endif
        apply_floating_point_environment(
            environment,
            "failed to apply caller floating-point environment");
        origin = ParallelFailureOrigin::Callback;
        const ParallelBlockContext context{block, slot};
        function(plan.bounds()[block], plan.bounds()[block + 1], context);
    } catch (...) {
        handler.record(handler.state, block, origin, std::current_exception());
    }
}

std::exception_ptr execute_plan_direct_deferred(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function,
    const ParallelFailureHandler& handler) {

#ifdef SCAR_PARALLEL_TESTING
    const auto captured_hook = environment_failure_hook;
#endif
    std::fenv_t entry_environment{};
    if (std::fegetenv(&entry_environment) != 0) {
        throw std::runtime_error(
            "failed to capture entry floating-point environment");
    }
    for (std::size_t block = 0; block < plan.block_count(); ++block) {
        execute_deferred_portion(
            plan, function, handler, entry_environment, block, 0
#ifdef SCAR_PARALLEL_TESTING
            , captured_hook
#endif
        );
    }
    try {
#ifdef SCAR_PARALLEL_TESTING
        inject_environment_failure(
            captured_hook, plan.block_count(), true,
            "failed to restore entry floating-point environment");
#endif
        apply_floating_point_environment(
            entry_environment,
            "failed to restore entry floating-point environment");
    } catch (...) {
        return std::current_exception();
    }
    return {};
}

}  // namespace

ParallelExecutionPlan::ParallelExecutionPlan(
    ParallelExecutionPlan&& other) noexcept
    : range_(other.range_),
      bounds_(std::move(other.bounds_)),
      block_count_(other.block_count_),
      worker_count_(other.worker_count_) {

    other.reset_to_empty();
}

ParallelExecutionPlan& ParallelExecutionPlan::operator=(
    const ParallelExecutionPlan& other) {

    if (this != &other) {
        ParallelExecutionPlan copy(other);
        swap(copy);
    }
    return *this;
}

ParallelExecutionPlan& ParallelExecutionPlan::operator=(
    ParallelExecutionPlan&& other) noexcept {

    if (this != &other) {
        ParallelExecutionPlan moved(std::move(other));
        swap(moved);
    }
    return *this;
}

void ParallelExecutionPlan::reset_to_empty() noexcept {
    range_ = {};
    bounds_.clear();
    block_count_ = 0;
    worker_count_ = 0;
}

void ParallelExecutionPlan::swap(ParallelExecutionPlan& other) noexcept {
    std::swap(range_, other.range_);
    bounds_.swap(other.bounds_);
    std::swap(block_count_, other.block_count_);
    std::swap(worker_count_, other.worker_count_);
}

ParallelExecutionPlan make_parallel_execution_plan(
    std::int64_t begin,
    std::int64_t end,
    std::size_t block_count,
    int worker_count) {

    const std::int64_t length = checked_range_length(begin, end);
    validate_thread_count(worker_count);
    if (length == 0) {
        if (block_count != 0) {
            throw std::invalid_argument(
                "empty parallel range requires zero blocks");
        }
        return ParallelExecutionPlan(
            {begin, end}, {begin}, 0, 0);
    }
    if (block_count == 0
        || block_count > static_cast<std::uint64_t>(length)) {
        throw std::invalid_argument(
            "parallel block count must be in [1, range length]");
    }
    std::size_t boundary_count = 0;
    std::size_t boundary_bytes = 0;
    if (!checked_size_add(block_count, std::size_t{1}, boundary_count)
        || !checked_size_mul(
            boundary_count, sizeof(std::int64_t), boundary_bytes)) {
        throw std::overflow_error(
            "parallel boundary storage size is not representable");
    }
    std::vector<std::int64_t> bounds;
    if (boundary_count > bounds.max_size()) {
        throw std::length_error("parallel boundary count is too large");
    }
    bounds.resize(boundary_count);
    bounds[0] = begin;
    const std::int64_t blocks = static_cast<std::int64_t>(block_count);
    const std::int64_t quotient = length / blocks;
    const std::int64_t remainder = length % blocks;
    std::int64_t current = begin;
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::int64_t block_size = quotient
            + (block < static_cast<std::size_t>(remainder) ? 1 : 0);
        current += block_size;
        bounds[block + 1] = current;
    }
    if (current != end || boundary_bytes == 0) {
        throw std::logic_error("parallel boundary construction failed");
    }
    return ParallelExecutionPlan(
        {begin, end},
        std::move(bounds),
        block_count,
        static_cast<std::size_t>(worker_count));
}

ParallelExecutionPlan make_parallel_execution_plan_from_bounds(
    const std::vector<std::int64_t>& bounds,
    int worker_count) {

    const auto validated = validate_execution_bounds(bounds, worker_count);
    return ParallelExecutionPlan(
        validated.range,
        bounds,
        validated.block_count,
        validated.worker_count);
}

ParallelExecutionPlan make_parallel_execution_plan_from_bounds(
    std::vector<std::int64_t>&& bounds,
    int worker_count) {

    const auto validated = validate_execution_bounds(bounds, worker_count);
    return ParallelExecutionPlan(
        validated.range,
        std::move(bounds),
        validated.block_count,
        validated.worker_count);
}

std::size_t parallel_execution_slots(const ParallelExecutionPlan& plan) noexcept {
    if (plan.empty()) {
        return 0;
    }
    if (in_runtime_worker || plan.block_count() == 1) {
        return 1;
    }
    return std::min(plan.block_count(), plan.worker_count());
}

namespace {

template <bool Deferred>
std::exception_ptr execute_parallel_plan_impl(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function,
    const ParallelFailureHandler* handler) {

    (void)handler;
    if (plan.empty()) {
        return {};
    }
    if (in_runtime_worker || plan.block_count() == 1) {
        if constexpr (Deferred) {
            return execute_plan_direct_deferred(plan, function, *handler);
        } else {
            execute_plan_direct(plan, function);
            return {};
        }
    }
    const std::size_t runner_count = parallel_execution_slots(plan);
    std::size_t runner_bytes = 0;
    if (runner_count == 0
        || !checked_size_mul(
            runner_count, sizeof(RunnerFunction), runner_bytes)) {
        throw std::overflow_error(
            "parallel runner storage size is not representable");
    }
    std::vector<RunnerFunction> runners;
    if (runner_count > runners.max_size()) {
        throw std::length_error("parallel runner count is too large");
    }
    runners.reserve(runner_count);
    const std::size_t quotient = plan.block_count() / runner_count;
    const std::size_t remainder = plan.block_count() % runner_count;
    std::size_t first_block = 0;
    for (std::size_t slot = 0; slot < runner_count; ++slot) {
        const std::size_t group_size = quotient + (slot < remainder ? 1 : 0);
        const std::size_t last_block = first_block + group_size;
        if constexpr (Deferred) {
            runners.emplace_back(
                [&plan, &function, first_block, last_block, slot, handler](Batch& batch) {
                    for (std::size_t block = first_block;
                         block < last_block;
                         ++block) {
                        execute_deferred_portion(
                            plan, function, *handler,
                            batch.floating_point_environment, block, slot
#ifdef SCAR_PARALLEL_TESTING
                            , batch.environment_hook
#endif
                        );
                    }
                });
        } else {
            runners.emplace_back(
                [&plan, &function, first_block, last_block, slot](Batch& batch) {
                    for (std::size_t block = first_block;
                         block < last_block;
                         ++block) {
                        try {
#ifdef SCAR_PARALLEL_TESTING
                            inject_environment_failure(
                                batch.environment_hook,
                                block,
                                false,
                                "failed to apply caller floating-point environment");
#endif
                            apply_floating_point_environment(
                                batch.floating_point_environment,
                                "failed to apply caller floating-point environment");
                            const ParallelBlockContext context{block, slot};
                            function(
                                plan.bounds()[block],
                                plan.bounds()[block + 1],
                                context);
                        } catch (...) {
                            batch.record_exception(std::current_exception());
                        }
                    }
                });
        }
        first_block = last_block;
    }
    if (first_block != plan.block_count() || runner_bytes == 0) {
        throw std::logic_error("parallel runner construction failed");
    }
    process_runtime().run(std::move(runners), runner_count);
    return {};
}

}  // namespace

void execute_parallel_plan(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function) {

    (void)execute_parallel_plan_impl<false>(plan, function, nullptr);
}

std::exception_ptr execute_parallel_plan_deferred(
    const ParallelExecutionPlan& plan,
    const PreparedParallelBlockFunction& function,
    const ParallelFailureHandler& handler) {

    const ParallelFailureHandler captured_handler = handler;
    if (captured_handler.record == nullptr) {
        throw std::invalid_argument("parallel failure recorder must not be null");
    }
    return execute_parallel_plan_impl<true>(plan, function, &captured_handler);
}

#ifdef SCAR_PARALLEL_TESTING
namespace parallel_testing {
void fail_after(Fault fault, std::size_t successful_attempts) {
    pending_fault = fault;
    fault_countdown = successful_attempts;
}

void clear_fault() noexcept {
    pending_fault = Fault::none;
    fault_countdown = 0;
}

void set_environment_failure_hook(EnvironmentFailureHook hook) noexcept {
    environment_failure_hook = hook;
}

void set_completion_notification_hook(NotificationHook hook) noexcept {
    completion_notification_hook.store(hook, std::memory_order_release);
}

void set_before_batch_wait_hook(NotificationHook hook) noexcept {
    before_batch_wait_hook.store(hook, std::memory_order_release);
}

void reset_completion_notification_count() noexcept {
    completion_notifications.store(0, std::memory_order_relaxed);
}

std::uint64_t completion_notification_count() noexcept {
    return completion_notifications.load(std::memory_order_relaxed);
}

void reset_ready_notification_counts() noexcept {
    ready_notify_one_notifications.store(0, std::memory_order_relaxed);
    ready_notify_all_notifications.store(0, std::memory_order_relaxed);
}

std::uint64_t ready_notify_one_count() noexcept {
    return ready_notify_one_notifications.load(std::memory_order_relaxed);
}

std::uint64_t ready_notify_all_count() noexcept {
    return ready_notify_all_notifications.load(std::memory_order_relaxed);
}
}  // namespace parallel_testing
#endif

void parallel_for_blocks(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t min_grain,
    int n_threads,
    const ParallelBlockFunction& function) {

    if (end < begin) {
        throw std::invalid_argument("parallel range end must be >= begin");
    }
    if (min_grain < 1) {
        throw std::invalid_argument("parallel min_grain must be >= 1");
    }
    validate_thread_count(n_threads);
    const std::int64_t length = checked_range_length(begin, end);
    if (length == 0) {
        return;
    }
    if (n_threads == 1 || length <= min_grain || in_runtime_worker) {
        function(begin, end, 0);
        return;
    }

    const auto grain_blocks = static_cast<std::size_t>(
        length / min_grain + (length % min_grain == 0 ? 0 : 1));
    const std::size_t block_count = static_cast<std::size_t>(
        limit_worker_count(n_threads, grain_blocks));
    if (block_count <= 1) {
        function(begin, end, 0);
        return;
    }

    const ParallelExecutionPlan plan = make_parallel_execution_plan(
        begin,
        end,
        block_count,
        static_cast<int>(block_count));
    execute_parallel_plan(
        plan,
        [&function](
            std::int64_t block_begin,
            std::int64_t block_end,
            const ParallelBlockContext& context) {
            function(block_begin, block_end, context.block_id);
        });
}

ParallelRuntimeInfo parallel_runtime_info() {
    ThreadPool* pool = runtime.load(std::memory_order_acquire);
    if (pool == nullptr) {
        return {};
    }
    if (pool->owner_pid() != current_pid()) {
        return {};
    }
    const ParallelRuntimeInfo info = pool->info();
    return info;
}

void shutdown_parallel_runtime() noexcept {
    ThreadPool* pool = runtime.exchange(nullptr, std::memory_order_acq_rel);
    if (pool != nullptr && pool->owner_pid() == current_pid()) {
        delete pool;
    }
}

}  // namespace scar_internal
