#include "scar/detail/parallel.hpp"

#include <array>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

using scar_internal::ParallelBlockContext;
using scar_internal::ParallelFailureHandler;
using scar_internal::ParallelFailureOrigin;
using scar_internal::PreparedParallelBlockFunction;
using scar_internal::execute_parallel_plan_deferred;
using scar_internal::make_parallel_execution_plan;
using scar_internal::parallel_runtime_info;
using scar_internal::parallel_testing::Fault;
using scar_internal::parallel_testing::set_environment_failure_hook;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[noreturn]] void timeout(const char* message) noexcept {
    std::fprintf(stderr, "parallel deferred timeout: %s\n", message);
    std::fflush(stderr);
    std::_Exit(124);
}

class Event {
public:
    void set() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = true;
        }
        changed_.notify_all();
    }

    void wait(const char* message) noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_for(lock, std::chrono::seconds(10),
                               [this] { return ready_; })) {
            timeout(message);
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool ready_ = false;
};

class Watchdog {
public:
    Watchdog() : worker_([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_for(lock, std::chrono::seconds(30),
                               [this] { return done_; })) {
            timeout("suite did not finish");
        }
    }) {}

    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        changed_.notify_one();
        worker_.join();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool done_ = false;
    std::thread worker_;
};

class RuntimeScope {
public:
    RuntimeScope() {
        scar_internal::shutdown_parallel_runtime();
        scar_internal::parallel_testing::clear_fault();
        set_environment_failure_hook(nullptr);
    }
    ~RuntimeScope() {
        set_environment_failure_hook(nullptr);
        scar_internal::parallel_testing::clear_fault();
        scar_internal::shutdown_parallel_runtime();
    }
};

class EnvironmentScope {
public:
    EnvironmentScope() {
        require(std::fegetenv(&saved_) == 0, "cannot capture test environment");
    }
    ~EnvironmentScope() { (void)std::fesetenv(&saved_); }

private:
    std::fenv_t saved_{};
};

class HookScope {
public:
    explicit HookScope(scar_internal::parallel_testing::EnvironmentFailureHook hook) {
        set_environment_failure_hook(hook);
    }
    ~HookScope() { set_environment_failure_hook(nullptr); }
};

class AsyncCall {
public:
    AsyncCall(std::function<void()> function, std::function<void()> release)
        : release_(std::move(release)), worker_([this, function = std::move(function)] {
            try {
                function();
            } catch (...) {
                failure_ = std::current_exception();
            }
            finished_.store(true);
        }) {}

    ~AsyncCall() {
        release_();
        join();
    }
    void join() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }
    bool finished() const noexcept { return finished_.load(); }
    void rethrow_failure() const {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

private:
    std::function<void()> release_;
    std::exception_ptr failure_;
    std::atomic<bool> finished_{false};
    std::thread worker_;
};

class BlockError : public std::exception {
public:
    explicit BlockError(std::size_t block) noexcept : block_(block) {}
    const char* what() const noexcept override { return "deferred block failure"; }
    std::size_t block() const noexcept { return block_; }

private:
    std::size_t block_;
};

struct FailureRecord {
    std::atomic<unsigned> count{0};
    ParallelFailureOrigin origin = ParallelFailureOrigin::EnvironmentApply;
    std::exception_ptr failure;
};

struct FailureStore {
    std::array<FailureRecord, 8> records{};
    std::atomic<unsigned> invalid_blocks{0};

    ParallelFailureHandler handler() noexcept { return {this, record}; }

    static void record(void* state, std::size_t block,
                       ParallelFailureOrigin origin,
                       std::exception_ptr failure) noexcept {
        auto& store = *static_cast<FailureStore*>(state);
        if (block >= store.records.size()) {
            store.invalid_blocks.fetch_add(1);
            return;
        }
        auto& entry = store.records[block];
        if (entry.count.fetch_add(1) == 0) {
            entry.origin = origin;
            entry.failure = std::move(failure);
        }
    }
};

void require_block_error(const std::exception_ptr& failure, std::size_t block) {
    require(failure != nullptr, "callback exception was not retained");
    try {
        std::rethrow_exception(failure);
    } catch (const BlockError& error) {
        require(error.block() == block, "callback exception lost its block ID");
    } catch (...) {
        throw std::runtime_error("callback exception type changed");
    }
}

void require_restore_error(const std::exception_ptr& failure) {
    require(failure != nullptr, "restore exception was not returned");
    try {
        std::rethrow_exception(failure);
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("restore") != std::string::npos,
                "returned exception is not a restore failure");
    } catch (...) {
        throw std::runtime_error("restore exception type changed");
    }
}

bool fail_apply_two(std::size_t block, bool restoring) noexcept {
    return !restoring && block == 2;
}

bool fail_restore(std::size_t, bool restoring) noexcept { return restoring; }

bool fail_apply_two_and_restore(std::size_t block, bool restoring) noexcept {
    return restoring || block == 2;
}

void validation_and_empty() {
    RuntimeScope runtime;
    FailureStore failures;
    const auto empty = make_parallel_execution_plan(0, 0, 0, 4);
    unsigned calls = 0;
    const PreparedParallelBlockFunction callback =
        [&](std::int64_t, std::int64_t, const ParallelBlockContext&) { ++calls; };
    require(!execute_parallel_plan_deferred(empty, callback, failures.handler()),
            "empty plan reported a restore failure");
    const ParallelFailureHandler stateless{
        nullptr,
        [](void*, std::size_t, ParallelFailureOrigin, std::exception_ptr) noexcept {},
    };
    require(!execute_parallel_plan_deferred(empty, callback, stateless),
            "stateless recorder was rejected");
    for (const auto& plan : {empty, make_parallel_execution_plan(0, 4, 4, 2)}) {
        bool rejected = false;
        try {
            (void)execute_parallel_plan_deferred(plan, callback, {});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "null recorder was accepted");
    }
    require(calls == 0 && !parallel_runtime_info().initialized,
            "empty or invalid deferred plan touched the runtime");
}

void queued_origins_and_siblings() {
    RuntimeScope runtime;
    EnvironmentScope environment;
    require(std::fesetround(FE_DOWNWARD) == 0, "cannot set caller rounding");
    const auto plan = make_parallel_execution_plan(0, 8, 8, 2);
    FailureStore failures;
    std::array<std::atomic<unsigned>, 8> calls{};
    for (auto& count : calls) {
        count.store(0);
    }
    HookScope hook(fail_apply_two);
    const auto before = parallel_runtime_info();
    const auto restore = execute_parallel_plan_deferred(plan,
        [&](std::int64_t begin, std::int64_t end, const ParallelBlockContext& context) {
            require(begin == static_cast<std::int64_t>(context.block_id)
                        && end == begin + 1 && context.worker_slot == context.block_id / 4,
                    "deferred logical geometry changed");
            calls[context.block_id].fetch_add(1);
            require(std::fegetround() == FE_DOWNWARD, "portion missed caller fenv");
            require(std::fesetround(FE_UPWARD) == 0, "cannot mutate worker rounding");
            if (context.block_id == 1 || context.block_id == 5) {
                throw BlockError(context.block_id);
            }
        }, failures.handler());
    const auto after = parallel_runtime_info();
    require(!restore && std::fegetround() == FE_DOWNWARD,
            "queued execution changed the caller environment");
    require(after.batches_submitted - before.batches_submitted == 1
                && after.tasks_submitted - before.tasks_submitted == 2,
            "deferred execution committed the wrong runner count");
    for (std::size_t block = 0; block < calls.size(); ++block) {
        require(calls[block].load() == (block == 2 ? 0U : 1U),
                "apply failure ran its callback or skipped a sibling");
        const bool failed = block == 1 || block == 2 || block == 5;
        require(failures.records[block].count.load() == (failed ? 1U : 0U),
                "deferred portion failure was lost or duplicated");
    }
    require(failures.records[2].origin == ParallelFailureOrigin::EnvironmentApply
                && failures.records[2].failure,
            "environment failure origin was lost");
    for (const std::size_t block : {std::size_t{1}, std::size_t{5}}) {
        require(failures.records[block].origin == ParallelFailureOrigin::Callback,
                "callback failure origin was lost");
        require_block_error(failures.records[block].failure, block);
    }
    require(failures.invalid_blocks.load() == 0, "recorder received an invalid block");
}

void direct_restore_failures() {
    RuntimeScope runtime;
    const auto plan = make_parallel_execution_plan(0, 1, 1, 4);
    for (const bool throw_primary : {false, true}) {
        EnvironmentScope environment;
        require(std::fesetround(FE_DOWNWARD) == 0, "cannot set direct rounding");
        FailureStore failures;
        HookScope hook(fail_restore);
        const auto restore = execute_parallel_plan_deferred(plan,
            [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                require(context.worker_slot == 0 && std::fegetround() == FE_DOWNWARD,
                        "direct deferred context or fenv changed");
                require(std::fesetround(FE_UPWARD) == 0, "cannot mutate direct rounding");
                if (throw_primary) {
                    throw BlockError(0);
                }
            }, failures.handler());
        require_restore_error(restore);
        require(std::fegetround() == FE_UPWARD,
                "failed restore was incorrectly reported as successful");
        require(failures.records[0].count.load() == (throw_primary ? 1U : 0U),
                "restore failure replaced the primary outcome");
        if (throw_primary) {
            require_block_error(failures.records[0].failure, 0);
        }
    }
    require(!parallel_runtime_info().initialized,
            "single-block deferred execution initialized workers");
}

void nested_geometry_and_restore() {
    RuntimeScope runtime;
    const auto plan = make_parallel_execution_plan(0, 6, 6, 4);
    std::array<FailureStore, 2> failures{};
    std::array<std::array<unsigned, 6>, 2> calls{};
    scar_internal::parallel_for_blocks(0, 2, 1, 2,
        [&](std::int64_t, std::int64_t, std::size_t outer) {
            EnvironmentScope environment;
            require(std::fesetround(FE_DOWNWARD) == 0, "cannot set nested rounding");
            HookScope hook(outer == 0 ? fail_apply_two : fail_apply_two_and_restore);
            const auto entry_thread = std::this_thread::get_id();
            const auto before = parallel_runtime_info();
            const auto restore = execute_parallel_plan_deferred(plan,
                [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                    require(context.worker_slot == 0 && std::this_thread::get_id() == entry_thread,
                            "nested deferred work left local slot zero");
                    ++calls[outer][context.block_id];
                    require(std::fegetround() == FE_DOWNWARD,
                            "nested portion did not reapply its entry fenv");
                    require(std::fesetround(FE_UPWARD) == 0,
                            "cannot mutate nested rounding");
                    if (context.block_id == 1) {
                        throw BlockError(1);
                    }
                }, failures[outer].handler());
            const auto after = parallel_runtime_info();
            require(before.batches_submitted == after.batches_submitted
                        && before.tasks_submitted == after.tasks_submitted,
                    "nested deferred execution enqueued work");
            if (outer == 0) {
                require(!restore && std::fegetround() == FE_DOWNWARD,
                        "nested entry environment was not restored");
            } else {
                require_restore_error(restore);
                require(std::fegetround() == FE_UPWARD,
                        "nested restore failure was concealed");
            }
        });
    for (std::size_t outer = 0; outer < failures.size(); ++outer) {
        for (std::size_t block = 0; block < 6; ++block) {
            require(calls[outer][block] == (block == 2 ? 0U : 1U),
                    "nested sibling continuation changed");
        }
        require_block_error(failures[outer].records[1].failure, 1);
        require(failures[outer].records[2].origin == ParallelFailureOrigin::EnvironmentApply
                    && failures[outer].records[2].count.load() == 1,
                "nested apply failure was not retained");
    }
    const auto info = parallel_runtime_info();
    require(info.batches_submitted == 1 && info.tasks_submitted == 2,
            "nested deferred work changed outer queue counters");
}

struct BlockingFailureState {
    FailureStore failures;
    Event entered;
    Event release;

    static void record(void* state, std::size_t block, ParallelFailureOrigin origin,
                       std::exception_ptr failure) noexcept {
        auto& self = *static_cast<BlockingFailureState*>(state);
        FailureStore::record(&self.failures, block, origin, std::move(failure));
        self.entered.set();
        self.release.wait("recorder was not released");
    }
};

struct CallbackCleanup {
    std::atomic<unsigned>& count;
    ~CallbackCleanup() { count.fetch_add(1); }
};

struct CaptureOwner {
    explicit CaptureOwner(std::atomic<bool>& destroyed) : destroyed_(destroyed) {}
    ~CaptureOwner() { destroyed_.store(true); }
    std::atomic<bool>& destroyed_;
};

void recorder_and_callback_drain() {
    RuntimeScope runtime;
    const auto plan = make_parallel_execution_plan(0, 4, 4, 2);
    BlockingFailureState state;
    Event callback_entered;
    Event callback_release;
    std::atomic<unsigned> cleanup_count{0};
    std::atomic<bool> captures_destroyed{false};
    unsigned cleanup_count_at_return = 0;
    const auto before = parallel_runtime_info();
    AsyncCall caller([&] {
        auto owner = std::make_shared<CaptureOwner>(captures_destroyed);
        PreparedParallelBlockFunction function =
            [&, owner](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                CallbackCleanup cleanup{cleanup_count};
                if (context.block_id == 0) {
                    throw BlockError(0);
                }
                if (context.block_id == 2) {
                    callback_entered.set();
                    callback_release.wait("callback was not released");
                }
                require(owner != nullptr, "callback lost its captured owner");
            };
        owner.reset();
        const auto restore = execute_parallel_plan_deferred(
            plan, function, {&state, BlockingFailureState::record});
        require(!restore, "queued callback drain reported a restore error");
        cleanup_count_at_return = cleanup_count.load();
        function = nullptr;
        require(captures_destroyed.load(), "callback capture was not released");
    }, [&] {
        callback_release.set();
        state.release.set();
    });
    state.entered.wait("recorder did not receive the callback error");
    callback_entered.wait("second runner did not reach its callback");
    require(!caller.finished(), "deferred caller returned with an active callback");
    callback_release.set();
    require(!caller.finished(), "deferred caller returned with an active recorder");
    state.release.set();
    caller.join();
    caller.rethrow_failure();
    require(cleanup_count_at_return == 4 && cleanup_count.load() == 4,
            "deferred completion preceded callback cleanup");
    require_block_error(state.failures.records[0].failure, 0);
    const auto after = parallel_runtime_info();
    require(after.batches_submitted - before.batches_submitted == 1
                && after.tasks_submitted - before.tasks_submitted == 2,
            "deferred drain changed queue accounting");
}

void infrastructure_failures_remain_ordinary() {
    for (const auto fault : {Fault::worker_creation, Fault::batch_allocation,
                             Fault::queue_insertion}) {
        RuntimeScope runtime;
        const auto plan = make_parallel_execution_plan(0, 4, 4, 2);
        FailureStore failures;
        std::atomic<unsigned> calls{0};
        const PreparedParallelBlockFunction function =
            [&](std::int64_t, std::int64_t, const ParallelBlockContext&) {
                calls.fetch_add(1);
            };
        scar_internal::parallel_testing::fail_after(
            fault, fault == Fault::queue_insertion ? 1 : 0);
        bool threw = false;
        try {
            (void)execute_parallel_plan_deferred(plan, function, failures.handler());
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        scar_internal::parallel_testing::clear_fault();
        require(threw && calls.load() == 0,
                "infrastructure failure became a deferred portion failure");
        require(parallel_runtime_info().batches_submitted == 0
                    && parallel_runtime_info().tasks_submitted == 0,
                "failed publication changed committed counters");
        for (const auto& record : failures.records) {
            require(record.count.load() == 0, "infrastructure failure reached recorder");
        }
        require(!execute_parallel_plan_deferred(plan, function, failures.handler())
                    && calls.load() == 4,
                "deferred runtime did not recover after publication failure");
    }
}

}  // namespace

int run_parallel_deferred_tests() {
    Watchdog watchdog;
    try {
        validation_and_empty();
        queued_origins_and_siblings();
        direct_restore_failures();
        nested_geometry_and_restore();
        recorder_and_callback_drain();
        infrastructure_failures_remain_ordinary();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "parallel deferred regression: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "parallel deferred regression: unknown exception\n");
        return 2;
    }
    return 0;
}
