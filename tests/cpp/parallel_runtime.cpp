#include "scar/detail/parallel.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cfenv>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using scar_internal::ParallelBlockFunction;
using scar_internal::ParallelRuntimeInfo;
using scar_internal::parallel_for_blocks;
using scar_internal::parallel_runtime_info;
using scar_internal::parallel_testing::Fault;
using scar_internal::parallel_testing::clear_fault;
using scar_internal::parallel_testing::completion_notification_count;
using scar_internal::parallel_testing::fail_after;
using scar_internal::parallel_testing::reset_completion_notification_count;
using scar_internal::parallel_testing::reset_ready_notification_counts;
using scar_internal::parallel_testing::ready_notify_all_count;
using scar_internal::parallel_testing::ready_notify_one_count;
using scar_internal::parallel_testing::set_before_batch_wait_hook;
using scar_internal::parallel_testing::set_completion_notification_hook;
using scar_internal::parallel_testing::set_environment_failure_hook;
using scar_internal::ParallelBlockContext;
using scar_internal::ParallelExecutionPlan;
using scar_internal::execute_parallel_plan;
using scar_internal::make_parallel_execution_plan;
using scar_internal::make_parallel_execution_plan_from_bounds;
using scar_internal::parallel_execution_slots;

constexpr auto wait_limit = std::chrono::seconds(10);

[[noreturn]] void timeout(const char* message) {
    std::fprintf(stderr, "parallel runtime timeout: %s\n", message);
    std::fflush(stderr);
    // A broken scheduler may still reference test storage. Do not unwind or
    // destroy that storage while its callbacks could still be executing.
    std::_Exit(124);
}

class Watchdog {
public:
    Watchdog() : worker_([this]() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_for(lock, std::chrono::seconds(30),
                               [this]() { return done_; })) {
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

class Event {
public:
    void set() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            set_ = true;
        }
        changed_.notify_all();
    }

    void wait(const char* message) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_for(lock, wait_limit,
                               [this]() { return set_; })) {
            timeout(message);
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool set_ = false;
};

class Arrivals {
public:
    void arrive() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++count_;
        }
        changed_.notify_all();
    }

    void wait(std::size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_for(lock, wait_limit,
                               [this, count]() { return count_ == count; })) {
            timeout("workers did not reach the blocking callback");
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t count_ = 0;
};

class AsyncCall {
public:
    AsyncCall(std::function<void()> call, std::function<void()> release)
        : release_(std::move(release)), worker_([this, call = std::move(call)]() {
            try {
                call();
            } catch (...) {
                exception_ = std::current_exception();
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

    bool finished() const { return finished_.load(); }
    std::exception_ptr exception() const { return exception_; }

private:
    std::function<void()> release_;
    std::exception_ptr exception_;
    std::atomic<bool> finished_{false};
    std::thread worker_;
};

class RuntimeScope {
public:
    RuntimeScope() {
        clear_fault();
        set_environment_failure_hook(nullptr);
        set_completion_notification_hook(nullptr);
        set_before_batch_wait_hook(nullptr);
        reset_completion_notification_count();
        reset_ready_notification_counts();
        scar_internal::shutdown_parallel_runtime();
    }

    ~RuntimeScope() {
        clear_fault();
        set_environment_failure_hook(nullptr);
        set_completion_notification_hook(nullptr);
        set_before_batch_wait_hook(nullptr);
        scar_internal::shutdown_parallel_runtime();
    }
};

class ScopedFault {
public:
    ScopedFault(Fault fault, std::size_t successful_attempts) {
        fail_after(fault, successful_attempts);
    }
    ~ScopedFault() { clear_fault(); }
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool rejects_submission(const ParallelBlockFunction& callback) {
    try {
        parallel_for_blocks(0, 4, 1, 4, callback);
    } catch (const std::bad_alloc&) {
        return true;
    }
    return false;
}

bool same_commits(const ParallelRuntimeInfo& lhs,
                  const ParallelRuntimeInfo& rhs) {
    return lhs.batches_submitted == rhs.batches_submitted
        && lhs.tasks_submitted == rhs.tasks_submitted
        && lhs.peak_queued_tasks == rhs.peak_queued_tasks;
}

void failed_submission(Fault fault, std::size_t successful_attempts,
                       std::size_t expected_workers) {
    std::atomic<int> rejected_callbacks{0};
    std::atomic<int> recovered_callbacks{0};
    const ParallelBlockFunction rejected =
        [&](std::int64_t, std::int64_t, std::size_t) {
            rejected_callbacks.fetch_add(1);
        };
    // Keep callback storage alive through recovery and shutdown even when an
    // old runtime incorrectly leaves rejected callbacks in its queue.
    RuntimeScope runtime;
    const auto before = parallel_runtime_info();
    ScopedFault injected(fault, successful_attempts);
    const bool rejected_with_bad_alloc = rejects_submission(rejected);
    const auto failed = parallel_runtime_info();
    const int callbacks_at_failure = rejected_callbacks.load();

    // The fault is still armed in this scope: recovery also verifies that it
    // was consumed exactly once by the failed submission.
    parallel_for_blocks(0, 4, 1, 4,
        [&](std::int64_t, std::int64_t, std::size_t) {
            recovered_callbacks.fetch_add(1);
        });
    const auto recovered = parallel_runtime_info();
    scar_internal::shutdown_parallel_runtime();

    require(rejected_with_bad_alloc, "submission did not propagate bad_alloc");
    require(callbacks_at_failure == 0 && rejected_callbacks.load() == 0,
            "rejected callback ran during failure or recovery");
    require(failed.initialized && failed.worker_count == expected_workers
            && failed.worker_start_events == expected_workers,
            "failed submission lost track of created workers");
    require(same_commits(before, failed),
            "failed submission changed commit or queue-peak counters");
    require(recovered_callbacks.load() == 4
            && recovered.worker_count == 4
            && recovered.worker_start_events == 4
            && recovered.batches_submitted == 1
            && recovered.tasks_submitted == 4
            && recovered.peak_queued_tasks == 4,
            "runtime did not recover after rejected submission");
}

ParallelRuntimeInfo wait_for_commits(std::uint64_t count) {
    const auto deadline = std::chrono::steady_clock::now() + wait_limit;
    while (true) {
        const auto info = parallel_runtime_info();
        if (info.batches_submitted == count) {
            return info;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            timeout("foreign batch was not committed");
        }
        // Observing the committed diagnostic under the queue mutex proves
        // publication. A delay or a producer-started event would not do so.
        std::this_thread::yield();
    }
}

void rollback_preserves_foreign_queue() {
    Event release;
    Arrivals arrived;
    std::atomic<int> rejected_callbacks{0};
    std::atomic<int> recovered_callbacks{0};
    std::array<std::atomic<int>, 4> foreign_callbacks{};
    for (auto& count : foreign_callbacks) {
        count.store(0);
    }
    const ParallelBlockFunction rejected =
        [&](std::int64_t, std::int64_t, std::size_t) {
            rejected_callbacks.fetch_add(1);
        };
    RuntimeScope runtime;
    AsyncCall blocking([&]() {
        parallel_for_blocks(0, 4, 1, 4,
            [&](std::int64_t, std::int64_t, std::size_t) {
                arrived.arrive();
                release.wait("blocking workers were not released");
            });
    }, [&]() { release.set(); });
    arrived.wait(4);

    // A fault armed on this caller must not affect the foreign producer.
    ScopedFault first_fault(Fault::queue_insertion, 0);
    AsyncCall foreign([&]() {
        parallel_for_blocks(0, 4, 1, 4,
            [&](std::int64_t, std::int64_t, std::size_t block) {
                foreign_callbacks[block].fetch_add(1);
            });
    }, [&]() { release.set(); });
    const auto queued = wait_for_commits(2);
    bool failures_ok = rejects_submission(rejected);
    bool counters_ok = same_commits(queued, parallel_runtime_info());
    for (const std::size_t successful_attempts : {std::size_t{1},
                                                 std::size_t{3}}) {
        ScopedFault fault(Fault::queue_insertion, successful_attempts);
        failures_ok = rejects_submission(rejected) && failures_ok;
        counters_ok = same_commits(queued, parallel_runtime_info())
            && counters_ok;
    }
    const bool no_early_callbacks = rejected_callbacks.load() == 0;
    release.set();
    blocking.join();
    foreign.join();
    parallel_for_blocks(0, 4, 1, 4,
        [&](std::int64_t, std::int64_t, std::size_t) {
            recovered_callbacks.fetch_add(1);
        });
    const auto recovered = parallel_runtime_info();
    scar_internal::shutdown_parallel_runtime();

    require(!blocking.exception() && !foreign.exception(),
            "a committed foreign batch failed");
    require(failures_ok && counters_ok,
            "rollback failed with a previously committed queue");
    require(no_early_callbacks && rejected_callbacks.load() == 0,
            "rolled-back callbacks survived into later work");
    for (const auto& count : foreign_callbacks) {
        require(count.load() == 1, "rollback changed foreign callbacks");
    }
    require(queued.worker_count == 4 && queued.tasks_submitted == 8
            && queued.peak_queued_tasks == 4
            && recovered_callbacks.load() == 4
            && recovered.batches_submitted == 3
            && recovered.tasks_submitted == 12
            && recovered.peak_queued_tasks == 4,
            "foreign-queue recovery counters are incorrect");
}

struct FirstException {};
struct LaterException {};

std::atomic<bool>* completion_seen = nullptr;

void mark_completion_seen() noexcept {
    completion_seen->store(true, std::memory_order_release);
}

void wait_for_completion_before_wait() noexcept {
    while (!completion_seen->load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void completion_notifies_once_per_queued_batch() {
    RuntimeScope runtime;
    std::atomic<bool> seen{false};
    completion_seen = &seen;
    set_completion_notification_hook(mark_completion_seen);
    set_before_batch_wait_hook(wait_for_completion_before_wait);
    std::atomic<int> visits{0};
    execute_parallel_plan(make_parallel_execution_plan(0, 34, 17, 4),
        [&](std::int64_t, std::int64_t, const ParallelBlockContext&) {
            visits.fetch_add(1);
        });
    set_before_batch_wait_hook(nullptr);
    set_completion_notification_hook(nullptr);
    completion_seen = nullptr;
    require(seen.load() && visits.load() == 17
            && completion_notification_count() == 1,
            "completion before caller wait did not notify exactly once");

    bool caught = false;
    try {
        execute_parallel_plan(make_parallel_execution_plan(0, 8, 4, 4),
            [](std::int64_t, std::int64_t, const ParallelBlockContext&) {
                throw FirstException{};
            });
    } catch (const FirstException&) {
        caught = true;
    }
    require(caught && completion_notification_count() == 2,
            "multiple runner failures changed completion notification count");

    execute_parallel_plan(make_parallel_execution_plan(0, 34, 17, 2),
        [](std::int64_t, std::int64_t, const ParallelBlockContext&) {});
    parallel_for_blocks(0, 4, 1, 4,
        [](std::int64_t, std::int64_t, std::size_t) {});
    parallel_for_blocks(0, 4, 1, 1,
        [](std::int64_t, std::int64_t, std::size_t) {});
    require(completion_notification_count() == 4,
            "alternating queued and direct calls changed completion notifications");

    execute_parallel_plan(make_parallel_execution_plan(0, 4, 4, 2),
        [](std::int64_t, std::int64_t, const ParallelBlockContext&) {
            execute_parallel_plan(make_parallel_execution_plan(0, 18, 9, 4),
                [](std::int64_t, std::int64_t, const ParallelBlockContext&) {});
        });
    require(completion_notification_count() == 5,
            "nested direct work emitted a completion notification");
}

void ready_notifications_follow_selected_threshold() {
    RuntimeScope runtime;
    const auto noop = [](
        std::int64_t, std::int64_t, const ParallelBlockContext&) {};

    parallel_for_blocks(0, 0, 1, 4,
        [](std::int64_t, std::int64_t, std::size_t) {});
    parallel_for_blocks(0, 4, 4, 4,
        [](std::int64_t, std::int64_t, std::size_t) {});
    parallel_for_blocks(0, 4, 1, 1,
        [](std::int64_t, std::int64_t, std::size_t) {});
    require(ready_notify_one_count() == 0 && ready_notify_all_count() == 0,
            "direct work emitted a ready notification");

    execute_parallel_plan(make_parallel_execution_plan(0, 4, 4, 4), noop);
    require(ready_notify_one_count() == 0 && ready_notify_all_count() == 1,
            "P4/J4 did not use one broad ready notification");

    execute_parallel_plan(make_parallel_execution_plan(0, 32, 32, 32), noop);
    require(ready_notify_one_count() == 0 && ready_notify_all_count() == 2,
            "P32/J32 did not use one broad ready notification");

    execute_parallel_plan(make_parallel_execution_plan(0, 4, 4, 4), noop);
    require(ready_notify_one_count() == 4 && ready_notify_all_count() == 2,
            "P32/J4 did not use four narrow ready notifications");

    execute_parallel_plan(make_parallel_execution_plan(0, 10, 10, 10), noop);
    require(ready_notify_one_count() == 14 && ready_notify_all_count() == 2,
            "P32/J10 did not stay below the broad-wake threshold");

    execute_parallel_plan(make_parallel_execution_plan(0, 11, 11, 11), noop);
    require(ready_notify_one_count() == 14 && ready_notify_all_count() == 3,
            "P32/J11 did not cross the broad-wake threshold");

    execute_parallel_plan(make_parallel_execution_plan(0, 24, 24, 24), noop);
    require(ready_notify_one_count() == 14 && ready_notify_all_count() == 4,
            "P32/J24 did not use one broad ready notification");

    execute_parallel_plan(make_parallel_execution_plan(0, 32, 32, 32), noop);
    require(ready_notify_one_count() == 14 && ready_notify_all_count() == 5,
            "P32/J32 changed after the pool was warm");

    execute_parallel_plan(make_parallel_execution_plan(0, 257, 257, 4), noop);
    require(ready_notify_one_count() == 18 && ready_notify_all_count() == 5,
            "prepared B greater than W did not notify for J only");
}

void concurrent_growth_and_producers_complete() {
    RuntimeScope runtime;
    constexpr std::array<std::size_t, 4> worker_counts{2, 8, 16, 32};
    std::array<std::atomic<int>, worker_counts.size()> visits{};
    std::array<std::exception_ptr, worker_counts.size()> failures{};
    std::array<std::thread, worker_counts.size()> producers;
    Arrivals ready;
    Event release;
    for (std::size_t index = 0; index < worker_counts.size(); ++index) {
        visits[index].store(0);
        producers[index] = std::thread([&, index]() {
            ready.arrive();
            release.wait("concurrent producers were not released");
            try {
                const auto count = worker_counts[index];
                execute_parallel_plan(
                    make_parallel_execution_plan(
                        0,
                        static_cast<std::int64_t>(count),
                        count,
                        static_cast<int>(count)),
                    [&](std::int64_t, std::int64_t,
                        const ParallelBlockContext&) {
                        visits[index].fetch_add(1);
                    });
            } catch (...) {
                failures[index] = std::current_exception();
            }
        });
    }
    ready.wait(worker_counts.size());
    release.set();
    for (auto& producer : producers) {
        producer.join();
    }

    for (std::size_t index = 0; index < worker_counts.size(); ++index) {
        require(!failures[index]
                && visits[index].load() == static_cast<int>(worker_counts[index]),
                "concurrent growth lost a producer or callback");
    }
    const auto info = parallel_runtime_info();
    require(info.worker_count == 32 && info.worker_start_events == 32
            && info.batches_submitted == worker_counts.size()
            && info.tasks_submitted == 58,
            "concurrent growth changed resident or submission accounting");

    // J=32 always broadcasts. Each smaller submission either broadcasts for
    // its observed pool size or contributes exactly J narrow calls.
    const auto one = ready_notify_one_count();
    const auto all = ready_notify_all_count();
    require((one & ~std::uint64_t{26}) == 0
            && (one & std::uint64_t{1}) == 0
            && all + ((one & 2) != 0 ? 1 : 0)
                   + ((one & 8) != 0 ? 1 : 0)
                   + ((one & 16) != 0 ? 1 : 0) == worker_counts.size(),
            "concurrent growth notification telemetry is inconsistent");
}

void committed_failure_drains_and_keeps_first_exception() {
    Event release_first;
    Event release_others;
    Arrivals arrived;
    std::atomic<int> completed_others{0};
    std::atomic<int> sentinels{0};
    std::atomic<int> recovered_callbacks{0};
    RuntimeScope runtime;
    AsyncCall throwing([&]() {
        parallel_for_blocks(0, 4, 1, 4,
            [&](std::int64_t, std::int64_t, std::size_t block) {
                arrived.arrive();
                if (block == 0) {
                    release_first.wait("first exception was not released");
                    throw FirstException{};
                }
                release_others.wait("remaining callbacks were not released");
                completed_others.fetch_add(1);
                if (block == 3) {
                    throw LaterException{};
                }
            });
    }, [&]() {
        release_first.set();
        release_others.set();
    });
    arrived.wait(4);
    release_first.set();
    // Only the worker that threw FirstException can execute this batch.
    // Its wrapper has therefore registered that exception before either
    // LaterException or normal completion of the three blocked callbacks.
    parallel_for_blocks(0, 2, 1, 2,
        [&](std::int64_t, std::int64_t, std::size_t) {
            sentinels.fetch_add(1);
        });
    const bool waited_for_all = !throwing.finished()
        && completed_others.load() == 0;
    release_others.set();
    throwing.join();
    bool kept_first = false;
    if (throwing.exception()) {
        try {
            std::rethrow_exception(throwing.exception());
        } catch (const FirstException&) {
            kept_first = true;
        } catch (...) {
        }
    }
    parallel_for_blocks(0, 4, 1, 4,
        [&](std::int64_t, std::int64_t, std::size_t) {
            recovered_callbacks.fetch_add(1);
        });
    const auto recovered = parallel_runtime_info();
    require(waited_for_all && completed_others.load() == 3,
            "committed callback exception escaped before batch drained");
    require(kept_first, "runtime replaced the first registered exception");
    require(sentinels.load() == 2 && recovered_callbacks.load() == 4
            && recovered.batches_submitted == 3
            && recovered.tasks_submitted == 10,
            "runtime did not recover after a throwing batch");
}

void verify_plan_execution(const ParallelExecutionPlan& plan) {
    const auto blocks = plan.block_count();
    const auto slots = parallel_execution_slots(plan);
    std::vector<std::atomic<int>> visits(blocks);
    std::vector<std::atomic<std::size_t>> observed_slots(blocks);
    std::vector<std::atomic<int>> busy(slots);
    std::vector<std::atomic<std::size_t>> scratch(slots);
    for (auto& value : visits) { value.store(0); }
    for (auto& value : observed_slots) { value.store(slots); }
    for (auto& value : busy) { value.store(0); }
    for (auto& value : scratch) { value.store(blocks); }
    std::atomic<int> active{0};
    std::atomic<int> peak{0};
    std::atomic<bool> invalid{false};
    std::atomic<bool> caller_executed{false};
    const auto caller = std::this_thread::get_id();
    execute_parallel_plan(plan,
        [&](std::int64_t begin, std::int64_t end, const ParallelBlockContext& context) {
            if (context.block_id >= blocks || context.worker_slot >= slots) {
                invalid.store(true);
                return;
            }
            const auto block = context.block_id;
            const auto slot = context.worker_slot;
            const int running = active.fetch_add(1) + 1;
            int old_peak = peak.load();
            while (old_peak < running && !peak.compare_exchange_weak(old_peak, running)) {}
            if (busy[slot].fetch_add(1) != 0 || begin != plan.bounds()[block]
                    || end != plan.bounds()[block + 1]) {
                invalid.store(true);
            }
            visits[block].fetch_add(1);
            observed_slots[block].store(slot);
            scratch[slot].store(block);
            std::this_thread::yield();
            if (scratch[slot].load() != block) { invalid.store(true); }
            busy[slot].fetch_sub(1);
            active.fetch_sub(1);
            if (std::this_thread::get_id() == caller) { caller_executed.store(true); }
        });
    require(!invalid.load() && active.load() == 0
            && peak.load() <= static_cast<int>(slots),
            "prepared callbacks exceeded their slots or shared active scratch");
    if (blocks == 0) {
        require(slots == 0 && peak.load() == 0, "empty plan executed a callback");
        return;
    }
    require(peak.load() >= 1 && caller_executed.load() == (blocks == 1),
            "prepared execution used the wrong caller or worker path");
    std::size_t block = 0;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto count = blocks / slots + (slot < blocks % slots ? 1 : 0);
        for (std::size_t offset = 0; offset < count; ++offset, ++block) {
            require(visits[block].load() == 1 && observed_slots[block].load() == slot,
                    "logical portion coverage or contiguous slot assignment changed");
        }
    }
}

void prepared_geometry_and_dispatch() {
    struct Case { std::size_t blocks; int workers; std::int64_t end; };
    for (const Case test : {Case{0, 4, -7}, Case{1, 4, 96}, Case{3, 4, 96},
                           Case{4, 4, 96}, Case{5, 4, 96}, Case{17, 1, 96},
                           Case{17, 2, 96}, Case{17, 4, 96}, Case{257, 4, 1024}}) {
        RuntimeScope runtime;
        const auto plan = make_parallel_execution_plan(-7, test.end, test.blocks, test.workers);
        require(plan.range().begin == -7 && plan.range().end == test.end
                && plan.bounds().size() == test.blocks + 1
                && plan.bounds().front() == -7 && plan.bounds().back() == test.end,
                "prepared plan has incorrect bounds or range");
        std::int64_t expected = -7;
        for (std::size_t block = 0; block < test.blocks; ++block) {
            const auto length = test.end + 7;
            expected += length / static_cast<std::int64_t>(test.blocks)
                + (block < static_cast<std::size_t>(length) % test.blocks ? 1 : 0);
            require(plan.bounds()[block + 1] == expected,
                    "prepared quotient/remainder partition changed");
        }
        require(!parallel_runtime_info().initialized, "plan creation initialized workers");
        verify_plan_execution(plan);
        const auto info = parallel_runtime_info();
        const auto runners = test.blocks < 2 ? 0 : parallel_execution_slots(plan);
        require(info.initialized == (runners != 0) && info.worker_count == runners
                && info.worker_start_events == runners && info.tasks_submitted == runners
                && info.batches_submitted == (runners == 0 ? 0 : 1),
                "prepared plan counted logical portions as queued runners");
    }
    RuntimeScope runtime;
    const std::vector<std::int64_t> bounds{-7, -6, 5, 12, 96};
    const auto custom = make_parallel_execution_plan_from_bounds(bounds, 2);
    require(custom.bounds() == bounds, "explicit boundaries were repartitioned");
    verify_plan_execution(custom);
}

void prepared_copy_and_move() {
    RuntimeScope runtime;
    auto original = make_parallel_execution_plan(-7, 96, 17, 4);
    const auto expected = original.bounds();
    auto copy = original;
    auto moved = std::move(original);
    auto assigned = make_parallel_execution_plan(2, 3, 1, 1);
    assigned = copy;
    copy = make_parallel_execution_plan(8, 8, 0, 2);
    auto move_assigned = make_parallel_execution_plan(9, 10, 1, 1);
    move_assigned = std::move(moved);
    for (const auto* empty : {&original, &moved}) {
        require(empty->empty() && empty->block_count() == 0 && empty->worker_count() == 0
                && empty->range().begin == 0 && empty->range().end == 0
                && empty->bounds().empty() && parallel_execution_slots(*empty) == 0,
                "moved-from plan retained executable metadata");
        verify_plan_execution(*empty);
    }
    for (const auto* owned : {&assigned, &move_assigned}) {
        require(owned->bounds() == expected && owned->block_count() == 17
                && owned->worker_count() == 4 && parallel_execution_slots(*owned) == 4,
                "plan copy or move lost immutable metadata");
        verify_plan_execution(*owned);
    }
}

void prepared_history_and_concurrent_calls() {
    Event release;
    Arrivals occupied;
    std::array<std::atomic<int>, 4> first{};
    std::array<std::atomic<int>, 33> visits{};
    for (auto& value : first) { value.store(0); }
    for (auto& value : visits) { value.store(0); }
    RuntimeScope runtime;
    parallel_for_blocks(0, 32, 1, 32, [](std::int64_t, std::int64_t, std::size_t) {});
    const auto before = parallel_runtime_info();
    const auto large = make_parallel_execution_plan(-7, 1024, 257, 4);
    verify_plan_execution(large);
    const auto after_large = parallel_runtime_info();
    require(after_large.worker_count == 32 && after_large.worker_start_events == 32
            && after_large.batches_submitted == before.batches_submitted + 1
            && after_large.tasks_submitted == before.tasks_submitted + 4,
            "resident history expanded the prepared runner budget");
    const auto long_plan = make_parallel_execution_plan(0, 99, 33, 4);
    AsyncCall long_call([&]() {
        execute_parallel_plan(long_plan,
            [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                require(context.block_id < visits.size() && context.worker_slot < first.size(),
                        "concurrent prepared context is out of bounds");
                visits[context.block_id].fetch_add(1);
                if (first[context.worker_slot].fetch_add(1) == 0) {
                    occupied.arrive();
                    release.wait("long prepared runners were not released");
                }
            });
    }, [&]() { release.set(); });
    occupied.wait(4);
    verify_plan_execution(make_parallel_execution_plan(-7, 96, 5, 2));
    const bool long_still_pending = !long_call.finished();
    release.set();
    long_call.join();
    const auto after = parallel_runtime_info();
    require(long_still_pending && !long_call.exception(),
            "neighboring call did not finish while long prepared runners were occupied");
    for (const auto& value : visits) {
        require(value.load() == 1, "concurrent prepared call lost logical portions");
    }
    require(after.batches_submitted == after_large.batches_submitted + 2
            && after.tasks_submitted == after_large.tasks_submitted + 6,
            "concurrent prepared calls counted queue entries incorrectly");
}

void prepared_exceptions_continue_siblings() {
    Event release_first;
    Event first_registered;
    std::array<std::atomic<int>, 17> visits{};
    for (auto& value : visits) { value.store(0); }
    RuntimeScope runtime;
    const auto plan = make_parallel_execution_plan(0, 34, 17, 2);
    AsyncCall call([&]() {
        execute_parallel_plan(plan,
            [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                visits[context.block_id].fetch_add(1);
                if (context.block_id == 0) {
                    release_first.wait("earlier logical exception was not released");
                    throw LaterException{};
                }
                if (context.block_id == 9) { throw FirstException{}; }
                if (context.block_id == 10) { first_registered.set(); }
            });
    }, [&]() { release_first.set(); });
    first_registered.wait("throwing portion prevented its next sibling");
    const bool waited = !call.finished();
    release_first.set();
    call.join();
    bool kept_first = false;
    if (call.exception()) {
        try { std::rethrow_exception(call.exception()); }
        catch (const FirstException&) { kept_first = true; }
        catch (...) {}
    }
    for (const auto& value : visits) {
        require(value.load() == 1, "throwing portion skipped a planned sibling");
    }
    require(waited && kept_first, "prepared exception transport changed registration order");
    verify_plan_execution(plan);
    const auto info = parallel_runtime_info();
    require(info.batches_submitted == 2 && info.tasks_submitted == 4,
            "prepared exception recovery changed runner accounting");
}

class EnvironmentScope {
public:
    EnvironmentScope() {
        require(std::fegetenv(&entry_) == 0, "cannot capture test floating-point environment");
    }
    ~EnvironmentScope() { (void)std::fesetenv(&entry_); }
private:
    std::fenv_t entry_{};
};

void install_environment(int rounding, int flags) {
    require(std::fesetround(rounding) == 0 && std::feclearexcept(FE_ALL_EXCEPT) == 0
            && std::feraiseexcept(flags) == 0, "cannot install test floating-point environment");
}

bool initial_environment() {
    return std::fegetround() == FE_DOWNWARD
        && std::fetestexcept(FE_ALL_EXCEPT) == FE_INEXACT;
}

void prepared_fenv_and_nested_paths() {
    RuntimeScope runtime;
    EnvironmentScope environment;
    install_environment(FE_DOWNWARD, FE_INEXACT);
    std::atomic<int> visits{0};
    std::atomic<bool> bad_environment{false};
    bool caught = false;
    try {
        execute_parallel_plan(make_parallel_execution_plan(0, 18, 9, 2),
            [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                if (!initial_environment()) { bad_environment.store(true); }
                visits.fetch_add(1);
                install_environment(FE_UPWARD, FE_INVALID);
                if (context.block_id == 1) { throw LaterException{}; }
            });
    } catch (const LaterException&) { caught = true; }
    require(caught && visits.load() == 9 && !bad_environment.load() && initial_environment(),
            "queued portions did not reapply the caller floating-point environment");
    execute_parallel_plan(make_parallel_execution_plan(0, 1, 1, 4),
        [](std::int64_t, std::int64_t, const ParallelBlockContext&) {
            install_environment(FE_UPWARD, FE_INVALID);
        });
    require(initial_environment(), "direct single-block plan failed to restore caller environment");
    const auto before = parallel_runtime_info();
    parallel_for_blocks(0, 2, 1, 2,
        [](std::int64_t, std::int64_t, std::size_t) {
            EnvironmentScope nested_environment;
            install_environment(FE_DOWNWARD, FE_INEXACT);
            const auto plan = make_parallel_execution_plan(0, 18, 9, 4);
            require(parallel_execution_slots(plan) == 1,
                    "nested scratch helper did not return one local slot");
            int nested_visits = 0;
            bool nested_caught = false;
            try {
                execute_parallel_plan(plan,
                    [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                        require(context.worker_slot == 0 && initial_environment(),
                                "nested prepared portion inherited sibling state");
                        ++nested_visits;
                        install_environment(FE_UPWARD, FE_INVALID);
                        if (context.block_id == 1) { throw LaterException{}; }
                    });
            } catch (const LaterException&) { nested_caught = true; }
            require(nested_caught && nested_visits == 9 && initial_environment(),
                    "nested prepared failure skipped work or failed to restore environment");
            int legacy_visits = 0;
            parallel_for_blocks(0, 9, 1, 4,
                [&](std::int64_t begin, std::int64_t end, std::size_t block) {
                    require(begin == 0 && end == 9 && block == 0,
                            "nested legacy callback was partitioned");
                    ++legacy_visits;
                    install_environment(FE_UPWARD, FE_INVALID);
                });
            require(legacy_visits == 1 && std::fegetround() == FE_UPWARD,
                    "nested legacy path changed its direct floating-point side effects");
        });
    const auto after = parallel_runtime_info();
    require(after.batches_submitted == before.batches_submitted + 1
            && after.tasks_submitted == before.tasks_submitted + 2,
            "nested prepared or legacy work published additional runners");
}

class EnvironmentHookScope {
public:
    explicit EnvironmentHookScope(scar_internal::parallel_testing::EnvironmentFailureHook hook) {
        set_environment_failure_hook(hook);
    }
    ~EnvironmentHookScope() { set_environment_failure_hook(nullptr); }
};

std::atomic<std::size_t> restore_block{0};
bool fail_apply_one(std::size_t block, bool restoring) noexcept {
    return !restoring && block == 1;
}
bool fail_restore(std::size_t block, bool restoring) noexcept {
    if (restoring) { restore_block.store(block); }
    return restoring;
}
bool fail_apply_and_restore(std::size_t block, bool restoring) noexcept {
    if (restoring) { restore_block.store(block); }
    return restoring || block == 0;
}

void prepared_environment_failures() {
    RuntimeScope runtime;
    EnvironmentScope environment;
    std::array<std::atomic<int>, 5> visits{};
    for (auto& value : visits) { value.store(0); }
    {
        EnvironmentHookScope hook(fail_apply_one);
        bool caught = false;
        try {
            execute_parallel_plan(make_parallel_execution_plan(0, 10, 5, 2),
                [&](std::int64_t, std::int64_t, const ParallelBlockContext& context) {
                    visits[context.block_id].fetch_add(1);
                });
        } catch (const std::runtime_error& failure) {
            caught = std::string(failure.what()).find("apply") != std::string::npos;
        }
        require(caught, "queued environment application failure was not reported");
    }
    for (std::size_t block = 0; block < visits.size(); ++block) {
        require(visits[block].load() == (block == 1 ? 0 : 1),
                "failed environment application invoked its callback or skipped siblings");
    }
    const auto single = make_parallel_execution_plan(0, 1, 1, 4);
    {
        EnvironmentHookScope hook(fail_restore);
        bool caught = false;
        try {
            execute_parallel_plan(single,
                [](std::int64_t, std::int64_t, const ParallelBlockContext&) {});
        } catch (const std::runtime_error& failure) {
            caught = std::string(failure.what()).find("restore") != std::string::npos;
        }
        require(caught && restore_block.load() == 1,
                "single restore failure lost its original exception or logical ID");
    }
    for (const bool primary_is_callback : {true, false}) {
        EnvironmentHookScope hook(primary_is_callback ? fail_restore : fail_apply_and_restore);
        bool caught = false;
        int callbacks = 0;
        try {
            execute_parallel_plan(single,
                [&](std::int64_t, std::int64_t, const ParallelBlockContext&) {
                    ++callbacks;
                    throw FirstException{};
                });
        } catch (const scar_internal::ParallelEnvironmentRestoreError& failure) {
            require(failure.primary_exception() && failure.restore_exception(),
                    "double environment failure discarded an exception pointer");
            bool primary_ok = false;
            try { failure.rethrow_primary(); }
            catch (const FirstException&) { primary_ok = primary_is_callback; }
            catch (const std::runtime_error& primary) {
                primary_ok = !primary_is_callback
                    && std::string(primary.what()).find("apply") != std::string::npos;
            }
            bool restore_ok = false;
            try { std::rethrow_exception(failure.restore_exception()); }
            catch (const std::runtime_error& restore) {
                restore_ok = std::string(restore.what()).find("restore") != std::string::npos;
            }
            caught = primary_ok && restore_ok;
        }
        require(caught && callbacks == (primary_is_callback ? 1 : 0),
                "double environment failure changed exception types or invoked unsafe work");
    }
    parallel_for_blocks(0, 2, 1, 2,
        [](std::int64_t, std::int64_t, std::size_t) {
            EnvironmentScope saved;
            install_environment(FE_DOWNWARD, FE_INEXACT);
            EnvironmentHookScope hook(fail_apply_one);
            int callbacks = 0;
            bool caught = false;
            try {
                execute_parallel_plan(make_parallel_execution_plan(0, 5, 5, 4),
                    [&](std::int64_t, std::int64_t, const ParallelBlockContext&) { ++callbacks; });
            } catch (const std::runtime_error&) { caught = true; }
            require(caught && callbacks == 4 && initial_environment(),
                    "nested apply failure did not restore entry state and continue siblings");
        });
}

template <typename Exception, typename Function>
void require_exception(Function&& function) {
    bool caught = false;
    try { function(); }
    catch (const Exception&) { caught = true; }
    require(caught, "invalid prepared input did not throw the expected exception");
}

void prepared_validation_and_legacy_serial() {
    RuntimeScope runtime;
    const auto minimum = std::numeric_limits<std::int64_t>::min();
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    require_exception<std::overflow_error>([&]() { (void)make_parallel_execution_plan(minimum, maximum, 4, 4); });
    require_exception<std::overflow_error>([&]() { (void)make_parallel_execution_plan_from_bounds({minimum, maximum}, 4); });
    require_exception<std::overflow_error>([&]() {
        (void)make_parallel_execution_plan(0, maximum,
            std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t), 4);
    });
    for (const int workers : {0, 257}) {
        require_exception<std::invalid_argument>([&]() { (void)make_parallel_execution_plan(0, 4, 4, workers); });
    }
    require_exception<std::invalid_argument>([]() { (void)make_parallel_execution_plan(2, 1, 1, 1); });
    require_exception<std::invalid_argument>([]() { (void)make_parallel_execution_plan(0, 1, 0, 1); });
    require_exception<std::invalid_argument>([]() { (void)make_parallel_execution_plan(0, 1, 2, 1); });
    require_exception<std::invalid_argument>([]() { (void)make_parallel_execution_plan(7, 7, 1, 1); });
    for (const auto& bounds : {std::vector<std::int64_t>{},
                               std::vector<std::int64_t>{1, 1},
                               std::vector<std::int64_t>{2, 1}}) {
        require_exception<std::invalid_argument>([&]() { (void)make_parallel_execution_plan_from_bounds(bounds, 2); });
    }
    int unexpected_callbacks = 0;
    require_exception<std::overflow_error>([&]() {
        parallel_for_blocks(minimum, maximum, 1, 1,
            [&](std::int64_t, std::int64_t, std::size_t) { ++unexpected_callbacks; });
    });
    require(unexpected_callbacks == 0 && !parallel_runtime_info().initialized,
            "invalid plan initialized workers or reached a callback");
    verify_plan_execution(make_parallel_execution_plan(minimum, minimum + 9, 4, 2));
    verify_plan_execution(make_parallel_execution_plan(maximum - 9, maximum, 4, 2));
    scar_internal::shutdown_parallel_runtime();
    parallel_for_blocks(0, 4, 1, 1,
        [](std::int64_t, std::int64_t, std::size_t) {
            verify_plan_execution(make_parallel_execution_plan(0, 10, 5, 2));
        });
    const auto info = parallel_runtime_info();
    require(info.worker_count == 2 && info.batches_submitted == 1 && info.tasks_submitted == 2,
            "legacy serial caller suppressed nested prepared parallel work");
}

}  // namespace

int run_parallel_runtime_tests() {
    Watchdog watchdog;
    try {
        failed_submission(Fault::worker_creation, 0, 0);
        failed_submission(Fault::worker_creation, 2, 2);
        failed_submission(Fault::batch_allocation, 0, 4);
        for (const std::size_t successful_attempts : {std::size_t{0},
                                                     std::size_t{1},
                                                     std::size_t{3}}) {
            failed_submission(Fault::queue_insertion, successful_attempts, 4);
        }
        rollback_preserves_foreign_queue();
        completion_notifies_once_per_queued_batch();
        ready_notifications_follow_selected_threshold();
        concurrent_growth_and_producers_complete();
        committed_failure_drains_and_keeps_first_exception();
        prepared_geometry_and_dispatch();
        prepared_copy_and_move();
        prepared_history_and_concurrent_calls();
        prepared_exceptions_continue_siblings();
        prepared_fenv_and_nested_paths();
        prepared_environment_failures();
        prepared_validation_and_legacy_serial();
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "parallel runtime regression: %s\n",
                     exception.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "parallel runtime regression: unknown exception\n");
        return 2;
    }
    return 0;
}
