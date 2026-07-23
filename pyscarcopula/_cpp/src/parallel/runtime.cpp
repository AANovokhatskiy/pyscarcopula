#include "scar/detail/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
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

std::uint64_t current_pid() noexcept {
#ifdef _WIN32
    return static_cast<std::uint64_t>(::_getpid());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

struct Batch {
    explicit Batch(std::size_t count) : remaining(count) {}

    std::mutex mutex;
    std::condition_variable completed;
    std::size_t remaining;
    std::exception_ptr exception;
};

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

    void run(std::vector<std::function<void()>> tasks, std::size_t workers) {
        ensure_workers(workers);
        auto batch = std::make_shared<Batch>(tasks.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& task : tasks) {
                queue_.emplace_back([batch, task = std::move(task)]() mutable {
                    try {
                        task();
                    } catch (...) {
                        std::lock_guard<std::mutex> batch_lock(batch->mutex);
                        if (!batch->exception) {
                            batch->exception = std::current_exception();
                        }
                    }
                    {
                        std::lock_guard<std::mutex> batch_lock(batch->mutex);
                        --batch->remaining;
                    }
                    batch->completed.notify_one();
                });
            }
            ++batches_submitted_;
            tasks_submitted_ += static_cast<std::uint64_t>(tasks.size());
            peak_queued_tasks_ = std::max(
                peak_queued_tasks_, queue_.size());
        }
        ready_.notify_all();

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

}  // namespace

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
    if (n_threads < 1) {
        throw std::invalid_argument("n_threads must be >= 1");
    }
    const std::int64_t length = end - begin;
    if (length == 0) {
        return;
    }
    if (n_threads == 1 || length <= min_grain || in_runtime_worker) {
        function(begin, end, 0);
        return;
    }

    const auto grain_blocks = static_cast<std::size_t>(
        length / min_grain + (length % min_grain == 0 ? 0 : 1));
    const std::size_t block_count = std::min(
        static_cast<std::size_t>(n_threads), grain_blocks);
    if (block_count <= 1) {
        function(begin, end, 0);
        return;
    }

    const std::int64_t quotient = length / static_cast<std::int64_t>(block_count);
    const std::int64_t remainder = length % static_cast<std::int64_t>(block_count);
    std::vector<std::function<void()>> tasks;
    tasks.reserve(block_count);
    std::int64_t block_begin = begin;
    for (std::size_t block = 0; block < block_count; ++block) {
        const std::int64_t block_size = quotient
            + (block < static_cast<std::size_t>(remainder) ? 1 : 0);
        const std::int64_t block_end = block_begin + block_size;
        tasks.emplace_back([=, &function]() {
            function(block_begin, block_end, block);
        });
        block_begin = block_end;
    }
    process_runtime().run(std::move(tasks), block_count);
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
