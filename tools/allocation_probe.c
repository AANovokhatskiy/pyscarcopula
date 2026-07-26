#define _GNU_SOURCE

#include <stdatomic.h>
#include <stddef.h>

/*
 * CI-only glibc allocation probe. It is loaded with LD_PRELOAD before Python
 * starts and enabled only around one native call. No production wheel links
 * to this helper.
 */
extern void* __libc_malloc(size_t size);
extern void* __libc_calloc(size_t count, size_t size);
extern void* __libc_realloc(void* pointer, size_t size);
extern void __libc_free(void* pointer);

static _Atomic int enabled = 0;
static _Atomic unsigned long long allocation_calls = 0;
static _Atomic unsigned long long allocation_bytes = 0;
static _Atomic unsigned long long free_calls = 0;

void* malloc(size_t size) {
    void* pointer = __libc_malloc(size);
    if (atomic_load_explicit(&enabled, memory_order_relaxed)) {
        atomic_fetch_add_explicit(
            &allocation_calls, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &allocation_bytes, (unsigned long long)size,
            memory_order_relaxed);
    }
    return pointer;
}

void* calloc(size_t count, size_t size) {
    void* pointer = __libc_calloc(count, size);
    if (atomic_load_explicit(&enabled, memory_order_relaxed)) {
        atomic_fetch_add_explicit(
            &allocation_calls, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &allocation_bytes, (unsigned long long)(count * size),
            memory_order_relaxed);
    }
    return pointer;
}

void* realloc(void* pointer, size_t size) {
    void* result = __libc_realloc(pointer, size);
    if (atomic_load_explicit(&enabled, memory_order_relaxed)) {
        atomic_fetch_add_explicit(
            &allocation_calls, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &allocation_bytes, (unsigned long long)size,
            memory_order_relaxed);
    }
    return result;
}

void free(void* pointer) {
    if (pointer != NULL
        && atomic_load_explicit(&enabled, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&free_calls, 1, memory_order_relaxed);
    }
    __libc_free(pointer);
}

void pysca_allocation_probe_reset(void) {
    atomic_store_explicit(&allocation_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&allocation_bytes, 0, memory_order_relaxed);
    atomic_store_explicit(&free_calls, 0, memory_order_relaxed);
}

void pysca_allocation_probe_enable(int value) {
    atomic_store_explicit(&enabled, value != 0, memory_order_seq_cst);
}

unsigned long long pysca_allocation_probe_calls(void) {
    return atomic_load_explicit(&allocation_calls, memory_order_relaxed);
}

unsigned long long pysca_allocation_probe_bytes(void) {
    return atomic_load_explicit(&allocation_bytes, memory_order_relaxed);
}

unsigned long long pysca_allocation_probe_frees(void) {
    return atomic_load_explicit(&free_calls, memory_order_relaxed);
}
