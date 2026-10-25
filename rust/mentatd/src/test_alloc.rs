//! In a test build, a cap on the memory the test binary holds at once. A
//! test that runs away then aborts with an allocation failure, where it
//! would otherwise take the machine's memory. macOS does not enforce
//! `ulimit -v`, so the cap lives in the allocator.
//!
//! `MENTAT_TEST_MEM_CAP` sets the cap in MiB. The default is 2048. Every
//! test thread counts against the one cap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[global_allocator]
static ALLOC: Capped = Capped;

struct Capped;

static HELD: AtomicUsize = AtomicUsize::new(0);
/// The cap in bytes, or 0 before it is read.
static CAP: AtomicUsize = AtomicUsize::new(0);

const DEFAULT_MIB: usize = 2048;

/// Reads the cap with `getenv` and parses it by hand: anything that
/// allocates here would call back into this allocator.
fn cap() -> usize {
    let c = CAP.load(Relaxed);
    if c != 0 {
        return c;
    }
    // SAFETY: getenv takes a NUL-terminated name and returns null or a
    // NUL-terminated string. Nothing in the test binary sets the variable
    // while tests run.
    let raw = unsafe { libc::getenv(c"MENTAT_TEST_MEM_CAP".as_ptr()) };
    let mut mib = 0usize;
    if !raw.is_null() {
        // SAFETY: as above, `raw` is a NUL-terminated string.
        for &b in unsafe { std::ffi::CStr::from_ptr(raw) }.to_bytes() {
            if !b.is_ascii_digit() {
                mib = 0;
                break;
            }
            mib = mib.saturating_mul(10).saturating_add((b - b'0') as usize);
        }
    }
    let bytes = if mib == 0 { DEFAULT_MIB } else { mib }.saturating_mul(1 << 20);
    CAP.store(bytes, Relaxed);
    bytes
}

/// Counts `size` more bytes, or refuses them when that passes the cap.
fn take(size: usize) -> bool {
    let held = HELD.fetch_add(size, Relaxed).saturating_add(size);
    if held <= cap() {
        return true;
    }
    HELD.fetch_sub(size, Relaxed);
    let why = b"mentatd tests: an allocation would pass MENTAT_TEST_MEM_CAP\n";
    // SAFETY: writes a static buffer to stderr.
    unsafe { libc::write(2, why.as_ptr().cast(), why.len()) };
    false
}

// SAFETY: every call goes to System with the caller's layout. The counter
// only decides whether to make the call.
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !take(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(layout) };
        if p.is_null() {
            HELD.fetch_sub(layout.size(), Relaxed);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !take(layout.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        if p.is_null() {
            HELD.fetch_sub(layout.size(), Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        HELD.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old = layout.size();
        if new_size > old && !take(new_size - old) {
            // The old block stays valid, as realloc's contract requires.
            return std::ptr::null_mut();
        }
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if p.is_null() {
            if new_size > old {
                HELD.fetch_sub(new_size - old, Relaxed);
            }
        } else if new_size < old {
            HELD.fetch_sub(old - new_size, Relaxed);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::HELD;
    use std::sync::atomic::Ordering::Relaxed;

    /// The counter rises with a live block. Other test threads allocate at
    /// the same time, so this checks a lower bound only.
    #[test]
    fn a_live_block_counts_against_the_cap() {
        let block = vec![0u8; 64 << 20];
        assert!(HELD.load(Relaxed) >= block.len());
        drop(block);
    }
}
