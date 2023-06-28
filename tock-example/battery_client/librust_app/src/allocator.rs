use alloc::alloc::{GlobalAlloc, Layout};
use core::ffi::{c_size_t, c_void};
use core::ptr::null_mut;

extern "C" {
    fn malloc_rs(n: c_size_t) -> *mut c_void;
    fn free_rs(p: *mut c_void);
}

pub struct Dummy;

unsafe impl GlobalAlloc for Dummy {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let bytes = layout.size();
        unsafe { malloc_rs(bytes as c_size_t) as *mut u8 }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe {
            free_rs(ptr as *mut c_void);
        }
    }
}

#[global_allocator]
pub static ALLOCATOR: Dummy = Dummy;
