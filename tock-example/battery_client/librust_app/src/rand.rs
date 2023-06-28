use crate::println;
use core::ffi::{c_int, c_size_t, c_uchar};
use core::num::NonZeroU32;
use getrandom::{register_custom_getrandom, Error};

extern "C" {
    fn rand_rs(buf: *mut c_int, len: c_size_t) -> c_int;
}

const NOT_ENOUGH_RANDOMNESS: u32 = Error::CUSTOM_START;
pub fn always_fail(buf: &mut [u8]) -> Result<(), Error> {
    let amount = unsafe { rand_rs(buf.as_mut_ptr() as *mut c_int, buf.len() as c_size_t) };
    if amount as usize != buf.len() {
        return Err(Error::from(NonZeroU32::new(NOT_ENOUGH_RANDOMNESS).unwrap()));
    }
    Ok(())
}

register_custom_getrandom!(always_fail);
