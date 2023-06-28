use core::ffi::{c_char, c_int};
use core::fmt::Write;

extern "C" {
    pub fn print_rs(str: *const c_char, len: c_int);
}

pub struct CPrinter;

impl Write for CPrinter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        // pad with a null terminator
        unsafe {
            print_rs(
                s.as_bytes().as_ptr() as *const i8,
                s.as_bytes().len() as i32,
            )
        };
        Ok(())
    }
}

pub fn _print(args: core::fmt::Arguments) {
    let mut printer = CPrinter;
    printer.write_fmt(args).unwrap();
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => ($crate::_print(format_args!($($arg)*)));
}

#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($($arg:tt)*) => ($crate::print!("{}\n", format_args!($($arg)*)));
}
