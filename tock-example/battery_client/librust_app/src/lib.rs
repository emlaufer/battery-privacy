#![no_std]
#![feature(error_in_core)]
#![feature(downcast_unchecked)]
#![feature(alloc_error_handler)]
#![feature(allocator_api)]
#![feature(c_size_t)]
#![feature(core_intrinsics)]

mod allocator;
mod client;
mod codec;
mod fft;
mod field;
mod flp;
mod fp;
mod polynomial;
mod prng;
mod rand;
mod utils;
mod vdaf;

#[macro_use]
extern crate alloc;
use alloc::alloc::Layout;
use alloc::boxed::Box;
use alloc::vec::Vec;
use field::{Field32, FieldElement};
use utils::*;

use core::panic::PanicInfo;

use client::{BatteryAggregator, BatteryClient, BatteryCollector, BatteryCoordinator};

#[no_mangle]
pub extern "C" fn run_the_client() -> bool {
    let mut ok = true;

    println!("entering rust main");

    let mut values = vec![];
    for i in 0..1 {
        values.push(i);
    }

    let max_power = vec![100];
    let max_energy = vec![100];
    let client1 = BatteryClient::new(values.len(), 100, max_power, max_energy);
    let mut verify_key = [0u8; 16];
    getrandom::getrandom(&mut verify_key).unwrap();

    let input_shares = client1.generate_shares(values);

    // do whatever you want with shares and proof
    println!("Share 1: {:x?}", input_shares[0]);
    println!("Share 2: {:x?}", input_shares[1]);

    ok
}

#[panic_handler]
fn panic(panic: &PanicInfo<'_>) -> ! {
    println!("Panic!: {}", panic);
    loop {}
}

#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    println!("allocation error!");
    loop {}
}
