use bprivacy::{
    server::BatteryServer, BatteryAggregator, BatteryClient, BatteryCollector, BatteryCoordinator,
    BatteryValue,
};
use prio::codec::Encode;
use rand::distributions::Standard;
use rand::prelude::*;
use std::env;
use std::time::{Duration, Instant};

const NUM_CHANGES: usize = 50;

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_shares = str::parse(&args[1]).unwrap();
    let max: usize = str::parse(&args[2]).unwrap();
    //let do_opt = &args.get(4).unwrap_or(&String::new()).as_str() == &"opt";
    let mut rng = thread_rng();

    // tracker for total comm cost
    let mut client_to_server_opt = 0;
    let mut server_to_server_opt = 0;
    let mut server_to_client_opt = 0;
    let mut client_to_server_bytes = 0;
    let mut server_to_server_bytes = 0;
    let mut server_to_client_bytes = 0;

    let mut client_time = Duration::new(0, 0);
    let mut client_opt_time = Duration::new(0, 0);
    let mut server_opt_time = Duration::new(0, 0);
    let mut verif_time = Duration::new(0, 0);
    let mut collect_time = Duration::new(0, 0);

    let max = max;
    let energy = max;
    let mut changes: Vec<usize> = rng
        .clone()
        .sample_iter(Standard)
        .take(NUM_CHANGES - 1)
        .map(|v: usize| v % std::cmp::max((num_shares / 2), 1))
        .collect();
    changes.sort();
    let mut schedule = Vec::with_capacity(num_shares);
    let mut c = 0;
    for i in 0..num_shares {
        if c < NUM_CHANGES - 1 && i % 2 == 0 && i / 2 >= changes[c] {
            schedule.push(1);
            c += 1;
        } else {
            schedule.push(0);
        }
    }
    //println!("SCHEDULE: {:?}", schedule);

    // prove schedules over aggregate
    let client_values: Vec<u32> = schedule;
    let num_values = client_values.len();
    //println!("NUM VALUES: {:?}", num_values);
    let client = BatteryClient::new(num_values, 1000, vec![max.clone()], vec![max.clone()]);

    let share_callback = |val: &[u8]| {};

    // DON"T Do all clients for benchmarks... no need to
    let start = Instant::now();
    let input_shares = client.generate_shares_callback(client_values, &share_callback);
    //let input_shares = client.generate_shares(client_values);
    let end = start.elapsed();
}
