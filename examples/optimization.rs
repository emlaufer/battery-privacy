use bprivacy::{
    server::BatteryServer, BatteryAggregator, BatteryClient, BatteryCollector, BatteryCoordinator,
    BatteryValue,
};
use prio::codec::Encode;
use rand::distributions::Standard;
use rand::prelude::*;
use std::env;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_clients = str::parse(&args[1]).unwrap();
    let num_shares = str::parse(&args[2]).unwrap();
    let max = str::parse(&args[3]).unwrap();
    let do_opt = &args.get(4).unwrap_or(&String::new()).as_str() == &"opt";
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

    let mut clients = vec![];
    let client_maxs = vec![max; num_clients];
    let client_energies = vec![max; num_clients];
    for i in 0..num_clients {
        // choose ~50 random indices to change at
        // we will do with replacement, doesn't really matter
        let mut changes: Vec<usize> = rng
            .clone()
            .sample_iter(Standard)
            .take(49)
            .map(|v: usize| v % num_shares)
            .collect();
        changes.sort();
        let mut schedule = Vec::with_capacity(num_shares);
        let mut c = 0;
        for i in 0..num_shares {
            if c < 49 && i >= changes[c] {
                schedule.push(1);
                c += 1;
            } else {
                schedule.push(0);
            }
        }
        println!("SCHEDULE: {:?}", schedule);
        clients.push(schedule);
    }

    // convert schedule to changes
    // changes_shares[client] = (share1, share2)
    let mut changes_shares = Vec::with_capacity(num_clients);
    for i in 0..num_clients {
        let start = Instant::now();
        let changes: Vec<bool> = (0..num_shares)
            .map(|j| j != 0 && clients[i][j] == clients[i][j - 1])
            .collect();
        let share1: Vec<bool> = rng.clone().sample_iter(Standard).take(num_shares).collect();
        if i == 0 {
            client_to_server_opt += changes.len() / 8;
        }
        let share2: Vec<bool> = (0..num_shares).map(|j| share1[j] ^ changes[j]).collect();
        changes_shares.push((share1, share2));
        if i == 0 {
            client_opt_time = start.elapsed();
        }
    }

    let aggregate_changes = if do_opt {
        let mut battery_operator =
            BatteryServer::new(num_clients, num_shares, vec![false; num_shares]);
        let mut battery_other = BatteryServer::new(num_clients, num_shares, vec![true; num_shares]);
        let mut a1 = vec![false; num_shares];
        let mut b1 = vec![false; num_shares];
        let mut c1 = vec![false; num_shares];
        let mut a2 = vec![false; num_shares];
        let mut b2 = vec![false; num_shares];
        let mut c2 = vec![false; num_shares];

        for i in 0..num_clients {
            let start = Instant::now();
            bprivacy::beaver::generate_triples(
                &mut a1, &mut a2, &mut b1, &mut b2, &mut c1, &mut c2,
            );

            battery_operator.collect_schedule_changes(&changes_shares[i].0, &a1, &b1, &c1);
            battery_other.collect_schedule_changes(&changes_shares[i].1, &a2, &b2, &c2);
            battery_operator.accumulate(&mut battery_other);
            let end = start.elapsed();
            if i == 0 {
                server_opt_time = end;
            }
        }
        server_to_server_opt += 3 * num_shares / 8;
        server_to_client_opt += num_shares / 8;
        let aggregate_changes = battery_operator.reveal(&mut battery_other);
        aggregate_changes
    } else {
        vec![false; num_shares]
    };

    // prove schedules over aggregate
    let client_values: Vec<Vec<u32>> = clients
        .iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .filter(|(i, v)| !aggregate_changes[*i])
                .map(|(i, v)| *v)
                .collect()
        })
        .collect();
    let num_values = client_values[0].len();
    println!("NUM VALUES: {:?}", num_values);

    // randomize max power for fun
    let mut aggregators = vec![];
    let verify_key = rng.gen();
    for i in 0..2 {
        aggregators.push(BatteryAggregator::new(
            i,
            num_values,
            &verify_key,
            client_maxs.clone(),
            client_energies.clone(),
        ));
    }
    let mut coordinator =
        BatteryCoordinator::new(num_values, client_maxs.clone(), client_energies.clone());
    let mut collector =
        BatteryCollector::new(num_values, client_maxs.clone(), client_energies.clone());
    // TODO: remove extra param 1024 max power, not used anymore
    let clients = vec![
        BatteryClient::new(
            num_values,
            1000,
            client_maxs.clone(),
            client_energies.clone()
        );
        num_clients
    ];
    // DON"T Do all clients for benchmarks... no need to
    for i in 0..1 {
        let start = Instant::now();
        let input_shares = clients[i].generate_shares(client_values[i].clone());
        let end = start.elapsed();

        if i == 0 {
            client_time += end;
            client_to_server_bytes += input_shares
                .iter()
                .map(|i| i.get_encoded().len())
                .sum::<usize>();
        }

        let nonce = rng.gen();
        for (j, (agg, share)) in aggregators
            .iter_mut()
            .zip(input_shares.into_iter())
            .enumerate()
        {
            let start = Instant::now();
            let prep_message = agg.add(share, i, &nonce);
            coordinator.add(prep_message.clone());
            let end = start.elapsed();

            if i == 0 && j == 0 {
                server_to_server_bytes += prep_message.get_encoded().len();
                verif_time = end;
            }
        }
        let prep_message = coordinator.perform_round();

        if i == 0 {
            server_to_server_bytes += prep_message.get_encoded().len();
        }

        let mut out_shares = vec![];
        for agg in aggregators.iter_mut() {
            out_shares.push(agg.finish_round(prep_message.clone()));
        }
        for (agg, out_share) in aggregators.iter_mut().zip(out_shares.into_iter()) {
            agg.aggregate(out_share);
        }
    }

    println!("Client -> Server: {}", client_to_server_bytes);
    println!("Server -> Server: {}", server_to_server_bytes);
    println!("Server -> Client: {}", server_to_client_bytes);
    println!("Client -> Server opt: {}", client_to_server_opt);
    println!("Server -> Server opt: {}", server_to_server_opt);
    println!("Server -> Client opt: {}", server_to_client_opt);

    for agg in aggregators {
        collector.add(agg.aggregate_shares.unwrap());
    }
    let mut expected = vec![0; client_values[0].len()];
    for vals in client_values {
        for (i, v) in vals.iter().enumerate() {
            expected[i] += v;
        }
    }
    let start = Instant::now();
    let aggregate = collector.aggregate();
    collect_time = start.elapsed();

    println!("Client time: {:?}", client_time.as_nanos());
    println!("Verify time: {:?}", verif_time.as_nanos());
    println!("Aggregate time: {:?}", collect_time.as_nanos());
    println!("Client Opt time: {:?}", client_opt_time.as_nanos());
    println!("Server Opt time: {:?}", server_opt_time.as_nanos());

    //assert_eq!(aggregate, expected);
}
