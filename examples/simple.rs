use bprivacy::{
    BatteryAggregator, BatteryClient, BatteryCollector, BatteryCoordinator, BatteryValue,
};
use rand::prelude::*;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_clients = str::parse(&args[1]).unwrap();
    let num_shares = str::parse(&args[2]).unwrap();

    let mut rng = thread_rng();
    let client_maxs = vec![thread_rng().gen::<usize>() % 256; num_clients];
    let clients = vec![BatteryClient::new(num_shares, 2000, client_maxs.clone()); num_clients];
    let mut aggregators = vec![];
    let verify_key = rng.gen();
    for i in 0..2 {
        aggregators.push(BatteryAggregator::new(
            i,
            num_shares,
            &verify_key,
            client_maxs.clone(),
        ));
    }
    let mut coordinator = BatteryCoordinator::new(num_shares, client_maxs.clone());
    let mut collector = BatteryCollector::new(num_shares, client_maxs.clone());

    for (i, client) in clients.iter().enumerate() {
        println!("CLIENT: {:?}", i);
        let start = Instant::now();
        let mut values = vec![];
        for v in 0..num_shares {
            values.push(v as BatteryValue % (client_maxs[i] as u32));
        }
        let input_shares = client.generate_shares(values);
        let end = Instant::now();
        let start = Instant::now();
        let nonce = rng.gen();
        for (j, (agg, share)) in aggregators
            .iter_mut()
            .zip(input_shares.into_iter())
            .enumerate()
        {
            let prep_message = agg.add(share, i, &nonce);

            coordinator.add(prep_message);
        }
        let prep_message = coordinator.perform_round();

        let mut out_shares = vec![];
        for agg in aggregators.iter_mut() {
            out_shares.push(agg.finish_round(prep_message.clone()));
        }
        for (agg, out_share) in aggregators.iter_mut().zip(out_shares.into_iter()) {
            agg.aggregate(out_share);
        }
    }
    let mut test_vec = Vec::new();
    //for agg in &aggregators {
    //    //test_vec.push(agg.get_index(13).unwrap());
    //}
    collector.aggregate_slice(test_vec);

    let start = Instant::now();
    for agg in aggregators {
        collector.add(agg.aggregate_shares.unwrap());
    }
    let _res = collector.aggregate();
    let end = Instant::now();
    println!("GOT FINAL TIME: {:?}", end - start);
    println!("res: {:?}", _res);
}
