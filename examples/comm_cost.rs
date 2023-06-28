use bprivacy::{
    BatteryAggregator, BatteryClient, BatteryCollector, BatteryCoordinator, BatteryValue,
};
use core::mem::size_of;
use prio::codec::Encode;
use rand::prelude::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_clients = str::parse(&args[1]).unwrap();
    let num_shares = str::parse(&args[2]).unwrap();

    let mut rng = thread_rng();
    let clients = vec![BatteryClient::new(num_shares, 300); num_clients];
    let mut aggregators = vec![];
    let verify_key = rng.gen();
    for i in 0..2 {
        aggregators.push(BatteryAggregator::new(i, num_shares, &verify_key));
    }
    let mut coordinator = BatteryCoordinator::new(num_shares);
    let mut collector = BatteryCollector::new(num_shares);

    print!("{},", num_shares);
    for (i, client) in clients[0..1].iter().enumerate() {
        let input_shares = client.generate_shares(vec![i as BatteryValue; num_shares]);
        let mut bytes = Vec::new();
        input_shares[0].encode(&mut bytes);
        // encode input shares
        println!("Client Data Size: {} bytes", bytes.len());

        let nonce = rng.gen();
        for (i, (agg, share)) in aggregators
            .iter_mut()
            .zip(input_shares.into_iter())
            .enumerate()
        {
            let prep_message = agg.add(share, &nonce);
            let mut bytes = Vec::new();
            // encode input shares
            prep_message.0.encode(&mut bytes);
            prep_message.1.encode(&mut bytes);
            println!("Prio Server Message Size: {} bytes", bytes.len());

            if i == 0 {
                //print!(
                //    "{},",
                //    prep_message.len()
                //        * (prep_message[0].1.get_encoded().len()
                //            + prep_message[0].0.get_encoded().len())
                //);
            }
            coordinator.add(prep_message);
        }
        let out_shares = coordinator.perform_round();
        println!("Output Share Size: {} bytes", out_shares.len() * 32);
        for (agg, out_share) in aggregators.iter_mut().zip(out_shares.into_iter()) {
            agg.aggregate(out_share);
        }
    }

    for agg in aggregators {
        collector.add(agg.aggregate_shares.unwrap());
    }
    //println!("Got: {:?}", collector.aggregate());
}
