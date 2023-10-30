use crate::{
    BatteryAggregateShare, BatteryAggregator, BatteryClient, BatteryCollector, BatteryCoordinator,
    BatteryInputShare, BatteryOutputShare, BatteryPrepareMessage, BatteryPrepareShare,
};
use prio::codec::ParameterizedDecode;
use prio::vdaf::{
    prg::PrgAes128,
    prio3::{Prio3, Prio3InputShare},
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
};

use rand::Rng;
use std::mem;

use prio::codec::Encode;

#[no_mangle]
pub extern "C" fn client_submit(
    data: *mut u32,
    data_len: u32,
    current_energy: u32,
    max_v: u32,
    max_e: u32,
    user: *mut libc::c_void,
    write_callback: extern "C" fn(*const u8, len: u32, num: u32, user: *mut libc::c_void),
) {
    let data_slice = unsafe { std::slice::from_raw_parts(data, data_len as usize) };

    let client = BatteryClient::new(
        data_len as usize,
        current_energy,
        vec![max_v.clone() as usize],
        vec![max_e.clone() as usize],
    );
    let input_shares = client.generate_shares(data_slice.to_vec());
    for (i, share) in input_shares.iter().enumerate() {
        let mut bytes = vec![];
        share.encode(&mut bytes);
        write_callback(bytes.as_ptr(), bytes.len() as u32, i as u32, user);
    }
}

#[no_mangle]
pub extern "C" fn new_aggregator(
    id: u32,
    schedule_length: u32,
    verify_key_ptr: *const u8,
    n_clients: u32,
    max_v: u32,
    max_e: u32,
) -> *mut libc::c_void {
    let verify_key = unsafe { std::slice::from_raw_parts(verify_key_ptr, 16) };

    let client_maxs = vec![max_v as usize; n_clients as usize];
    let client_max_es = vec![max_e as usize; n_clients as usize];

    let mut agg = BatteryAggregator::new(
        id as usize,
        schedule_length as usize,
        verify_key.try_into().unwrap(),
        client_maxs,
        client_max_es,
    );
    let agg_box = Box::new(agg);

    return Box::into_raw(agg_box) as *mut libc::c_void;
}

fn aggregate_inner(
    agg: &mut BatteryAggregator,
    input_share: *const u8,
    input_share_len: u32,
    client_num: u32,
) -> BatteryPrepareShare {
    let mut rng = rand::thread_rng();
    let nonce = rng.gen();

    let input_bytes = unsafe { std::slice::from_raw_parts(input_share, input_share_len as usize) };
    let input_share =
        BatteryInputShare::get_decoded_with_param(&(&agg.vdaf, agg.id), input_bytes).unwrap();
    return agg.add(input_share, client_num as usize, &nonce);

    // TODO: return this in bytes to send to collector...
}

#[no_mangle]
pub extern "C" fn aggregate(
    agg_ptr: *mut libc::c_void,
    input_share: *const u8,
    input_share_len: u32,
    client_num: u32,
    user: *mut libc::c_void,
    write_callback: extern "C" fn(*const u8, len: u32, num: u32, user: *mut libc::c_void),
) {
    let agg = unsafe { &mut *(agg_ptr as *mut BatteryAggregator) };
    let mut bytes = vec![];
    let prep_share = aggregate_inner(agg, input_share, input_share_len, client_num);
    prep_share.encode(&mut bytes);

    write_callback(bytes.as_ptr(), bytes.len() as u32, client_num, user);
}

#[no_mangle]
pub extern "C" fn new_collector(
    id: u32,
    schedule_length: u32,
    verify_key_ptr: *const u8,
    n_clients: u32,
    max_v: u32,
    max_e: u32,
) -> *mut libc::c_void {
    let verify_key = unsafe { std::slice::from_raw_parts(verify_key_ptr, 16) };

    let client_maxs = vec![max_v as usize; n_clients as usize];
    let client_max_es = vec![max_e as usize; n_clients as usize];

    let mut agg = BatteryAggregator::new(
        id as usize,
        schedule_length as usize,
        verify_key.try_into().unwrap(),
        client_maxs.clone(),
        client_max_es.clone(),
    );
    let mut coord = BatteryCoordinator::new(
        schedule_length as usize,
        client_maxs.clone(),
        client_max_es.clone(),
    );
    let mut coll = BatteryCollector::new(
        schedule_length as usize,
        client_maxs.clone(),
        client_max_es.clone(),
    );
    let agg_box = Box::new((agg, coord, coll));

    return Box::into_raw(agg_box) as *mut libc::c_void;
}

#[no_mangle]
pub extern "C" fn aggregate_coll(
    agg_ptr: *mut libc::c_void,
    input_share: *const u8,
    input_share_len: u32,
    client_num: u32,
) {
    let agg = unsafe {
        &mut *(agg_ptr as *mut (BatteryAggregator, BatteryCoordinator, BatteryCollector))
    };
    let prep_share = aggregate_inner(&mut agg.0, input_share, input_share_len, client_num);
    agg.1.add(prep_share.clone());
}

#[no_mangle]
pub extern "C" fn add_prep(agg_ptr: *mut libc::c_void, prep_share: *const u8, prep_share_len: u32) {
    let agg = unsafe {
        &mut *(agg_ptr as *mut (BatteryAggregator, BatteryCoordinator, BatteryCollector))
    };
    let input_bytes = unsafe { std::slice::from_raw_parts(prep_share, prep_share_len as usize) };
    let prep_share =
        BatteryPrepareShare::get_decoded_with_param(&agg.0.state.as_ref().unwrap(), input_bytes)
            .unwrap();
    agg.1.add(prep_share);
}

#[no_mangle]
pub extern "C" fn perform_round(
    agg_ptr: *mut libc::c_void,
    user: *mut libc::c_void,
    write_callback: extern "C" fn(*const u8, len: u32, user: *mut libc::c_void),
) {
    let agg = unsafe {
        &mut *(agg_ptr as *mut (BatteryAggregator, BatteryCoordinator, BatteryCollector))
    };
    let mut bytes = vec![];
    let prep_message = agg.1.perform_round();
    let out_message = agg.0.finish_round(prep_message.clone());

    // send prep_message
    prep_message.encode(&mut bytes);
    write_callback(bytes.as_ptr(), bytes.len() as u32, user);

    agg.0.aggregate(out_message.clone());

    // send out_share...
    let bytes: Vec<u8> = out_message.into();
    write_callback(bytes.as_ptr(), bytes.len() as u32, user);
}

#[no_mangle]
pub extern "C" fn agg_finish_round(
    agg: *mut libc::c_void,
    input_share: *const u8,
    input_share_len: u32,
    user: *mut libc::c_void,
    write_callback: extern "C" fn(*const u8, len: u32, user: *mut libc::c_void),
) {
    let agg = unsafe { &mut *(agg as *mut BatteryAggregator) };
    let input_bytes = unsafe { std::slice::from_raw_parts(input_share, input_share_len as usize) };
    let prep_message =
        BatteryPrepareMessage::get_decoded_with_param(&agg.state.clone().unwrap(), input_bytes)
            .unwrap();
    let out_message = agg.finish_round(prep_message);
    agg.aggregate(out_message.clone());

    let bytes: Vec<u8> = out_message.into();
    write_callback(bytes.as_ptr(), bytes.len() as u32, user);
}

fn agg_finish_inner(agg: &mut BatteryAggregator, input_share: *const u8, input_share_len: u32) {
    let input_bytes = unsafe { std::slice::from_raw_parts(input_share, input_share_len as usize) };
    let out_share: BatteryOutputShare = input_bytes.try_into().unwrap();
    agg.aggregate(out_share);
}

#[no_mangle]
pub extern "C" fn aggregate_finish_coll(
    agg_ptr: *mut libc::c_void,
    input_share: *const u8,
    input_share_len: u32,
) {
    let agg = unsafe {
        &mut *(agg_ptr as *mut (BatteryAggregator, BatteryCoordinator, BatteryCollector))
    };
    agg_finish_inner(&mut agg.0, input_share, input_share_len);
}

#[no_mangle]
pub extern "C" fn aggregate_finish(
    agg_ptr: *mut libc::c_void,
    input_share: *const u8,
    input_share_len: u32,
) {
    let agg = unsafe { &mut *(agg_ptr as *mut BatteryAggregator) };
    agg_finish_inner(agg, input_share, input_share_len);
}

#[no_mangle]
pub extern "C" fn send_agg_share(
    agg_ptr: *mut libc::c_void,
    user: *mut libc::c_void,
    write_callback: extern "C" fn(*const u8, len: u32, user: *mut libc::c_void),
) {
    let agg = unsafe { &mut *(agg_ptr as *mut BatteryAggregator) };
    let bytes: Vec<u8> = agg.aggregate_shares.clone().unwrap().into();
    write_callback(bytes.as_ptr(), bytes.len() as u32, user);
}

#[no_mangle]
pub extern "C" fn collect(
    agg_ptr: *mut libc::c_void,
    input_share: *const u8,
    input_share_len: u32,
) {
    let agg = unsafe {
        &mut *(agg_ptr as *mut (BatteryAggregator, BatteryCoordinator, BatteryCollector))
    };
    let input_bytes = unsafe { std::slice::from_raw_parts(input_share, input_share_len as usize) };
    let agg_share: BatteryAggregateShare = input_bytes.try_into().unwrap();
    agg.2.add(agg_share);
    let aggregate = agg.2.aggregate();
    println!("GOT: {:?}", aggregate);
}
