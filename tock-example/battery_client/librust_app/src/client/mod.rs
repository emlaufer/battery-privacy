use crate::field::{Field128, Field32, Field64};
use crate::flp::gadgets::{BlindPolyEval, ParallelSum};
use crate::vdaf::prio3::{Prio3PrepareMessage, Prio3PrepareShare, Prio3PrepareState};
use crate::vdaf::{
    prg::PrgAes128,
    prio3::{Prio3, Prio3InputShare},
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
};
//use rand::prelude::*;
//
use alloc::vec::Vec;

//mod client;
//mod gadgets;
mod power_type;
//mod sum_vec;

use power_type::LinearPower;

static NUM_SHARES: usize = 2;
static NUM_BITS: usize = 14;

type BatteryField = Field32;
pub type BatteryValue = u32;
//type BatteryVdaf = Prio3<
//    LinearPower<BatteryField, ParallelSum<BatteryField, BlindPolyEval<BatteryField>>>,
//    PrgAes128,
//    16,
//>;
//#[cfg(not(feature = "ram"))]
type BatteryType =
    LinearPower<BatteryField, ParallelSum<BatteryField, BlindPolyEval<BatteryField>>>;
//#[cfg(feature = "ram")]
//type BatteryType = RAMPower<BatteryField, ParallelSum<BatteryField, BlindPolyEval<BatteryField>>>;
type BatteryVdaf = Prio3<BatteryType, PrgAes128, 16>;
type BatteryInputShare = Prio3InputShare<BatteryField, 16>;
type BatteryPrepareState = Prio3PrepareState<BatteryField, 16>;
type BatteryPrepareShare = Prio3PrepareShare<BatteryField, 16>;
type BatteryPrepareMessage = Prio3PrepareMessage<16>;
type BatteryOutputShare = OutputShare<BatteryField>;
type BatteryAggregateShare = AggregateShare<BatteryField>;

#[derive(Clone)]
pub struct BatteryClient {
    vdaf: BatteryVdaf,
    current_energy: u32,
}

impl BatteryClient {
    pub fn new(
        schedule_length: usize,
        current_energy: u16,
        client_maxs: Vec<usize>,
        client_energies: Vec<usize>,
    ) -> Self {
        BatteryClient {
            vdaf: Prio3::new(
                NUM_SHARES.try_into().unwrap(),
                BatteryType::new(
                    schedule_length,
                    NUM_BITS.into(),
                    client_maxs,
                    client_energies,
                )
                .unwrap(),
            )
            .unwrap(),
            current_energy: current_energy.into(),
        }
    }

    /// Generates data shares for each aggregator
    ///
    /// Returns a list of shares, one for each aggregator
    pub fn generate_shares(&self, values: Vec<BatteryValue>) -> Vec<BatteryInputShare> {
        // vector of vectors, each containing shares for each aggregator
        let mut aggr_input_shares: Vec<BatteryInputShare> = Vec::new();

        //for value in &self.values {
        let (_, value_shares) = self
            .vdaf
            .shard(&(self.current_energy.into(), values))
            .unwrap();

        value_shares
    }
}

#[derive(Debug)]
pub struct BatteryAggregator {
    // TODO: add state here so we can update
    id: usize,
    vdaf: BatteryVdaf,
    verify_key: [u8; 16],
    pub aggregate_shares: Option<BatteryAggregateShare>,
    num_measurements: usize,
    state: Option<BatteryPrepareState>,
}

impl BatteryAggregator {
    pub fn new(
        id: usize,
        schedule_length: usize,
        verify_key: &[u8; 16],
        client_maxs: Vec<usize>,
        client_energies: Vec<usize>,
    ) -> Self {
        BatteryAggregator {
            id,
            // TODO: clean up
            verify_key: verify_key.clone(),
            vdaf: Prio3::new(
                NUM_SHARES.try_into().unwrap(),
                BatteryType::new(
                    schedule_length,
                    NUM_BITS.into(),
                    client_maxs,
                    client_energies,
                )
                .unwrap(),
            )
            .unwrap(),
            aggregate_shares: None,
            num_measurements: 0,
            state: None,
            //prep_states: Vec::new(),
            //prep_shares: Vec::new(),
        }
    }

    // add shares from one client ....
    pub fn add(
        &mut self,
        input_shares: BatteryInputShare,
        client_num: usize,
        nonce: &[u8; 16],
    ) -> BatteryPrepareShare {
        self.vdaf.typ.client_num = client_num;

        // preparation stage
        let (state, msg) = self
            .vdaf
            .prepare_init(&self.verify_key, self.id, &(), nonce, &(), &input_shares)
            .unwrap();
        self.state = Some(state);
        msg
    }

    pub fn finish_round(&mut self, prep_message: BatteryPrepareMessage) -> BatteryOutputShare {
        let out_share = match self
            .vdaf
            .prepare_step(self.state.clone().unwrap(), prep_message.clone())
            .unwrap()
        {
            PrepareTransition::Finish(out_share) => out_share,
            _ => panic!("Unexpected prepare transition!"),
        };
        self.state = None;
        out_share
    }

    pub fn aggregate(&mut self, output_share: BatteryOutputShare) {
        if let Some(ref mut share) = self.aggregate_shares {
            share.accumulate(&output_share).unwrap();
        } else {
            let aggregate_share = self
                .vdaf
                .aggregate(&(), vec![output_share.clone()])
                .unwrap();
            self.aggregate_shares = Some(aggregate_share);
        }
        self.num_measurements += 1;
    }

    pub fn get_index(&self, index: usize) -> Option<BatteryAggregateShare> {
        self.aggregate_shares
            .as_ref()
            .map(|share| AggregateShare::from(vec![share.0[index]]))
    }
}

pub struct BatteryCoordinator {
    // a vector of prepare shares/states, indexed by aggregator
    shares: Vec<BatteryPrepareShare>,
    vdaf: BatteryVdaf,
}

impl BatteryCoordinator {
    pub fn new(
        schedule_length: usize,
        client_maxs: Vec<usize>,
        client_energies: Vec<usize>,
    ) -> BatteryCoordinator {
        BatteryCoordinator {
            shares: vec![],
            vdaf: Prio3::new(
                NUM_SHARES.try_into().unwrap(),
                BatteryType::new(
                    schedule_length,
                    NUM_BITS.into(),
                    client_maxs,
                    client_energies,
                )
                .unwrap(),
            )
            .unwrap(),
        }
    }

    // TODO: ensure ordering is correct...
    pub fn add(&mut self, message: BatteryPrepareShare) {
        self.shares.push(message);
        //self.states.push(state);
    }

    //Returns a list of lists of output shares, one for each aggregator
    pub fn perform_round(&mut self) -> BatteryPrepareMessage {
        // for each share across aggregators
        let prep_message = self.vdaf.prepare_preprocess(self.shares.clone()).unwrap();
        self.shares = vec![];
        prep_message
        //for (agg, state) in self.states.iter().enumerate() {
        //    let out_share = match self
        //        .vdaf
        //        .prepare_step(state.clone(), prep_message.clone())
        //        .unwrap()
        //    {
        //        PrepareTransition::Finish(out_share) => out_share,
        //        _ => panic!("Unexpected prepare transition!"),
        //    };
        //    result.push(out_share);
        //}

        //// reset state
        //self.states = vec![];

        //result
    }
}

pub struct BatteryCollector {
    // list of list of aggregate shares, one for each aggregator
    aggregate_shares: Vec<BatteryAggregateShare>,
    vdaf: BatteryVdaf,
    num_measurements: usize,
}

impl BatteryCollector {
    pub fn new(
        schedule_length: usize,
        client_maxs: Vec<usize>,
        client_energies: Vec<usize>,
    ) -> BatteryCollector {
        BatteryCollector {
            aggregate_shares: vec![],
            vdaf: Prio3::new(
                NUM_SHARES.try_into().unwrap(),
                BatteryType::new(
                    schedule_length,
                    NUM_BITS.into(),
                    client_maxs,
                    client_energies,
                )
                .unwrap(),
            )
            .unwrap(),
            num_measurements: 5,
        }
    }

    pub fn add(&mut self, share: BatteryAggregateShare) {
        self.aggregate_shares.push(share);
    }

    pub fn aggregate(&mut self) -> Vec<BatteryValue> {
        // aggregates shares across aggregators
        let result = self
            .vdaf
            .unshard(&(), self.aggregate_shares.drain(..), self.num_measurements)
            .unwrap();
        result
    }

    pub fn aggregate_slice(&mut self, shares: Vec<BatteryAggregateShare>) -> u32 {
        // manually unshard...
        return 0;
    }
}
