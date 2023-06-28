use crate::beaver::{generate_triples, BeaverServer};
use rand::Fill;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

// TODO: alright, all that is left is to:
// 1. write a type that proves over the schedule intervals
// 2. reveal aggregate and send to client
// 3.

pub struct BatteryServer {
    // accumulator for aggregate schedule changes
    pub changes_acc: Vec<bool>,
    beaver: Rc<RefCell<BeaverServer>>,
    num_clients: usize,
    len: usize,
}

impl BatteryServer {
    pub fn new(num_clients: usize, len: usize, initial_acc: Vec<bool>) -> BatteryServer {
        let beaver = Rc::new(RefCell::new(BeaverServer::new(len)));
        BatteryServer {
            changes_acc: initial_acc,
            beaver,
            num_clients,
            len,
        }
    }

    // refresh acc with shares of true
    pub fn reset(&mut self, initial_acc: Vec<bool>) {
        self.changes_acc = initial_acc;
    }

    // collect boolean shares of schedule changes from clients
    pub fn collect_schedule_changes(
        &mut self,
        schedule_shares: &[bool],
        a: &[bool],
        b: &[bool],
        c: &[bool],
    ) {
        self.beaver.borrow_mut().xs = schedule_shares.to_vec();
        self.beaver.borrow_mut().ys = self.changes_acc.to_vec();
        self.beaver.borrow_mut().a = a.to_vec();
        self.beaver.borrow_mut().b = b.to_vec();
        self.beaver.borrow_mut().c = c.to_vec();
        self.beaver.borrow_mut().done = false;
    }

    // accumulate with server
    pub fn accumulate(&mut self, other: &mut BatteryServer) {
        self.beaver.borrow_mut().other = Some(other.beaver.clone());
        self.beaver.borrow_mut().multiply();

        self.changes_acc = self.beaver.borrow().a.clone();
        other.changes_acc = other.beaver.borrow().a.clone();
    }

    pub fn reveal(&mut self, other: &mut BatteryServer) -> Vec<bool> {
        let mut v = vec![];
        // TODO: could open to other as well
        //       OR open to each client individually
        for i in 0..self.len {
            v.push(self.changes_acc[i] ^ other.changes_acc[i]);
        }
        v
    }
}

#[cfg(test)]
mod test {}
