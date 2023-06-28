use rand::Fill;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

// Simple, but gross wrapper around a beaver triple multiply algorithm
pub struct BeaverServer {
    pub other: Option<Rc<RefCell<BeaverServer>>>,

    pub len: usize,
    pub xs: Vec<bool>,
    pub ys: Vec<bool>,
    pub a: Vec<bool>, // also stores result when done
    pub b: Vec<bool>,
    pub c: Vec<bool>,
    pub done: bool,
}

impl BeaverServer {
    pub fn new(len: usize) -> BeaverServer {
        BeaverServer {
            other: None,
            len,
            xs: vec![],
            ys: vec![],
            a: vec![],
            b: vec![],
            c: vec![],
            done: false,
        }
    }

    pub fn multiply(&mut self) {
        let mut As = Vec::with_capacity(self.len);
        let mut Bs = Vec::with_capacity(self.len);
        for i in 0..self.len {
            As.push(self.xs[i] ^ self.a[i]);
            Bs.push(self.ys[i] ^ self.b[i]);
        }

        self.other
            .as_ref()
            .unwrap()
            .borrow_mut()
            .reveal(&mut As, &mut Bs);

        self.finish(&As, &Bs)
    }

    fn finish(&mut self, As: &[bool], Bs: &[bool]) {
        // now As and Bs are revealed
        // compute result
        for i in 0..self.len {
            self.a[i] = (As[i] & self.ys[i]) ^ (Bs[i] & self.a[i]) ^ self.c[i];
        }
        self.done = true;
    }

    // called by other server...passes in their shares
    // Expected: modify A to reveal our share as well
    fn reveal(&mut self, As: &mut [bool], Bs: &mut [bool]) {
        for i in 0..self.len {
            As[i] ^= self.xs[i] ^ self.a[i];
            Bs[i] ^= self.ys[i] ^ self.b[i];
        }

        self.finish(As, Bs);
    }
}

pub fn generate_triples(
    a1: &mut [bool],
    a2: &mut [bool],
    b1: &mut [bool],
    b2: &mut [bool],
    c1: &mut [bool],
    c2: &mut [bool],
) {
    let mut rng = rand::thread_rng();

    rng.fill(a1);
    rng.fill(b1);
    rng.fill(c1);
    rng.fill(a2);
    rng.fill(b2);

    for i in 0..a1.len() {
        c2[i] = ((a1[i] ^ a2[i]) & (b1[i] ^ b2[i])) ^ c1[i]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        //  shares of 0 1 0 0 1
        let xs1 = [false, false, true, false, true];
        let xs2 = [false, true, true, false, false];

        // shares of 0 1 0 0 0
        let ys1 = [true, true, true, false, true];
        let ys2 = [true, false, true, false, true];

        // should get 0 1 0 0 0
        let expected = [false, true, false, false, false];

        let mut a1 = [false; 5];
        let mut a2 = [false; 5];
        let mut b1 = [false; 5];
        let mut b2 = [false; 5];
        let mut c1 = [false; 5];
        let mut c2 = [false; 5];
        generate_triples(&mut a1, &mut a2, &mut b1, &mut b2, &mut c1, &mut c2);

        let mut beaver_follower = BeaverServer {
            other: None,
            len: 5,
            xs: xs1.to_vec(),
            ys: ys1.to_vec(),
            a: a1.to_vec(),
            b: b1.to_vec(),
            c: c1.to_vec(),
            done: false,
        };
        let rc = Rc::new(RefCell::new(beaver_follower));

        let z1 = {
            let mut beaver_primary = BeaverServer {
                other: Some(rc.clone()),
                len: 5,
                xs: xs2.to_vec(),
                ys: ys2.to_vec(),
                a: a2.to_vec(),
                b: b2.to_vec(),
                c: c2.to_vec(),
                done: false,
            };

            beaver_primary.multiply();
            assert!(beaver_primary.done);
            let res = beaver_primary.a;
            res
        };

        for i in 0..5 {
            assert_eq!(z1[i] ^ rc.borrow().a[i], expected[i]);
        }
    }
}
