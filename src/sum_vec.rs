//pub type Prio3Aes128SumVec = Prio3<SumVec<
//

use prio::{field::FieldElement, flp::FlpError};

// Returns the degree of polynomial `p`.
pub fn poly_deg<F: FieldElement>(p: &[F]) -> usize {
    let mut d = p.len();
    while d > 0 && p[d - 1] == F::zero() {
        d -= 1;
    }
    d.saturating_sub(1)
}

// Multiplies polynomials `p` and `q` and returns the result.
pub fn poly_mul<F: FieldElement>(p: &[F], q: &[F]) -> Vec<F> {
    let p_size = poly_deg(p) + 1;
    let q_size = poly_deg(q) + 1;
    let mut out = vec![F::zero(); p_size + q_size];
    for i in 0..p_size {
        for j in 0..q_size {
            out[i + j] += p[i] * q[j];
        }
    }
    out.truncate(poly_deg(&out) + 1);
    out
}

// Returns a polynomial that evaluates to `0` if the input is in range `[start, end)`. Otherwise,
// the output is not `0`.
pub(crate) fn poly_range_check<F: FieldElement>(start: usize, end: usize) -> Vec<F> {
    let mut p = vec![F::one()];
    let mut q = [F::zero(), F::one()];
    for i in start..end {
        q[0] = -F::from(F::Integer::try_from(i).unwrap());
        p = poly_mul(&p, &q);
    }
    p
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct SumVec<F: FieldElement> {
    bits: usize,
    range_checker: Vec<F>,
}

impl<F: FieldElement> SumVec<F> {
    pub fn new(bits: usize) -> Result<Self, FlpError> {
        //if !F::valid_integer_bitlength(bits) {
        //    return Err(FlpError::Encode(
        //        "invalid bits: number of bits exceeds maximum number of bits in this field"
        //            .to_string(),
        //    ));
        //}
        Ok(Self {
            bits,
            range_checker: poly_range_check(0, 2),
        })
    }
}

use prio::flp::{Gadget, Type};
impl<F: FieldElement> Type for SumVec<F> {
    const ID: u32 = 0xbbbb;
    type Measurement = F::Integer;
    type AggregateResult = F::Integer;
    type Field = F;

    fn encode_measurement(&self, value: &F::Integer) -> Result<Vec<F>, FlpError> {
        unimplemented!();
    }

    fn decode_result(&self, data: &[F], num_measurements: usize) -> Result<F::Integer, FlpError> {
        unimplemented!();
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        unimplemented!();
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        unimplemented!();
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        unimplemented!();
    }

    fn input_len(&self) -> usize {
        unimplemented!()
    }

    fn proof_len(&self) -> usize {
        unimplemented!()
    }

    fn verifier_len(&self) -> usize {
        unimplemented!()
    }

    fn output_len(&self) -> usize {
        unimplemented!()
    }

    fn joint_rand_len(&self) -> usize {
        unimplemented!()
    }

    fn prove_rand_len(&self) -> usize {
        unimplemented!()
    }

    fn query_rand_len(&self) -> usize {
        unimplemented!()
    }
}

