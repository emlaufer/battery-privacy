use prio::field::{FieldElement, FieldError};
use prio::flp::{gadgets::PolyEval, FlpError, Gadget, Type};
use std::any::Any;
use std::marker::PhantomData;

// Evaluate a polynomial using Horner's method.
pub fn poly_eval<F: FieldElement>(poly: &[F], eval_at: F) -> F {
    if poly.is_empty() {
        return F::zero();
    }

    let mut result = poly[poly.len() - 1];
    for i in (0..poly.len() - 1).rev() {
        result *= eval_at;
        result += poly[i];
    }

    result
}

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

fn poly_add_assign<F: FieldElement>(p: &mut [F], q: &[F]) {
    assert!(
        p.len() >= q.len(),
        "Cannot add into polynomial of smaller size (got {} < {})",
        p.len(),
        q.len()
    );

    for i in 0..p.len() {
        p[i] += q[i];
    }
}

fn poly_sub_assign<F: FieldElement>(p: &mut [F], q: &[F]) {
    assert!(
        p.len() >= q.len(),
        "Cannot add into polynomial of smaller size (got {} < {})",
        p.len(),
        q.len()
    );

    for i in 0..p.len() {
        p[i] -= q[i];
    }
}

fn poly_mul_scalar<F: FieldElement>(p: &[F], coeff: F) -> Vec<F> {
    let p_size = poly_deg(p) + 1;
    let mut out = vec![F::zero(); p_size];
    for i in 0..p_size {
        out[i] = p[i] * coeff;
    }
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

pub(crate) fn call_gadget_on_vec_entries<F: FieldElement>(
    g: &mut Box<dyn Gadget<F>>,
    input: &[F],
    rnd: F,
) -> Result<F, FlpError> {
    let mut range_check = F::zero();
    let mut r = rnd;
    for chunk in input.chunks(1) {
        range_check += r * g.call(chunk)?;
        r *= rnd;
    }
    Ok(range_check)
}

/// Compute the length of the wire polynomial constructed from the given number of gadget calls.
#[inline]
pub(crate) fn wire_poly_len(num_calls: usize) -> usize {
    (1 + num_calls).next_power_of_two()
}

/// Compute the length of the gadget polynomial for a gadget with the given degree and from wire
/// polynomials of the given length.
#[inline]
pub(crate) fn gadget_poly_len(gadget_degree: usize, wire_poly_len: usize) -> usize {
    gadget_degree * (wire_poly_len - 1) + 1
}

#[inline]
fn gadget_poly_fft_mem_len(degree: usize, num_calls: usize) -> usize {
    gadget_poly_len(degree, wire_poly_len(num_calls)).next_power_of_two()
}

fn gadget_call_check<F: FieldElement, G: Gadget<F>>(
    gadget: &G,
    in_len: usize,
) -> Result<(), FlpError> {
    if in_len != gadget.arity() {
        return Err(FlpError::Gadget(format!(
            "unexpected number of inputs: got {}; want {}",
            in_len,
            gadget.arity()
        )));
    }

    if in_len == 0 {
        return Err(FlpError::Gadget("can't call an arity-0 gadget".to_string()));
    }

    Ok(())
}

// Check that the input parameters of g.call_poly() are well-formed.
fn gadget_call_poly_check<F: FieldElement, G: Gadget<F>>(
    gadget: &G,
    outp: &[F],
    inp: &[Vec<F>],
) -> Result<(), FlpError>
where
    G: Gadget<F>,
{
    gadget_call_check(gadget, inp.len())?;

    for i in 1..inp.len() {
        if inp[i].len() != inp[0].len() {
            return Err(FlpError::Gadget(
                "gadget called on wire polynomials with different lengths".to_string(),
            ));
        }
    }

    let expected = gadget_poly_len(gadget.degree(), inp[0].len()).next_power_of_two();
    if outp.len() != expected {
        return Err(FlpError::Gadget(format!(
            "incorrect output length: got {}; want {}",
            outp.len(),
            expected
        )));
    }

    Ok(())
}

/// Test
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sub<F: FieldElement> {
    /// The number of times this gadget will be called.
    num_calls: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldElement> Sub<F> {
    /// Return a new multiplier gadget. `num_calls` is the number of times this gadget will be
    /// called by the validity circuit.
    pub fn new(num_calls: usize) -> Self {
        Self {
            num_calls,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldElement> Gadget<F> for Sub<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, FlpError> {
        gadget_call_check(self, inp.len())?;
        Ok(inp[0] - inp[1])
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_call_poly_check(self, outp, inp)?;

        for i in 0..outp.len() {
            outp[i] = inp[0][i] - inp[1][i];
        }

        Ok(())
    }

    fn arity(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        1
    }

    fn calls(&self) -> usize {
        self.num_calls
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

//trait ComparableGadget<F>: Gadget<F> + Clone + Eq + PartialEq {}
// Two methods:
// 1. Create gate G = r0 * f(x) + r1 * b0 + ...
//      - Degree is ~ f(x) right?

// Creates a composite gadget constructed by passing the outputs from multiple
// gadgets as the inputs to another.
//#[derive(Clone, Debug, Eq, PartialEq)]
//pub struct Compose<F: FieldElement, G: Gadget<F>> {
//    inputs: Vec<Box<dyn Any>>,
//    into: G,
//    extra_inputs: usize,
//    _mark: PhantomData<F>,
//}
//
//impl<F: FieldElement, G: Gadget<F>> Compose<F, G> {
//    pub fn new(inputs: Vec<Box<dyn Gadget<F>>>, extra_inputs: usize, into: G) -> Compose<F, G> {
//        Self {
//            inputs,
//            into,
//            extra_inputs,
//            _mark: PhantomData,
//        }
//    }
//}
//
//impl<F: FieldElement, G: Gadget<F> + 'static> Gadget<F> for Compose<F, G> {
//    fn call(&mut self, inp: &[F]) -> Result<F, FlpError> {
//        let mut intermediate_values = vec![F::zero(); self.inputs.len()];
//        let mut inp_cursor = 0;
//
//        for i in 0..self.inputs.len() {
//            let gadget_arity = self.inputs[i].arity();
//            intermediate_values[i] =
//                self.inputs[i].call(&inp[inp_cursor..inp_cursor + gadget_arity])?;
//            inp_cursor += gadget_arity;
//        }
//        // add extra inputs to the end...
//        intermediate_values.extend_from_slice(&inp[inp_cursor..]);
//
//        self.into.call(&intermediate_values)
//    }
//
//    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
//        let mut intermediate_values = vec![Vec::new(); self.inputs.len()];
//        let mut inp_cursor = 0;
//        for (i, value) in intermediate_values.iter_mut().enumerate() {
//            value.resize(self.inputs[i].degree(), F::zero());
//        }
//
//        for i in 0..self.inputs.len() {
//            let gadget_arity = self.inputs[i].arity();
//            self.inputs[i].call_poly(
//                &mut intermediate_values[i],
//                &inp[inp_cursor..inp_cursor + gadget_arity],
//            )?;
//            inp_cursor += gadget_arity;
//        }
//        // add extra inputs to the end...
//        intermediate_values.extend_from_slice(&inp[inp_cursor..]);
//
//        // TODO: fix
//        self.into.call_poly(outp, &intermediate_values)
//    }
//
//    fn calls(&self) -> usize {
//        self.into.calls()
//    }
//
//    // TODO: correct?
//    fn degree(&self) -> usize {
//        self.inputs.iter().map(|g| g.degree()).max().unwrap() * self.into.degree()
//    }
//
//    fn arity(&self) -> usize {
//        self.inputs.iter().map(|g| g.arity()).sum::<usize>() + self.extra_inputs
//    }
//
//    fn as_any(&mut self) -> &mut dyn Any {
//        self
//    }
//}
//
//

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootHash<F: FieldElement> {
    len: usize,
    num_calls: usize,
    _mark: std::marker::PhantomData<F>,
}

impl<F: FieldElement> RootHash<F> {
    pub fn new(len: usize, num_calls: usize) -> Self {
        Self {
            len,
            num_calls,
            _mark: std::marker::PhantomData,
        }
    }
}

impl<F: FieldElement> Gadget<F> for RootHash<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, FlpError> {
        gadget_call_check(self, inp.len())?;

        let r = inp[0];
        let values = &inp[1..];

        let mut res = F::one();
        for v in values {
            res *= r - *v;
        }

        Ok(res)
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_call_poly_check(self, outp, inp)?;

        let r = &inp[0];
        let values = &inp[1..];

        let mut res = vec![F::zero(); outp.len()];
        res[0] = F::one();

        let mut curr_hash = vec![F::zero(); r.len()];
        for i in 0..values.len() {
            for j in 0..curr_hash.len() {
                curr_hash[j] = r[j] - values[i][j];
            }

            res = poly_mul(&res, &curr_hash);
        }

        outp[..res.len()].copy_from_slice(&res);
        Ok(())
    }

    fn arity(&self) -> usize {
        self.len + 1
    }

    fn degree(&self) -> usize {
        self.len
    }

    fn calls(&self) -> usize {
        self.num_calls
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// An arity-n gadget that returns the linear combination of the inputs.
/// The first half of the input is the value vector, the second half
/// is the coefficient vector.
///
/// inps[n+1] * inps[0] + inps[n+2] * inps[1] + ... inps[n]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearCombination<F: FieldElement> {
    len: usize,
    num_calls: usize,
    _mark: std::marker::PhantomData<F>,
}

impl<F: FieldElement> LinearCombination<F> {
    pub fn new(len: usize, num_calls: usize) -> Self {
        Self {
            len,
            num_calls,
            _mark: std::marker::PhantomData,
        }
    }
}

impl<F: FieldElement> Gadget<F> for LinearCombination<F> {
    fn call(&mut self, inp: &[F]) -> Result<F, FlpError> {
        gadget_call_check(self, inp.len())?;

        let values = &inp[..self.len];
        let coefficients = &inp[self.len..];
        let mut res = F::zero();
        for i in 0..self.len {
            res += coefficients[i] * values[i];
        }

        Ok(res)
    }

    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
        gadget_call_poly_check(self, outp, inp)?;

        for i in 0..outp.len() {
            outp[i] = F::zero();
        }

        let mut res = vec![F::zero(); outp.len()];
        res[0] = F::one();

        let values = &inp[..self.len];
        let coefficients = &inp[self.len..];
        for i in 0..self.len {
            //res = poly_mul(&res, poly_sub_
        }

        Ok(())
    }

    fn arity(&self) -> usize {
        2 * self.len
    }

    fn degree(&self) -> usize {
        2
    }

    fn calls(&self) -> usize {
        self.num_calls
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

// We dont need this...operation is affine!
/// An arity-n gadget returns 0 if bits is the bit decomposition of n
///
/// Returns 0 if n = 2^0 * b0 + 2^1 * b1 ...
/// NOTE: This does not constraint b0, b1, ... as bits. Use the poly_range_check
///       gadget to check that...
//#[derive(Clone, Debug, Eq, PartialEq)]
//pub struct BitSplit<F: FieldElement> {
//    bits: usize,
//    /// Size of buffer for FFT operations.
//    //n: usize,
//    /// Inverse of `n` in `F`.
//    //n_inv: F,
//    /// The number of times this gadget will be called.
//    num_calls: usize,
//    _mark: std::marker::PhantomData<F>,
//}
//
//impl<F: FieldElement> BitSplit<F> {
//    pub fn new(bits: usize, num_calls: usize) -> Self {
//        BitSplit {
//            bits,
//            num_calls,
//            _mark: std::marker::PhantomData,
//        }
//    }
//}
//
//impl<F: FieldElement> Gadget<F> for BitSplit<F> {
//    // must be of the form f,b0,...,bn
//    fn call(&mut self, inp: &[F]) -> Result<F, FlpError> {
//        gadget_call_check(self, inp.len())?;
//
//        // res = 0 iff f == b0 + 2 * b1 + ...
//        let mut res = F::zero();
//        let mut coeff = F::one();
//        let mut two = F::from(F::Integer::try_from(2).unwrap());
//        for i in 0..(self.bits) as usize {
//            res += coeff * inp[i + 1];
//            coeff *= two;
//        }
//        res -= inp[0];
//
//        Ok(res)
//    }
//
//    fn call_poly(&mut self, outp: &mut [F], inp: &[Vec<F>]) -> Result<(), FlpError> {
//        gadget_call_poly_check(self, outp, inp)?;
//
//        for i in 0..outp.len() {
//            outp[i] = F::zero();
//        }
//
//        // TODO: add fft version?
//        let mut res = F::zero();
//        let mut coeff = F::one();
//        let mut two = F::from(F::Integer::try_from(2).unwrap());
//
//        for i in 1..(self.bits + 1) as usize {
//            for c in 0..outp.len() {
//                outp[c] += inp[i][c] * coeff;
//            }
//            coeff *= two;
//        }
//        for c in 0..outp.len() {
//            outp[c] -= inp[0][c];
//        }
//
//        Ok(())
//    }
//
//    fn arity(&self) -> usize {
//        // 1 input per bit, and 1 input for field element
//        self.bits as usize + 1
//    }
//
//    fn degree(&self) -> usize {
//        1
//    }
//
//    fn calls(&self) -> usize {
//        self.num_calls
//    }
//
//    fn as_any(&mut self) -> &mut dyn Any {
//        self
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;
    use prio::field::Field32 as TestField;
    use rand::prelude::*;

    //#[test]
    //fn bit_split() {
    //    let mut g: BitSplit<TestField> = BitSplit::new(16, 1000);
    //    let mut rng = rand::thread_rng();

    //    // valid
    //    for i in 0..65535 {
    //        let mut inp = vec![TestField::zero(); 17];

    //        let bits = i.view_bits::<Lsb0>();
    //        for b in 0..16 {
    //            inp[b + 1] = TestField::from(if bits[b] { 1 } else { 0 });
    //        }
    //        inp[0] = TestField::from(i);

    //        let res = g.call(&inp).unwrap();
    //        assert_eq!(
    //            res,
    //            TestField::zero(),
    //            "Bit decomp validity is {} when it should be 0 for {} (bits:{:?})",
    //            res,
    //            i,
    //            inp
    //        );
    //    }

    //    // invalid
    //    for i in 0..65535 {
    //        let mut inp = vec![TestField::zero(); 17];

    //        let i_bits = i.view_bits::<Lsb0>();
    //        let real_bits: Vec<u32> = (0..16).map(|i| if i_bits[i] { 1 } else { 0 }).collect();
    //        let mut bits: Vec<u32> = (0..16).map(|x| rng.gen_range(0..2)).collect();
    //        // try again if somehow equal...
    //        while bits == real_bits {
    //            bits = (0..16).map(|x| rng.gen_range(0..2)).collect();
    //        }

    //        for b in 0..16 {
    //            inp[b] = TestField::from(bits[b]);
    //        }
    //        inp[16] = TestField::from(i);

    //        let res = g.call(&inp).unwrap();
    //        assert_ne!(
    //            res,
    //            TestField::zero(),
    //            "Bit decomp validity is {} when it should be invalid for {} (bits: {:?} and {:?})",
    //            res,
    //            i,
    //            inp,
    //            real_bits,
    //        );
    //    }
    //}

    //#[test]
    //fn bit_split_poly() {
    //    let mut g: BitSplit<TestField> = BitSplit::new(16, 1000);
    //    gadget_test(&mut g, 1000);
    //}
    //
    #[test]
    fn root_hash() {
        let mut g: RootHash<TestField> = RootHash::new(1000, 10);
        gadget_test(&mut g, 10);
    }

    // TODO:
    // Test that calling g.call_poly() and evaluating the output at a given point is equivalent
    // to evaluating each of the inputs at the same point and applying g.call() on the results.
    fn gadget_test<F: FieldElement, G: Gadget<F>>(g: &mut G, num_calls: usize) {
        let wire_poly_len = (1 + num_calls).next_power_of_two();
        let mut prng = prio::prng::Prng::new().unwrap();
        let mut inp = vec![F::zero(); g.arity()];
        let mut gadget_poly = vec![F::zero(); gadget_poly_fft_mem_len(g.degree(), num_calls)];
        let mut wire_polys = vec![vec![F::zero(); wire_poly_len]; g.arity()];

        let r = prng.get();
        for i in 0..g.arity() {
            for j in 0..wire_poly_len {
                wire_polys[i][j] = prng.get();
            }
            inp[i] = poly_eval(&wire_polys[i], r);
        }

        g.call_poly(&mut gadget_poly, &wire_polys).unwrap();
        let got = poly_eval(&gadget_poly, r);
        let want = g.call(&inp).unwrap();
        assert_eq!(got, want);

        // Repeat the call to make sure that the gadget's memory is reset properly between calls.
        g.call_poly(&mut gadget_poly, &wire_polys).unwrap();
        let got = poly_eval(&gadget_poly, r);
        assert_eq!(got, want);
    }
}
