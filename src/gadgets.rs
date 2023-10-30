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
