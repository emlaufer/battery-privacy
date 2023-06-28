use crate::field::FieldElement;
use alloc::vec::Vec;

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

pub fn poly_range_check<F: FieldElement>(start: usize, end: usize) -> Vec<F> {
    let mut p = vec![F::one()];
    let mut q = [F::zero(), F::one()];
    for i in start..end {
        q[0] = -F::from(F::Integer::try_from(i).unwrap());
        p = poly_mul(&p, &q);
    }
    p
}
