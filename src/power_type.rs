use crate::gadgets::poly_range_check; //, BitSplit};
use crate::NUM_BITS;
//1 + 2 * self.len
//2 * self.len + self.bits
use prio::field::{FieldElement, FieldError};
use prio::flp::gadgets::{Mul, ParallelSumGadget};
use prio::flp::{
    gadgets::{BlindPolyEval, PolyEval},
    FlpError, Gadget, LazyEncode, Type, VecLike,
};
use prio::vdaf::prg::Prg;
use prio::vdaf::prio3::Prio3;
use prio::vdaf::Aggregatable;
use prio::vdaf::VdafError;
use prio::vdaf::{AggregateShare, OutputShare};
use std::any::Any;
use std::clone::Clone;
use std::marker::PhantomData;

// Trait to allow unsharding of individual entries from the vector
pub trait IndexCollector<T: Type> {
    fn unshard_index<It: IntoIterator<Item = AggregateShare<T::Field>>>(
        &self,
        _agg_param: &(),
        agg_shares: It,
        num_measurements: usize,
    ) -> Result<T::AggregateResult, VdafError>;
}

impl<
        F: FieldElement,
        S: ParallelSumGadget<F, BlindPolyEval<F>> + Eq + 'static + Clone,
        P: Prg<L>,
        const L: usize,
    > IndexCollector<LinearPower<F, S>> for Prio3<LinearPower<F, S>, P, L>
{
    fn unshard_index<It: IntoIterator<Item = AggregateShare<F>>>(
        &self,
        _agg_param: &(),
        agg_shares: It,
        num_measurements: usize,
    ) -> Result<<LinearPower<F, S> as Type>::AggregateResult, VdafError> {
        let mut agg_share = AggregateShare::from(vec![F::zero()]);
        for share in agg_shares {
            agg_share.merge(&share)?;
        }

        Ok(self.typ.decode_result(&agg_share.0, num_measurements)?)
    }
}

impl<
        F: FieldElement,
        S: ParallelSumGadget<F, BlindPolyEval<F>> + Eq + 'static + Clone,
        P: Prg<L>,
        const L: usize,
    > IndexCollector<RAMPower<F, S>> for Prio3<RAMPower<F, S>, P, L>
{
    fn unshard_index<It: IntoIterator<Item = AggregateShare<F>>>(
        &self,
        _agg_param: &(),
        agg_shares: It,
        num_measurements: usize,
    ) -> Result<<RAMPower<F, S> as Type>::AggregateResult, VdafError> {
        let mut agg_share = AggregateShare::from(vec![F::zero()]);
        for share in agg_shares {
            agg_share.merge(&share)?;
        }

        Ok(self.typ.decode_result(&agg_share.0, num_measurements)?)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RAMPower<F: FieldElement, S: ParallelSumGadget<F, BlindPolyEval<F>>> {
    len: usize,
    bits: usize,
    chunk_len: usize,
    gadget_calls: usize,
    range_checker: Vec<F>,
    pub client_num: usize,
    client_maxs: Vec<usize>,
    client_energies: Vec<usize>,
    _mark: PhantomData<F>,
    _mark2: PhantomData<S>,
}

impl<F: FieldElement, S: ParallelSumGadget<F, BlindPolyEval<F>>> RAMPower<F, S> {
    pub fn new(
        len: usize,
        bits: usize,
        client_maxs: Vec<usize>,
        client_energies: Vec<usize>,
    ) -> Result<Self, FlpError> {
        let chunk_len = std::cmp::max(1, ((len + 2usize.pow(bits as u32)) as f64).sqrt() as usize);

        let mut gadget_calls = (len + 2usize.pow(bits as u32)) / chunk_len;
        if len % chunk_len != 0 {
            gadget_calls += 1;
        }

        Ok(Self {
            len,
            bits,
            chunk_len,
            gadget_calls,
            range_checker: poly_range_check(0, 2),
            client_num: 0,
            client_maxs,
            client_energies,
            _mark: PhantomData,
            _mark2: PhantomData,
        })
    }

    pub fn set_client_num(&mut self, num: usize) {
        self.client_num = num;
    }

    pub fn gadget_calls(&self) -> usize {
        let chunk_len = self.chunk_len;

        let mut gadget_calls = (self.len + self.client_maxs[self.client_num]) / chunk_len;
        if self.len % chunk_len != 0 {
            gadget_calls += 1;
        }
        gadget_calls
    }

    pub fn chunk_len(&self) -> usize {
        let chunk_len = std::cmp::max(
            1,
            ((self.len + self.client_maxs[self.client_num]) as f64).sqrt() as usize,
        );
        chunk_len
    }

    fn permutation_check(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        s: F,
        r: F,
        diffs: &[F],
        sorted_diffs: &[F],
    ) -> Result<F, FlpError> {
        let mut diff_root_hash = s.clone();
        for v in diffs {
            diff_root_hash = g[0].call(&[diff_root_hash, r * s - *v])?;
        }

        let mut sorted_root_hash = s.clone();
        for v in sorted_diffs.iter() {
            sorted_root_hash = g[0].call(&[sorted_root_hash, r * s - *v])?;
        }
        Ok(diff_root_hash - sorted_root_hash)
    }

    fn sorted_check(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        s: F,
        joint_rand: &[F],
        sorted_diffs: &[F],
    ) -> Result<F, FlpError> {
        //// sorted check
        let mut sorted_check = F::zero();
        let sorted_increments: Vec<F> = (1..sorted_diffs.len())
            .map(|i| sorted_diffs[i] - sorted_diffs[i - 1])
            .collect();
        let mut r = joint_rand[0];

        let mut padded_chunk = vec![F::zero(); 2 * self.chunk_len];
        for (i, chunk) in sorted_increments.chunks(self.chunk_len).enumerate() {
            let d = chunk.len();
            for i in 0..self.chunk_len {
                if i < d {
                    padded_chunk[2 * i] = chunk[i];
                } else {
                    // If the chunk is smaller than the chunk length, then copy the last element of
                    // the chunk into the remaining slots.
                    padded_chunk[2 * i] = chunk[d - 1];
                }
                padded_chunk[2 * i + 1] = r * s;
                r *= joint_rand[0];
            }

            sorted_check += g[1].call(&padded_chunk)?;
        }
        Ok(sorted_check)
    }
}

impl<F: FieldElement, S: ParallelSumGadget<F, BlindPolyEval<F>> + Eq + 'static + Clone> Type
    for RAMPower<F, S>
{
    const ID: u32 = 0x00000002;
    type Measurement = (F::Integer, Vec<F::Integer>);
    type LazyMeasurement = Vec<F>;
    type AggregateResult = Vec<F::Integer>;
    type Field = F;

    fn encode_measurement(
        &self,
        measurement: &(F::Integer, Vec<F::Integer>),
    ) -> Result<Vec<F>, FlpError> {
        let mut current_energy = measurement.0;
        let mut measurements = measurement.1.clone();
        if measurements.len() != self.len {
            return Err(FlpError::Encode(format!(
                "unexpected measurement length: got {}; want {}",
                measurements.len(),
                self.len
            )));
        }

        let mut res = vec![];
        res.extend(measurements.iter().map(|value| F::from(*value)));
        for i in 0..self.client_maxs[self.client_num] + 1 {
            measurements.push(F::Integer::try_from(i as usize).unwrap());
        }
        measurements.sort();
        res.extend(measurements.iter().map(|value| F::from(*value)));

        // encode energies
        let mut energies = (0..measurement.1.len())
            .map(|i| {
                measurement.1[0..i]
                    .iter()
                    .cloned()
                    .fold(F::Integer::try_from(0).unwrap(), |a, m| a + m)
            })
            .map(|value| current_energy - value)
            .collect::<Vec<_>>();
        for i in 0..self.client_energies[self.client_num] + 1 {
            energies.push(F::Integer::try_from(i as usize).unwrap());
        }
        energies.sort();
        res.extend(energies.iter().map(|value| F::from(*value)));
        res.push(F::from(current_energy));

        Ok(res)
    }

    fn decode_result(
        &self,
        data: &[F],
        _num_measurements: usize,
    ) -> Result<Vec<F::Integer>, FlpError> {
        Ok(data.iter().map(|elem| F::Integer::from(*elem)).collect())
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![
            // TODO:...
            //Box::new(BitSplit::new(self.bits, self.len)),
            //Box::new(Mul::new(1 * (self.len + 2usize.pow(self.bits as u32)))),
            Box::new(Mul::new(
                2 * (self.len + self.client_maxs[self.client_num] + 1)
                    + 2 * (self.len + self.client_energies[self.client_num] + 1),
            )),
            //Box::new(Mul::new(2 * self.len)),
            Box::new(S::new(
                BlindPolyEval::new(self.range_checker.clone(), 2 * self.gadget_calls),
                self.chunk_len,
            )),
            //Box::new(RootHash::new(self.len, 1)),
        ]
        //vec![]
    }

    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        self.valid_call_check(input, joint_rand)?;
        let s = F::from(F::Integer::try_from(num_shares).unwrap()).inv();
        let mut r = joint_rand[0];

        let diffs_start = 0;
        let diffs_end = self.len;
        let diffs = input[diffs_start..diffs_end]
            .iter()
            .map(|v| v.clone())
            .chain(
                (0..self.client_maxs[self.client_num] + 1)
                    .map(|v| s * F::from(F::Integer::try_from(v).unwrap())),
            )
            .collect::<Vec<_>>();

        let sorted_start = diffs_end;
        let sorted_end = diffs_end + self.len + self.client_maxs[self.client_num] + 1; //+ 2usize.pow(self.bits as u32);
        let sorted_diffs = &input[sorted_start..sorted_end];

        let current_energy = input[input.len() - 1];
        let energy_diffs_start = sorted_end;
        let energy_diffs_end = energy_diffs_start;
        let energy_diffs = (diffs_start..diffs_end)
            // compute partial sums
            .map(|i| {
                input[diffs_start..i]
                    .iter()
                    .cloned()
                    .fold(F::zero(), |a, m| a + m)
            })
            // subtract from current energy
            .map(|value| current_energy - value)
            // add all values from 0..max_energy
            .chain(
                (0..self.client_energies[self.client_num] + 1)
                    .map(|v| s * F::from(F::Integer::try_from(v).unwrap())),
            )
            .collect::<Vec<_>>();

        let energy_sorted_start = energy_diffs_end;
        let energy_sorted_end =
            energy_sorted_start + self.len + self.client_energies[self.client_num] + 1; //+ 2usize.pow(self.bits as u32);
        let energy_sorted_diffs = &input[energy_sorted_start..energy_sorted_end];

        let permutation_check = self.permutation_check(g, s, r, &diffs, sorted_diffs)?;
        let sorted_check = self.sorted_check(g, s, joint_rand, sorted_diffs)?;

        let energy_permutation_check =
            self.permutation_check(g, s, r, &energy_diffs, energy_sorted_diffs)?;
        let energy_sorted_check = self.sorted_check(g, s, joint_rand, energy_sorted_diffs)?;

        // max check
        let min_check = sorted_diffs[0];
        let max_check = sorted_diffs[sorted_diffs.len() - 1]
            - s * F::from(F::Integer::try_from(self.client_maxs[self.client_num]).unwrap());
        let energy_min_check = energy_sorted_diffs[0];
        let energy_max_check = energy_sorted_diffs[energy_sorted_diffs.len() - 1]
            - s * F::from(F::Integer::try_from(self.client_energies[self.client_num]).unwrap());

        return Ok(permutation_check
            + sorted_check
            + energy_permutation_check
            + energy_sorted_check
            + min_check
            + max_check
            + energy_min_check
            + energy_max_check);
    }

    fn get_wire_values(
        &self,
        gadgets: &Vec<Box<dyn Gadget<Self::Field>>>,
        outp: &mut [Self::Field],
        input: &Self::LazyMeasurement,
        joint_rand: &[Self::Field],
        num_shares: usize,
        gadget_idx: usize,
        wire_idx: usize,
    ) -> Result<(), FlpError> {
        unimplemented!();
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input[0..self.len].to_vec())
    }

    fn input_len(&self) -> usize {
        //2 * self.len + 2usize.pow(self.bits as u32)
        3 * self.len + self.client_maxs[self.client_num] + self.client_energies[self.client_num] + 3
    }

    fn proof_len(&self) -> usize {
        //8 * 2usize.pow(self.bits as u32) + 1
        //4 * 2usize.pow(self.bits as u32) + 1
        //2 * (self.len + 1)
        //2
        //131073
        //524289
        //1048577
        //524289
        //526505
        //526463
        //1050839
        //526505
        println!(
            "len: {:?}, bits: {:?}, chunk_len: {:?}, gadget_calls: {:?}, another: {:?}",
            self.len,
            self.bits,
            self.chunk_len,
            self.gadget_calls,
            2 * (self.len + 2usize.pow(self.bits as u32))
        );
        //4 * ((self.len + 1).next_power_of_two())
        4 * (self.len
            + self.client_maxs[self.client_num]
            + 1
            + self.len
            + self.client_energies[self.client_num])
            .next_power_of_two()
            + 1
            + 2 * self.chunk_len
            + 6 * ((1 + self.gadget_calls()).next_power_of_two() - 1)
            + 4
        //8 * (self.len + 2usize.pow(self.bits as u32))
        //self.len * (self.len + 1) + 2 * 2usize.pow(self.bits as u32)
        //+ 2 * self.chunk_len
        //+ 3 * ((1 + self.gadget_calls).next_power_of_two() - 1)
        //1
    }

    fn verifier_len(&self) -> usize {
        //self.len + 3
        //4
        //645
        //733
        //687
        2 * self.chunk_len + 5
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn joint_rand_len(&self) -> usize {
        4
    }

    fn prove_rand_len(&self) -> usize {
        2 * self.chunk_len + 2
        //2
    }

    fn query_rand_len(&self) -> usize {
        4
    }
}

/// Prio type for Power summation.
///
/// Each measurement is an integer between [0, 2^bits) and the aggregate is
/// the sum of measurements...
///
/// The proof is reduced to size \sqrt{N} using [[BBCG+19], Corollary 4.9]
/// as in the Prio CountVec gadget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearPower<F: FieldElement, S: ParallelSumGadget<F, BlindPolyEval<F>>> {
    len: usize,
    bits: usize,
    chunk_len: usize,
    gadget_calls: usize,
    range_checker: Vec<F>,
    pub client_num: usize,
    client_maxs: Vec<usize>,
    client_energies: Vec<usize>,
    _marker: PhantomData<S>,
}

impl<F: FieldElement, S: ParallelSumGadget<F, BlindPolyEval<F>>> LinearPower<F, S> {
    /// Return a new [`Sum`] type parameter. Each value of this type is an integer in range `[0,
    /// 2^bits)`.
    pub fn new(
        len: usize,
        bits: usize,
        client_maxs: Vec<usize>,
        client_energies: Vec<usize>,
    ) -> Result<Self, FlpError> {
        let chunk_len = std::cmp::max(1, ((len * 3 * NUM_BITS) as f64).sqrt() as usize);

        let mut gadget_calls = (len * 3 * NUM_BITS) / chunk_len;
        if (len * 3 * NUM_BITS) % chunk_len != 0 {
            gadget_calls += 1;
        }

        Ok(Self {
            len,
            bits,
            chunk_len,
            gadget_calls,
            range_checker: poly_range_check(0, 2),
            client_num: 0,
            _marker: PhantomData,
            client_maxs,
            client_energies,
        })
    }

    pub fn encode_index(
        data: &(F::Integer, Vec<F::Integer>, F::Integer, F::Integer),
        o_idx: usize,
    ) -> F {
        let current_energy = data.0;
        let measurements = &data.1;
        let max_power = data.2;
        let max_energy = data.3;
        let mut idx = o_idx;

        if idx == 0 {
            // TODO: technically no need for this...
            return F::from(current_energy);
        }
        idx -= 1;

        if idx < measurements.len() {
            return F::from(measurements[idx]);
        }
        idx -= measurements.len();

        // if data.len() < idx < data.len() + measure_len * num_bits, get correct bit...
        if idx < measurements.len() * NUM_BITS {
            let measurement = idx / NUM_BITS;
            let bit = F::Integer::try_from(idx % NUM_BITS).unwrap();

            return F::from((measurements[measurement] >> bit) & F::Integer::try_from(1).unwrap());
        }
        idx -= measurements.len() * NUM_BITS;

        if idx < measurements.len() * NUM_BITS {
            let measurement = idx / NUM_BITS;
            let bit = F::Integer::try_from(idx % NUM_BITS).unwrap();

            return F::from(
                ((max_power - measurements[measurement]) >> bit) & F::Integer::try_from(1).unwrap(),
            );
        }
        idx -= measurements.len() * NUM_BITS;

        if idx < measurements.len() * NUM_BITS {
            let measurement = idx / NUM_BITS;
            let current_energy = current_energy
                - measurements[0..measurement]
                    .iter()
                    .fold(F::Integer::try_from(0).unwrap(), |a, v| a + *v);
            let bit = F::Integer::try_from(idx % NUM_BITS).unwrap();

            return F::from(
                ((max_energy - (current_energy)) >> bit) & F::Integer::try_from(1).unwrap(),
            );
            //    current_energy = current_energy - measurements[i];
        }
        panic!("Index {} out of bounds!", o_idx);
    }
}

impl<F: FieldElement, S: ParallelSumGadget<F, BlindPolyEval<F>> + Eq + 'static + Clone> Type
    for LinearPower<F, S>
{
    const ID: u32 = 0x00000001;
    type Measurement = (F::Integer, Vec<F::Integer>);
    type LazyMeasurement = LazyEncode<(F::Integer, Vec<F::Integer>, F::Integer, F::Integer), F>;
    type AggregateResult = Vec<F::Integer>;
    type Field = F;

    fn encode_measurement(
        &self,
        measurement: &(F::Integer, Vec<F::Integer>),
    ) -> Result<Self::LazyMeasurement, FlpError> {
        let (current_energy, measurements) = measurement.clone();
        let max_power = F::Integer::try_from(self.client_maxs[self.client_num]).unwrap();
        let max_energy = F::Integer::try_from(self.client_energies[self.client_num]).unwrap();
        Ok(LazyEncode {
            data: (current_energy, measurements, max_power, max_energy),
            len: self.len * (3 * self.bits + 1) + 1,
            indexer: Self::encode_index,
        })
        // BELOW: Eager version...
        //let mut current_energy = measurement.0;
        //let measurements = &measurement.1;
        //if measurements.len() != self.len {
        //    return Err(FlpError::Encode(format!(
        //        "unexpected measurement length: got {}; want {}",
        //        measurements.len(),
        //        self.len
        //    )));
        //}

        //let max_power = F::Integer::try_from(self.client_maxs[self.client_num]).unwrap();
        //let max_energy = F::Integer::try_from(self.client_energies[self.client_num]).unwrap();
        //let mut res: Vec<F> = Vec::with_capacity(1 + (1 + 3 * self.bits) * measurements.len());
        //println!("EXPECT: {}", 1 + (1 + 3 * self.bits) * measurements.len());
        //res.push(F::from(current_energy));
        //println!("CURRENT ENERGY: {:?}", F::from(current_energy));
        //// TODO: get rid of this...
        //for i in 0..measurements.len() {
        //    res.push(F::from(measurements[i]));
        //}
        //for i in 0..measurements.len() {
        //    res.extend(
        //        encode_into_bitvector_representation::<F>(&measurements[i], self.bits as usize)
        //            .expect(&format!(
        //                "Cannot encode ({:?} - {:?}) within {} bits! Invalid!",
        //                max_power, measurements[i], self.bits
        //            )),
        //    );
        //}
        //for i in 0..measurements.len() {
        //    res.extend(
        //        encode_into_bitvector_representation::<F>(
        //            &(max_power - measurements[i]),
        //            self.bits as usize,
        //        )
        //        .expect(&format!(
        //            "Cannot encode ({:?} - {:?}) within {} bits! Invalid!",
        //            max_power, measurements[i], self.bits
        //        )),
        //    );
        //}
        //for i in 0..measurements.len() {
        //    res.extend(
        //        encode_into_bitvector_representation::<F>(
        //            &(max_energy - (current_energy - measurements[i])),
        //            self.bits as usize,
        //        )
        //        .expect(&format!(
        //            "Cannot encode ({:?} - {:?}) within {} bits! Invalid!",
        //            max_power, measurements[i], self.bits
        //        )),
        //    );
        //    current_energy = current_energy - measurements[i];
        //}
        //println!("TOTAL VALUES: {:?}", res.len());
        //println!("bytes CAP: {:?}", res.capacity() * std::mem::size_of::<F>());
        //println!("got: {:?}", std::mem::size_of::<F>());
        //Ok(res)
    }

    fn decode_result(
        &self,
        data: &[F],
        _num_measurements: usize,
    ) -> Result<Vec<F::Integer>, FlpError> {
        //if data.len() != self.len {
        //    return Err(FlpError::Decode("unexpected input length".into()));
        //}
        Ok(data.iter().map(|elem| F::Integer::from(*elem)).collect())
    }

    fn gadget(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![
            // TODO:...
            //Box::new(BitSplit::new(self.bits, self.len)),
            Box::new(S::new(
                BlindPolyEval::new(self.range_checker.clone(), self.gadget_calls),
                self.chunk_len,
            )),
        ]
        //vec![]
    }

    // For now...we will give the client power as an extra field element
    // for each input...
    fn valid(
        &self,
        g: &mut Vec<Box<dyn Gadget<F>>>,
        input: &[F],
        joint_rand: &[F],
        num_shares: usize,
    ) -> Result<F, FlpError> {
        //println!("{}", input.len());
        self.valid_call_check(input, joint_rand)?;

        let current_energy = input[0];
        let values = &input[1..self.len + 1];
        let value_sums = (1..self.len + 1)
            .map(|i| input[1..i].iter().cloned().fold(F::zero(), |a, m| a + m))
            .collect::<Vec<_>>();
        let bits = &input[1 + self.len..];
        //println!("len: {}", self.len);
        //println!("values len: {}", values.len());
        //println!("bits len: {}", bits.len());

        // check sums of value bits
        // check the sums are correct using a random linear combination of
        // bitsplit checks...
        let mut r = joint_rand[0];
        let mut res = F::zero();
        let two = F::from(F::Integer::try_from(2).unwrap());
        for i in 0..self.len {
            let mut value = F::zero();
            let mut bit_coeff = F::one();

            for bit in &bits[i * self.bits..(i + 1) * self.bits] {
                value += bit_coeff * *bit;
                bit_coeff *= two;
            }
            value -= values[i];
            res += r * value;
            r *= joint_rand[0];
        }

        // check sums of m - v bits
        // check the sums are correct using a random linear combination of
        // bitsplit checks...
        let s = F::from(F::Integer::try_from(num_shares).unwrap()).inv();
        let two = F::from(F::Integer::try_from(2).unwrap());
        for i in 0..self.len {
            let mut value = F::zero();
            let mut bit_coeff = F::one();

            for bit in &bits[(self.len + i) * self.bits..(self.len + i + 1) * self.bits] {
                value += bit_coeff * *bit;
                bit_coeff *= two;
            }
            value -= s * F::from(F::Integer::try_from(self.client_maxs[self.client_num]).unwrap())
                - values[i];
            res += r * value;
            r *= joint_rand[0];
        }

        // check sums of sums bits
        // check the sums are correct using a random linear combination of
        // bitsplit checks...
        let mut r = joint_rand[0];
        let mut res = F::zero();
        let two = F::from(F::Integer::try_from(2).unwrap());
        for i in 0..self.len {
            let mut value = F::zero();
            let mut bit_coeff = F::one();

            for bit in &bits[(i + 2 * self.len) * self.bits..(i + 2 * self.len + 1) * self.bits] {
                value += bit_coeff * *bit;
                bit_coeff *= two;
            }
            value -=
                s * F::from(F::Integer::try_from(self.client_energies[self.client_num]).unwrap())
                    - (current_energy - value_sums[i]);
            res += r * value;
            r *= joint_rand[0];
        }

        // check that each bit is 0 or 1 using parallel sum gadget
        let mut r = joint_rand[0];
        let mut outp = F::zero();
        let mut padded_chunk = vec![F::zero(); 2 * self.chunk_len];
        let mut test = 0;
        for chunk in bits.chunks(self.chunk_len) {
            let d = chunk.len();
            for i in 0..self.chunk_len {
                if i < d {
                    padded_chunk[2 * i] = chunk[i];
                } else {
                    // If the chunk is smaller than the chunk length, then copy the last element of
                    // the chunk into the remaining slots.
                    padded_chunk[2 * i] = chunk[d - 1];
                }
                padded_chunk[2 * i + 1] = r * s;
                r *= joint_rand[0];
            }

            test += 1;
            outp += g[0].call(&padded_chunk)?;
        }

        Ok(res + outp)
    }

    fn get_wire_values(
        &self,
        gadgets: &Vec<Box<dyn Gadget<Self::Field>>>,
        outp: &mut [Self::Field],
        inp: &Self::LazyMeasurement,
        joint_rand: &[Self::Field],
        num_shares: usize,
        gadget_idx: usize,
        wire_idx: usize,
    ) -> Result<(), FlpError> {
        // TODO: Fix this!
        let bits_off = self.len + 1;
        let num_bits = self.len * (3 * self.bits); //inp.len() - (self.len + 1);

        let mut r = joint_rand[0];
        // check that each bit is 0 or 1 using parallel sum gadget
        //let mut outp = F::zero();
        //let mut padded_chunk = vec![F::zero(); 2 * self.chunk_len];
        let s = F::from(F::Integer::try_from(num_shares).unwrap()).inv();
        for idx in 0..outp.len() {
            for i in 0..wire_idx / 2 {
                r *= joint_rand[0];
            }
            // get the wire values for each chunk
            if wire_idx % 2 == 0 {
                let inp_idx = bits_off + (idx * self.chunk_len) + wire_idx / 2;
                if inp_idx < inp.len() {
                    outp[idx] = inp.index(bits_off + (idx * self.chunk_len) + wire_idx / 2);
                } else {
                    outp[idx] = inp.index(inp.len() - 1);
                }
            } else {
                outp[idx] = r * s
            }
            for i in wire_idx / 2..self.chunk_len {
                r *= joint_rand[0];
            }
        }

        Ok(())
        //padded_chunk[
        //for chunk in bits.chunks(self.chunk_len) {
        //    let d = chunk.len();
        //    for i in 0..self.chunk_len {
        //        if i < d {
        //            padded_chunk[2 * i] = chunk[i];
        //        } else {
        //            // If the chunk is smaller than the chunk length, then copy the last element of
        //            // the chunk into the remaining slots.
        //            padded_chunk[2 * i] = chunk[d - 1];
        //        }
        //        padded_chunk[2 * i + 1] = r * s;
        //        r *= joint_rand[0];
        //    }

        //    println!("chunk: {:?}", chunk);
        //    outp += g[0].call(&padded_chunk)?;
        //}
    }

    fn truncate(&self, input: Vec<F>) -> Result<Vec<F>, FlpError> {
        self.truncate_call_check(&input)?;
        Ok(input[1..self.output_len() + 1].to_vec())
    }

    fn input_len(&self) -> usize {
        self.len * (3 * self.bits + 1) + 1
    }

    fn proof_len(&self) -> usize {
        //println!("GOT: {}", (self.len + 1).next_power_of_two());
        //println!("GOT: {}", self.chunk_len * 2);
        //panic!(
        //    "PROOF LEN: {}",
        //    16 + (self.len + 1).next_power_of_two()
        //        + self.chunk_len * 2
        //        + 3 * ((1 + self.gadget_calls).next_power_of_two() - 1)
        //        + 1
        //);
        //16 + (self.len + 1).next_power_of_two()
        //    + self.chunk_len * 2
        2 * self.chunk_len + 3 * ((1 + self.gadget_calls).next_power_of_two() - 1) + 1
        //+ 2 * (self.len).next_power_of_two() * (self.bits + 1)
        //0
    }

    fn verifier_len(&self) -> usize {
        2 * self.chunk_len + 2
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn joint_rand_len(&self) -> usize {
        16
    }

    fn prove_rand_len(&self) -> usize {
        //2 * self.len + 2
        2 * self.chunk_len
    }

    fn query_rand_len(&self) -> usize {
        16
    }
}

/// Encode `input` as bitvector of elements of `Self`. Output is written into the `output` slice.
/// If `output.len()` is smaller than the number of bits required to respresent `input`,
/// an error is returned.
///
/// # Arguments
///
/// * `input` - The field element to encode
/// * `output` - The slice to write the encoded bits into. Least signicant bit comes first
fn fill_with_bitvector_representation<F: FieldElement>(
    input: &F::Integer,
    output: &mut [F],
) -> Result<(), FieldError> {
    // Create a mutable copy of `input`. In each iteration of the following loop we take the
    // least significant bit, and shift input to the right by one bit.
    let mut i = *input;

    let one = F::Integer::from(F::one());
    for bit in output.iter_mut() {
        let w = F::from(i & one);
        *bit = w;
        i = i >> one;
    }

    // If `i` is still not zero, this means that it cannot be encoded by `bits` bits.
    if i != F::Integer::from(F::zero()) {
        return Err(FieldError::InputSizeMismatch);
    }

    Ok(())
}

/// Encode `input` as `bits`-bit vector of elements of `Self` if it's small enough
/// to be represented with that many bits.
///
/// # Arguments
///
/// * `input` - The field element to encode
/// * `bits` - The number of bits to use for the encoding
fn encode_into_bitvector_representation<F: FieldElement>(
    input: &F::Integer,
    bits: usize,
) -> Result<Vec<F>, FieldError> {
    let mut result = vec![F::zero(); bits];
    fill_with_bitvector_representation(input, &mut result)?;
    Ok(result)
}

/// Decode the bitvector-represented value `input` into a simple representation as a single
/// field element.
///
/// # Errors
///
/// This function errors if `2^input.len() - 1` does not fit into the field `Self`.
fn decode_from_bitvector_representation<F: FieldElement>(input: &[F]) -> Result<F, FieldError> {
    let fi_one = F::Integer::from(F::one());

    if !valid_integer_bitlength::<F>(input.len()) {
        return Err(FieldError::ModulusOverflow);
    }

    let mut decoded = F::zero();
    for (l, bit) in input.iter().enumerate() {
        let fi_l = F::Integer::try_from(l).map_err(|_| FieldError::IntegerTryFrom)?;
        let w = fi_one << fi_l;
        decoded += F::from(w) * *bit;
    }
    Ok(decoded)
}

fn valid_integer_bitlength<F: FieldElement>(bits: usize) -> bool {
    if bits >= 8 * F::ENCODED_SIZE {
        return false;
    }
    if let Ok(bits_int) = F::Integer::try_from(bits) {
        if F::modulus() >> bits_int != F::Integer::from(F::zero()) {
            return true;
        }
    }
    false
}
