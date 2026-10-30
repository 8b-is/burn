use crate::quantization::QuantValue;
use alloc::vec::Vec;
use num_traits::PrimInt;

/// Pack signed 8-bit integer values into a sequence of unsigned 32-bit integers.
pub fn pack_i8s_to_u32s(values: Vec<i8>) -> Vec<u32> {
    // Shift and combine groups of four 8-bit values into a u32.
    // Same as doing this:
    //     let result = (d_u8 & 0xFF) << 24 | (c_u8 & 0xFF) << 16 | (b_u8 & 0xFF) << 8 | (a_u8 & 0xFF);
    #[cfg(target_endian = "big")]
    {
        values
            .chunks(4)
            .map(|x| {
                x.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, x)| acc | (*x as u32 & 0xFF) << (i * 8))
            })
            .collect()
    }

    // The order of bytes in little endian matches the above description, we just need to
    // handle padding when the number of values is not a factor of 4
    #[cfg(target_endian = "little")]
    {
        let mut values = values;
        let remainder = values.len() % 4;
        if remainder != 0 {
            // Pad with zeros
            values.extend(core::iter::repeat_n(0, 4 - remainder));
        }

        let len = values.len() / 4;
        let capacity = values.capacity() / 4;

        // Pre-forget the old vec and re-interpret as u32
        let mut values = core::mem::ManuallyDrop::new(values);
        let ptr = values.as_mut_ptr() as *mut u32;

        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }
}

/// Unpack integer values into a sequence of signed 8-bit integers.
pub(crate) fn unpack_q_to_i8s<Q: PrimInt>(
    values: &[Q],
    numel: usize,
    value: &QuantValue,
) -> Vec<i8> {
    let size_store = size_of::<Q>() * 8;
    let size_quant = value.size_bits();
    let num_quants = size_store / size_quant;
    let mask = Q::from((1 << size_quant) - 1).unwrap();
    let sign_shift = 8 - size_quant; // sign extension for sub-byte values
    values
        .iter()
        .enumerate()
        .flat_map(|(i, &packed)| {
            // A single u32 could contain less than four 8-bit values...
            let n = core::cmp::min(num_quants, numel - i * num_quants);
            // Extract each 8-bit segment from u32 and cast back to i8
            // Same as doing this (when 4 values are fully packed):
            //     let a = (packed & 0xFF) as i8;
            //     let b = ((packed >> 8) & 0xFF) as i8;
            //     let c = ((packed >> 16) & 0xFF) as i8;
            //     let d = ((packed >> 24) & 0xFF) as i8;
            (0..n).map(move |i| {
                let raw = (packed >> (i * size_quant) & mask).to_u8().unwrap();
                ((raw << sign_shift) as i8) >> sign_shift
            })
        })
        .collect()
}

/// Pack signed 4-bit integer values (`[-8, 7]`) into Zero-Shuffle Block-Planar `Q4_G64` `u32` words
/// (`8 × u32` = 32 bytes per 64-element group, offset `+8`).
///
/// Within each 64-element group:
/// - `words[0..4]` (`bytes[0..16]`): byte `i` holds `low_nibble = w[i] + 8`, `high_nibble = w[16 + i] + 8`
/// - `words[4..8]` (`bytes[16..32]`): byte `16 + i` holds `low_nibble = w[32 + i] + 8`, `high_nibble = w[48 + i] + 8`
///
/// This layout allows 128-bit SIMD registers (ARM NEON / AVX2 / WGSL) to unpack contiguous 16-element
/// activation lanes via simple `& 0x0F` and `>> 4` masks with zero `UZP1`/`UZP2` register shuffles.
pub fn pack_i8s_to_u32s_planar_q4s_g64(values: &[i8]) -> Vec<u32> {
    let num_groups = values.len().div_ceil(64);
    let mut out = Vec::with_capacity(num_groups * 8);

    for g in 0..num_groups {
        let base = g * 64;
        let mut group = [0i8; 64];
        let valid = (values.len().saturating_sub(base)).min(64);
        group[..valid].copy_from_slice(&values[base..base + valid]);

        let mut bytes = [0u8; 32];
        for i in 0..16 {
            let q0 = ((group[i].clamp(-8, 7) + 8) as u8) & 0x0F;
            let q1 = ((group[16 + i].clamp(-8, 7) + 8) as u8) & 0x0F;
            let q2 = ((group[32 + i].clamp(-8, 7) + 8) as u8) & 0x0F;
            let q3 = ((group[48 + i].clamp(-8, 7) + 8) as u8) & 0x0F;
            bytes[i] = q0 | (q1 << 4);
            bytes[16 + i] = q2 | (q3 << 4);
        }

        for chunk in bytes.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }
    out
}

/// Unpack Zero-Shuffle Block-Planar `Q4_G64` `u32` words back into signed `i8` values (`[-8, 7]`).
pub fn unpack_u32s_to_i8s_planar_q4s_g64(packed: &[u32], numel: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(numel.div_ceil(64) * 64);

    for words in packed.chunks(8) {
        let mut bytes = [0x88u8; 32];
        for (w_idx, &word) in words.iter().enumerate() {
            let le = word.to_le_bytes();
            bytes[w_idx * 4..w_idx * 4 + 4].copy_from_slice(&le);
        }
        let mut group = [0i8; 64];
        for i in 0..16 {
            let b_lo = bytes[i];
            let b_hi = bytes[16 + i];
            group[i] = ((b_lo & 0x0F) as i8) - 8;
            group[16 + i] = ((b_lo >> 4) as i8) - 8;
            group[32 + i] = ((b_hi & 0x0F) as i8) - 8;
            group[48 + i] = ((b_hi >> 4) as i8) - 8;
        }
        out.extend_from_slice(&group);
    }
    out.truncate(numel);
    out
}

/// Pack ternary `i8` values (`{-1, 0, +1}`) into Zero-Shuffle Block-Planar `Q2S_G64` `u32` words
/// (`4 × u32` = 16 bytes per 64-element group) using the branchless `AyeOS` encoding:
/// - `0b00 -> 0`, `0b01 -> +1`, `0b10 -> -1` (`trit = (code & 1) - (code >> 1)`).
/// - Byte `i` (`0..16`) holds `(w[i], w[16 + i], w[32 + i], w[48 + i])` in bits `[1:0], [3:2], [5:4], [7:6]`.
pub fn pack_i8s_to_u32s_planar_q2s_g64(values: &[i8]) -> Vec<u32> {
    let num_groups = values.len().div_ceil(64);
    let mut out = Vec::with_capacity(num_groups * 4);

    for g in 0..num_groups {
        let base = g * 64;
        let mut group = [0i8; 64];
        let valid = (values.len().saturating_sub(base)).min(64);
        group[..valid].copy_from_slice(&values[base..base + valid]);

        let mut bytes = [0u8; 16];
        for i in 0..16 {
            let encode = |t: i8| -> u8 { ((t > 0) as u8) | (((t < 0) as u8) << 1) };
            let c0 = encode(group[i]);
            let c1 = encode(group[16 + i]);
            let c2 = encode(group[32 + i]);
            let c3 = encode(group[48 + i]);
            bytes[i] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
        }

        for chunk in bytes.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }
    out
}

/// Unpack Zero-Shuffle Block-Planar `Q2S_G64` `u32` words back into ternary `i8` values (`{-1, 0, +1}`)
/// using the branchless identity `trit = ((code & 1) as i8) - ((code >> 1) as i8)`.
pub fn unpack_u32s_to_i8s_planar_q2s_g64(packed: &[u32], numel: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(numel.div_ceil(64) * 64);

    for words in packed.chunks(4) {
        let mut bytes = [0u8; 16];
        for (w_idx, &word) in words.iter().enumerate() {
            let le = word.to_le_bytes();
            bytes[w_idx * 4..w_idx * 4 + 4].copy_from_slice(&le);
        }
        let mut group = [0i8; 64];
        for i in 0..16 {
            let b = bytes[i];
            let c0 = b & 0x03;
            let c1 = (b >> 2) & 0x03;
            let c2 = (b >> 4) & 0x03;
            let c3 = (b >> 6) & 0x03;
            group[i] = ((c0 & 1) as i8) - ((c0 >> 1) as i8);
            group[16 + i] = ((c1 & 1) as i8) - ((c1 >> 1) as i8);
            group[32 + i] = ((c2 & 1) as i8) - ((c2 >> 1) as i8);
            group[48 + i] = ((c3 & 1) as i8) - ((c3 >> 1) as i8);
        }
        out.extend_from_slice(&group);
    }
    out.truncate(numel);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn should_pack_i8s_to_u32() {
        let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127]);

        assert_eq!(packed, vec![2147287680]);
    }

    #[test]
    fn should_pack_i8s_to_u32_padded() {
        let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55]);
        let packed_padded = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55, 0, 0, 0]);

        assert_eq!(packed, vec![2147287680, 55]);
        assert_eq!(packed, packed_padded);
    }

    #[test]
    fn should_unpack_u32s_to_i8s() {
        let unpacked = unpack_q_to_i8s(&[2147287680u32], 4, &QuantValue::Q8S);

        assert_eq!(unpacked, vec![-128, 2, -3, 127]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_padded() {
        let unpacked = unpack_q_to_i8s(&[55u32], 1, &QuantValue::Q8S);

        assert_eq!(unpacked, vec![55]);
    }

    #[test]
    fn should_unpack_u32s_to_i8s_arange() {
        let unpacked = unpack_q_to_i8s(
            &[
                0u32, 286331136, 286331153, 572657937, 572662306, 857874978, 858993459, 858993459,
                1145324612, 1145324612, 1431655748, 1431655765, 1717982549, 1717986918, 2003199590,
                2004318071,
            ],
            128,
            &QuantValue::Q4S,
        );

        assert_eq!(
            unpacked,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
            ]
        );
    }

    #[test]
    fn should_roundtrip_planar_q4s_g64() {
        let original: Vec<i8> = (0..128).map(|i| ((i % 16) as i8) - 8).collect();
        let packed = pack_i8s_to_u32s_planar_q4s_g64(&original);
        assert_eq!(packed.len(), 16);
        let unpacked = unpack_u32s_to_i8s_planar_q4s_g64(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    #[test]
    fn should_roundtrip_planar_q2s_g64() {
        let original: Vec<i8> = (0..128)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let packed = pack_i8s_to_u32s_planar_q2s_g64(&original);
        assert_eq!(packed.len(), 8);
        let unpacked = unpack_u32s_to_i8s_planar_q2s_g64(&packed, original.len());
        assert_eq!(unpacked, original);
    }
}
