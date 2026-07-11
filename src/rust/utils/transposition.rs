//! Byte-plane Transposition Utilities
//!
//! Transpose bytes by grouping same-position bytes together.
//! For item_size=2: [A0 A1 B0 B1 C0 C1] -> [A0 B0 C0 A1 B1 C1]
//! This improves compression by putting similar bytes together.

/// Transpose bytes by grouping same-position bytes together.
/// For item_size=2: [A0 A1 B0 B1 C0 C1] -> [A0 B0 C0 A1 B1 C1]
#[inline]
pub fn transpose_bytes(data: &[u8], item_size: usize) -> Vec<u8> {
    if item_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let num_elements = data.len() / item_size;
    if num_elements == 0 {
        return data.to_vec();
    }

    let mut transposed = vec![0u8; data.len()];

    // Transpose: plane[i] contains the i-th byte of each element
    for plane in 0..item_size {
        for elem in 0..num_elements {
            transposed[plane * num_elements + elem] = data[elem * item_size + plane];
        }
    }

    transposed
}

/// Inverse transpose to restore original byte order.
/// [A0 B0 C0 A1 B1 C1] -> [A0 A1 B0 B1 C0 C1]
#[inline]
pub fn inverse_transpose_bytes(data: &[u8], item_size: usize) -> Vec<u8> {
    if item_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let num_elements = data.len() / item_size;
    if num_elements == 0 {
        return data.to_vec();
    }

    let mut result = vec![0u8; data.len()];

    for plane in 0..item_size {
        for elem in 0..num_elements {
            result[elem * item_size + plane] = data[plane * num_elements + elem];
        }
    }

    result
}

/// Regroup bytes for BF16: [Exponent] plane + [Sign+Mantissa] plane.
/// Input: slice of u16 (as bytes).
/// Output: [All Exponents, All Sign+Mantissas]
/// BF16: S(1) E(8) M(7)
/// Exponent byte: E(8)
/// SignMantissa byte: S(1) M(7)
#[inline]
pub fn regroup_bytes(data: &[u8]) -> Vec<u8> {
    if data.len() < 2 {
        return data.to_vec();
    }

    let num_elements = data.len() / 2;
    let mut result = vec![0u8; data.len()];

    // We assume Little Endian input for u16: [Low, High]
    // Low byte:  E_low(1) + M(7)
    // High byte: S(1) + E_high(7)
    //
    // u16 bits:
    // 15: S
    // 14-7: E (8 bits)
    // 6-0: M (7 bits)

    let (exponents, mantissas) = result.split_at_mut(num_elements);

    for i in 0..num_elements {
        // Read u16 from data (Little Endian)
        let low = data[i * 2] as u16;
        let high = data[i * 2 + 1] as u16;
        let val = (high << 8) | low;

        // Extract Exponent (bits 7-14) -> 8 bits
        // Shift right by 7, mask 0xFF
        exponents[i] = ((val >> 7) & 0xFF) as u8;

        // Extract Sign (bit 15) + Mantissa (bits 0-6) -> 8 bits
        // Sign to bit 7: (val & 0x8000) >> 8
        // Mantissa to bits 0-6: (val & 0x7F)
        mantissas[i] = (((val & 0x8000) >> 8) | (val & 0x7F)) as u8;
    }

    result
}

/// Inverse regroup: restore BF16 bytes from [Exponent] + [Sign+Mantissa] planes.
#[inline]
pub fn inverse_regroup_bytes(data: &[u8]) -> Vec<u8> {
    if data.len() < 2 {
        return data.to_vec();
    }

    let num_elements = data.len() / 2;
    let mut result = vec![0u8; data.len()];

    let (exponents, mantissas) = data.split_at(num_elements);

    for i in 0..num_elements {
        let exp = exponents[i] as u16;
        let sm = mantissas[i] as u16;

        // Reconstruct u16
        // S: bit 7 of sm -> bit 15
        // E: exp -> bits 7-14
        // M: bits 0-6 of sm -> bits 0-6

        let s = (sm & 0x80) << 8;
        let e = exp << 7;
        let m = sm & 0x7F;

        let val = s | e | m;

        // Write Little Endian
        result[i * 2] = (val & 0xFF) as u8;
        result[i * 2 + 1] = (val >> 8) as u8;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_bytes() {
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let transposed = transpose_bytes(&data, 2);
        assert_eq!(transposed, vec![0x01, 0x03, 0x02, 0x04]);
        let restored = inverse_transpose_bytes(&transposed, 2);
        assert_eq!(restored, data);
    }

    #[test]
    fn test_regroup_bytes() {
        // Test with a known BF16 value
        // 1.0 in BF16: 0x3F80
        // Binary: 0011 1111 1000 0000
        // S=0, E=01111111 (127), M=0000000
        //
        // regroup logic:
        // Exp = 127 (0x7F)
        // SM = 0 | 0 = 0
        //
        // Input bytes (LE): 0x80, 0x3F

        let data = vec![0x80, 0x3F];
        let regroups = regroup_bytes(&data);
        assert_eq!(regroups, vec![0x7F, 0x00]);

        let restored = inverse_regroup_bytes(&regroups);
        assert_eq!(restored, data);

        // Test with -2.0
        // -2.0 in BF16: 0xC000
        // Binary: 1100 0000 0000 0000
        // S=1, E=10000000 (128), M=0000000
        //
        // regroup logic:
        // Exp = 128 (0x80)
        // SM = 1<<7 | 0 = 0x80

        let data2 = vec![0x00, 0xC0]; // LE
        let regroups2 = regroup_bytes(&data2);
        assert_eq!(regroups2, vec![0x80, 0x80]);

        let restored2 = inverse_regroup_bytes(&regroups2);
        assert_eq!(restored2, data2);
    }
}
