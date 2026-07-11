//! ZigZag Encoding Utilities
//!
//! Efficiently maps signed integers to unsigned integers so that numbers with small absolute value
//! (positive or negative) are mapped to small unsigned integers.

/// ZigZag encode a signed 16-bit integer to unsigned.
#[inline(always)]
pub fn zigzag_encode_i16(n: i16) -> u16 {
    ((n << 1) ^ (n >> 15)) as u16
}

/// ZigZag decode an unsigned 16-bit integer back to signed.
#[inline(always)]
pub fn zigzag_decode_i16(n: u16) -> i16 {
    ((n >> 1) as i16) ^ (-((n & 1) as i16))
}

/// ZigZag encode a signed 32-bit integer to unsigned.
#[inline(always)]
pub fn zigzag_encode_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// ZigZag decode an unsigned 32-bit integer back to signed.
#[inline(always)]
pub fn zigzag_decode_i32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Batch ZigZag encode i16 array in-place (as u16)
pub fn zigzag_encode_i16_batch(data: &mut [u8]) {
    let len = data.len() / 2;
    for i in 0..len {
        let val = i16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let encoded = zigzag_encode_i16(val);
        let bytes = encoded.to_le_bytes();
        data[i * 2] = bytes[0];
        data[i * 2 + 1] = bytes[1];
    }
}

/// Batch ZigZag decode u16 array in-place (as i16)
pub fn zigzag_decode_i16_batch(data: &mut [u8]) {
    let len = data.len() / 2;
    for i in 0..len {
        let val = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        let decoded = zigzag_decode_i16(val);
        let bytes = decoded.to_le_bytes();
        data[i * 2] = bytes[0];
        data[i * 2 + 1] = bytes[1];
    }
}

/// Batch ZigZag encode i32 array in-place (as u32)
pub fn zigzag_encode_i32_batch(data: &mut [u8]) {
    let len = data.len() / 4;
    for i in 0..len {
        let val = i32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
        let encoded = zigzag_encode_i32(val);
        let bytes = encoded.to_le_bytes();
        data[i * 4] = bytes[0];
        data[i * 4 + 1] = bytes[1];
        data[i * 4 + 2] = bytes[2];
        data[i * 4 + 3] = bytes[3];
    }
}

/// Batch ZigZag decode u32 array in-place (as i32)
pub fn zigzag_decode_i32_batch(data: &mut [u8]) {
    let len = data.len() / 4;
    for i in 0..len {
        let val = u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]);
        let decoded = zigzag_decode_i32(val);
        let bytes = decoded.to_le_bytes();
        data[i * 4] = bytes[0];
        data[i * 4 + 1] = bytes[1];
        data[i * 4 + 2] = bytes[2];
        data[i * 4 + 3] = bytes[3];
    }
}
