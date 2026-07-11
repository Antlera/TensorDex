use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// BitCountSketch — High-performance bit-level CountSketch for Hamming distance
// ============================================================================
//
// Algorithm:
//   Each u16 element is expanded to 16 bits. Standard CountSketch is applied
//   to the resulting binary vector {0,1}^(n*16).
//
// Key property: For binary {0,1} vectors, L2² = L1 = Hamming weight.
//   So L2²(sketch_A - sketch_B) estimates Hamming distance(A, B).
//
// Parameters:
//   d = number of independent hash rows (median reduces variance)
//   w = number of buckets per row (larger → lower variance per row)
//
// Typical config: d=7, w=512 → 7*512*8 = 28KB per sketch
//
// Composable: sketch each tensor independently, combine via subtraction.

/// SplitMix64 hash finalizer — used only for seed generation, not in hot path.
#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Fast multiply-shift hash for CountSketch bucket+sign assignment.
///
/// Theory: h(x) = (a·x) where a is a random odd constant provides
/// 2-wise independence via the upper bits (Dietzfelbinger et al. 1997).
/// This is provably sufficient for unbiased CountSketch estimation.
///
/// Cost: 1 multiply + 1 xor-shift ≈ 2 cycles (vs splitmix64's ~10 cycles).
#[inline(always)]
fn fast_hash(x: u64, seed: u64) -> u64 {
    let h = x.wrapping_mul(seed);
    h ^ (h >> 31)
}

// Note: BIT_XORS / BIT_MULTIPLIERS no longer needed.
// V8 collapsed algorithm derives per-bit signs from base_h bits directly.

/// Pre-computed per-row seed (odd constant for multiply-shift hash)
struct BcsRowSeed {
    seed: u64,
}

/// BitCountSketch: CountSketch on bit representation for Hamming distance estimation
///
/// Memory layout: table is a flat Vec<f64> of size d * w, row-major.
/// table[row * w + col] = accumulated signed bit values for (row, col).
pub struct BitCountSketch {
    pub d: usize,
    pub w: usize,
    pub n_elements: usize,
    pub n_bits: usize, // n_elements * 16
    pub table: Vec<f64>,
}

impl BitCountSketch {
    fn make_row_seeds(d: usize, base_seeds: &[u64]) -> Vec<BcsRowSeed> {
        assert!(base_seeds.len() >= 2 * d, "Need at least 2*d base seeds");
        (0..d)
            .map(|row| BcsRowSeed {
                seed: splitmix64(base_seeds[2 * row]) | 1,
            })
            .collect()
    }

    /// Create BitCountSketch from raw byte data (interpreted as u16 little-endian).
    ///
    /// Collapsed CountSketch with popcount: all 16 bits of the same element
    /// map to the SAME bucket; per-bit signs are derived from different bits
    /// of base_h. Net contribution = 2·popcount(val & sign_mask) − popcount(val).
    ///
    /// This is ~4× faster than per-bit hashing because it replaces the
    /// ~8-iteration bit-extraction loop + per-bit hash with 1 AND + 2 popcount
    /// + 1 table write per row.
    ///
    /// Math: diff_i = Σ_{b∈D_i} sign(b) → E[diff_i²] = |D_i| = Hamming(A_i,B_i)
    pub fn from_bytes(data: &[u8], d: usize, w: usize, base_seeds: &[u64]) -> Self {
        assert!(
            data.len() % 2 == 0,
            "Data length must be even (u16 elements)"
        );
        let n_elements = data.len() / 2;

        if n_elements >= 100_000 {
            return Self::from_bytes_parallel(data, d, w, base_seeds);
        }

        let n_bits = n_elements * 16;
        let row_seeds = Self::make_row_seeds(d, base_seeds);
        let w_mask = w - 1;
        debug_assert!(w.is_power_of_two(), "w must be a power of 2");

        let mut itable = vec![0i32; d * w];
        let data_ptr = data.as_ptr() as *const u16;
        let tbl = itable.as_mut_ptr();

        if d == 2 {
            let s0 = row_seeds[0].seed;
            let s1 = row_seeds[1].seed;
            for i in 0..n_elements {
                let val = unsafe { data_ptr.add(i).read_unaligned() };
                if val == 0 {
                    continue;
                }
                let idx = i as u64;
                let pc = val.count_ones() as i32;

                let bh0 = fast_hash(idx, s0);
                let net0 = 2 * (val & (bh0 >> 1) as u16).count_ones() as i32 - pc;
                unsafe {
                    *tbl.add(((bh0 >> 32) as usize) & w_mask) += net0;
                }

                let bh1 = fast_hash(idx, s1);
                let net1 = 2 * (val & (bh1 >> 1) as u16).count_ones() as i32 - pc;
                unsafe {
                    *tbl.add(w + (((bh1 >> 32) as usize) & w_mask)) += net1;
                }
            }
        } else {
            for i in 0..n_elements {
                let val = unsafe { data_ptr.add(i).read_unaligned() };
                if val == 0 {
                    continue;
                }
                let idx = i as u64;
                let pc = val.count_ones() as i32;
                for r in 0..d {
                    let bh = fast_hash(idx, row_seeds[r].seed);
                    let net = 2 * (val & (bh >> 1) as u16).count_ones() as i32 - pc;
                    unsafe {
                        *tbl.add(r * w + (((bh >> 32) as usize) & w_mask)) += net;
                    }
                }
            }
        }

        let table: Vec<f64> = itable.iter().map(|&v| v as f64).collect();
        BitCountSketch {
            d,
            w,
            n_elements,
            n_bits,
            table,
        }
    }

    /// Parallel construction for large tensors using rayon.
    /// Collapsed CountSketch with popcount, matching from_bytes.
    fn from_bytes_parallel(data: &[u8], d: usize, w: usize, base_seeds: &[u64]) -> Self {
        let n_elements = data.len() / 2;
        let n_bits = n_elements * 16;
        let row_seeds = Self::make_row_seeds(d, base_seeds);
        let w_mask = w - 1;
        debug_assert!(w.is_power_of_two(), "w must be a power of 2");

        let num_threads = rayon::current_num_threads();
        let chunk_size = (n_elements + num_threads - 1) / num_threads;

        let partial_tables: Vec<Vec<i32>> = (0..num_threads)
            .into_par_iter()
            .map(|t| {
                let mut lt = vec![0i32; d * w];
                let start = t * chunk_size;
                let end = std::cmp::min(start + chunk_size, n_elements);
                let data_ptr = data.as_ptr() as *const u16;
                let tbl = lt.as_mut_ptr();

                if d == 2 {
                    let s0 = row_seeds[0].seed;
                    let s1 = row_seeds[1].seed;
                    for i in start..end {
                        let val = unsafe { data_ptr.add(i).read_unaligned() };
                        if val == 0 {
                            continue;
                        }
                        let idx = i as u64;
                        let pc = val.count_ones() as i32;

                        let bh0 = fast_hash(idx, s0);
                        let net0 = 2 * (val & (bh0 >> 1) as u16).count_ones() as i32 - pc;
                        unsafe {
                            *tbl.add(((bh0 >> 32) as usize) & w_mask) += net0;
                        }

                        let bh1 = fast_hash(idx, s1);
                        let net1 = 2 * (val & (bh1 >> 1) as u16).count_ones() as i32 - pc;
                        unsafe {
                            *tbl.add(w + (((bh1 >> 32) as usize) & w_mask)) += net1;
                        }
                    }
                } else {
                    for i in start..end {
                        let val = unsafe { data_ptr.add(i).read_unaligned() };
                        if val == 0 {
                            continue;
                        }
                        let idx = i as u64;
                        let pc = val.count_ones() as i32;
                        for r in 0..d {
                            let bh = fast_hash(idx, row_seeds[r].seed);
                            let net = 2 * (val & (bh >> 1) as u16).count_ones() as i32 - pc;
                            unsafe {
                                *tbl.add(r * w + (((bh >> 32) as usize) & w_mask)) += net;
                            }
                        }
                    }
                }
                lt
            })
            .collect();

        // Merge partial tables (i32 → f64)
        let table_len = d * w;
        let mut table = vec![0.0f64; table_len];
        for local in &partial_tables {
            for i in 0..table_len {
                table[i] += local[i] as f64;
            }
        }

        BitCountSketch {
            d,
            w,
            n_elements,
            n_bits,
            table,
        }
    }

    /// Estimate Hamming distance between two BCS sketches.
    ///
    /// Returns median of per-row L2² estimates.
    /// The median provides robustness against outlier hash collisions.
    pub fn estimate_hamming(a: &Self, b: &Self) -> f64 {
        assert_eq!(a.d, b.d, "Sketch depth mismatch");
        assert_eq!(a.w, b.w, "Sketch width mismatch");

        let d = a.d;
        let w = a.w;
        let mut row_l2sq = Vec::with_capacity(d);

        for row in 0..d {
            let offset = row * w;
            let mut sum_sq = 0.0f64;
            for col in 0..w {
                let diff = a.table[offset + col] - b.table[offset + col];
                sum_sq += diff * diff;
            }
            row_l2sq.push(sum_sq);
        }

        // Median
        row_l2sq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if d % 2 == 0 {
            (row_l2sq[d / 2 - 1] + row_l2sq[d / 2]) / 2.0
        } else {
            row_l2sq[d / 2]
        }
    }

    /// Estimate normalized Hamming distance (fraction of differing bits).
    ///
    /// Result is in [0, 0.5] for typical data (p ≈ 0 means identical,
    /// p ≈ 0.5 means random/maximally different).
    pub fn estimate_norm_hamming(a: &Self, b: &Self) -> f64 {
        let hamming = Self::estimate_hamming(a, b);
        hamming / a.n_bits as f64
    }

    /// Memory footprint of the sketch in bytes
    pub fn size_bytes(&self) -> usize {
        // Header (d, w, n_elements, n_bits) + table data
        4 * 8 + self.d * self.w * 8
    }
}

// ============================================================================
// Entropy Prediction Model: Poly2 on 8·H(p)
// ============================================================================
//
// Theory (i.i.d. bit flips with probability p):
//   byte_entropy ≈ 8 · H(p) where H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
//
// In practice, BF16 bit flips are NOT perfectly i.i.d. (mantissa bits flip
// more than exponent/sign bits), so we use a quadratic correction:
//   entropy = a · [8H(p)]² + b · [8H(p)] + c
//
// This "Poly2 on 8H(p)" model achieves R² > 0.99 in large-scale validation.

/// Binary entropy function: H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
///
/// Returns 0 for p ≤ 0 or p ≥ 1 (boundary cases).
/// Maximum is H(0.5) = 1.0.
#[inline]
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 1e-15 || p >= 1.0 - 1e-15 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Predict byte entropy from normalized Hamming distance.
///
/// Model: entropy = coeffs[0]·t² + coeffs[1]·t + coeffs[2]
///        where t = 8·H(p) and p = norm_hamming
///
/// # Arguments
/// * `norm_hamming` - Normalized Hamming distance (bit flip rate p)
/// * `coeffs` - Polynomial coefficients [a, b, c] (highest degree first)
#[inline]
pub fn predict_entropy_poly2_8hp(norm_hamming: f64, coeffs: &[f64; 3]) -> f64 {
    let t = 8.0 * binary_entropy(norm_hamming);
    coeffs[0] * t * t + coeffs[1] * t + coeffs[2]
}

/// Fit Poly2 on 8H(p) model using least-squares regression.
///
/// Given (x_i, y_i) pairs where x = norm_hamming and y = entropy,
/// fits y = a·[8H(x)]² + b·[8H(x)] + c via normal equations.
///
/// Returns [a, b, c] coefficients (highest degree first).
pub fn fit_poly2_on_8hp(x: &[f64], y: &[f64]) -> [f64; 3] {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    // Transform x → t = 8·H(x)
    let t: Vec<f64> = x.iter().map(|&p| 8.0 * binary_entropy(p)).collect();

    // Build normal equations for [a, b, c]:
    // [ Σt⁴  Σt³  Σt² ] [a]   [Σt²y]
    // [ Σt³  Σt²  Σt  ] [b] = [Σty ]
    // [ Σt²  Σt   n   ] [c]   [Σy  ]
    let mut s = [0.0f64; 9]; // t^0 through t^4, plus cross terms
                             // s[0]=Σ1, s[1]=Σt, s[2]=Σt², s[3]=Σt³, s[4]=Σt⁴
                             // s[5]=Σy, s[6]=Σty, s[7]=Σt²y
    for i in 0..n {
        let ti = t[i];
        let yi = y[i];
        if !ti.is_finite() || !yi.is_finite() {
            continue;
        }
        let t2 = ti * ti;
        let t3 = t2 * ti;
        let t4 = t3 * ti;
        s[0] += 1.0;
        s[1] += ti;
        s[2] += t2;
        s[3] += t3;
        s[4] += t4;
        s[5] += yi;
        s[6] += ti * yi;
        s[7] += t2 * yi;
    }

    // Solve 3x3 system using Cramer's rule
    let a11 = s[4];
    let a12 = s[3];
    let a13 = s[2];
    let a21 = s[3];
    let a22 = s[2];
    let a23 = s[1];
    let a31 = s[2];
    let a32 = s[1];
    let a33 = s[0];
    let b1 = s[7];
    let b2 = s[6];
    let b3 = s[5];

    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);

    if det.abs() < 1e-30 {
        return [0.0, 1.0, 0.0]; // Fallback: identity-like
    }

    let det_a =
        b1 * (a22 * a33 - a23 * a32) - a12 * (b2 * a33 - a23 * b3) + a13 * (b2 * a32 - a22 * b3);

    let det_b =
        a11 * (b2 * a33 - a23 * b3) - b1 * (a21 * a33 - a23 * a31) + a13 * (a21 * b3 - b2 * a31);

    let det_c =
        a11 * (a22 * b3 - b2 * a32) - a12 * (a21 * b3 - b2 * a31) + b1 * (a21 * a32 - a22 * a31);

    [det_a / det, det_b / det, det_c / det]
}

// ============================================================================
// Utility functions
// ============================================================================

/// Shannon byte entropy (bits per byte) from raw data.
///
/// Counts byte frequencies and computes H = -Σ p_i·log₂(p_i).
/// Returns value in [0, 8].
pub fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Exact Hamming distance between two byte slices (interpreted as u16 LE).
///
/// Returns total number of differing bits across all u16 elements.
pub fn exact_hamming_u16(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");
    assert!(a.len() % 2 == 0, "Length must be even (u16 elements)");
    let n = a.len() / 2;
    let a_ptr = a.as_ptr() as *const u16;
    let b_ptr = b.as_ptr() as *const u16;
    let mut total = 0u64;
    for i in 0..n {
        let va = unsafe { a_ptr.add(i).read_unaligned() };
        let vb = unsafe { b_ptr.add(i).read_unaligned() };
        total += (va ^ vb).count_ones() as u64;
    }
    total
}

/// Exact Hamming distance with parallel computation for large data.
pub fn exact_hamming_u16_parallel(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 2 == 0);
    let n = a.len() / 2;

    if n < 500_000 {
        return exact_hamming_u16(a, b);
    }

    // Process in parallel chunks using 64-bit XOR + popcount
    let n8 = a.len() / 8;
    let chunk_size = 65536usize;
    let n_chunks = (n8 + chunk_size - 1) / chunk_size;

    let total: u64 = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk_size;
            let end = std::cmp::min(start + chunk_size, n8);
            let a_ptr = a.as_ptr() as *const u64;
            let b_ptr = b.as_ptr() as *const u64;
            let mut subtotal = 0u64;
            for i in start..end {
                let va = unsafe { a_ptr.add(i).read_unaligned() };
                let vb = unsafe { b_ptr.add(i).read_unaligned() };
                subtotal += (va ^ vb).count_ones() as u64;
            }
            subtotal
        })
        .sum();

    // Handle remaining bytes
    let processed = n8 * 8;
    let remaining = a.len() - processed;
    if remaining >= 2 {
        let n_rem = remaining / 2;
        let a_rem =
            unsafe { std::slice::from_raw_parts(a.as_ptr().add(processed) as *const u16, n_rem) };
        let b_rem =
            unsafe { std::slice::from_raw_parts(b.as_ptr().add(processed) as *const u16, n_rem) };
        let rem_total: u64 = a_rem
            .iter()
            .zip(b_rem.iter())
            .map(|(&va, &vb)| (va ^ vb).count_ones() as u64)
            .sum();
        total + rem_total
    } else {
        total
    }
}

/// Compute evaluation metrics: R², MAE, Spearman correlation
pub struct EvalMetrics {
    pub n: usize,
    pub r2: f64,
    pub mae: f64,
    pub spearman: f64,
    pub mape: f64,
}

impl EvalMetrics {
    /// Compute metrics from actual and predicted arrays
    pub fn compute(actual: &[f64], predicted: &[f64]) -> Self {
        assert_eq!(actual.len(), predicted.len());

        // Filter out non-finite values
        let pairs: Vec<(f64, f64)> = actual
            .iter()
            .zip(predicted.iter())
            .filter(|(a, p)| a.is_finite() && p.is_finite())
            .map(|(&a, &p)| (a, p))
            .collect();

        let n = pairs.len();
        if n < 3 {
            return EvalMetrics {
                n: 0,
                r2: f64::NAN,
                mae: f64::NAN,
                spearman: f64::NAN,
                mape: f64::NAN,
            };
        }

        // MAE
        let mae: f64 = pairs.iter().map(|(a, p)| (a - p).abs()).sum::<f64>() / n as f64;

        // R²
        let mean_a = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
        let ss_res: f64 = pairs.iter().map(|(a, p)| (a - p).powi(2)).sum();
        let ss_tot: f64 = pairs.iter().map(|(a, _)| (a - mean_a).powi(2)).sum();
        let r2 = if ss_tot > 1e-15 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        // Spearman correlation (rank-based)
        let spearman = spearman_correlation(&pairs);

        // MAPE (only for actual > 0.1)
        let nz_pairs: Vec<_> = pairs.iter().filter(|(a, _)| *a > 0.1).collect();
        let mape = if !nz_pairs.is_empty() {
            nz_pairs.iter().map(|(a, p)| (a - p).abs() / a).sum::<f64>() / nz_pairs.len() as f64
                * 100.0
        } else {
            f64::NAN
        };

        EvalMetrics {
            n,
            r2,
            mae,
            spearman,
            mape,
        }
    }
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={}, R²={:.4}, MAE={:.4}, ρ={:.4}, MAPE={:.1}%",
            self.n, self.r2, self.mae, self.spearman, self.mape
        )
    }
}

/// Compute Spearman rank correlation coefficient
fn spearman_correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    if n < 3 {
        return f64::NAN;
    }

    let ranks_a = compute_ranks(&pairs.iter().map(|(a, _)| *a).collect::<Vec<_>>());
    let ranks_b = compute_ranks(&pairs.iter().map(|(_, p)| *p).collect::<Vec<_>>());

    // Pearson correlation on ranks
    let mean_ra = ranks_a.iter().sum::<f64>() / n as f64;
    let mean_rb = ranks_b.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for i in 0..n {
        let da = ranks_a[i] - mean_ra;
        let db = ranks_b[i] - mean_rb;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// Compute fractional ranks (average rank for ties)
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ============================================================================
// XOR delta byte entropy (matches delta_entropy.rs)
// ============================================================================

/// Compute XOR delta byte entropy between two byte slices.
/// This is the ground truth that the BCS model predicts.
pub fn xor_byte_entropy(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    // Process 8 bytes at a time
    let n8 = a.len() / 8;
    let a64 = a.as_ptr() as *const u64;
    let b64 = b.as_ptr() as *const u64;

    for i in 0..n8 {
        let xor = unsafe { a64.add(i).read_unaligned() ^ b64.add(i).read_unaligned() };
        let bytes = xor.to_le_bytes();
        for &byte in &bytes {
            counts[byte as usize] += 1;
        }
    }

    // Handle remaining bytes
    for i in (n8 * 8)..a.len() {
        counts[(a[i] ^ b[i]) as usize] += 1;
    }

    let total = a.len() as f64;
    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute SUB delta byte entropy between two byte slices (u16 LE elements).
/// target - base as i16 wrapping subtraction, then compute byte entropy.
pub fn sub_byte_entropy_u16(target: &[u8], base: &[u8]) -> f64 {
    assert_eq!(target.len(), base.len());
    assert!(target.len() % 2 == 0);
    let n = target.len() / 2;
    if n == 0 {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    let t_ptr = target.as_ptr() as *const u16;
    let b_ptr = base.as_ptr() as *const u16;

    for i in 0..n {
        let tv = unsafe { t_ptr.add(i).read_unaligned() };
        let bv = unsafe { b_ptr.add(i).read_unaligned() };
        let delta = tv.wrapping_sub(bv);
        let bytes = delta.to_le_bytes();
        counts[bytes[0] as usize] += 1;
        counts[bytes[1] as usize] += 1;
    }

    let total = target.len() as f64; // 2 bytes per element
    let mut entropy = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

// ============================================================================
// Python binding for BCS fingerprint computation
// ============================================================================

/// Default BCS seeds used across the codebase (d_max=16, 2*d_max seeds).
/// Generated by LCG chain starting from seed=42, matching bcs_bench.rs.
pub const BCS_DEFAULT_SEEDS: [u64; 32] = {
    let mut seeds = [0u64; 32];
    let mut state: u64 = 42;
    let mut i = 0;
    while i < 32 {
        // LCG: same as generate_bcs_seeds() in bcs_bench.rs
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seeds[i] = state;
        i += 1;
    }
    seeds
};

/// Compute BCS (BitCountSketch) fingerprint from raw tensor bytes.
///
/// Takes raw bytes (interpreted as u16 little-endian), computes a BCS
/// fingerprint with the given d and w, and returns it as an int32 array
/// (table values cast from f64 to i32).
///
/// Returns: numpy int32 array of shape (d * w,)
#[pyfunction]
#[pyo3(signature = (data, d=2, w=1024))]
pub fn compute_bcs_fingerprint_py(
    py: Python,
    data: &[u8],
    d: usize,
    w: usize,
) -> PyResult<Py<PyArray1<i32>>> {
    let sk = BitCountSketch::from_bytes(data, d, w, &BCS_DEFAULT_SEEDS);

    // Cast f64 table to i32 (values are integer-like: sums of +1/-1)
    let int_table: Vec<i32> = sk.table.iter().map(|&v| v.round() as i32).collect();
    Ok(PyArray1::from_vec(py, int_table).to_owned())
}

/// Zero-copy variant: accepts a numpy uint16 array directly, avoiding
/// the Python-side .tobytes() allocation (~60ms saved per 117MB tensor).
#[pyfunction]
#[pyo3(signature = (data, d=2, w=1024))]
pub fn compute_bcs_fingerprint_u16_py(
    py: Python,
    data: numpy::PyReadonlyArray1<u16>,
    d: usize,
    w: usize,
) -> PyResult<Py<PyArray1<i32>>> {
    let slice = data.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array must be contiguous: {}", e))
    })?;
    let byte_slice =
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 2) };
    let sk = BitCountSketch::from_bytes(byte_slice, d, w, &BCS_DEFAULT_SEEDS);
    let int_table: Vec<i32> = sk.table.iter().map(|&v| v.round() as i32).collect();
    Ok(PyArray1::from_vec(py, int_table).to_owned())
}

/// Compute BCS fingerprints for multiple tensors in batch.
///
/// Takes a list of (tensor_id, raw_bytes) tuples and returns a list of
/// (tensor_id, int32_fingerprint_bytes) tuples.
#[pyfunction]
#[pyo3(signature = (items, d=2, w=1024))]
pub fn compute_bcs_fingerprints_batch_py(
    py: Python,
    items: Vec<(String, Vec<u8>)>,
    d: usize,
    w: usize,
) -> PyResult<Vec<(String, Py<PyArray1<i32>>)>> {
    let results: Vec<(String, Vec<i32>)> = items
        .into_par_iter()
        .map(|(tid, data)| {
            let sk = BitCountSketch::from_bytes(&data, d, w, &BCS_DEFAULT_SEEDS);
            let int_table: Vec<i32> = sk.table.iter().map(|&v| v.round() as i32).collect();
            (tid, int_table)
        })
        .collect();

    let py_results: Vec<(String, Py<PyArray1<i32>>)> = results
        .into_iter()
        .map(|(tid, table)| {
            let arr = PyArray1::from_vec(py, table).to_owned();
            (tid, arr)
        })
        .collect();

    Ok(py_results)
}
