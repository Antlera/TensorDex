//! Table 3 / Fig 1-right — codec throughput benchmark (in-memory, CPU-only).
//!
//! Measures pure codec compute the way the paper did: (target, base) tensor
//! pairs are split into 2 MB sub-chunks and every chunk is compressed /
//! decompressed in parallel across all cores with the *shipped* kernels
//! (zstd level 1), outputs dropped as they complete.
//!
//! Two modes:
//!
//!   synthetic (no download):
//!       make bench-table3
//!       cargo run --release --example table3_bench -- [size_mb] [delta_range]
//!
//!   real model pair — THE PAPER'S TABLE-3 SETUP (Qwen2.5-7B vs -Instruct):
//!       make bench-table3-real        # downloads the two models, then runs:
//!       cargo run --release --example table3_bench -- \
//!           --base-model <dir-with-safetensors> --target-model <dir>
//!
//! Paper reference on a c6a.48xlarge (192 vCPU), real Qwen2.5-7B pair:
//! TensorX ~22.9 GB/s compress · ~28.4 GB/s decompress at 59.4% reduction.
//! Numbers are hardware-dependent; the round-trip check must PASS everywhere.
use rayon::prelude::*;
use std::time::Instant;
use tensordex_ops::kernels::tensorx::{compress_tensorx, decompress_tensorx};

const CHUNK: usize = 2 * 1024 * 1024; // sub-chunk size, mirrors the paper bench
const ITERS: usize = 5;
const WARMUP: usize = 2;
const LEVEL: i32 = 1; // TensorX zstd level — fixed constant of the published trace

fn synth(n: usize, delta_range: u32) -> (Vec<u8>, Vec<u8>) {
    let mut base = vec![0u8; n];
    let mut tgt = vec![0u8; n];
    let mut x: u32 = 12345;
    for i in (0..n).step_by(2) {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        let b = (x >> 8) as u16;
        let d = (x % (2 * delta_range + 1)) as i16 - delta_range as i16;
        let t = (b as i16).wrapping_add(d) as u16;
        base[i] = b as u8;
        base[i + 1] = (b >> 8) as u8;
        tgt[i] = t as u8;
        tgt[i + 1] = (t >> 8) as u8;
    }
    (tgt, base)
}

/// name -> raw little-endian bytes for every 2-byte-dtype tensor in a
/// directory of .safetensors files.
fn load_model(dir: &str) -> std::collections::HashMap<String, Vec<u8>> {
    let mut out = std::collections::HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .safetensors files in {dir}");
    for f in files {
        let buf = std::fs::read(&f).unwrap();
        let st = safetensors::SafeTensors::deserialize(&buf).unwrap();
        for (name, view) in st.tensors() {
            use safetensors::Dtype::*;
            if matches!(view.dtype(), BF16 | F16 | I16 | U16) {
                out.insert(name.to_string(), view.data().to_vec());
            }
        }
    }
    out
}

fn bench<F: Fn() + Sync>(name: &str, gib: f64, f: F) {
    for _ in 0..WARMUP {
        f();
    }
    let mut best = f64::MAX;
    let mut sum = 0.0;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        sum += dt;
    }
    println!(
        "  {:<12} avg {:6.1} GB/s   peak {:6.1} GB/s",
        name,
        gib * ITERS as f64 / sum,
        gib / best
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Gather (target, base) byte buffers for either mode.
    let (pairs, label): (Vec<(Vec<u8>, Vec<u8>)>, String) =
        if let Some(i) = args.iter().position(|a| a == "--base-model") {
            let base_dir = args.get(i + 1).expect("--base-model DIR").clone();
            let j = args
                .iter()
                .position(|a| a == "--target-model")
                .expect("--target-model DIR required with --base-model");
            let tgt_dir = args.get(j + 1).expect("--target-model DIR").clone();
            eprintln!("loading {base_dir} …");
            let mut base = load_model(&base_dir);
            eprintln!("loading {tgt_dir} …");
            let tgt = load_model(&tgt_dir);
            let mut pairs = Vec::new();
            let mut names: Vec<_> = tgt.keys().cloned().collect();
            names.sort();
            for name in names {
                if let (Some(t), Some(b)) = (tgt.get(&name), base.get(&name)) {
                    if t.len() == b.len() && !t.is_empty() {
                        pairs.push((t.clone(), base.remove(&name).unwrap()));
                    }
                }
            }
            (pairs, "real model pair".to_string())
        } else {
            let size_mb: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(4096);
            let delta: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
            let (t, b) = synth(size_mb * 1024 * 1024, delta);
            (vec![(t, b)], format!("synthetic, delta ±{delta}"))
        };

    // Split every pair into 2 MB sub-chunks (the unit of parallelism).
    let ranges: Vec<(&[u8], &[u8])> = pairs
        .iter()
        .flat_map(|(t, b)| {
            (0..t.len()).step_by(CHUNK).map(move |s| {
                let e = (s + CHUNK).min(t.len());
                (&t[s..e], &b[s..e])
            })
        })
        .collect();
    let n: usize = ranges.iter().map(|(t, _)| t.len()).sum();
    let gib = n as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "TensorX codec throughput — {} ({} MB across {} chunks, {} threads, zstd L{})",
        label,
        n >> 20,
        ranges.len(),
        rayon::current_num_threads(),
        LEVEL
    );

    // Round-trip integrity first (timing means nothing without it).
    let comp: Vec<Vec<u8>> = ranges
        .par_iter()
        .map(|&(t, b)| compress_tensorx(t, b, 2, LEVEL).unwrap())
        .collect();
    let comp_bytes: usize = comp.iter().map(|c| c.len()).sum();
    let ok = ranges
        .par_iter()
        .zip(comp.par_iter())
        .all(|(&(t, b), c)| decompress_tensorx(c, b, 2).unwrap() == t);
    assert!(ok, "round-trip mismatch");
    println!(
        "  reduction    {:.3}x ({:.1}% saved)   round-trip byte-exact ✅",
        comp_bytes as f64 / n as f64,
        (1.0 - comp_bytes as f64 / n as f64) * 100.0
    );

    bench("compress", gib, || {
        ranges.par_iter().for_each(|&(t, b)| {
            let _ = compress_tensorx(t, b, 2, LEVEL).unwrap();
        })
    });
    bench("decompress", gib, || {
        ranges.par_iter().zip(comp.par_iter()).for_each(|(&(_, b), c)| {
            let _ = decompress_tensorx(c, b, 2).unwrap();
        })
    });

    println!(
        "\npaper reference (c6a.48xlarge, 192 vCPU, real Qwen2.5-7B pair): \
         22.9 / 28.4 GB/s @ 59.4%"
    );
    println!(
        "baselines: make bench-baselines re-runs ZipNN, OpenZL, and ZipLLM's \
         official\nBitX implementation; FM-Delta cites its paper (0.1/0.1 GB/s)"
    );
    println!("RESULT: PASS ✅  round-trip byte-exact; throughput measured");
}
