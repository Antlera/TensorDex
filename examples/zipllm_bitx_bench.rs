//! ZipLLM/BitX baseline throughput — runs the OFFICIAL BitX implementation
//! (vendored verbatim from github.com/ds2-lab/ZipLLM, Apache-2.0; see
//! third_party/zipllm_bitx/) on the same real model pair as
//! `make bench-table3-real`: tensor pairs driven in parallel as ZipLLM's
//! pipeline does (`bitx_compress` / `bitx_decompress`, each internally
//! parallel: rayon XOR/unzip + multithreaded zstd L3).
//!
//!     cargo run --release --example zipllm_bitx_bench -- \
//!         --base-model <dir-with-safetensors> --target-model <dir>
//!
//! Paper reference (ZipLLM's published figures): ~5.9 GB/s compress,
//! ~8.0 GB/s decompress.
use rayon::prelude::*;
use std::time::Instant;

#[path = "../third_party/zipllm_bitx/bitx_bytes.rs"]
mod bitx_bytes;
use bitx_bytes::{bitx_compress, bitx_decompress};

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let i = args.iter().position(|a| a == "--base-model").expect("--base-model DIR");
    let base_dir = args[i + 1].clone();
    let j = args.iter().position(|a| a == "--target-model").expect("--target-model DIR");
    let tgt_dir = args[j + 1].clone();

    eprintln!("loading {base_dir} …");
    let mut base = load_model(&base_dir);
    eprintln!("loading {tgt_dir} …");
    let tgt = load_model(&tgt_dir);
    let mut names: Vec<_> = tgt.keys().cloned().collect();
    names.sort();
    let mut pairs = Vec::new();
    for name in names {
        if let (Some(t), Some(b)) = (tgt.get(&name), base.get(&name)) {
            if t.len() == b.len() && t.len() >= 2 {
                pairs.push((t.clone(), base.remove(&name).unwrap()));
            }
        }
    }
    let n: usize = pairs.iter().map(|(t, _)| t.len()).sum();
    let gib = n as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "ZipLLM BitX (official implementation) — real model pair, {} MB, {} tensor pairs",
        n >> 20,
        pairs.len()
    );

    // Round-trip first, then timed passes (2 warmup + 5 measured, best/avg).
    let comp: Vec<(Vec<u8>, Vec<u8>)> = pairs
        .par_iter()
        .map(|(t, b)| bitx_compress(b, t))
        .collect();
    let comp_bytes: usize = comp.iter().map(|(e, s)| e.len() + s.len()).sum();
    let ok = pairs
        .par_iter()
        .zip(comp.par_iter())
        .all(|((t, b), (e, s))| bitx_decompress(b, e, s) == *t);
    assert!(ok, "official BitX round-trip mismatch");
    println!(
        "  reduction    {:.3}x ({:.1}% saved)   round-trip byte-exact ✅",
        comp_bytes as f64 / n as f64,
        (1.0 - comp_bytes as f64 / n as f64) * 100.0
    );

    let time_pass = |f: &dyn Fn()| {
        for _ in 0..2 { f(); }
        let (mut best, mut sum) = (f64::MAX, 0.0);
        for _ in 0..5 {
            let t0 = Instant::now();
            f();
            let dt = t0.elapsed().as_secs_f64();
            best = best.min(dt);
            sum += dt;
        }
        (gib * 5.0 / sum, gib / best)
    };
    let (avg, peak) = time_pass(&|| {
        pairs.par_iter().for_each(|(t, b)| { let _ = bitx_compress(b, t); });
    });
    println!("  compress     avg {avg:6.1} GB/s   peak {peak:6.1} GB/s");
    let (avg, peak) = time_pass(&|| {
        pairs.par_iter().zip(comp.par_iter()).for_each(|((_, b), (e, s))| {
            let _ = bitx_decompress(b, e, s);
        });
    });
    println!("  decompress   avg {avg:6.1} GB/s   peak {peak:6.1} GB/s");
    println!("\npaper reference (ZipLLM's published figures): 5.9 / 8.0 GB/s");
    println!("RESULT: PASS ✅  round-trip byte-exact; throughput measured");
}
