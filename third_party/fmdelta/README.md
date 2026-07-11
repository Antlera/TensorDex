# Vendored FM-Delta (for the FM++ codec)

`libfmdelta.a` is a **prebuilt static library** of **FM-Delta** — the lossless
delta coder that TensorDex's **FM++** codec extends (paper ref *[71]*, Ning et
al., *FM-Delta: Lossless Compression for Storing Massive Fine-tuned Foundation
Models*, NeurIPS 2024). It is built into the extension only when you pass
`--features fmpp`; the default build does not use it.

- **Platform:** compiled for **x86_64-linux** (the paper's eval platform,
  c6a.48xlarge). On other platforms, rebuild from source (below).
- **Why prebuilt:** FM++ reproduces the published `fratio`/`fbytes_out` columns
  bit-for-bit; shipping the verified object avoids every reviewer needing to
  fetch and build a C++ dependency. It is `type_ = 2` (HALF), 1-D — see
  `src/rust/kernels/fmpp.rs`.

## ⚠️ Licensing

FM-Delta is third-party. Before redistributing this `.a` in a public repo,
confirm FM-Delta's license permits it. If it does not, **remove the `.a`** and
have users build it from source (below) — the `fmpp` feature and `fmpp.rs` FFI
work unchanged against a locally built `libfmdelta.a`.

## Rebuild from source

Obtain the FM-Delta source (`package/src` + `package/include` with
`write.cpp read.cpp rcencoder.cpp rcdecoder.cpp rcqsmodel.cpp error.cpp` and the
`fmd_*` FFI), then:

```bash
c++ -O3 -std=c++17 -DNDEBUG -Ipackage/include -c package/src/*.cpp
ar rcs libfmdelta.a *.o
cp libfmdelta.a third_party/fmdelta/libfmdelta.a
```

The FFI TensorDex relies on: `fmd_write_to_buffer`, `fmd_write_header`,
`fmd_write`, `fmd_write_close` (declared in `src/rust/kernels/fmpp.rs`).
