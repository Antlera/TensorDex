//! FM++ codec — reduction-oriented delta compression, extended from FM-Delta [71].
//!
//! Thin FFI over the FM-Delta arithmetic residual coder (vendored prebuilt at
//! `third_party/fmdelta/libfmdelta.a`; built only under the `fmpp` cargo
//! feature). Treats the tensor as a 1-D HALF array (`type_ = 2`), matching the
//! encode path that produced the published `fratio` / `fbytes_out` columns — so
//! `compress_fmpp_rust` reproduces them bit-for-bit.

use pyo3::prelude::*;

#[repr(C)]
struct FMD {
    type_: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
}

extern "C" {
    fn fmd_write_to_buffer(buffer: *mut u8, size: usize) -> *mut FMD;
    fn fmd_write_header(fmd: *mut FMD) -> i32;
    fn fmd_write(fmd: *mut FMD, base_data: *const u8, finetuned_data: *const u8) -> usize;
    fn fmd_write_close(fmd: *mut FMD);
}

/// Encode `target` as an FM++ delta against `base`; returns the compressed bytes.
/// `item_size` is the element width in bytes (2 for bf16/fp16).
#[pyfunction]
#[pyo3(signature = (target, base, item_size=2))]
pub fn compress_fmpp_rust(
    py: Python,
    target: &[u8],
    base: &[u8],
    item_size: usize,
) -> PyResult<PyObject> {
    if base.len() != target.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fmpp: base and target must have equal byte length",
        ));
    }
    let n_elements = base.len() / item_size;
    let buf_size = n_elements * item_size + 28 + 1024; // header + slack
    let mut buffer = vec![0u8; buf_size];

    let out = unsafe {
        let p = fmd_write_to_buffer(buffer.as_mut_ptr(), buf_size);
        if p.is_null() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "fmpp: fmd_write_to_buffer returned null",
            ));
        }
        (*p).type_ = 2; // FMD_TYPE_HALF
        (*p).nx = n_elements as i32;
        (*p).ny = 1;
        (*p).nz = 1;
        (*p).nf = 1;
        if fmd_write_header(p) == 0 {
            fmd_write_close(p);
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "fmpp: fmd_write_header failed",
            ));
        }
        let n = fmd_write(p, base.as_ptr(), target.as_ptr());
        fmd_write_close(p);
        n
    };
    Ok(pyo3::types::PyBytes::new(py, &buffer[..out]).to_object(py))
}
