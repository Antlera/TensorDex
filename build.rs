use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=Cargo.toml");

    // Strip symbols on release builds.
    if env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-link-arg=-Wl,-s");
    }

    // FM++ codec: link the vendored FM-Delta static lib. Only when the `fmpp`
    // feature is on, so the default build needs neither the lib nor a C++ toolchain.
    if env::var("CARGO_FEATURE_FMPP").is_ok() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("third_party/fmdelta");
        let lib = dir.join("libfmdelta.a");
        if !lib.exists() {
            panic!(
                "feature `fmpp` needs {}; it is a prebuilt x86_64-linux static lib \
                 of FM-Delta — see third_party/fmdelta/README.md to obtain/rebuild it.",
                lib.display()
            );
        }
        println!("cargo:rerun-if-changed={}", lib.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=fmdelta");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
