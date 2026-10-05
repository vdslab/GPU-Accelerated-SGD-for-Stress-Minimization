use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=cuda/bfs.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let home = PathBuf::from(env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into()));
    let nvcc = env::var("NVCC").unwrap_or_else(|_| home.join("bin/nvcc").display().to_string());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("bfs.o");
    let status = Command::new(&nvcc)
        .args([
            "-O3",
            "-std=c++14",
            "-Xcompiler",
            "-fPIC",
            "-c",
            "cuda/bfs.cu",
            "-o",
        ])
        .arg(&object)
        .status()
        .expect("CUDA feature requires nvcc; set CUDA_HOME or NVCC");
    assert!(status.success(), "nvcc compilation failed");
    assert!(Command::new("ar")
        .arg("crs")
        .arg(out.join("libpivot_bfs.a"))
        .arg(object)
        .status()
        .unwrap()
        .success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!(
        "cargo:rustc-link-search=native={}",
        home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=static=pivot_bfs");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
