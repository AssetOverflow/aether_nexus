//! Build script for nexus-core
//!
//! Compiles Metal shaders:
//!   - weaver_kernel.metal → weaver.metallib (Weaver decode kernel)
//!   - ops.metal → ops.metallib (transformer ops)
//!
//! Resulting metallib paths are exposed via environment variables
//! for runtime loading.

use std::env;
use std::process::Command;

/// Compile a Metal shader: MSL → AIR → metallib
fn compile_metal_shader(source: &str, name: &str, out_dir: &str) -> Option<String> {
    let air = format!("{}/{}.air", out_dir, name);
    let metallib = format!("{}/{}.metallib", out_dir, name);

    // Step 1: Compile MSL → AIR
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal",
               "-c", source,
               "-o", &air,
               "-std=metal3.0",
               "-O2"])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning={} shader compiled successfully", name);
        }
        Ok(s) => {
            println!("cargo:warning={} shader compilation failed (exit: {:?})", name, s.code());
            return None;
        }
        Err(e) => {
            println!("cargo:warning=xcrun not found: {}. Skipping {} shader.", e, name);
            return None;
        }
    }

    // Step 2: Link AIR → metallib
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib",
               &air, "-o", &metallib])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning={} library linked successfully", name);
            Some(metallib)
        }
        Ok(s) => {
            println!("cargo:warning={} metallib linking failed (exit: {:?})", name, s.code());
            None
        }
        Err(e) => {
            println!("cargo:warning={} metallib linking error: {}", name, e);
            None
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/weaver_kernel.metal");
    println!("cargo:rerun-if-changed=src/ops.metal");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Compile Weaver decode kernel
    if let Some(path) = compile_metal_shader("src/weaver_kernel.metal", "weaver", &out_dir) {
        println!("cargo:rustc-env=WEAVER_METALLIB={}", path);
    }

    // Compile transformer ops kernels
    if let Some(path) = compile_metal_shader("src/ops.metal", "ops", &out_dir) {
        println!("cargo:rustc-env=OPS_METALLIB={}", path);
    }
}
