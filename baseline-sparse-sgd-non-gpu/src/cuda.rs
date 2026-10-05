use crate::preprocessing::{Csr, DistanceBackend};
use anyhow::{bail, Result};
use std::{
    ffi::{c_char, c_void, CStr},
    ptr::NonNull,
};
extern "C" {
    fn pivot_bfs_create(
        n: u32,
        edges: u32,
        rows: *const u32,
        cols: *const u32,
        ctx: *mut *mut c_void,
        transfer: *mut f64,
    ) -> i32;
    fn pivot_bfs_run(ctx: *mut c_void, source: u32, output: *mut u32, transfer: *mut f64) -> i32;
    fn pivot_bfs_destroy(ctx: *mut c_void);
    fn pivot_bfs_error(code: i32) -> *const c_char;
}
fn check(code: i32) -> Result<()> {
    if code != 0 {
        bail!(
            "CUDA BFS: {}",
            unsafe { CStr::from_ptr(pivot_bfs_error(code)) }.to_string_lossy()
        );
    }
    Ok(())
}
pub struct CudaBfs {
    ctx: NonNull<c_void>,
    n: usize,
    transfer: f64,
}
impl CudaBfs {
    pub fn new(csr: &Csr) -> Result<Self> {
        csr.validate()?;
        let mut ctx = std::ptr::null_mut();
        let mut transfer = 0.0;
        check(unsafe {
            pivot_bfs_create(
                (csr.rows.len() - 1) as u32,
                csr.columns.len() as u32,
                csr.rows.as_ptr(),
                csr.columns.as_ptr(),
                &mut ctx,
                &mut transfer,
            )
        })?;
        Ok(Self {
            ctx: NonNull::new(ctx).expect("successful CUDA allocation"),
            n: csr.rows.len() - 1,
            transfer,
        })
    }
}
impl DistanceBackend for CudaBfs {
    fn distances(&mut self, source: usize) -> Result<Vec<u32>> {
        if source >= self.n {
            bail!("source out of range");
        }
        let mut row = vec![u32::MAX; self.n];
        let mut transfer = 0.0;
        check(unsafe {
            pivot_bfs_run(
                self.ctx.as_ptr(),
                source as u32,
                row.as_mut_ptr(),
                &mut transfer,
            )
        })?;
        self.transfer += transfer;
        Ok(row)
    }
    fn transfer_seconds(&self) -> f64 {
        self.transfer
    }
}
impl Drop for CudaBfs {
    fn drop(&mut self) {
        unsafe {
            pivot_bfs_destroy(self.ctx.as_ptr());
        }
    }
}
