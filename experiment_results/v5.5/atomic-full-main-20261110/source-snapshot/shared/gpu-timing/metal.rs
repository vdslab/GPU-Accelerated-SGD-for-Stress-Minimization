#![allow(unexpected_cfgs)] // objc 0.2 macros test the historical cargo-clippy feature.
use metal::CommandBufferRef;
use objc::{msg_send, sel, sel_impl};
use std::time::Duration;

/// A command contains one compute encoder only. Read timestamps after completion.
/// https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime
pub fn command_time(command: &CommandBufferRef) -> Option<Duration> {
    if std::env::var("SGD_GPU_TIMESTAMPS").as_deref() == Ok("0") {
        return None;
    }
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    (start.is_finite() && end.is_finite() && start > 0.0 && end >= start)
        .then(|| Duration::from_secs_f64(end - start))
}
