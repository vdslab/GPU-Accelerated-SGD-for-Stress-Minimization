//! Pass timestamps, resolved after the final coordinate readback. Query sets are
//! paged to respect WebGPU's 8192-query limit without changing SGD submissions.
use std::time::Duration;

pub struct PassTimer {
    pages: Vec<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, u32)>,
    used: usize,
    ends: Vec<usize>,
}
impl PassTimer {
    pub fn new(device: &wgpu::Device, passes: usize) -> Self {
        let mut pages = Vec::new();
        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            for start in (0..passes * 2).step_by(8192) {
                let count = (passes * 2 - start).min(8192) as u32;
                let query = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("SGD pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count,
                });
                let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp resolve"),
                    size: count as u64 * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp readback"),
                    size: count as u64 * 8,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                pages.push((query, resolve, read, count));
            }
        }
        Self {
            pages,
            used: 0,
            ends: Vec::new(),
        }
    }
    pub fn next(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if self.pages.is_empty() {
            return None;
        }
        let index = self.used * 2;
        self.used += 1;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.pages[index / 8192].0,
            beginning_of_pass_write_index: Some((index % 8192) as u32),
            end_of_pass_write_index: Some((index % 8192 + 1) as u32),
        })
    }
    pub fn end_iteration(&mut self) {
        self.ends.push(self.used);
    }
    pub fn finish(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> anyhow::Result<Option<Vec<Duration>>> {
        if self.pages.is_empty() {
            return Ok(None);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resolve timing"),
        });
        for (i, (query, resolve, read, capacity)) in self.pages.iter().enumerate() {
            let count = (self.used * 2)
                .saturating_sub(i * 8192)
                .min(*capacity as usize) as u32;
            if count > 0 {
                encoder.resolve_query_set(query, 0..count, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, read, 0, count as u64 * 8);
            }
        }
        queue.submit([encoder.finish()]);
        for (_, _, read, _) in &self.pages {
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let mut durations = Vec::with_capacity(self.used);
        for (i, (_, _, read, capacity)) in self.pages.iter().enumerate() {
            let count = (self.used * 2)
                .saturating_sub(i * 8192)
                .min(*capacity as usize);
            let mapped = read.slice(..).get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&mapped);
            for pair in ticks[..count].chunks_exact(2) {
                anyhow::ensure!(pair[1] >= pair[0], "Invalid GPU timestamps");
                durations.push(Duration::from_secs_f64(
                    (pair[1] - pair[0]) as f64 * queue.get_timestamp_period() as f64 * 1e-9,
                ));
            }
            drop(mapped);
            read.unmap();
        }
        let mut start = 0;
        Ok(Some(
            self.ends
                .into_iter()
                .map(|end| {
                    let duration = durations[start..end].iter().copied().sum();
                    start = end;
                    duration
                })
                .collect(),
        ))
    }
}

/// Allocate first, upload only after pipelines and all resources are ready.
pub fn buffer(
    device: &wgpu::Device,
    label: Option<&str>,
    size: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size: size.max(4).div_ceil(4) as u64 * 4,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
