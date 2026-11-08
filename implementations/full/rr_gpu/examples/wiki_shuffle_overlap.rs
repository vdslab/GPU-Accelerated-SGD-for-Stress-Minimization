//! Isolated paired comparison: foreground shuffle vs an ahead-of-time CPU producer.
#[path = "../src/graph.rs"] mod graph;
#[path = "../src/gpu.rs"] mod production;
#[path = "wiki_shuffle_overlap/gpu.rs"] mod experimental;
use anyhow::{ensure, Result};
use experiment_common::{OutputFormat, positions_sha256_f32, positions_sha256_f64};
use serde_json::json;
use std::{path::Path, time::Instant};

fn copy_params(p: &graph::SgdParams) -> graph::SgdParams {
    graph::SgdParams { etas:p.etas.clone(), positions:p.positions.clone(), pairs:p.pairs.clone(), center:p.center }
}
fn ms(t:std::time::Duration)->f64 {t.as_secs_f64()*1000.0}
fn main()->Result<()> {
    ensure!(std::env::var_os("SGD_TRACE_DIR").is_none(), "disable traces");
    let args:Vec<_>=std::env::args().collect();
    let input=args.get(1).expect("input path");
    let smoke=args.iter().any(|a|a=="--smoke");
    let graph=graph::Graph::from_mtx(Path::new(input))?;
    ensure!(graph.node_size==7066,"expected canonical wiki-Vote LCC");
    let baseline=production::GpuContext::new(production::KernelConfig{phase_chunks:1,..Default::default()})?;
    let context=experimental::GpuContext::new(experimental::KernelConfig{phase_chunks:1,..Default::default()})?;
    let schedule=production::build_schedule((graph.node_size as u32).div_ceil(1024));
    for seed in 0..if smoke {1} else {5} {
        let params=graph.prepare_sgd_params(15,0.1,true,seed);
        let initial_hash=positions_sha256_f64(&params.positions);
        let control=baseline.execute_sgd(copy_params(&params),&schedule,seed,false,OutputFormat::Json)?;
        let expected=positions_sha256_f32(&control.positions);
        for repetition in 0..if smoke {1} else {6} {
            let modes=if (seed+repetition)%2==0 {[false,true]} else {[true,false]};
            for background in modes {
                let input_params=copy_params(&params);
                let start=Instant::now();
                let result=context.execute_sgd(input_params,&schedule,seed,false,OutputFormat::Json,background)?;
                let execute_wall=ms(start.elapsed());
                let final_hash=positions_sha256_f32(&result.positions);
                ensure!(final_hash==expected,"coordinate mismatch: seed={seed}, rep={repetition}, background={background}");
                println!("{}",json!({
                    "seed":seed,"repetition":repetition,"warmup":repetition==0,
                    "method":if background {"background_shuffle"} else {"foreground_shuffle"},
                    "nodes":graph.node_size,"rounds_per_iteration":schedule.len(),"iterations":15,
                    "phase_chunks":1,"tile_size":1024,"adapter":context.adapter_name,"backend":context.backend,
                    "initial_positions_sha256":initial_hash,"final_positions_sha256":final_hash,
                    "production_final_positions_sha256":expected,"production_match":true,
                    "sgd_total_ms":ms(result.iteration_time),"mean_iteration_ms":ms(result.iteration_time)/15.0,
                    "sgd_to_cpu_ms":ms(result.sgd_to_cpu_time),"readback_ms":ms(result.readback_time),
                    "upload_setup_ms":ms(result.upload_time),"execute_wall_ms":execute_wall,
                    "shuffle_cpu_ms":ms(result.shuffle_time),"plan_wait_ms":ms(result.plan_wait_time),
                    "dispatch_cpu_ms":ms(result.dispatch_time),"gpu_wait_cpu_ms":ms(result.gpu_wait_time),
                    "gpu_timestamp_span_ms":result.gpu_device_time.map(ms),"dispatches":result.dispatches,
                    "plan_events_ms":result.plan_events,"iteration_events_ms":result.iteration_events
                }));
            }
        }
    }
    Ok(())
}
