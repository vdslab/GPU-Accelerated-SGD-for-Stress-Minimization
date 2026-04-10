mod graph;
mod gpu;

use anyhow::Result;
use chrono::Local;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    env_logger::init();

    let mtx_path = Path::new("../data/bcspwr10.mtx");
    let graph = graph::Graph::from_mtx(mtx_path).expect("MTX ファイルの読み込みに失敗しました");

    println!("グラフ読み込み完了: nodes={}, edges={}", graph.node_size, graph.edge_size);

    let total_start = Instant::now();

    let sgd_params = graph.prepare_sgd_params(15, 0.1, true);
    let num_iterations = sgd_params.etas.len();

    let ctx = gpu::GpuContext::new()?;

    let sgd_start = Instant::now();
    let (init_pos, final_pos) = ctx.execute_sgd(sgd_params)?;
    let sgd_duration = sgd_start.elapsed();

    let total_duration = total_start.elapsed();

    println!("全体合計時間 (BFS+SGD): {:.3}s", total_duration.as_secs_f64());
    println!(
        "SGD時間 (GPU-CPU伝送含む): {:.3}s / iter平均: {:.1}ms",
        sgd_duration.as_secs_f64(),
        sgd_duration.as_secs_f64() / num_iterations as f64 * 1000.0
    );

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let data_name = mtx_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();

    // ---- 初期座標を保存 ----
    let path_init = format!("../output/rr-gpu-{}-{}-0.txt", data_name, timestamp);
    save_result(&path_init, "rr-gpu - Initial (Randomized)", &graph, &init_pos)?;
    println!("初期座標を保存: {}", path_init);

    // ---- 最終座標を保存 ----
    let path_final = format!("../output/rr-gpu-{}-{}-1.txt", data_name, timestamp);
    save_result(&path_final, "rr-gpu - Processed", &graph, &final_pos)?;
    println!("最終座標を保存: {}", path_final);

    Ok(())
}

fn save_result(
    path: &str,
    label: &str,
    graph: &graph::Graph,
    positions: &[[f32; 2]],
) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "# Rust GPU Result ({label})")?;
    writeln!(file, "# Timestamp: {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(file, "# Node count: {}", graph.node_size)?;
    writeln!(file, "# Edge count: {}", graph.edge_size)?;
    writeln!(file)?;
    writeln!(file, "# Edges (source target)")?;
    for i in 0..graph.edge_size {
        writeln!(file, "{} {}", graph.edge_src[i], graph.edge_dst[i])?;
    }
    writeln!(file)?;
    writeln!(file, "# Positions (x y)")?;
    for pos in positions {
        writeln!(file, "{} {}", pos[0], pos[1])?;
    }

    Ok(())
}
