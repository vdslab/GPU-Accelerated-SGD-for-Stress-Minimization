"""Presentation of measured intervals; never reinterpret historical wall timers."""
from __future__ import annotations

PRIMARY = (
    ("iteration_times_ms", "反復時間"),
    ("iteration_total_time_ms", "反復時間合計"),
    ("total_compute_time_ms", "総計算時間"),
)
DETAIL = (
    ("preprocess_time_ms", "前処理時間"),
    ("upload_time_ms", "初期転送時間"),
    ("gpu_prepare_time_ms", "GPU準備時間"),
    ("sgd_time_ms", "SGD実行時間（反復時間合計＋GPU計算外の経過時間）"),
    ("readback_time_ms", "結果取得時間"),
    ("draw_time_ms", "描画時間"),
    ("total_with_draw_time_ms", "総計算時間（描画込み）"),
)


def timing_lines(record: dict) -> list[str]:
    lines = []
    cpu = record.get("method") in {"sgd", "sparse_sgd"}
    for heading, fields in (("主要な計測", PRIMARY), ("内訳・描画", DETAIL)):
        lines.append(heading)
        for key, label in fields:
            if cpu and key in {"upload_time_ms", "gpu_prepare_time_ms", "readback_time_ms"}:
                continue
            if cpu and key == "sgd_time_ms":
                label = "SGD実行時間"
            value = record.get(key)
            if isinstance(value, list):
                formatted = ", ".join(f"{i}: {t:.3f}" for i, t in enumerate(value, 1)) + " ms"
            elif isinstance(value, (int, float)):
                formatted = f"{value:.3f} ms"
            else:
                formatted = "未計測"
            lines.append(f"  {label}: {formatted}")
    return lines
