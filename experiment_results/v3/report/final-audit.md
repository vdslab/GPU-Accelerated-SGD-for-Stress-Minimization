# v3 最終記録監査

公式SNAP原配布復旧の90条件を補完した有効cohortの監査。core原本3210件＋recovery原本90件＝3300記録を保持し、有効IDは3210件。元90件は未起動unavailableで、supersededとして保持する。実起動済み条件の上書きはない。

coreとrecoveryをそれぞれ独立にplan/config/native commitへ照合した。67 native/sharedファイルの集合・SHA、runner 1ファイルのSHA、全binary SHA、limitsの一致を確認してから合算した。commitはbase別に区別する。正式起動・warmup・未起動・正式失敗・RSS超過は両baseを加算し、元の未起動90件だけは有効欠測から差し引いた。

原配布は同名の公式SNAP sourceと公開ノード数・unique edge数を検証したもの。SuiteSparse MTXとのbyte/edgewise同一性は未検証。power197k未取得・G_n_pin_pout規模除外は変更していない。

- [core単独監査](../tables/audit-v3-core.json) · [recovery単独監査](../tables/audit-v3-recovery.json)
- [採用検証provenance](../tables/cohort.json) · [全原記録の有効/置換状態](../tables/cohort-source-records.csv) · [元の未起動90条件](../tables/cohort-superseded.csv)

監査状態：success。計画 3210 条件、確定ID 3210 件。

確定recordを持つ正式起動 2370 件、再利用 120 条件、未起動 720 条件。
計画条件数 = 確定recordを持つ正式起動数 + 再利用条件数 + 未起動条件数（全条件の確定時）。暫定監査では未確定の条件数も別途残る。
追加の中断起動attempt 1 件を加え、正式nativeの総起動attemptは 2371 件。
総起動attempt = 確定recordを持つ正式起動数 + 追加の中断attempt数。中断後に同じrun_idの条件が完了しても、途中attemptは別枠で残す。pilot・prebuffer-aborted・warmupはこの総数へ加算しない。
正式の実失敗 49 件。別枠warmupは実起動 468 件、実失敗 126 件。

成否：`{"success": 2426, "memory_limit": 46, "warmup_failed": 720, "unavailable": 15, "timeout": 3}`。
warmupによる未起動の理由：`{"physical_warmup_failure": 690, "size_excluded": 15}`。

記録整合性エラー 0 件。警告分類：`{"native_git_dirty": 2663}`。
git_dirtyはnative測定時の作業場所の状態を示す。固定snapshot/binary/recordの照合とは分けて記録する。

## ホスト再起動による追加中断attempt

中断receiptは計画条件や確定失敗の追加行として扱わない。receiptのrun_idがplan内にあること、退避artifactと記載ファイルの存在・サイズ、command.jsonのrun_idを照合する。

| run_id | 確定条件の結果 | 中断証拠 | 退避artifact | 証拠整合 |
|---|---|---|---|---|
| v3-basic-web-NotreDame-sparse_sgd-s3-p200-tNone | success | [receipt](../interruption-reboot-20261024/receipt.json) | [interruption-reboot-20261024/artifacts/v3-basic-web-NotreDame-sparse_sgd-s3-p200-tNone](../interruption-reboot-20261024/artifacts/v3-basic-web-NotreDame-sparse_sgd-s3-p200-tNone) | True |

## 実失敗の観測段階

| Scope | Stage | Count |
|---|---|---:|
| formal | before_first_iteration_completed | 10 |
| formal | before_trace_or_trace_missing | 36 |
| formal | during_iterations | 3 |
| warmup | before_trace_or_trace_missing | 120 |
| warmup | during_iterations | 6 |

traceなしは前処理・buffer準備中などの可能性があり、詳細な停止命令は特定しない。反復0は最初の反復完了前、1〜14は反復途中、15は全15反復完了後の段階。反復15でもstress評価や保存の完了を意味しない。

## 成功した実行のRSS閾値超過

成功した正式実行とwarmupのうち、終了後の最大RSSが閾値8 GiBを超えるものは 24 件。再利用条件は重複加算していない。
監視は間隔観測であり、実行中の瞬間最大RSSが閾値以下だったことを保証しない。成功statusを閾値超過だけで書き換えない。RSSを専用VRAM量とは解釈しない。

## 証拠と監査範囲

- [監査JSON](../tables/audit-v3.json)
- [status別時間・RSS](../tables/audit-v3-status.csv)
- [起動数と条件数](../tables/audit-v3-execution-counts.csv)
- [追加中断attemptの証拠](../tables/audit-v3-interrupted-attempts.csv)
- [実失敗・trace段階](../tables/audit-v3-failures.csv)
- [成功時RSS閾値超過](../tables/audit-v3-success-rss-exceedances.csv)
- [未起動理由](../tables/audit-v3-unexecuted.csv)
- [再利用条件](../tables/audit-v3-reused.csv)

planの全ID・重複・余分・設定、configの固定hash、各実起動recordのinput/binary hash、成功native commit、Atomic更新完了、archive状態を照合する。座標の全再計算・gzip再読込・binary再hashは行わない。archiveは実行時の検証metadataと保存ファイルの存在を確認する。

この文書はresource-incidents.mdの履歴を上書きせず、最終監査として別に保存する。
