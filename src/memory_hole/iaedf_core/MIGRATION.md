# IAEDF-Simからの移管記録

2026-10-01にローカルのIAEDF-Sim `backend/bkmcore` と、必要な3個のCSV資産をSCAへ移管した。
移管元にはGitメタデータがないため、元ファイルのSHA-256を `provenance.json` に記録した。
CSVのデータ出典は各ファイル先頭のコメントに保持している。

元のKCL、シース、衝突、磁場、Laplace、空間電荷、TPMCの数値アルゴリズムは維持した。
SCA向けに以下を追加・調整した。

- パッケージを `memory_hole.iaedf_core` へ配置。内部の相対importを維持。
- 1D到達時の速度を `(vy, vz, vx)`、2D到達時の速度を `(接線速度, vz, 内向き法線速度)` のSCA座標で保存。
- 2D到達位相を、既存の表面交差補間時刻から記録。乱数呼出しや速度更新は追加していない。
- raw NPZに3成分速度、2D到達位相、座標系の識別を追加。
- 入力の有限値・未知フィールドを検査。公開入口の数値範囲と資産名は `upstream.validated_config` で検査。
- SCAの書式規約に合わせ、警告にstacklevel、zipにstrict指定を追加。

外側の `upstream.py` と `upstream_api.py` がSCAの保存先・ワーカー管理・取消・入口適用を担当する。
移管元のWeb画面、SQLite、認証層、環境設定はモデルの依存関係に含めていない。

元の物理テストは `tests/iaedf_core` へ移管した。追加の結合テストは `tests/test_upstream.py` と
`tests/test_upstream_storage.py` にある。 `scripts/verify_iaedf_migration.py` は移管元を読み取り専用で
読み込み、1D/2D各2設定・各2圧力について既存出力の完全一致を検査する。

IAEDFの分布と位相は一方向でホールへ渡す。ホール帯電を上流シースへ戻す連成計算は行わない。
絶対流束は密度・Bohm速度・到達率・コレクタ表面長に基づくモデル推定で、実測校正ではない。
