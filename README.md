# LJ Stabilization Sims (2D 3-body / 3D 4-body)

Lennard–Jones (6,12) ポテンシャルで 2D 3 体系・3D 4 体系の挙動を調べるスクリプト群です。積分は 4 次の Yoshida スキーム（速度 Verlet 合成）で、単位は LJ 既約単位系（m = σ = ε = 1）。

## ファイル構成

- `src/lj3_2d.py` : 2D・3体の最小例。左右の粒子を±x 方向に振り、中央が安定化するかを見る。
- `src/lj4_3d.py` : 3D・4体（複数の平衡配置に対応）。数値積分とプロットを行う CLI。
- `src/lj4_3d_anim.py` : 3D・4体の軌道をインタラクティブに再生する簡易ビューワ。
- `src/lj4_storage.py` : 4 体系シミュレーションの保存・復元・キャッシュのユーティリティ。

## クイックスタート

### 2D 3 体系（単発）

```bash
uv run src/lj3_2d.py --x0 1.24 --vb 0.20 --y0 0.02 --dt 0.002 --T 120 \
  --save_traj results/2d_run.csv --save_metrics results/2d_run.json --plot results/2d_run.png
```

### 2D 3 体系（vb スイープ）

```bash
uv run src/lj3_2d.py --x0 1.20 --y0 0.02 --dt 0.002 --T 120 \
  --sweep_vb 0.14 0.28 8 --out_prefix results/2d_sweep
```

### 3D 4 体系（計算して保存 → 再利用）

```bash
# 計算してバンドル保存
uv run src/lj4_3d.py --config triangle_center --modes 0 --mode-displacement 0.02 --mode-velocity 0.00 \
  --dt 0.002 --T 80 --thin 10 --use-cache

# 保存済みバンドルを使ってプロットだけ出力（再計算なし）
uv run src/lj4_3d.py --config triangle_center --modes 0 --mode-displacement 0.02 --mode-velocity 0.00 \
  --dt 0.002 --T 80 --thin 10 --use-cache \
  --plot-energies results/plots/energies.png --plot-modal results/plots/modal.png --plot-dihedral results/plots/dihedral.png

# 保存済みバンドルをインタラクティブ再生
uv run src/lj4_3d_anim.py --config triangle_center --modes 0 --mode-displacement 0.02 --mode-velocity 0.00 \
  --dt 0.002 --T 60 --thin 5 --use-cache
```

`--use-cache`（デフォルトで有効）により、パラメータから自動生成されるハッシュディレクトリ（既定: `results/cache/{config}/{hash}/`）に保存し、同じパラメータで再実行すると即座に読み込みます。キャッシュを無効にしたい場合は `--no-cache`。任意の場所に保存したい場合は `--save-bundle パス`、既存バンドルを直接読む場合は `--load-bundle パス` を使います。

## 出力形式

- 2D (`lj3_2d.py`)
  - `--save_traj`: CSV（`t,y,energy`）
  - `--save_metrics`: JSON（最大変位、RMS、エネルギードリフトなど）
  - `--plot`: PNG（y(t)）
- 3D (`lj4_3d.py`)
  - バンドル（デフォルトはキャッシュに自動保存）: `metadata.json` + `series.npz`
    - `metadata.json`: 平衡配置情報、入力パラメータ、モード選択、エネルギー概要など
    - `series.npz`: 時系列配列（位置、モード座標、エネルギー、二面角など一式）
  - オプション出力: CSV（`--save-traj`）、サマリー JSON（`--save-summary`）、各種プロット PNG
- 3D アニメーション (`lj4_3d_anim.py`)
  - バンドルを読んで軌道を可視化（保存も `--use-cache` または `--save-bundle` で可能）

## 主なパラメータ

### 共通事項
- `--dt` : 時間刻み。小さすぎると計算量増、大きすぎると解が壊れる。
- `--T` : 積分する総時間。
- `--thin` / `--save_stride` : 何ステップごとにスナップショットを保存するか。
- `--use-cache` : パラメータから自動ハッシュを作り、既存なら読み込み・無ければ保存（デフォルト有効、`--no-cache` で無効化）。
- `--cache-dir` : キャッシュのルート（既定: `results/cache`）。

### 2D 3 体系 (`lj3_2d.py`)
- `--x0` : 左右粒子の初期 x 座標（±x0）。
- `--vb` : 左右粒子の初期速度の絶対値（±vb, x 方向）。
- `--y0` : 中央粒子の初期 y 変位。
- `--save_every` : 出力の間引き間隔（ステップ数）。
- `--sweep_vb min max N` : vb を線形に N 分割して一括実行。
- 出力: `--save_traj`(CSV), `--save_metrics`(JSON), `--plot`(PNG)。

### 3D 4 体系 (`lj4_3d.py`)
- `--config` : 平衡配置キー。`tetrahedron` / `rhombus` / `square` / `triangle_center` / `isosceles_interior` / `linear_chain`。
- `--modes` : 安定モードのインデックスをカンマ区切りで指定（0 が最小安定固有値）。
- `--mode-displacement` / `--mode-velocity` : 選択モードごとの初期変位・速度係数（カンマ区切り。1 個だけなら全モードに適用）。
- `--center-mass` : `triangle_center` の中心粒子質量（その他配置では全質量 1）。
- `--modal-kick-energy` : 最初に見つかるモード方向へ与える運動エネルギー（0 で無効）。
- 出力: `--save-bundle`/`--load-bundle`/`--use-cache`, `--save-traj`(CSV), `--save-summary`(JSON), `--plot-energies`/`--plot-modal`/`--plot-dihedral`(PNG)。

### 3D 4 体系アニメ (`lj4_3d_anim.py`)
- 主要パラメータは `lj4_3d.py` と同じ（`--config` など）。
- `--fps` : インタラクティブ表示の目安フレームレート。
- `--trace-index` : 軌跡を残す粒子インデックス（未指定なら配置ごとの推奨値）。
- バンドルを使う場合は `--use-cache` または `--load-bundle` を指定。

## 注意点
- 時間刻み `dt` と初期配置が小さすぎると粒子重なりにより発散するので、既定値から徐々に調整してください。
- エネルギー保存は理想的ではないため、長時間積分では `--plot-energies` などで漂いを確認すると安全です。
