---
marp: true
title: 3 次元 4 体 Lennard-Jones 系の遷移状態における形状の安定性
description: Stability analysis of a four-body Lennard-Jones cluster with one central and three peripheral particles.
paginate: true
backgroundColor: #ffffff
math: katex
---

<style>
  .two-column {
    display: flex;
    gap: 24px;
    align-items: center;
  }
  .two-column > div {
    flex: 1;
  }
</style>

<!-- _class: lead -->

# 3 次元 4 体 Lennard-Jones 系の遷移状態における形状の安定性

コロキウム / 2025-10-09
発表者: 金地亮弥

---

## Lennard-Jones ポテンシャルについて

<div class="two-column">
<div>

- (6, 12)型 Lennard-Jones(LJ) ポテンシャル: $V(d) = 4[(1/d)^{12} - (1/d)^6]$
- 原子間の引力・斥力をモデル化
- 最小値は $d = 2^{1/6} \approx 1.122$

</div>
<div>

![w:520](lj_potential.png)

</div>
</div>

---

## 背景: Lennard-Jones 系と安定化

- 2 次元 3 体の直線配置は、ポテンシャルの鞍点 (遷移状態)となるため不安定
- つまり、より安定な三角形配置に向かうと考えられる
- しかし、基準振動励起により配置が長時間維持できると報告 [1]
- 3 次元 4 体 系にも類似の安定化メカニズムがあるのかを検証

<div class="two-column">
<div>
<video src="lj3_2d_unstable.mov" width="520" controls></video>
</div>
<div>
<video src="lj3_2d_stable.mov" width="520" controls></video>
</div>

</div>

[1] Y. Y. Yamaguchi, _Phys. Rev. E_ **111**, 024204 (2025).

---

## 復習: 直線 3 体系の線形解析

| モード種別  | $\lambda$     | 特徴                           |
| :---------- | :------------ | :----------------------------- |
| 不安定      | $-0.22$       | 面外へ曲がるモードで指数的発散 |
| 安定 (2 本) | $58.2, 176.1$ | 伸縮・呼吸の実振動モード       |
| ゼロ (3 本) | $\approx 0$   | 並進・回転対称性               |

- 先行研究 [1] は安定モードへの励起で直線維持を実現
- 4 体系における不安定方向が何かを特定することが第一歩

---

## 4 体系の幾何と平衡半径

- 正三角形の頂点に 3 粒子、中心に 1 粒子を配置した平衡点を考える
- 正三角形の外周粒子間距離: $s = \sqrt{3}\,r$
- 全ポテンシャル: $U(r) = 3V(r) + 3V(\sqrt{3}\,r)$
- 平衡条件 $U'(r) = 0$ から
  $$
  r^{*6} = \frac{2 \left(1 + 1/3^{6}\right)}{1 + 1/3^{3}} = \frac{365}{189}, \quad r^* \approx 1.116
  $$
- 2 体最適距離よりわずかに短く、中心との引力が平衡距離を縮める

---

## 4 体系の正規モード解析

| モード種別 | $\lambda$   | 物理像                                   |
| :--------- | :---------- | :--------------------------------------- |
| 不安定 (1) | $-1.42$     | 中心と周辺が逆符号で面外傘状に変形       |
| 不安定 (2) | $-1.39$     | 正三角形が面内でひし形へ崩壊 (縮退 2 重) |
| 安定 (1)   | $62.1$      | 面内同相の呼吸モード                     |
| 安定 (2)   | $160.7$     | 中心が面内で振れ、外周が追随 (縮退 2 重) |
| ゼロ (6)   | $\approx 0$ | 並進 3、回転 3(?)                        |

- 平衡点は 3 本の不安定方向を持つ **鞍点**
- 面外変形の曲率が最も負で、中心粒子が抜けやすい
- まずは、面内呼吸モードを励起して形状安定性を調べる

---

## シミュレーション設定

- 運動方程式: 等質量 4 粒子、LJ ポテンシャルの相互作用が各粒子に働く、束縛条件なし
- 初期配置: 周辺 3 粒子を半径 $r$ の正三角形 (中心原点)、中心粒子を $z_0 = 0.02$ にシフト
- 初期速度: 0
- 数値積分: 4 次シンプレクティック法、時間刻み $\Delta t = 0.002$, 追跡時間 $T = 80$
- 構造崩壊判定: 相対距離変化がしきい値 $\delta$ を超える最初の時刻

---

## シミュレーション結果

左: 平衡点 ($r = r^{*}$, $z_0 = 0.02$)
右: 面内安定モード励起 ($r = r^{*} \times 1.12$, $z_0 = 0.02$)

<div class="two-column">
<div>
<video src="lj4_3d_unstable.mov" width="420" controls></video>
</div>
<div>
<video src="lj4_3d_stable.mov" width="420" controls></video>
</div>
</div >

---

## パラメータ掃引と評価指標

- `side_scale` : 正三角形の中心から頂点までの距離をスケール (0.8〜1.6, 295 点)
  - 1.0 で $r=r^{*}$ の平衡点、<1 で圧縮、>1 で伸長
- 閾値 $\delta = 0.1, 0.2, 0.3, 0.4, 0.5$ を設定し崩壊時間を記録
  - 崩壊: 相対距離変化が $\delta$ を超えた最初の時刻
- 判定不能 (崩壊なし) の場合は $t = T$ で打ち切り

---

## 結果: `side_scale` と崩壊時間

<div style="display:flex; gap:24px; align-items:center;">
<div style="flex:1;">

- `side_scale` = 1.0 は振動を与えていない状態での崩壊時間
- `side_scale` ≈ 0.92–0.95 と 1.08–1.18 に緩い安定化領域
- `side_scale` > 1.2 では全閾値で寿命が減少

</div>
<div style="flex:1;">

![w:620](side_scale_break.png)

</div>
</div>

---

## 考察

- 伸長により崩壊時間を長くできるが、2 次元の場合と異なり、より長時間の形状安定化は見られない

---

## 今後のステップ案

- 特定ペアにのみ相互作用を与えた系での挙動を調べる
- 他モード励起時の形状安定性を調べる

---

## 参考文献

- [1] Y. Y. Yamaguchi, _Phys. Rev. E_ **111**, 024204 (2025).
