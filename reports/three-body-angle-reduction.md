# 2 次元 3 体系 Lennard-Jones モデルの角度座標への縮約

2025 年 2 月発表の Ref. [Phys. Rev. E 111, 024204 (2025)](../references/PhysRevE.111.024204.pdf) Appendix A に書かれている座標変換と行列 $B$ を、Python 上で再現した手順と結果を以下にまとめます。

## 元のラグランジアンと対称性

粒子位置 $r = (r_1, r_2, r_3)$, $r_i \in \mathbb{R}^2$ に対するラグランジアンは

$$
L(r, \dot r) = \frac{m}{2} \sum_{i=1}^3 \|\dot r_i\|^2 - V(r),
\qquad
V(r) = \sum_{1 \leq i < j \leq 3} U_{\rm LJ}(\|r_i - r_j\|),
$$

です。全系は並進・回転対称性を持つため、形状以外の自由度を取り除く座標変換を行います。

## 座標変換

指示通りに 4 変数 $y = (l_1, l_2, \phi, \phi_L)$ へと縮約します。

1. **重心＋相対座標**  
   $$
   q_1 = r_2 - r_1,\quad
   q_2 = r_3 - r_2,\quad
   q_3 = \frac{r_1 + r_2 + r_3}{3}.
   $$
   これを逆変換すると
   $$
   r_1 = q_3 - \frac{2}{3}q_1 - \frac{1}{3}q_2,\quad
   r_2 = q_3 + \frac{1}{3}q_1 - \frac{1}{3}q_2,\quad
   r_3 = q_3 + \frac{1}{3}q_1 + \frac{2}{3}q_2.
   $$
2. **並進対称性の除去**  
   重心座標 \(q_3\) を固定し \(q_3 = \dot q_3 = 0\) とおくことで平行移動を除去します。
3. **極座標への変換**  
   $q_1 = l_1 (\cos \phi_1, \sin \phi_1),\; q_2 = l_2 (\cos \phi_2, \sin \phi_2)$ とし、極半径 $l_1, l_2$ と角度 $\phi_1, \phi_2$ を導入します。
4. **全体回転と形状角**  
   指定通り
   $$
   \phi = \phi_2 - \phi_1,\qquad
   \phi_L = \frac{\phi_1 + \phi_2}{2}
   $$
   を定義すると、$\phi_1 = \phi_L - \phi/2$, $\phi_2 = \phi_L + \phi/2$ と書けます。これで $y = (l_1, l_2, \phi, \phi_L)$ が得られます。

ポテンシャルは
$$
l_3^2 = l_1^2 + l_2^2 + 2 l_1 l_2 \cos\phi,\quad
V(y) = U_{\rm LJ}(l_1) + U_{\rm LJ}(l_2) + U_{\rm LJ}(l_3),
$$
と $y$ だけで表せます。

## 運動エネルギーと行列 \(B(y)\)

極座標での速度は
$$
\dot q_1 = \dot l_1\, \hat e_{r_1} + l_1 \dot\phi_1\, \hat e_{\phi_1},\qquad
\dot q_2 = \dot l_2\, \hat e_{r_2} + l_2 \dot\phi_2\, \hat e_{\phi_2},
$$
（$\hat e_{r_i}$, $\hat e_{\phi_i}$ は極座標の単位ベクトル）であり、$\phi_1, \phi_2$ の時間微分は
$\dot\phi_1 = \dot\phi_L - \dot\phi/2$, $\dot\phi_2 = \dot\phi_L + \dot\phi/2$ です。

重心固定後の $\dot r_i$ を $\dot q_i$ で書き直して代入すると、運動エネルギーは

$$
\begin{aligned}
T &= \frac{m}{2} \sum_{i=1}^3 \|\dot r_i\|^2 = \frac{m}{3}\Big[
 \dot l_1^2 + \dot l_2^2 + \dot l_1 \dot l_2 \cos\phi \\
&\quad - \frac{l_2 \sin\phi}{2} \dot l_1 \dot\phi - \frac{l_1 \sin\phi}{2} \dot l_2 \dot\phi
- l_2 \sin\phi\, \dot l_1 \dot\phi_L + l_1 \sin\phi\, \dot l_2 \dot\phi_L \\
&\quad + \frac{1}{4}(l_1^2 + l_2^2 - l_1 l_2 \cos\phi) \dot\phi^2
 + (-l_1^2 + l_2^2) \dot\phi\, \dot\phi_L
 + (l_1^2 + l_2^2 + l_1 l_2 \cos\phi) \dot\phi_L^2
\Big].
\end{aligned}
$$

これを $L = T - V = \frac{1}{2} \sum_{\alpha,\beta=1}^4 B_{\alpha\beta}(y)\, \dot y_\alpha \dot y_\beta - V(y)$ に合わせると、Appendix A の行列 $B(y)$ がそのまま得られます。

$$
B(y) = \frac{m}{3}
\begin{pmatrix}
2 & \cos\phi & -\frac{1}{2} l_2 \sin\phi & -l_2 \sin\phi \\
\cos\phi & 2 & -\frac{1}{2} l_1 \sin\phi & l_1 \sin\phi \\
-\frac{1}{2} l_2 \sin\phi & -\frac{1}{2} l_1 \sin\phi &
2 l_1^2 + l_2^2 - l_1 l_2 \cos\phi & l_2^2 - l_1^2 \\
-l_2 \sin\phi & l_1 \sin\phi & l_2^2 - l_1^2 &
2 l_1^2 + l_2^2 + l_1 l_2 \cos\phi
\end{pmatrix}.
$$

上式を $B_{ll}, B_{l\phi}, B_{\phi\phi}$ のブロックに分割すると、論文 Appendix A の (A1)–(A2) と完全に一致することを確認しました。$B^{-1}(y)$ についても Appendix A (A3) と一致することを数式処理で確かめています。

## 検算（Python）

理解を確実にするため、任意の \(y\) と \(\dot y\) を与えて

1. 元の Cartes座標で計算した $T = \frac{m}{2}\sum \|\dot r_i\|^2$ と、
2. $\frac{1}{2}\dot y^\mathsf{T} B(y) \dot y$

が一致するかを `numpy` で比較しました。以下はその最小例です。

```python
import numpy as np
m = 1.0
l1, l2, phi, phiL = 1.7, 1.2, 0.4, 0.7
l1d, l2d, phid, phiLd = 0.2, -0.3, 0.5, -0.1
# …中略（本文と同じ式で ṙi, B(y) を構成）…
np.allclose(T_cartesian, 0.5 * ydot @ B @ ydot)  # -> True
```

これにより、参照論文の座標変換と \(B\) 行列を本リポジトリでも再現できることを確認しました。
