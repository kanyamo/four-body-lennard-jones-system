# LJ Stabilization Sims (2D 3-body / 3D 4-body)

Two CLI-ready scripts for the Lennard–Jones (6,12) stabilization experiments with a 4th-order symplectic (Yoshida) integrator.

## Files

- `lj3_2d.py` : 2D, 3-body, all pairs interact (autonomous). Outer two start with opposite x-velocities (“breathing”), center starts near the line.
- `lj4_3d.py` : 3D, 4-body (equilateral triangle + center), all pairs interact (autonomous). Triangle vertices start with alternating in-plane radial velocities.

Both use reduced LJ units (`m = σ = ε = 1`).

## Quick start

```bash
# 2D 3-body (single run)
uv run src/lj3_2d.py --x0 1.24 --vb 0.20 --y0 0.02 --dt 0.002 --T 120 \
  --save_traj results/run.csv --save_metrics results/run.json --plot results/run.png

# 2D 3-body (vb sweep)
uv run src/lj3_2d.py --x0 1.20 --vb 0.18 --sweep_vb 0.14 0.28 8 --out_prefix results/sweep_case

# 3D 4-body triangle+center (single run)
uv run src/lj4_3d.py --side_scale 1.03 --vb 0.22 --z0 0.005 --dt 0.002 --T 80 \
  --save_traj results/tri.csv --save_metrics results/tri.json --plot results/tri.png

# 3D 4-body triangle+center (vb sweep)
uv run src/lj4_3d.py --side_scale 1.00 --z0 0.02 --sweep_vb 0.16 0.26 7 --out_prefix results/tri_sweep

# 3D 4-body triangle+center (animation)
uv run src/lj4_3d_anim.py --side_scale 1.03 --vb 0.22 --z0 0.005 --dt 0.002 --T 60 \
  --fps 30 --thin 5 --outfile results/tri_anim.mp4
```

## Outputs

- CSVs: time series (2D: `t,y,energy`; 3D: `t,x,y,z,rho`)
- JSON metrics: `y_maxabs`, `y_rms`, `frac(|y|<0.02)`, `energy_rel_drift` (2D) / `rho_rms`, `rho_max`, `frac(rho<0.02)` (3D)
- Optional PNG plots if `--plot` is provided (no seaborn, simple Matplotlib).

## Notes

- Integrator is symplectic, but the systems with driving in earlier experiments were non-autonomous; the current scripts are fully autonomous. Energy drift should be small; monitor the reported relative drift in 2D.
- A tiny symmetry-breaking noise is injected into transverse velocity components to avoid perfectly symmetric, measure-zero motions.
- LJ singularity: the step size `dt` and initial separations must be chosen to avoid overlaps; the provided defaults are safe for the listed examples.
