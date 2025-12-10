# LJ single run
# uv run src/lj4_3d.py \
#     --config rhombus \
#     --mode-displacement 0.16 \
#     --mode-velocity 0.0 \
#     --T 80 \
#     --thin 10 \
#     --modal-kick-energy 0.00001 \
#     --modes 0 \
#     --random-kick-energy 0.00000 \
#     --repulsive-exp 18 \
#     --attractive-exp 9 \
#     --plot-modal results/modal_plot.png \
#     --plot-modal-categories stable,unstable \
#     --plot-energies results/energy_plot.png \
#     --plot-dihedral results/dihedral_plot.png \

# displacement collapse scan
# uv run src/scan_displacement_collapse.py \
#     --config rhombus \
#     --disp-min 0.0 \
#     --disp-max 0.2 \
#     --disp-samples 100 \
#     --T 300 \
#     --thin 10 \
#     --modal-kick-energy 0.00001 \
#     --mode-index 0 \
#     --repulsive-exp 18 \
#     --attractive-exp 9 \

# spring single run
uv run src/spring4_3d_anim.py \
    --mode-displacement 0.2,0.1 \
    --mode-velocity 0.0 \
    --T 200 \
    --thin 10 \
    --kick-energy 0.001 \
    --modes 1,2 \
    # --plot-modal results/modal_plot.png \
    # --plot-modal-categories stable,unstable \
    # --plot-energies results/energy_plot.png \
    # --plot-dihedral results/dihedral_plot.png \
