# MPC, PI and deep reinforcement learning for unsteady flow regulation in the South Mirzachul main canal

Code, data and results for the manuscript *"Offset-free model predictive control of unsteady flow in a large irrigation main canal using SCADA measurements: a benchmark against PI and deep reinforcement learning"* (target: Water Resources Management).

## Layout

```
data/structures.csv     check structures and offtakes from the canal passport (Seytov 2018)
LICENSE, LICENSE-DATA.md  MIT (code); CC BY 4.0 (data, results, figures)
src/gated_model.py      Saint-Venant model, Preissmann scheme, gated checks (from the South Mirzachul repository)
src/canal.py            normal operating point (260 m3/s, constant upstream levels)
src/env.py              control environment: 7 check gates, 30-min decisions, 48-h episodes, scenarios
src/controllers.py      Fixed, PI, MPC (full state), MPC-KF (Kalman filter, SCADA only), residual DRL wrapper
src/rl.py               PPO and TD3 in NumPy (MLP with manual back-propagation, Adam)
src/tune_pi.py          grid search of the PI gains
src/train.py            training of PPO / TD3 (residual or pure)
src/val_baselines.py    validation returns of the classical controllers
src/evaluate.py         test scenarios T1-T6 and 20 random scenarios, ASCE indicators
src/sensitivity.py      MPC-KF horizon / move-weight sensitivity
src/tables.py           summary tables (results/tables.md, results/summary.json)
src/figures.py          figures (figures/*.png; journal versions in figures/journal)
(manuscript sources are kept outside this repository until publication)
run_training.sh         residual PPO/TD3 (5 seeds), pure PPO/TD3 seed 0
run_training2.sh        pure PPO (5 seeds, 1200 episodes), pure TD3 seeds 1-2
results/                gains, logs, trained policies (.pkl), evaluation, sensitivity, timing
```

## Reproduce (about 4 h on two CPU cores; Python 3, NumPy, SciPy, pandas, Matplotlib)

```
cd src
python tune_pi.py            # PI grid (edit KP/KI to the stage you want); final gains in results/pi_gains.json
bash ../run_training.sh
bash ../run_training2.sh
python val_baselines.py
python evaluate.py named & python evaluate.py random
python sensitivity.py
python tables.py
python figures.py
```

## Notes

- The hydraulic model is built from the canal passport only; it has not been calibrated against field data.
- PPO and TD3 are implemented in NumPy because no deep-learning library was available in the computing environment; Stable-Baselines3 can be used to cross-check them.
- The PI gains were selected in several grid stages (results/pi_grid_*.csv): stage 1-2 with an uncapped quadratic cost, the final stages with the capped stage cost of the manuscript.
