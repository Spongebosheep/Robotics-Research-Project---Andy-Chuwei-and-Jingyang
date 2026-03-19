# Robotics-Research-Project---Andy-Chuwei-and-Jingyang
Optimal Deployment of Seismic Sensor Nodes for Planetary Surface Monitoring

# RRP-Simulation

A research-style Python project for **terrain-aware sensor-layout evaluation and optimization** for **TDoA-based source localization**.

The repository contains two connected parts:

1. **`Initial-simulation/`** – a step-by-step development history of the simulation model, from a minimal geometry-only baseline to a more realistic terrain-constrained, geodesic, amplitude-aware, fairness-controlled, and robustness-tested pipeline.
2. **`LayoutStudy/`** – the main reusable workflow for searching, reevaluating, optimizing, and analyzing **8-node sensor layouts** under multiple terrain realizations.

The core idea is simple:

- start from a 2D candidate sensor layout,
- deploy it onto a terrain surface while approximately preserving the nominal net geometry,
- simulate many source events on that surface,
- generate noisy arrival-time/TDoA observations,
- recover the source location from those observations,
- score the layout by localization accuracy, robustness, and deployment deformation,
- then search for better layouts.

---

## What the model does

For each candidate layout, the code evaluates performance on several terrain types (`flat`, `hill`, `rough`). The evaluation pipeline is:

1. **Layout definition**  
   A layout is an `n x 2` array of sensor coordinates in a normalized square domain `[0, 1] x [0, 1]`.

2. **Terrain generation**  
   The code supports:
   - **flat** terrain,
   - **hill** terrain via a Gaussian bump,
   - **rough** terrain via a random smooth field.

3. **Terrain-conforming deployment**  
   The nominal planar layout is "draped" onto the terrain surface. The deployment solver tries to preserve:
   - Delaunay-edge lengths,
   - centroid position,
   - global orientation,
   - while penalizing excessive node drift.

4. **Path-length model**  
   Later stages use **surface geodesic distance** instead of straight-line 3D distance. A terrain grid is converted into a graph, and shortest paths are computed with Dijkstra.

5. **Forward event simulation**  
   For many random source events, the code simulates:
   - event origin time,
   - propagation travel time,
   - timing noise,
   - optional amplitude attenuation,
   - optional SNR-based detection masks.

6. **Inverse localization**  
   Source location is estimated on the terrain surface by solving a weighted TDoA least-squares problem from multiple starting points.

7. **Metric aggregation**  
   Performance is aggregated per terrain and then into a universal layout score.

---

## Objective function

The main scoring logic lives in:

- `LayoutStudy/layout_objective_evaluator.py`

For each terrain, the code computes a terrain-specific score:

```text
J_t = 0.45 * mean_err_3d
    + 0.30 * p95_err_3d
    + 0.15 * (1 - valid_localization_rate)
    + 0.10 * dgeom_star
```

where:

- `mean_err_3d` = average 3D localization error,
- `p95_err_3d` = 95th percentile 3D localization error,
- `valid_localization_rate` = fraction of all events that produce a valid localization,
- `dgeom_star` = normalized deployment deformation penalty built from edge distortion and terrain height spread.

The universal score is then:

```text
J_universal = 0.75 * mean(J_t across terrains) + 0.25 * max(J_t across terrains)
```

So **lower is better**.

This means the project does **not** optimize only average accuracy. It also penalizes:

- bad worst-case tails,
- low localization success rate,
- layouts that deploy poorly on terrain.

---

## Repository structure

```text
RRP-Simulation/
├── Initial-simulation/
│   ├── Step1/
│   ├── Step2/
│   ├── Step3/
│   ├── Step4/
│   ├── Step5&6a/
│   ├── Step6b/
│   ├── Step7/
│   ├── Step8/
│   ├── Step9/
│   ├── Step10/
│   └── Step11/
└── └── LayoutStudy/
    ├── RunThis(TopLayer).py
    ├── layout_objective_evaluator.py
    ├── reevaluate_top50_layouts.py
    ├── cmaes_optimize_from_top_seeds.py
    ├── analyze_layout_features_complete.py
    ├── run_symmetric_layout_candidate.py
    ├── Result/
    ├── feature_analysis_complete/
    ├── single_layout_symmetric_eval/
    └── Legacy/
```

---

## Development history in `Initial-simulation/`

The `Initial-simulation` folder is not just old code: it records how the modeling assumptions evolved.

| Step | Main idea |
|---|---|
| Step 1 | Minimal geometry-to-TDoA experiment using simple terrain-conforming deployment and localization error. |
| Step 2 | Source is explicitly constrained to lie on the terrain surface. |
| Step 3 | Terrain models become explicit: `flat`, `hill`, `rough`, with added deployment visualization. |
| Step 4 | Stable deployment solver with edge, centroid, orientation, and drift regularization. |
| Step 6a | Arrival-time model using 3D Euclidean chord distance. |
| Step 6b | Arrival-time model upgraded to surface geodesic distance via graph shortest paths. |
| Step 7 | Dynamic TDoA reference node selection instead of a fixed reference sensor. |
| Step 8 | Amplitude attenuation, SNR, and threshold-based detection are added. |
| Step 9 | Monte Carlo design is frozen with shared event banks for fair layout comparison. |
| Step 10 | Post-processing adds normalized geometry-deformation metrics and fairness tables. |
| Step 11 | Robustness study over different timing-noise levels and ranking-flip analysis. |

Notes:

- The numbering reflects the research progression, so not every intermediate step exists as a separate standalone file.
- `Step11_standalone_optimized.py` is a speed-optimized version of the robustness pipeline.

---

## Main workflow in `LayoutStudy/LayoutStudy/`

This is the part to use if you want the full search-and-analysis pipeline.

### 1) Coarse random search

**Script:** `RunThis(TopLayer).py`

This script:

- samples many feasible random 8-node layouts,
- filters them using boundary, pairwise-distance, and hull-area constraints,
- evaluates each layout with the main objective,
- ranks layouts by `J_universal`,
- saves CSV tables, a top-10 figure, and an `.npz` of the best layouts.

Typical output files:

- `Result/random_layout_5000_all_results.csv`
- `Result/random_layout_5000_top10.csv`
- `Result/random_layout_5000_top10.png`
- `Result/random_layout_5000_top10_layouts.npz`

### 2) Reevaluate the top layouts

**Script:** `reevaluate_top50_layouts.py`

The coarse search is intentionally relatively cheap. This script takes the best layouts from the random search and reevaluates them multiple times using different terrain and Monte Carlo seeds, then summarizes:

- `mean_J`,
- `std_J`,
- `p95_J`,
- confidence intervals,
- terrain-specific metrics.

Typical output files:

- `Result/reeval_top50_summary.csv`
- `Result/reeval_top50_detail.csv`
- `Result/reeval_top50_top10.png`
- `Result/reeval_top50_layouts.npz`
- `Result/reeval_top50_meta.json`

### 3) Optimize from the best seeds using CMA-ES

**Script:** `cmaes_optimize_from_top_seeds.py`

This script:

- loads top seed layouts,
- repairs candidate layouts to maintain feasibility,
- runs an in-repo CMA-ES implementation,
- tracks best objective values,
- performs final reevaluation of optimized layouts.

Typical output files:

- `Result/cmaes_multistart_best.csv`
- `Result/cmaes_multistart_history.csv`
- `Result/cmaes_multistart_best_layouts.png`
- `Result/cmaes_multistart_final_layouts.png`
- `Result/cmaes_multistart_final_reeval.csv`
- `Result/cmaes_multistart_meta.json`

### 4) Analyze which geometry features matter

**Script:** `analyze_layout_features_complete.py`

This script extracts interpretable layout features such as:

- pairwise spacing,
- radius statistics,
- distance to boundary,
- angular-gap regularity,
- anisotropy,
- convex hull area/perimeter,

and then links them to performance using:

- Pearson/Spearman correlation,
- bootstrap statistics,
- random forest importance,
- binned nonlinear trend analysis,
- top-layout galleries.

Outputs are saved under:

- `feature_analysis_complete/`

### 5) Evaluate one hand-designed symmetric candidate

**Script:** `run_symmetric_layout_candidate.py`

This script evaluates a manually specified symmetric 8-node layout, saves:

- a layout plot,
- raw per-seed results,
- terrain summary,
- summary JSON.

Outputs are saved under:

- `single_layout_symmetric_eval/`

---

## Key scripts and their roles

### `layout_objective_evaluator.py`

This is the **core engine**. It contains:

- layout validation,
- terrain generation,
- terrain-conforming deployment,
- surface graph / geodesic construction,
- event-bank generation,
- amplitude and SNR model,
- TDoA observation building,
- multi-start localization,
- terrain summary and final objective calculation.

If you only want one file to understand the project logic, start here.

### `RunThis(TopLayer).py`

Top-level entry point for random layout search.

### `reevaluate_top50_layouts.py`

Turns coarse ranking into more statistically stable ranking.

### `cmaes_optimize_from_top_seeds.py`

Searches locally around good seeds instead of relying only on random layouts.

### `analyze_layout_features_complete.py`

Explains *why* some layouts perform better.

---

## Bundled result folders

This repository already includes generated outputs.

### `Result/`

Contains representative search / reevaluation / optimization outputs, for example:

- coarse search over 5000 random layouts,
- reevaluation of the top 50,
- multistart CMA-ES optimization from top seeds.

From the bundled result files:

- the coarse random search reached top layouts around **`J_universal ≈ 0.082`**, and
- the reevaluated top layouts are around **`mean_J ≈ 0.105`**, which is expected because reevaluation is stricter and less noisy than the coarse pass.

### `feature_analysis_complete/`

Contains the generated feature-analysis package:

- feature tables,
- correlation tables,
- RF importance plots,
- nonlinear binned trends,
- top-layout galleries,
- a text summary.

The included summary suggests that layout quality is strongly influenced by a mix of:

- spacing regularity,
- distance to the domain boundary,
- anisotropy,
- nearest-neighbor structure.

### `single_layout_symmetric_eval/`

Contains outputs for a manually designed symmetric candidate layout.

---

## Installation

This is a script-based repository; there is no package installer.

Use Python 3.10+ and install the main dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

No external CMA-ES dependency is required because the optimizer is implemented directly inside `cmaes_optimize_from_top_seeds.py`.

---

## Quick start

Run commands from:

```bash
cd LayoutStudy/LayoutStudy
```

### Coarse random search

```bash
python "RunThis(TopLayer).py" \
  --n-layouts 5000 \
  --n-events 20 \
  --n-seeds 1 \
  --geo-grid-n 21 \
  --search-grid-n 7 \
  --n-multi-start 3 \
  --rough-grid-n 65 \
  --output-dir Result
```

### Reevaluate the top 50 layouts

```bash
python reevaluate_top50_layouts.py \
  --result-dir Result \
  --top-k 50 \
  --n-events 100 \
  --n-seeds 5 \
  --n-repeats 2 \
  --geo-grid-n 21 \
  --search-grid-n 7 \
  --n-multi-start 3 \
  --rough-grid-n 65
```

### Optimize top layouts with CMA-ES

```bash
python cmaes_optimize_from_top_seeds.py \
  --result-dir Result \
  --top-k-seeds 5 \
  --max-gens 25 \
  --sigma0 0.05 \
  --n-events 100 \
  --n-seeds 5
```

### Analyze geometry-performance relationships

```bash
python analyze_layout_features_complete.py \
  --all-csv Result/random_layout_5000_all_results.csv \
  --reeval-csv Result/reeval_top50_summary.csv \
  --outdir feature_analysis_complete
```

### Evaluate the hand-designed symmetric layout

```bash
python run_symmetric_layout_candidate.py
```

For a faster sanity check:

```bash
python run_symmetric_layout_candidate.py --quick
```

---

## Important modeling assumptions

- The domain is normalized to a unit square.
- Layouts are mainly evaluated for **8 sensors**.
- Terrain deployment is not rigid; nodes can move in-plane to conform to the surface.
- Rough terrain is sampled randomly, so repeated evaluations matter.
- Later-stage scripts enforce fairness by sharing event banks across terrain/layout comparisons.
- Detection is SNR-threshold-based in the later model stages.
- Localization is performed on the terrain surface, not in unconstrained 3D space.

---

## Practical reading order

If you are new to the codebase, this is the most efficient order:

1. `LayoutStudy/layout_objective_evaluator.py`  
   Understand the full evaluation logic.
2. `LayoutStudy/RunThis(TopLayer).py`  
   See how candidate layouts are generated and ranked.
3. `LayoutStudy/reevaluate_top50_layouts.py`  
   See how unstable coarse rankings are cleaned up.
4. `LayoutStudy/cmaes_optimize_from_top_seeds.py`  
   See how the search is refined.
5. `LayoutStudy/analyze_layout_features_complete.py`  
   See how performance is interpreted.
6. `Initial-simulation/`  
   Read the historical steps only if you want to follow the model evolution from first principles.

---

## Limitations

- The project is organized as research scripts, not as a packaged library.
- There is no automated test suite yet.
- Some metadata files still contain original local Windows paths from the authoring environment.
- Some scripts are computationally heavy at full settings.
- The code is tightly centered on normalized 8-node layouts in a square domain.

---

## Suggested next improvements

- factor the common simulation code into a small reusable package,
- add unit tests for terrain generation, deployment, and objective evaluation,
- separate config from code via YAML/JSON,
- add a single master pipeline script,
- store result metadata more consistently,
- add notebooks or examples for visualization and reproducibility.

---

## Summary

This repository is best understood as a **complete experimental pipeline for terrain-aware layout design**:

- it starts from physically constrained deployment,
- uses geodesic-aware TDoA localization,
- scores layouts with both accuracy and robustness terms,
- compares layouts fairly through frozen Monte Carlo banks,
- refines the best layouts by reevaluation and CMA-ES,
- and finally analyzes which geometric features drive good performance.

If you are cloning the repo to reproduce the main logic, focus on the `LayoutStudy` folder first.
