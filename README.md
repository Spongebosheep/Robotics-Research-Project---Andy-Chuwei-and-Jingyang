# Robotics-Research-Project---Andy-Chuwei-and-Jingyang
Optimal Deployment of Seismic Sensor Nodes for Planetary Surface Monitoring

# Demo Video
A short YouTube demonstration video is also available, showing the 70 × 70 cm deformable sensor net, the 8 piezo sensors, the sensor-to-ESP32-to-computer signal path, the standardized 205 g drop test, flat-ground and sloped-surface experiments, the mapping-based calibration idea, and the final simulation-guided layout comparison: [Detailed Experiment Video](https://youtu.be/zfnjO2ME83k).

# RRP-Simulation

A research-style Python project for **terrain-aware sensor-layout evaluation and optimization** for **TDoA-based source localization**, plus a **live physical prototype** for real-time impact sensing on an 8-sensor net.

The repository now contains **three connected layers**:

1. **`Initial-simulation/`** – a step-by-step development history of the simulation model, from a minimal geometry-only baseline to a more realistic terrain-constrained, geodesic, amplitude-aware, fairness-controlled, and robustness-tested pipeline.
2. **`LayoutStudy/`** – the main reusable workflow for searching, reevaluating, optimizing, and analyzing **8-node sensor layouts** under multiple terrain realizations.
3. **`physical_prototype.py`** – a real-time prototype GUI for the physical net, reading 8-channel sensor data from an ESP32 over serial, detecting impact events, and estimating impact position using either a physics-based solver or a calibration-template mapping mode.

The core idea is:

- start from a 2D candidate sensor layout,
- deploy it onto a terrain surface while approximately preserving the nominal net geometry,
- simulate many source events on that surface,
- generate noisy arrival-time/TDoA observations,
- recover the source location from those observations,
- score the layout by localization accuracy, robustness, and deployment deformation,
- then search for better layouts,
- and finally test the idea on a real prototype with live sensor streams.

---

## What the project does

### Simulation / layout-design side

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

### Physical prototype side

The real prototype script turns the research idea into a live demonstration system:

- reads **8 analog channels** from an ESP32 through serial,
- maintains a rolling baseline and detects impact events,
- extracts first-arrival times and peak amplitudes from each event,
- estimates location in an extended search area around a **70 x 70 central net**,
- visualizes sensors, calibration points, and predicted impact position in a PyQtGraph GUI.

The prototype supports **two runtime localization modes** and one calibration mode:

1. **Physics-based localization**  
   Searches over a 2D grid and a bank of wave-speed candidates, then minimizes a score combining:
   - relative arrival-time mismatch,
   - amplitude-pattern mismatch,
   - arrival-order mismatch.

2. **Template mapping localization**  
   Uses manually collected calibration taps at predefined positions outside the net, builds robust pairwise time-difference templates, and matches new events against those templates.

3. **Mapping calibration mode**  
   Guides the user through repeated taps at each calibration point, stores valid features, rejects outliers, and builds a reusable JSON calibration map.

This means the repository is not only about simulation-based layout search; it also contains a **working hardware inference layer** for physical validation.
For a visual overview of the hardware experiment, see the accompanying YouTube demonstration video: [Detailed Experiment Video](https://youtu.be/zfnjO2ME83k).

---

## Objective function

The main scoring logic for the simulation study lives in:

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
├── LayoutStudy/
│   ├── RunThis(TopLayer).py
│   ├── layout_objective_evaluator.py
│   ├── reevaluate_top50_layouts.py
│   ├── cmaes_optimize_from_top_seeds.py
│   ├── analyze_layout_features_complete.py
│   ├── run_symmetric_layout_candidate.py
│   ├── Result/
│   ├── feature_analysis_complete/
│   ├── single_layout_symmetric_eval/
│   └── Legacy/
├── physical_prototype.py
└── Test-waveform.ino
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

## Main workflow in `LayoutStudy/`

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

## Physical prototype: architecture and logic

The physical prototype code is a standalone real-time application intended for the **actual 8-sensor net**.

### Hardware and signal assumptions

The script assumes:

- **8 sensor channels**,
- serial streaming at **115200 baud**,
- one row per sample containing **8 comma-separated numeric values**,
- an ESP32-side pin mapping for sensors `S1` to `S8`,
- a sampling rate of **800 Hz**,
- sensor coordinates fixed in a measured local layout that is then shifted into a larger global search frame.

A central design choice is that the searchable area is **larger than the net itself**:

- outer search domain: `x, y in [-20, 120]`,
- physical net domain: `70 x 70`, offset to `[15, 85] x [15, 85]`.

This allows the GUI to predict impacts both **inside the net** and **outside the net boundary**, which matches the calibration strategy used later in the script.

### Real-time event pipeline

For every incoming batch of serial data, the script does the following:

1. **Read and clean rows**  
   Ignores malformed rows, comments, headers, and incomplete packets.

2. **Maintain baseline**  
   Stores quiet-time samples in a rolling baseline buffer and estimates baseline with a per-channel median.

3. **Trigger event capture**  
   Starts event capture when the maximum relative response exceeds a trigger threshold and at least two channels rise above a secondary threshold.

4. **Analyze captured event**  
   Computes:
   - per-channel local baseline,
   - robust noise estimate,
   - positive signal envelope,
   - peak amplitude,
   - first threshold-crossing arrival time.

5. **Reject weak events**  
   Events are discarded if too few channels have meaningful peaks or valid arrivals.

6. **Localize event**  
   Depending on the mode, the event is routed to either:
   - mapping calibration,
   - mapping-template runtime localization,
   - physics-based runtime localization.

7. **Update GUI**  
   The plot refreshes live sensor intensity, predicted impact position, mapping-point state, and diagnostic text such as score, velocity estimate, and arrivals.

### Physics-based runtime localization

When mapping mode is not enabled, the script performs brute-force localization over a 2D search grid.

For each candidate point it computes:

- distance from candidate point to each valid sensor,
- a predicted amplitude pattern using inverse-distance decay,
- a predicted arrival-time pattern for each candidate propagation speed,
- a composite score made from:
  - time-difference error,
  - amplitude mismatch,
  - arrival-order rank error.

The best `(x, y, v)` combination is retained.

This solver is lightweight and practical for a prototype because it avoids explicit PDE modeling while still using three physically meaningful cues:

- **when** each channel responds,
- **how strongly** each channel responds,
- **in what order** channels respond.

### Mapping calibration and template matching

A second inference mode is based on **empirical calibration**, which is useful when the real prototype behaves less ideally than the simulation.

The mapping workflow is:

1. The user taps a set of predefined calibration points.
2. Each accepted event is converted into a pairwise arrival-time-difference feature vector.
3. Multiple taps are collected per calibration point.
4. Outliers are rejected using robust z-score filtering.
5. A median template and spread are stored for each point.
6. At runtime, a new event is matched against the saved templates.
7. The top few template matches are combined by inverse-score weighting to interpolate position.

In the current script, the calibration points are placed around the outside of the net:

- `LT`, `LM`, `LB`,
- `TM`,
- `RT`, `RM`, `RB`,
- `BM`.

This makes the mapping mode especially suitable for **outside-impact classification/localization**.

### GUI behavior

The interface is implemented with **PyQt6 + PyQtGraph** and provides:

- live visualization of sensor responses,
- sensor labels and current values,
- the central net boundary and larger outer boundary,
- calibration-point markers and progress counters,
- buttons for starting calibration, advancing points, building a map, saving a map, and loading a map,
- a runtime checkbox to switch to mapping-based inference.

The prototype script therefore acts as both:

- a **demo/experiment tool**, and
- a **debugging interface** for real hardware testing.

### Relationship to the simulation study

The physical prototype is conceptually downstream of the simulation work:

- the simulation identifies promising 8-node layouts and studies geometry/performance trade-offs,
- the prototype fixes one concrete sensor layout,
- the runtime code tests whether arrival-time logic can work on real signals,
- the mapping mode compensates for real-world effects not fully captured by the idealized forward model.

So the repository can be read as a progression from:

**simulation -> layout selection -> hardware implementation -> live localization**.

---

## Key scripts and their roles

### `layout_objective_evaluator.py`

This is the **core simulation engine**. It contains:

- layout validation,
- terrain generation,
- terrain-conforming deployment,
- surface graph / geodesic construction,
- event-bank generation,
- amplitude and SNR model,
- TDoA observation building,
- multi-start localization,
- terrain summary and final objective calculation.

If you only want one file to understand the simulation logic, start here.

### `RunThis(TopLayer).py`

Top-level entry point for random layout search.

### `reevaluate_top50_layouts.py`

Turns coarse ranking into more statistically stable ranking.

### `cmaes_optimize_from_top_seeds.py`

Searches locally around good seeds instead of relying only on random layouts.

### `analyze_layout_features_complete.py`

Explains *why* some layouts perform better.

### `physical_prototype.py`

Top-level real-time prototype application for the physical system. It contains:

- serial acquisition,
- baseline and trigger logic,
- event segmentation,
- arrival/peak extraction,
- brute-force physics-based localization,
- mapping calibration and JSON map persistence,
- live PyQt visualization.

If you want to understand how the simulation idea was transferred into a working hardware demo, read this file after `layout_objective_evaluator.py`.

### `Test-waveform.ino`

A lightweight ESP32-side diagnostic script for the physical prototype. It reads the same 8 analogue sensor channels used by **`physical_prototype.py`**, estimates a startup baseline for each channel, 
and streams baseline-relative channel responses over serial as comma-separated values at 115200 baud.

This script is intended as a quick hardware-response check rather than a full localization firmware. Its main purpose is to verify that all sensor channels are connected correctly, 
that controlled impacts produce observable multi-channel responses, and that the ESP32-to-PC serial path is functioning before running the full Python GUI pipeline.

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

Use Python 3.10+.

### Simulation dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### Physical prototype dependencies

```bash
pip install numpy pyserial PyQt6 pyqtgraph
```

If you want both environments available in one place, install the union of both dependency sets.

---

## Quick start

### Simulation workflow

Run commands from:

```bash
cd LayoutStudy
```

#### Coarse random search

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

#### Reevaluate the top 50 layouts

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

#### Optimize top layouts with CMA-ES

```bash
python cmaes_optimize_from_top_seeds.py \
  --result-dir Result \
  --top-k-seeds 5 \
  --max-gens 25 \
  --sigma0 0.05 \
  --n-events 100 \
  --n-seeds 5
```

#### Analyze geometry-performance relationships

```bash
python analyze_layout_features_complete.py \
  --all-csv Result/random_layout_5000_all_results.csv \
  --reeval-csv Result/reeval_top50_summary.csv \
  --outdir feature_analysis_complete
```

#### Evaluate the hand-designed symmetric layout

```bash
python run_symmetric_layout_candidate.py
```

For a faster sanity check:

```bash
python run_symmetric_layout_candidate.py --quick
```

### Physical prototype workflow

Run from the repository root:

```bash
python physical_prototype.py
```

Before running, update the configuration section if needed:

- `SERIAL_PORT`
- `BAUD_RATE`
- `DISABLED_SENSORS`
- `sensor_coords_local`
- trigger / arrival thresholds
- search-grid size and velocity candidates

Typical usage:

1. Connect the ESP32 and start serial streaming.
2. Launch the GUI.
3. Verify that live channel values are updating.
4. Use physics mode directly, or start mapping calibration.
5. For mapping mode:
   - click **Start mapping calibration**,
   - tap each calibration point multiple times,
   - click **Build map**,
   - optionally save the map as JSON,
   - enable **Use mapping at runtime**.

---

## Important modeling assumptions

### Simulation assumptions

- The domain is normalized to a unit square.
- Layouts are mainly evaluated for **8 sensors**.
- Terrain deployment is not rigid; nodes can move in-plane to conform to the surface.
- Rough terrain is sampled randomly, so repeated evaluations matter.
- Later-stage scripts enforce fairness by sharing event banks across terrain/layout comparisons.
- Detection is SNR-threshold-based in the later model stages.
- Localization is performed on the terrain surface, not in unconstrained 3D space.

### Prototype assumptions

- The incoming serial stream already contains synchronized 8-channel sensor values.
- The physical sensor order must match the hard-coded sensor coordinates.
- Arrival extraction is threshold-based, so sensor gain/noise consistency matters.
- Mapping-based localization depends on the quality and repeatability of calibration taps.
- The current runtime configuration is especially oriented toward **outside-net** localization experiments.

---

## Practical reading order

If you are new to the codebase, this is the most efficient order:

1. `LayoutStudy/layout_objective_evaluator.py`  
   Understand the full simulation and scoring logic.
2. `LayoutStudy/LayoutStudy/RunThis(TopLayer).py`  
   See how candidate layouts are generated and ranked.
3. `LayoutStudy/reevaluate_top50_layouts.py`  
   See how unstable coarse rankings are cleaned up.
4. `LayoutStudy/cmaes_optimize_from_top_seeds.py`  
   See how the search is refined.
5. `LayoutStudy/analyze_layout_features_complete.py`  
   See how performance is interpreted.
6. `physical_prototype.py`  
   See how the selected layout and timing logic are used on real hardware.
7. `Initial-simulation/`  
   Read the historical steps if you want to follow the model evolution from first principles.

---

## Limitations

- The project is organized as research scripts, not as a packaged library.
- There is no automated test suite yet.
- Some metadata files still contain original local Windows paths from the authoring environment.
- Some scripts are computationally heavy at full settings.
- The simulation code is tightly centered on normalized 8-node layouts in a square domain.
- The prototype code currently hard-codes serial configuration, sensor coordinates, thresholds, and calibration points.
- The prototype GUI is meant for lab/demo use rather than production deployment.

---

## Suggested next improvements

- factor the common simulation code into a small reusable package,
- move prototype configuration into a JSON/YAML file,
- add unit tests for terrain generation, deployment, event parsing, and objective evaluation,
- separate hardware IO from localization logic,
- add a single master pipeline script,
- store result metadata more consistently,
- add notebooks or examples for visualization and reproducibility,
- add documentation for the ESP32 firmware packet format.

---

## Summary

This repository is best understood as a **complete experimental pipeline for impact localization**:

- it starts from terrain-aware simulation and layout optimization,
- uses geodesic-aware TDoA localization in the virtual study,
- scores layouts with both accuracy and robustness terms,
- compares layouts fairly through frozen Monte Carlo banks,
- refines the best layouts by reevaluation and CMA-ES,
- then carries the chosen logic into a real 8-sensor prototype,
- and supports live localization through both a physics-based solver and a calibration-template mode.

So the full story of the repo is:

**model the problem -> optimize layout -> analyze geometry -> build hardware -> localize real impacts**.
