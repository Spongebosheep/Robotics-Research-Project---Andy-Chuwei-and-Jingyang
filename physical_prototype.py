import sys
import json
import serial
import numpy as np
from collections import deque
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg


# =========================================================
# CONFIG
# =========================================================
SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUD_RATE = 115200
NUM_SENSORS = 8
READ_BATCH = 30

# ESP32 pin mapping (firmware side)
# S1 -> GPIO34
# S2 -> GPIO35
# S3 -> GPIO32
# S4 -> GPIO33
# S5 -> GPIO25
# S6 -> GPIO26
# S7 -> GPIO27
# S8 -> GPIO12

# Disabled channels (0-based index)
DISABLED_SENSORS = set()


# =========================================================
# GLOBAL SEARCH / DISPLAY AREA
# Allows prediction outside the net
# =========================================================
GRID_X_MIN = -20
GRID_X_MAX = 120
GRID_Y_MIN = -20
GRID_Y_MAX = 120
GRID_W = GRID_X_MAX - GRID_X_MIN
GRID_H = GRID_Y_MAX - GRID_Y_MIN

# Central net: 70 x 70
NET_SIZE = 70
NET_OFFSET_X = 15
NET_OFFSET_Y = 15
NET_X_MIN = NET_OFFSET_X
NET_X_MAX = NET_OFFSET_X + NET_SIZE
NET_Y_MIN = NET_OFFSET_Y
NET_Y_MAX = NET_OFFSET_Y + NET_SIZE


# =========================================================
# SAMPLING / EVENT CONFIG
# =========================================================
SAMPLE_RATE_HZ = 800.0
DT = 1.0 / SAMPLE_RATE_HZ

TRIGGER_THRESHOLD = 8.0
ARRIVAL_THRESHOLD = 2.0
PRE_SAMPLES = 8
POST_SAMPLES = 80
BASELINE_LEN = 50


# =========================================================
# PHYSICS-BASED LOCALIZATION CONFIG
# =========================================================
TIME_WEIGHT = 1.0
AMP_WEIGHT = 0.35
RANK_WEIGHT = 0.25

AMP_ALPHA = 1.2
EPS = 1e-6

VELOCITY_CANDIDATES = np.linspace(80, 600, 20)

SEARCH_NX = 61
SEARCH_NY = 61

MIN_VALID_ARRIVALS = 3
MIN_PEAK_CHANNELS = 3
MIN_PEAK_VALUE = 2.0


# =========================================================
# MAPPING / CALIBRATION CONFIG
# =========================================================
MAPPING_TAPS_PER_POINT = 5

CALIBRATION_POINTS = [
    {"label": "LT", "xy": [8, 92]},
    {"label": "LM", "xy": [8, 50]},
    {"label": "LB", "xy": [8, 8]},
    {"label": "TM", "xy": [50, 92]},
    {"label": "RT", "xy": [92, 92]},
    {"label": "RM", "xy": [92, 50]},
    {"label": "RB", "xy": [92, 8]},
    {"label": "BM", "xy": [50, 8]},
]


# =========================================================
# SENSOR LAYOUT
# Local coordinates inside the 70x70 net
# IMPORTANT: final positions should match the real prototype
# =========================================================
sensor_coords_local = np.array([
    [12, 28],  # S1
    [17, 42],  # S2
    [21, 55],  # S3
    [30, 12],  # S4
    [38, 64],  # S5
    [39, 25],  # S6
    [54, 35],  # S7
    [56, 14],  # S8
], dtype=float)

sensor_coords = sensor_coords_local + np.array([NET_OFFSET_X, NET_OFFSET_Y], dtype=float)
sensor_names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


# =========================================================
# SERIAL
# =========================================================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.03)


# =========================================================
# RUNTIME STATE
# =========================================================
latest_values = np.zeros(NUM_SENSORS)

baseline_buffer = deque(maxlen=BASELINE_LEN)
pre_trigger_buffer = deque(maxlen=PRE_SAMPLES)

capture_mode = False
captured_rows = []

last_est_xy = None
last_est_score = None
last_arrivals = None
last_peaks = None
last_best_v = None
last_region_text = "unknown"
last_max_rel = 0.0
last_status_text = "idle"


# =========================================================
# MAPPING STATE
# =========================================================
mapping_calibration_mode = False
mapping_runtime_enabled = False
mapping_ready = False

mapping_current_point_idx = 0
mapping_current_tap_count = 0

mapping_samples = [[] for _ in range(len(CALIBRATION_POINTS))]
mapping_templates = []


# =========================================================
# GENERAL HELPERS
# =========================================================
def mask_disabled_channels(arr):
    arr = np.array(arr, dtype=float, copy=True)
    if arr.ndim == 1:
        for idx in DISABLED_SENSORS:
            if 0 <= idx < len(arr):
                arr[idx] = 0.0
    elif arr.ndim == 2:
        for idx in DISABLED_SENSORS:
            if 0 <= idx < arr.shape[1]:
                arr[:, idx] = 0.0
    return arr


def read_serial_batch():
    rows = []
    for _ in range(READ_BATCH):
        try:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue
            if raw == "START":
                continue
            if raw.startswith("s1,"):
                continue
            if raw.startswith("#"):
                continue

            parts = raw.split(",")
            if len(parts) != 8:
                continue

            vals = [float(x) for x in parts]
            rows.append(vals)

        except Exception:
            continue

    return rows


def get_baseline():
    if len(baseline_buffer) < 5:
        return np.zeros(NUM_SENSORS)

    arr = np.array(baseline_buffer, dtype=float)
    arr = mask_disabled_channels(arr)
    return np.median(arr, axis=0)


def detect_trigger(row, baseline):
    rel = row - baseline
    rel = mask_disabled_channels(rel)
    return (np.max(rel) > TRIGGER_THRESHOLD) and (np.sum(rel > 3.0) >= 2)


def analyze_event(event_rows):
    arr = np.array(event_rows, dtype=float)
    arr = mask_disabled_channels(arr)

    pre = arr[:max(PRE_SAMPLES, 3), :]
    baseline = np.median(pre, axis=0)

    noise = 1.4826 * np.median(np.abs(pre - baseline), axis=0) + 1e-6
    noise = mask_disabled_channels(noise)

    signal = arr - baseline
    signal = np.maximum(signal, 0.0)
    signal = mask_disabled_channels(signal)

    peaks = np.max(signal, axis=0)
    peaks = mask_disabled_channels(peaks)

    arrivals = np.full(NUM_SENSORS, np.nan)
    valid = np.zeros(NUM_SENSORS, dtype=bool)

    for i in range(NUM_SENSORS):
        if i in DISABLED_SENSORS:
            arrivals[i] = np.nan
            valid[i] = False
            continue

        th = max(ARRIVAL_THRESHOLD, 3.0 * noise[i])
        idx = np.where(signal[:, i] > th)[0]
        if len(idx) > 0:
            arrivals[i] = idx[0] * DT
            valid[i] = True

    return arrivals, peaks, valid, noise, signal


def normalize_peaks(peaks):
    p = np.array(peaks, dtype=float)
    p = mask_disabled_channels(p)
    p = np.maximum(p, 0.0)

    s = np.sum(p)
    if s < 1e-9:
        return np.zeros_like(p)

    return p / s


def classify_region(x, y):
    inside_x = (NET_X_MIN <= x <= NET_X_MAX)
    inside_y = (NET_Y_MIN <= y <= NET_Y_MAX)

    if inside_x and inside_y:
        return "inside net"

    horiz = ""
    vert = ""

    if x < NET_X_MIN:
        horiz = "left outside"
    elif x > NET_X_MAX:
        horiz = "right outside"

    if y < NET_Y_MIN:
        vert = "bottom outside"
    elif y > NET_Y_MAX:
        vert = "top outside"

    if horiz and vert:
        return f"{vert} + {horiz}"
    elif horiz:
        return horiz
    elif vert:
        return vert
    else:
        return "boundary / unknown"


def arrival_rank_error(measured_arrivals, pred_arrivals):
    ma = np.array(measured_arrivals, dtype=float)
    pa = np.array(pred_arrivals, dtype=float)

    if len(ma) < 3:
        return 0.0

    order_m = np.argsort(ma)
    order_p = np.argsort(pa)

    rank_m = np.empty_like(order_m)
    rank_p = np.empty_like(order_p)

    rank_m[order_m] = np.arange(len(ma))
    rank_p[order_p] = np.arange(len(pa))

    return np.mean((rank_m - rank_p) ** 2)


def estimate_location(arrivals, peaks, valid_mask, region_mode="all"):
    valid_idx = np.where(valid_mask)[0]
    valid_idx = np.array([i for i in valid_idx if i not in DISABLED_SENSORS], dtype=int)

    if len(valid_idx) < MIN_VALID_ARRIVALS:
        return None, None, None

    arr_valid = arrivals[valid_idx]
    t0 = np.nanmin(arr_valid)
    measured_dt = arr_valid - t0
    measured_peaks = normalize_peaks(peaks[valid_idx])

    xs = np.linspace(GRID_X_MIN, GRID_X_MAX, SEARCH_NX)
    ys = np.linspace(GRID_Y_MIN, GRID_Y_MAX, SEARCH_NY)

    best_score = np.inf
    best_xy = None
    best_v = None

    for x in xs:
        for y in ys:
            inside = (NET_X_MIN <= x <= NET_X_MAX) and (NET_Y_MIN <= y <= NET_Y_MAX)

            if region_mode == "outside_only" and inside:
                continue
            if region_mode == "inside_only" and not inside:
                continue

            p = np.array([x, y], dtype=float)
            dists = np.linalg.norm(sensor_coords[valid_idx] - p, axis=1) + EPS

            pred_amp = 1.0 / (dists ** AMP_ALPHA)
            pred_amp = pred_amp / (np.sum(pred_amp) + EPS)
            amp_err = np.mean((pred_amp - measured_peaks) ** 2)

            for v in VELOCITY_CANDIDATES:
                pred_t = dists / v
                pred_dt = pred_t - np.min(pred_t)

                time_err = np.mean((pred_dt - measured_dt) ** 2)
                rank_err = arrival_rank_error(measured_dt, pred_dt)

                score = (
                    TIME_WEIGHT * time_err
                    + AMP_WEIGHT * amp_err
                    + RANK_WEIGHT * rank_err
                )

                if score < best_score:
                    best_score = score
                    best_xy = (x, y)
                    best_v = v

    return best_xy, best_score, best_v


def clamp_xy(xy):
    if xy is None:
        return None

    x, y = xy
    x = max(GRID_X_MIN, min(GRID_X_MAX, x))
    y = max(GRID_Y_MIN, min(GRID_Y_MAX, y))
    return (x, y)


# =========================================================
# MAPPING HELPERS
# =========================================================
def is_inside_net_xy(x, y):
    return (NET_X_MIN <= x <= NET_X_MAX) and (NET_Y_MIN <= y <= NET_Y_MAX)


def build_dt_feature(arrivals, valid):
    valid_idx = np.where(valid)[0]
    valid_idx = np.array([i for i in valid_idx if i not in DISABLED_SENSORS], dtype=int)

    if len(valid_idx) < MIN_VALID_ARRIVALS:
        return None

    arr_valid = arrivals[valid_idx]
    if np.any(np.isnan(arr_valid)):
        return None

    pairs = []
    values = []

    for ii in range(len(valid_idx)):
        for jj in range(ii + 1, len(valid_idx)):
            i = valid_idx[ii]
            j = valid_idx[jj]
            pairs.append((int(i), int(j)))
            values.append(float(arrivals[i] - arrivals[j]))

    if len(values) == 0:
        return None

    return {
        "pairs": pairs,
        "values": np.array(values, dtype=float),
        "valid_idx": valid_idx.tolist(),
    }


def feature_to_pair_dict(feature):
    return {tuple(p): float(v) for p, v in zip(feature["pairs"], feature["values"])}


def add_mapping_sample(point_idx, arrivals, valid):
    global mapping_current_tap_count, mapping_current_point_idx
    global mapping_samples, mapping_calibration_mode, last_status_text

    feat = build_dt_feature(arrivals, valid)
    if feat is None:
        last_status_text = f"mapping sample rejected at {CALIBRATION_POINTS[point_idx]['label']}"
        return False

    mapping_samples[point_idx].append({
        "pairs": feat["pairs"],
        "values": feat["values"].tolist(),
        "valid_idx": feat["valid_idx"],
    })

    mapping_current_tap_count = len(mapping_samples[point_idx])
    last_status_text = (
        f"mapping {CALIBRATION_POINTS[point_idx]['label']}: "
        f"{mapping_current_tap_count}/{MAPPING_TAPS_PER_POINT}"
    )

    if mapping_current_tap_count >= MAPPING_TAPS_PER_POINT:
        if mapping_current_point_idx < len(CALIBRATION_POINTS) - 1:
            mapping_current_point_idx += 1
            mapping_current_tap_count = len(mapping_samples[mapping_current_point_idx])
            last_status_text = (
                f"{CALIBRATION_POINTS[point_idx]['label']} done -> "
                f"move to {CALIBRATION_POINTS[mapping_current_point_idx]['label']}"
            )
        else:
            mapping_calibration_mode = False
            last_status_text = "all mapping points collected; click Build Map"

    return True


def finalize_point_template(samples_for_one_point):
    if len(samples_for_one_point) == 0:
        return None

    common_pairs = None
    for s in samples_for_one_point:
        pair_set = set(tuple(p) for p in s["pairs"])
        if common_pairs is None:
            common_pairs = pair_set
        else:
            common_pairs = common_pairs.intersection(pair_set)

    if not common_pairs:
        return None

    common_pairs = sorted(list(common_pairs))
    mat = []

    for s in samples_for_one_point:
        d = {tuple(p): float(v) for p, v in zip(s["pairs"], s["values"])}
        row = [d[p] for p in common_pairs]
        mat.append(row)

    mat = np.array(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return None

    med = np.median(mat, axis=0)
    mad = np.median(np.abs(mat - med), axis=0) + 1e-9
    robust_z = np.max(np.abs(mat - med) / (1.4826 * mad + 1e-9), axis=1)

    keep = robust_z < 4.0
    if np.sum(keep) >= 1:
        mat = mat[keep]

    template = np.median(mat, axis=0)
    spread = np.std(mat, axis=0) + 1e-6

    return {
        "pairs": [list(p) for p in common_pairs],
        "template": template.tolist(),
        "spread": spread.tolist(),
        "num_samples": int(mat.shape[0]),
    }


def build_mapping_templates():
    global mapping_templates, mapping_ready

    templates = []

    for idx, cp in enumerate(CALIBRATION_POINTS):
        tpl = finalize_point_template(mapping_samples[idx])
        if tpl is None:
            continue

        templates.append({
            "point_index": int(idx),
            "label": cp["label"],
            "xy": [float(cp["xy"][0]), float(cp["xy"][1])],
            "pairs": tpl["pairs"],
            "template": tpl["template"],
            "spread": tpl["spread"],
            "num_samples": tpl["num_samples"],
        })

    mapping_templates = templates
    mapping_ready = len(mapping_templates) >= 3
    return mapping_ready


def match_map(arrivals, valid):
    feat = build_dt_feature(arrivals, valid)
    if feat is None or not mapping_ready or len(mapping_templates) == 0:
        return None, None, None

    feat_dict = feature_to_pair_dict(feat)

    best_score = np.inf
    best_xy = None
    all_scores = []

    for tpl in mapping_templates:
        tpl_pairs = [tuple(p) for p in tpl["pairs"]]
        shared = [p for p in tpl_pairs if p in feat_dict]

        if len(shared) < 3:
            continue

        tpl_dict = {tuple(p): float(v) for p, v in zip(tpl["pairs"], tpl["template"])}
        spd_dict = {tuple(p): float(v) for p, v in zip(tpl["pairs"], tpl["spread"])}

        diffs = []
        for p in shared:
            diffs.append((feat_dict[p] - tpl_dict[p]) / (spd_dict[p] + 1e-6))

        diffs = np.array(diffs, dtype=float)
        score = float(np.mean(diffs ** 2))

        tpl_xy = np.array(tpl["xy"], dtype=float)
        all_scores.append((score, tpl_xy))

        if score < best_score:
            best_score = score
            best_xy = tuple(tpl["xy"])

    if best_xy is None:
        return None, None, None

    all_scores.sort(key=lambda x: x[0])
    topk = all_scores[:min(3, len(all_scores))]

    weights = np.array([1.0 / (s + 1e-6) for s, _ in topk], dtype=float)
    pts = np.array([xy for _, xy in topk], dtype=float)
    xy_interp = np.sum(pts * weights[:, None], axis=0) / np.sum(weights)

    if is_inside_net_xy(float(xy_interp[0]), float(xy_interp[1])):
        xy_final = best_xy
    else:
        xy_final = (float(xy_interp[0]), float(xy_interp[1]))

    return xy_final, best_score, len(topk)


def save_mapping_map(filepath):
    payload = {
        "calibration_points": CALIBRATION_POINTS,
        "taps_per_point": int(MAPPING_TAPS_PER_POINT),
        "templates": mapping_templates,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_mapping_map(filepath):
    global mapping_templates, mapping_ready

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping_templates = payload.get("templates", [])
    mapping_ready = len(mapping_templates) >= 3
    return mapping_ready


# =========================================================
# QT WINDOW
# =========================================================
class LiveMapWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Impact Localization (Irregular Node Layout)")
        self.resize(1180, 1000)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)

        # -------------------------
        # Controls
        # -------------------------
        ctrl_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_layout)

        self.btn_start_mapping = QtWidgets.QPushButton("Start mapping calibration")
        self.btn_next_point = QtWidgets.QPushButton("Next point")
        self.btn_build_map = QtWidgets.QPushButton("Build map")
        self.btn_save_map = QtWidgets.QPushButton("Save map")
        self.btn_load_map = QtWidgets.QPushButton("Load map")
        self.chk_use_mapping = QtWidgets.QCheckBox("Use mapping at runtime")

        ctrl_layout.addWidget(self.btn_start_mapping)
        ctrl_layout.addWidget(self.btn_next_point)
        ctrl_layout.addWidget(self.btn_build_map)
        ctrl_layout.addWidget(self.btn_save_map)
        ctrl_layout.addWidget(self.btn_load_map)
        ctrl_layout.addWidget(self.chk_use_mapping)

        self.btn_start_mapping.clicked.connect(self.start_mapping_calibration)
        self.btn_next_point.clicked.connect(self.next_mapping_point)
        self.btn_build_map.clicked.connect(self.build_map_clicked)
        self.btn_save_map.clicked.connect(self.save_map_clicked)
        self.btn_load_map.clicked.connect(self.load_map_clicked)
        self.chk_use_mapping.stateChanged.connect(self.toggle_mapping_runtime)

        # -------------------------
        # Status labels
        # -------------------------
        self.mapping_hint_label = QtWidgets.QLabel("Mapping: idle")
        self.mapping_hint_label.setStyleSheet(
            "font-size: 14px; padding: 6px; color: rgb(0, 220, 120);"
        )
        layout.addWidget(self.mapping_hint_label)

        self.info_label = QtWidgets.QLabel("Waiting for data...")
        self.info_label.setStyleSheet("font-size: 15px; padding: 6px;")
        layout.addWidget(self.info_label)

        # -------------------------
        # Plot
        # -------------------------
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)

        self.plot.setXRange(GRID_X_MIN, GRID_X_MAX)
        self.plot.setYRange(GRID_Y_MIN, GRID_Y_MAX)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)

        # Outer boundary
        self.plot.plot(
            [GRID_X_MIN, GRID_X_MAX, GRID_X_MAX, GRID_X_MIN, GRID_X_MIN],
            [GRID_Y_MIN, GRID_Y_MIN, GRID_Y_MAX, GRID_Y_MAX, GRID_Y_MIN],
            pen=pg.mkPen((180, 180, 180), width=2)
        )

        # Net boundary
        self.plot.plot(
            [NET_X_MIN, NET_X_MAX, NET_X_MAX, NET_X_MIN, NET_X_MIN],
            [NET_Y_MIN, NET_Y_MIN, NET_Y_MAX, NET_Y_MAX, NET_Y_MIN],
            pen=pg.mkPen((0, 255, 255), width=2)
        )

        self.net_text = pg.TextItem(
            text="70x70 net",
            color=(0, 255, 255),
            anchor=(0, 1)
        )
        self.net_text.setPos(NET_X_MIN + 1, NET_Y_MAX - 1)
        self.plot.addItem(self.net_text)

        # Sensor scatter
        self.sensor_scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.sensor_scatter)

        self.text_items = []
        for i, (x, y) in enumerate(sensor_coords):
            if i in DISABLED_SENSORS:
                txt = pg.TextItem(
                    text=f"{sensor_names[i]}: DISABLED",
                    color=(180, 180, 180),
                    anchor=(0, 1)
                )
            else:
                txt = pg.TextItem(
                    text=f"{sensor_names[i]}: 0",
                    color="w",
                    anchor=(0, 1)
                )

            txt.setPos(x + 1, y + 1)
            self.plot.addItem(txt)
            self.text_items.append(txt)

        # Predicted impact point
        self.impact_scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.impact_scatter)

        self.impact_text = pg.TextItem(text="", color="y", anchor=(0.5, 1.2))
        self.plot.addItem(self.impact_text)

        # Mapping points
        self.mapping_scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.mapping_scatter)

        self.mapping_text_items = []
        for idx, cp in enumerate(CALIBRATION_POINTS):
            x, y = cp["xy"]
            txt = pg.TextItem(
                text=cp["label"],
                color=(0, 220, 120),
                anchor=(0.5, 0)
            )
            txt.setPos(x, y + 1.5)
            self.plot.addItem(txt)
            self.mapping_text_items.append(txt)

        self.refresh_mapping_visuals()
        self.refresh_mapping_hint()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def start_mapping_calibration(self):
        global mapping_calibration_mode, mapping_current_point_idx, mapping_current_tap_count
        global mapping_samples, mapping_templates, mapping_ready, last_status_text

        mapping_calibration_mode = True
        mapping_current_point_idx = 0
        mapping_current_tap_count = 0
        mapping_samples = [[] for _ in range(len(CALIBRATION_POINTS))]
        mapping_templates = []
        mapping_ready = False

        last_status_text = f"mapping calibration started at {CALIBRATION_POINTS[0]['label']}"

        self.refresh_mapping_visuals()
        self.refresh_mapping_hint()

    def next_mapping_point(self):
        global mapping_current_point_idx, mapping_current_tap_count
        global mapping_calibration_mode, last_status_text

        if mapping_current_point_idx < len(CALIBRATION_POINTS) - 1:
            mapping_current_point_idx += 1
            mapping_current_tap_count = len(mapping_samples[mapping_current_point_idx])
            mapping_calibration_mode = True
            last_status_text = f"manual next -> {CALIBRATION_POINTS[mapping_current_point_idx]['label']}"
        else:
            mapping_calibration_mode = False
            last_status_text = "already at last point"

        self.refresh_mapping_visuals()
        self.refresh_mapping_hint()

    def build_map_clicked(self):
        global last_status_text

        ok = build_mapping_templates()
        counts = [len(s) for s in mapping_samples]
        built_labels = [tpl["label"] for tpl in mapping_templates]

        print("====================================")
        print("Build map clicked")
        print("sample counts per point:", counts)
        print("templates built:", len(mapping_templates))
        print("template labels:", built_labels)
        print("====================================")

        if ok:
            last_status_text = f"mapping ready with {len(mapping_templates)} templates"
            QtWidgets.QMessageBox.information(
                self,
                "Build Map",
                f"Build success.\n"
                f"Templates built: {len(mapping_templates)}\n"
                f"Sample counts: {counts}\n"
                f"Labels: {built_labels}"
            )
        else:
            last_status_text = f"mapping build failed; valid templates={len(mapping_templates)}"
            QtWidgets.QMessageBox.warning(
                self,
                "Build Map",
                f"Build failed.\n"
                f"Templates built: {len(mapping_templates)}\n"
                f"Sample counts: {counts}\n"
                f"Usually this means some points did not get enough valid taps."
            )

        self.refresh_mapping_visuals()
        self.refresh_mapping_hint()

    def save_map_clicked(self):
        global last_status_text

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save mapping map",
            "outside_mapping_map.json",
            "JSON Files (*.json)"
        )

        if not path:
            return

        try:
            save_mapping_map(path)
            last_status_text = f"map saved: {path}"
        except Exception as e:
            last_status_text = f"save failed: {e}"

        self.refresh_mapping_hint()

    def load_map_clicked(self):
        global last_status_text

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load mapping map",
            "",
            "JSON Files (*.json)"
        )

        if not path:
            return

        try:
            ok = load_mapping_map(path)
            if ok:
                last_status_text = f"map loaded: {path}"
            else:
                last_status_text = "map loaded but not ready"
        except Exception as e:
            last_status_text = f"load failed: {e}"

        self.refresh_mapping_visuals()
        self.refresh_mapping_hint()

    def toggle_mapping_runtime(self, state):
        global mapping_runtime_enabled, last_status_text

        mapping_runtime_enabled = bool(state)
        last_status_text = f"mapping runtime = {mapping_runtime_enabled}"
        self.refresh_mapping_hint()

    def refresh_mapping_visuals(self):
        spots = []

        for idx, cp in enumerate(CALIBRATION_POINTS):
            x, y = cp["xy"]
            n = len(mapping_samples[idx])

            if idx == mapping_current_point_idx and mapping_calibration_mode:
                brush = pg.mkBrush(255, 180, 0, 220)
                size = 20
            elif n >= MAPPING_TAPS_PER_POINT:
                brush = pg.mkBrush(0, 220, 120, 220)
                size = 16
            else:
                brush = pg.mkBrush(80, 120, 80, 180)
                size = 14

            spots.append({
                "pos": (float(x), float(y)),
                "size": size,
                "brush": brush,
                "pen": pg.mkPen("k", width=1.2)
            })

            self.mapping_text_items[idx].setText(f"{cp['label']}({n}/{MAPPING_TAPS_PER_POINT})")

        self.mapping_scatter.setData(spots)

    def refresh_mapping_hint(self):
        if mapping_calibration_mode:
            cp = CALIBRATION_POINTS[mapping_current_point_idx]
            count = len(mapping_samples[mapping_current_point_idx])

            self.mapping_hint_label.setText(
                f"Mapping: calibration mode | current point = {cp['label']} at {cp['xy']} | "
                f"taps = {count}/{MAPPING_TAPS_PER_POINT}"
            )

        elif mapping_runtime_enabled and mapping_ready:
            self.mapping_hint_label.setText(
                f"Mapping: runtime ENABLED | templates = {len(mapping_templates)} | outside-only map"
            )

        elif mapping_ready:
            self.mapping_hint_label.setText(
                f"Mapping: map ready | templates = {len(mapping_templates)} | runtime not enabled"
            )

        else:
            self.mapping_hint_label.setText("Mapping: idle / not ready")

    def update_frame(self):
        global latest_values
        global capture_mode, captured_rows
        global last_est_xy, last_est_score, last_arrivals, last_peaks
        global last_best_v, last_region_text, last_max_rel, last_status_text
        global mapping_calibration_mode, mapping_current_point_idx

        batch = read_serial_batch()
        if len(batch) == 0:
            self.info_label.setText("No serial data...")
            self.refresh_mapping_hint()
            return

        for row in batch:
            row = np.array(row, dtype=float)
            latest_values = row.copy()

            baseline = get_baseline()
            rel = row - baseline
            rel = mask_disabled_channels(rel)
            last_max_rel = float(np.max(rel))

            if not capture_mode:
                pre_trigger_buffer.append(row.copy())

                quiet = np.max(np.abs(rel)) < 5.0
                if quiet:
                    baseline_buffer.append(mask_disabled_channels(row.copy()))

                if detect_trigger(row, baseline):
                    print(f"TRIGGERED | max_rel={last_max_rel:.2f} | row={row.tolist()}")
                    last_status_text = "triggered"
                    capture_mode = True
                    captured_rows = list(pre_trigger_buffer)
                    captured_rows.append(row.copy())

            else:
                captured_rows.append(row.copy())

                if len(captured_rows) >= PRE_SAMPLES + POST_SAMPLES:
                    event_arr = np.array(captured_rows, dtype=float)
                    event_arr_masked = mask_disabled_channels(event_arr)

                    print("event max by channel:", np.max(event_arr_masked, axis=0).tolist())
                    print("event min by channel:", np.min(event_arr_masked, axis=0).tolist())

                    arrivals, peaks, valid, noise, signal = analyze_event(captured_rows)

                    num_peak_channels = int(np.sum(peaks > MIN_PEAK_VALUE))
                    num_valid_arrivals = int(np.sum(valid))

                    print("local noise:", np.round(noise, 2).tolist())
                    print("peaks:", np.round(peaks, 2).tolist())
                    print("valid arrivals:", num_valid_arrivals)
                    print("peak channels:", num_peak_channels)

                    if num_peak_channels < MIN_PEAK_CHANNELS or num_valid_arrivals < MIN_VALID_ARRIVALS:
                        print(
                            "Rejected empty/weak event",
                            "peak_channels=", num_peak_channels,
                            "valid_arrivals=", num_valid_arrivals
                        )
                        last_status_text = (
                            f"rejected weak event (peaks={num_peak_channels}, arrivals={num_valid_arrivals})"
                        )
                        capture_mode = False
                        captured_rows = []
                        pre_trigger_buffer.clear()
                        self.refresh_mapping_hint()
                        continue

                    # -------------------------------------------------
                    # Mode 1: Mapping calibration
                    # -------------------------------------------------
                    if mapping_calibration_mode:
                        current_label = CALIBRATION_POINTS[mapping_current_point_idx]["label"]
                        current_xy = CALIBRATION_POINTS[mapping_current_point_idx]["xy"]

                        last_arrivals = arrivals
                        last_peaks = peaks
                        last_est_xy = tuple(current_xy)
                        last_region_text = f"mapping {current_label}"
                        last_est_score = None
                        last_best_v = None

                        ok = add_mapping_sample(mapping_current_point_idx, arrivals, valid)

                        if ok:
                            print(f"Mapping sample accepted for {current_label}")
                        else:
                            print(f"Mapping sample rejected for {current_label}")

                        self.refresh_mapping_visuals()
                        self.refresh_mapping_hint()

                    # -------------------------------------------------
                    # Mode 2: Mapping runtime localization
                    # -------------------------------------------------
                    elif mapping_runtime_enabled and mapping_ready:
                        est_xy, score, used_k = match_map(arrivals, valid)

                        last_arrivals = arrivals
                        last_peaks = peaks
                        last_est_xy = clamp_xy(est_xy)
                        last_est_score = score
                        last_best_v = None

                        if last_est_xy is not None:
                            last_region_text = f"mapping match outside (k={used_k})"
                            last_status_text = "valid event (mapping)"
                        else:
                            last_region_text = "unknown"
                            last_status_text = "mapping failed"

                        arr_ms = []
                        for i in range(NUM_SENSORS):
                            if i in DISABLED_SENSORS:
                                arr_ms.append("DIS")
                            elif np.isnan(arrivals[i]):
                                arr_ms.append("--")
                            else:
                                arr_ms.append(f"{arrivals[i] * 1000:.2f}")

                        print("====================================")
                        print("MAPPING Predicted XY:", last_est_xy)
                        print("Region      :", last_region_text)
                        print("Score       :", last_est_score)
                        print("Arrivals ms :", arr_ms)
                        print("Peaks       :", np.round(peaks, 2).tolist())
                        print("====================================")

                    # -------------------------------------------------
                    # Mode 3: Physics-based localization
                    # -------------------------------------------------
                    else:
                        est_xy, score, best_v = estimate_location(
                            arrivals,
                            peaks,
                            valid,
                            region_mode="outside_only"
                        )

                        last_arrivals = arrivals
                        last_peaks = peaks
                        last_est_xy = clamp_xy(est_xy)
                        last_est_score = score
                        last_best_v = best_v

                        if last_est_xy is not None:
                            last_region_text = classify_region(last_est_xy[0], last_est_xy[1])
                            last_status_text = "valid event"
                        else:
                            last_region_text = "unknown"
                            last_status_text = "valid event but no location"

                        arr_ms = []
                        for i in range(NUM_SENSORS):
                            if i in DISABLED_SENSORS:
                                arr_ms.append("DIS")
                            elif np.isnan(arrivals[i]):
                                arr_ms.append("--")
                            else:
                                arr_ms.append(f"{arrivals[i] * 1000:.2f}")

                        print("====================================")
                        print("Predicted XY:", last_est_xy)
                        print("Region      :", last_region_text)
                        print("Score       :", last_est_score)
                        print("Best v      :", last_best_v)
                        print("Arrivals ms :", arr_ms)
                        print("Peaks       :", np.round(peaks, 2).tolist())
                        print("====================================")

                    capture_mode = False
                    captured_rows = []
                    pre_trigger_buffer.clear()

        baseline = get_baseline()
        rel_now = np.maximum(latest_values - baseline, 0.0)
        rel_now = mask_disabled_channels(rel_now)
        norm = rel_now / (np.max(rel_now) + 1e-6)

        spots = []
        for i, (x, y) in enumerate(sensor_coords):
            if i in DISABLED_SENSORS:
                spots.append({
                    "pos": (x, y),
                    "size": 18,
                    "brush": pg.mkBrush(120, 120, 120, 180),
                    "pen": pg.mkPen((220, 220, 220), width=1.5)
                })
                self.text_items[i].setText(f"{sensor_names[i]}: DISABLED")
                continue

            intensity = float(norm[i])
            size = 16 + 22 * intensity
            brush = pg.mkBrush(int(255 * intensity), 100, 255, 220)

            spots.append({
                "pos": (x, y),
                "size": size,
                "brush": brush,
                "pen": pg.mkPen("w", width=1.5)
            })

            self.text_items[i].setText(f"{sensor_names[i]}: {int(latest_values[i])}")

        self.sensor_scatter.setData(spots)

        self.refresh_mapping_visuals()
        self.refresh_mapping_hint()

        if last_est_xy is not None:
            x, y = last_est_xy

            if "inside" in last_region_text:
                point_brush = pg.mkBrush(255, 255, 0, 230)
            elif "mapping" in last_region_text:
                point_brush = pg.mkBrush(0, 255, 120, 230)
            else:
                point_brush = pg.mkBrush(255, 80, 80, 230)

            self.impact_scatter.setData([{
                "pos": (x, y),
                "size": 24,
                "brush": point_brush,
                "pen": pg.mkPen("k", width=2)
            }])

            self.impact_text.setText(f"Predicted\n({x:.1f}, {y:.1f})")
            self.impact_text.setPos(x, y)

            arr_text = []
            if last_arrivals is not None:
                for i in range(NUM_SENSORS):
                    if i in DISABLED_SENSORS:
                        arr_text.append(f"{sensor_names[i]}: DIS")
                    elif np.isnan(last_arrivals[i]):
                        arr_text.append(f"{sensor_names[i]}: --")
                    else:
                        arr_text.append(f"{sensor_names[i]}: {last_arrivals[i] * 1000:.1f} ms")

            mapping_info = ""
            if mapping_calibration_mode:
                cp = CALIBRATION_POINTS[mapping_current_point_idx]
                mapping_info = (
                    f" | CAL {cp['label']}=({cp['xy'][0]:.0f},{cp['xy'][1]:.0f}) "
                    f"{len(mapping_samples[mapping_current_point_idx])}/{MAPPING_TAPS_PER_POINT}"
                )
            elif mapping_runtime_enabled and mapping_ready:
                mapping_info = " | runtime=mapping(outside)"
            elif mapping_ready:
                mapping_info = " | map ready"

            best_v_text = "--" if last_best_v is None else f"{last_best_v:.1f}"
            score_text = "--" if last_est_score is None else f"{last_est_score:.6f}"

            self.info_label.setText(
                f"Predicted = ({x:.1f}, {y:.1f}) | {last_region_text} | "
                f"score = {score_text} | best_v = {best_v_text} | "
                f"max_rel = {last_max_rel:.1f} | {last_status_text}{mapping_info}\n"
                + " | ".join(arr_text)
            )

        else:
            self.impact_scatter.setData([])
            self.impact_text.setText("")

            mapping_info = ""
            if mapping_calibration_mode:
                cp = CALIBRATION_POINTS[mapping_current_point_idx]
                mapping_info = (
                    f" | CAL {cp['label']}=({cp['xy'][0]:.0f},{cp['xy'][1]:.0f}) "
                    f"{len(mapping_samples[mapping_current_point_idx])}/{MAPPING_TAPS_PER_POINT}"
                )
            elif mapping_runtime_enabled and mapping_ready:
                mapping_info = " | runtime=mapping(outside)"
            elif mapping_ready:
                mapping_info = " | map ready"

            self.info_label.setText(
                f"idle | max_rel = {last_max_rel:.1f} | trigger = {TRIGGER_THRESHOLD:.1f} | "
                f"{'(capturing event)' if capture_mode else last_status_text}{mapping_info}"
            )


# =========================================================
# MAIN
# =========================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = LiveMapWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()