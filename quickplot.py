import h5py
import numpy as np
import time
import math

import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore

from pathlib import Path
import sys
import os

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # Running from a PyInstaller EXE
    ROOT = Path(sys.executable).resolve().parent
else:
    # Running from source (python quickplot.py)
    ROOT = Path(__file__).resolve().parent

DATA_DIR = (ROOT / "data").resolve()

# Discover .h5 files in ./data (sorted for stable prev/next)
H5_FILES = sorted([p for p in DATA_DIR.glob("*.h5")])
if not H5_FILES:
    raise FileNotFoundError(f"No .h5 files found in: {DATA_DIR}")


def get_segment_keys(f):
    if "segments" not in f:
        raise KeyError("Missing '/segments' group.")
    seg_keys = sorted(list(f["segments"].keys()))
    if not seg_keys:
        raise ValueError("No segments found under /segments.")
    return seg_keys


def load_segment(path, seg_name=None):
    path = str(path)
    with h5py.File(path, "r") as f:
        # Quick sanity checks (helps detect if you're always loading the same file)
        src = f.attrs.get("source_file", "(missing)")
        seg_keys = get_segment_keys(f)
        nseg_attr = (
            int(np.array(f.attrs.get("nSegments", 0)).squeeze())
            if "nSegments" in f.attrs
            else 0
        )
        nseg = nseg_attr if nseg_attr > 0 else len(seg_keys)

        # Use a stable ordering (h5py key iteration order can be arbitrary)
        if seg_name is None or seg_name not in seg_keys:
            seg_name = seg_keys[0]
        seg = f["segments"][seg_name]

        I = seg["current"][:].squeeze()
        V = seg["voltage"][:].squeeze()

        # AE is optional
        AE = None
        if "ae" in seg.keys():
            AE = seg["ae"][:].squeeze()

        # Extra debug so you can confirm two files are truly different
        def stats(x):
            x = np.asarray(x)
            return (
                float(np.min(x)),
                float(np.max(x)),
                float(np.mean(x)),
                float(np.std(x)),
            )

        # Commented out debug prints to reduce noise when switching files
        # print("source_file:", src)
        # print("nSegments:", nseg)
        # print("segment:", seg_name)
        # print("t_start_sec:", float(np.array(seg["t_start_sec"][()]).squeeze()))
        # print("I stats (min,max,mean,std):", stats(I))
        # print("V stats (min,max,mean,std):", stats(V))
        # print("I head:", I[:5].tolist())
        # print("V head:", V[:5].tolist())
        # if AE is not None:
        #     print("AE stats (min,max,mean,std):", stats(AE))
        #     print("AE head:", AE[:5].tolist())

        Fs = float(np.array(f.attrs["Fs"]).squeeze())
        t0 = float(np.array(seg["t_start_sec"][()]).squeeze())

    # time axis in seconds (float64)
    t = t0 + np.arange(I.size, dtype=np.float64) / Fs
    return t, V, I, AE, seg_name, Fs, src, nseg, seg_keys


def main():
    app = QtWidgets.QApplication([])

    # --- Main window ---
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("EDM Signal Viewer")

    central = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout(central)
    vbox.setContentsMargins(8, 8, 8, 8)
    vbox.setSpacing(6)

    # --- Top controls ---
    top = QtWidgets.QHBoxLayout()
    btn_prev = QtWidgets.QPushButton("◀ Prev")
    btn_home = QtWidgets.QPushButton("⌂ Home")
    btn_next = QtWidgets.QPushButton("Next ▶")
    combo = QtWidgets.QComboBox()
    lbl_seg = QtWidgets.QLabel("Seg:")
    combo_seg = QtWidgets.QComboBox()

    # Populate dropdown with filenames
    for p in H5_FILES:
        combo.addItem(p.name)

    top.addWidget(btn_prev)
    top.addWidget(btn_home)
    top.addWidget(btn_next)
    top.addWidget(combo, 1)
    top.addWidget(lbl_seg)
    top.addWidget(combo_seg)

    vbox.addLayout(top)

    readout = QtWidgets.QLabel("t: — sec | V: — | I: — | AE: —")
    readout.setStyleSheet("color: white;")
    readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    vbox.addWidget(readout)

    # --- Plot widget ---
    glw = pg.GraphicsLayoutWidget()
    vbox.addWidget(glw, 1)

    win.setCentralWidget(central)
    win.resize(1400, 900)

    # Disable OpenGL so LinearRegionItem overlays render reliably.
    pg.setConfigOptions(useOpenGL=False, antialias=False)

    # Hold plot items so we can link/update
    state = {
        "p1": None,
        "p2": None,
        "p3": None,
        "x0": None,
        "x1": None,
        "t": None,
        "V": None,
        "I": None,
        "AE": None,
        "src": None,
        "cross": None,
        "scene_proxy": None,
        "set_cross_x": None,
        "cross_updating": False,
    }

    slope_tool = {
        "active_channel": None,
        "active_plotitem": None,
        "plot_by_channel": {},
        "base_titles": {},
        "last_titles": {},
        "region": None,
        "region_plot": None,
        "slope": None,
        "slope_valid": False,
        "region_signals_connected": False,
        "press_scene_pos": None,
        "press_x_data": None,
        "ctrl_selecting": False,
        "selecting": False,
        "last_update_ts": 0.0,
        "drag_threshold_px": 8,
        "min_points": 2,
        "max_fit_points": 5000,
        "throttle_s": 0.10,
    }

    def get_channel_data(channel):
        if channel == "V":
            return state.get("V")
        if channel == "I":
            return state.get("I")
        if channel == "AE":
            return state.get("AE")
        return None

    def set_plot_title(channel, text):
        plot = slope_tool.get("plot_by_channel", {}).get(channel)
        if plot is None:
            return
        last_titles = slope_tool.get("last_titles", {})
        if text == last_titles.get(channel, ""):
            return
        plot.setTitle(text)
        last_titles[channel] = text
        slope_tool["last_titles"] = last_titles

    def refresh_titles(active_only=True):
        # Cache titles to avoid hammering the UI on every mouse move.
        plot_by_channel = slope_tool.get("plot_by_channel", {})
        base_titles = slope_tool.get("base_titles", {})
        active = slope_tool.get("active_channel")
        has_region = slope_tool.get("region") is not None

        def build_title(channel):
            text = base_titles.get(channel, "")
            if channel == active:
                text += " (active)"
                if has_region:
                    if slope_tool.get("slope_valid", False):
                        slope_txt = f"{slope_tool['slope']:.6g}"
                    else:
                        slope_txt = "—"
                    text += f" — slope: {slope_tool['slope']:.2f} /μs"
            return text

        if active_only:
            if active in plot_by_channel:
                set_plot_title(active, build_title(active))
            return

        for channel in base_titles:
            if channel in plot_by_channel:
                set_plot_title(channel, build_title(channel))

    def attach_region_to_active_plot():
        region = slope_tool.get("region")
        if region is None:
            return
        active_plot = slope_tool.get("active_plotitem")
        if active_plot is None:
            return
        if slope_tool.get("region_plot") is active_plot:
            return
        bounds = region.getRegion()
        old_plot = slope_tool.get("region_plot")
        if old_plot is not None:
            old_plot.removeItem(region)
        active_plot.addItem(region, ignoreBounds=True)
        slope_tool["region_plot"] = active_plot
        region.setRegion(bounds)

    def update_region_bounds(x0, x1):
        region = slope_tool.get("region")
        if region is None:
            return
        if x0 > x1:
            x0, x1 = x1, x0
        xx0 = state.get("x0")
        xx1 = state.get("x1")
        if xx0 is not None and xx1 is not None:
            x0 = max(xx0, min(xx1, x0))
            x1 = max(xx0, min(xx1, x1))
        region.setRegion((x0, x1))

    def recompute_slope(throttled=False):
        region = slope_tool.get("region")
        if region is None:
            slope_tool["slope"] = None
            slope_tool["slope_valid"] = False
            refresh_titles(active_only=True)
            return
        if throttled:
            now = time.monotonic()
            if now - slope_tool.get("last_update_ts", 0.0) < slope_tool.get("throttle_s", 0.10):
                return
            slope_tool["last_update_ts"] = now

        t = state.get("t")
        y = get_channel_data(slope_tool.get("active_channel"))
        if t is None or y is None:
            slope_tool["slope"] = None
            slope_tool["slope_valid"] = False
            refresh_titles(active_only=True)
            return

        x0, x1 = region.getRegion()
        if x0 > x1:
            x0, x1 = x1, x0

        idx0 = int(np.searchsorted(t, x0, side="left"))
        idx1 = int(np.searchsorted(t, x1, side="right"))
        t_sel = t[idx0:idx1]
        y_sel = y[idx0:idx1]

        min_points = slope_tool.get("min_points", 2)
        if t_sel.size < min_points:
            slope_tool["slope"] = None
            slope_tool["slope_valid"] = False
            refresh_titles(active_only=True)
            return

        mask = np.isfinite(t_sel) & np.isfinite(y_sel)
        if int(mask.sum()) < min_points:
            slope_tool["slope"] = None
            slope_tool["slope_valid"] = False
            refresh_titles(active_only=True)
            return

        t_sel = t_sel[mask]
        y_sel = y_sel[mask]
        max_fit_points = slope_tool.get("max_fit_points", 5000)
        if t_sel.size > max_fit_points:
            # Cap polyfit points to keep dragging responsive.
            step = max(1, t_sel.size // max_fit_points)
            t_sel = t_sel[::step]
            y_sel = y_sel[::step]
        if t_sel.size < min_points:
            slope_tool["slope"] = None
            slope_tool["slope_valid"] = False
            refresh_titles(active_only=True)
            return

        # Scale time to microseconds so slope is per-us.
        t_fit = t_sel * 1e6
        m, _b = np.polyfit(t_fit, y_sel, 1)
        slope_tool["slope"] = float(m)
        slope_tool["slope_valid"] = True
        refresh_titles(active_only=True)

    def on_region_changed():
        recompute_slope(throttled=True)

    def on_region_change_finished():
        recompute_slope(throttled=False)

    def ensure_region():
        if slope_tool.get("region") is not None:
            return slope_tool["region"]
        region = pg.LinearRegionItem(
            values=(0, 0),
            orientation="vertical",
            movable=True,
            brush=pg.mkBrush(0, 150, 255, 60),
            pen=pg.mkPen((0, 150, 255, 180), width=1),
        )
        region.setZValue(5)
        if not slope_tool.get("region_signals_connected", False):
            region.sigRegionChanged.connect(on_region_changed)
            if hasattr(region, "sigRegionChangeFinished"):
                region.sigRegionChangeFinished.connect(on_region_change_finished)
            elif hasattr(region, "sigRegionChangedFinished"):
                region.sigRegionChangedFinished.connect(on_region_change_finished)
            else:
                # Fall back to sigRegionChanged-only updates if no finished signal exists.
                pass
            slope_tool["region_signals_connected"] = True
        slope_tool["region"] = region
        slope_tool["region_plot"] = None
        attach_region_to_active_plot()
        refresh_titles(active_only=True)
        return region

    def set_active_channel(channel):
        plot_by_channel = slope_tool.get("plot_by_channel", {})
        if channel not in plot_by_channel:
            return
        if (
            slope_tool.get("active_channel") == channel
            and slope_tool.get("active_plotitem") is plot_by_channel[channel]
        ):
            return
        # Active plot/channel follows the last click/drag inside a plot.
        slope_tool["active_channel"] = channel
        slope_tool["active_plotitem"] = plot_by_channel[channel]
        attach_region_to_active_plot()
        refresh_titles(active_only=False)
        if slope_tool.get("region") is not None:
            recompute_slope(throttled=False)

    def clear_slope_region():
        region = slope_tool.get("region")
        if region is not None:
            region_plot = slope_tool.get("region_plot")
            if region_plot is not None:
                region_plot.removeItem(region)
        slope_tool["region"] = None
        slope_tool["region_plot"] = None
        slope_tool["slope"] = None
        slope_tool["slope_valid"] = False
        slope_tool["region_signals_connected"] = False
        slope_tool["ctrl_selecting"] = False
        slope_tool["selecting"] = False
        slope_tool["press_scene_pos"] = None
        slope_tool["press_x_data"] = None
        slope_tool["last_update_ts"] = 0.0
        refresh_titles(active_only=True)

    def get_active_viewbox():
        active_plot = slope_tool.get("active_plotitem")
        if active_plot is None:
            active_plot = state.get("p1")
        if active_plot is None:
            return None
        return active_plot.vb

    class SlopeViewBox(pg.ViewBox):
        def __init__(self, channel, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.channel = channel

        def mousePressEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                set_active_channel(self.channel)
            super().mousePressEvent(ev)

        def mouseClickEvent(self, ev):
            if ev.button() == QtCore.Qt.LeftButton:
                set_active_channel(self.channel)
            super().mouseClickEvent(ev)

        def mouseDragEvent(self, ev):
            if ev.button() != QtCore.Qt.LeftButton:
                super().mouseDragEvent(ev)
                return

            if ev.isStart():
                set_active_channel(self.channel)

            is_ctrl = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
            if is_ctrl or slope_tool.get("ctrl_selecting") or slope_tool.get("selecting"):
                # Ctrl+drag is reserved for the slope tool; accept to prevent panning.
                ev.accept()
                if ev.isStart():
                    if not is_ctrl:
                        return
                    slope_tool["press_scene_pos"] = ev.scenePos()
                    slope_tool["press_x_data"] = float(self.mapSceneToView(ev.scenePos()).x())
                    slope_tool["ctrl_selecting"] = True
                    slope_tool["selecting"] = False
                    return

                if not slope_tool.get("ctrl_selecting"):
                    return

                if ev.isFinish():
                    if slope_tool.get("selecting"):
                        recompute_slope(throttled=False)
                    slope_tool["ctrl_selecting"] = False
                    slope_tool["selecting"] = False
                    slope_tool["press_scene_pos"] = None
                    slope_tool["press_x_data"] = None
                    return

                press_pos = slope_tool.get("press_scene_pos")
                if press_pos is None:
                    return

                delta = ev.scenePos() - press_pos
                if not slope_tool.get("selecting"):
                    # Ctrl+drag gesture: wait until the cursor moves enough to start a region.
                    if math.hypot(delta.x(), delta.y()) < slope_tool.get("drag_threshold_px", 8):
                        return
                    slope_tool["selecting"] = True
                    ensure_region()

                x0 = slope_tool.get("press_x_data")
                if x0 is None:
                    return
                x1 = float(self.mapSceneToView(ev.scenePos()).x())
                update_region_bounds(x0, x1)
                recompute_slope(throttled=True)
                return

            super().mouseDragEvent(ev)

    def clear_plots():
        clear_slope_region()
        glw.clear()
        state["p1"] = None
        state["p2"] = None
        state["p3"] = None
        state["x0"] = None
        state["x1"] = None
        # Removed clearing data arrays here to fix bug
        # state["t"] = None
        # state["V"] = None
        # state["I"] = None
        # state["AE"] = None
        # state["src"] = None
        state["cross"] = None
        state["set_cross_x"] = None
        state["cross_updating"] = False
        slope_tool["plot_by_channel"] = {}
        slope_tool["base_titles"] = {}
        slope_tool["last_titles"] = {}
        slope_tool["active_channel"] = None
        slope_tool["active_plotitem"] = None

    def populate_segments_for_file(p: Path, prefer_seg=None):
        path = str(p)
        try:
            with h5py.File(path, "r") as f:
                seg_keys = get_segment_keys(f)
        except Exception as exc:
            combo_seg.blockSignals(True)
            combo_seg.clear()
            combo_seg.blockSignals(False)
            QtWidgets.QMessageBox.critical(
                win, "Segments not found", f"{p.name}\n{exc}"
            )
            return None

        selected = prefer_seg if prefer_seg in seg_keys else seg_keys[0]
        combo_seg.blockSignals(True)
        combo_seg.clear()
        combo_seg.addItems(seg_keys)
        combo_seg.setCurrentIndex(seg_keys.index(selected))
        combo_seg.blockSignals(False)
        return selected

    def plot_file(p: Path, seg_name=None):
        # Rebuild plots (clear previous widgets/items first)
        clear_plots()

        # Load data
        try:
            t, V, I, AE, seg_name, Fs, src, nseg, seg_keys = load_segment(p, seg_name)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                win, "Failed to load segment", f"{p.name}\n{exc}"
            )
            return

        # Keep arrays in state for crosshair readout
        state["t"] = t
        state["V"] = V
        state["I"] = I
        state["AE"] = AE
        state["src"] = src

        def y_range(x, margin=0.2):
            x = np.asarray(x)
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            span = xmax - xmin
            if span == 0:
                span = abs(xmax) if xmax != 0 else 1.0
            pad = span * margin
            return xmin - pad, xmax + pad

        # Title with metadata
        seg_idx = seg_keys.index(seg_name) if seg_name in seg_keys else 0
        seg_total = nseg if nseg > 0 else len(seg_keys)
        win.setWindowTitle(
            f"EDM Signal Viewer | {p.name} | source={src} | seg={seg_name} ({seg_idx + 1}/{seg_total}) | Fs={Fs/1e6:.1f} MHz"
        )

        vb1 = SlopeViewBox("V")
        vb2 = SlopeViewBox("I")
        vb3 = SlopeViewBox("AE") if AE is not None else None

        p1 = glw.addPlot(row=0, col=0, viewBox=vb1)
        p1.setLabel("left", "Voltage (V)")
        p1.showGrid(x=True, y=True, alpha=0.25)
        c1 = p1.plot(
            t,
            V,
            pen=pg.mkPen(color=(0, 120, 255), width=1.5),
        )  # blue
        # Performance (rendering only): keep full data, but draw efficiently
        c1.setClipToView(True)
        c1.setDownsampling(auto=True, method="peak")
        vmin, vmax = y_range(V)
        p1.setYRange(vmin, vmax, padding=0)

        p2 = glw.addPlot(row=1, col=0, viewBox=vb2)
        p2.setLabel("left", "Current (A)")
        p2.showGrid(x=True, y=True, alpha=0.25)
        c2 = p2.plot(
            t,
            I,
            pen=pg.mkPen(color=(220, 50, 50), width=1.5),
        )  # red
        # Performance (rendering only)
        c2.setClipToView(True)
        c2.setDownsampling(auto=True, method="peak")
        imin, imax = y_range(I)
        p2.setYRange(imin, imax, padding=0)

        p3 = None
        if AE is not None:
            p3 = glw.addPlot(row=2, col=0, viewBox=vb3)
            p3.setLabel("left", "AE (V)")
            p3.setLabel("bottom", "Time (ms)")
            p3.showGrid(x=True, y=True, alpha=0.25)
            c3 = p3.plot(
                t,
                AE,
                pen=pg.mkPen(color=(240, 240, 240), width=1.2),
            )  # white
            # Performance (rendering only)
            c3.setClipToView(True)
            c3.setDownsampling(auto=True, method="peak")
            aemin, aemax = y_range(AE)
            p3.setYRange(aemin, aemax, padding=0)
        else:
            p2.setLabel("bottom", "Time (ms)")

        if p3 is not None:
            p1.setLabel("bottom", "Time (ms)")
            p2.setLabel("bottom", "Time (ms)")

        # Link X axes
        p2.setXLink(p1)
        if p3 is not None:
            p3.setXLink(p1)

        plots = [p1, p2] + ([p3] if p3 is not None else [])
        for p in plots:
            p.getAxis("bottom").enableAutoSIPrefix(False)

        # X-only zoom
        p1.setMouseEnabled(x=True, y=False)
        p2.setMouseEnabled(x=True, y=False)
        if p3 is not None:
            p3.setMouseEnabled(x=True, y=False)

        # Full X range
        x0, x1 = float(t[0]), float(t[-1])
        p1.setXRange(x0, x1, padding=0)
        p2.setXRange(x0, x1, padding=0)
        if p3 is not None:
            p3.setXRange(x0, x1, padding=0)

        state["x0"] = x0
        state["x1"] = x1
        state["p1"] = p1
        state["p2"] = p2
        state["p3"] = p3
        slope_tool["plot_by_channel"] = {"V": p1, "I": p2}
        slope_tool["base_titles"] = {"V": "Voltage", "I": "Current"}
        if p3 is not None:
            slope_tool["plot_by_channel"]["AE"] = p3
            slope_tool["base_titles"]["AE"] = "AE"
        slope_tool["last_titles"] = {key: "" for key in slope_tool["base_titles"]}
        set_active_channel("V")

        # Shared vertical crosshair (click to place + drag to move)
        cross_pen = pg.mkPen(color=(255, 255, 0), width=1)  # yellow, high contrast

        # Movable lines so you can drag from any subplot
        l1 = pg.InfiniteLine(angle=90, movable=True, pen=cross_pen)
        l2 = pg.InfiniteLine(angle=90, movable=True, pen=cross_pen)
        l3 = None

        l1.setZValue(10)
        l2.setZValue(10)

        p1.addItem(l1, ignoreBounds=True)
        p2.addItem(l2, ignoreBounds=True)

        if p3 is not None:
            l3 = pg.InfiniteLine(angle=90, movable=True, pen=cross_pen)
            l3.setZValue(10)
            p3.addItem(l3, ignoreBounds=True)

        state["cross"] = (l1, l2, l3)

        def set_cross_x(x: float):
            # Prevent recursive updates when syncing multiple lines
            if state.get("cross_updating", False):
                return
            state["cross_updating"] = True
            try:
                # Clamp to data range
                xx0, xx1 = state.get("x0"), state.get("x1")
                if xx0 is None or xx1 is None:
                    return
                if x < xx0:
                    x = xx0
                elif x > xx1:
                    x = xx1

                # Sync all lines
                l1.setPos(x)
                l2.setPos(x)
                if l3 is not None:
                    l3.setPos(x)

                # Readout at nearest sample
                t_arr = state.get("t")
                V_arr = state.get("V")
                I_arr = state.get("I")
                AE_arr = state.get("AE")
                if t_arr is None or V_arr is None or I_arr is None:
                    return

                j = int(np.searchsorted(t_arr, x))
                if j <= 0:
                    j = 0
                elif j >= t_arr.size:
                    j = t_arr.size - 1

                tm_ms = (t_arr[j] - t_arr[0]) * 1000.0
                v = float(V_arr[j])
                i = float(I_arr[j])
                ae_txt = "—" if AE_arr is None else f"{float(AE_arr[j]):.3g}"
                readout.setText(
                    f"t: {tm_ms:.3f} sec | V: {v:.3g} | I: {i:.3g} | AE: {ae_txt}"
                )
            finally:
                state["cross_updating"] = False

        state["set_cross_x"] = set_cross_x

        def on_line_moved():
            # Use l1 as the source; it will already be near the final x
            set_cross_x(float(l1.value()))

        l1.sigPositionChanged.connect(on_line_moved)
        l2.sigPositionChanged.connect(on_line_moved)
        if l3 is not None:
            l3.sigPositionChanged.connect(on_line_moved)

        # Initialize at start
        set_cross_x(x0)

    def update_crosshair_from_scene(pos):
        p1 = state.get("p1")
        if p1 is None:
            return

        t = state.get("t")
        V = state.get("V")
        I = state.get("I")
        AE = state.get("AE")
        cross = state.get("cross")
        if t is None or V is None or I is None or cross is None:
            return

        # Map scene position to data coordinates using the active plot's ViewBox
        vb = get_active_viewbox()
        if vb is None:
            return
        mouse_point = vb.mapSceneToView(pos)
        x = float(mouse_point.x())

        # Clamp to data range
        x0 = state.get("x0")
        x1 = state.get("x1")
        if x0 is None or x1 is None:
            return
        if x < x0:
            x = x0
        elif x > x1:
            x = x1

        # Nearest index in time vector
        idx = int(np.searchsorted(t, x))
        if idx <= 0:
            idx = 0
        elif idx >= t.size:
            idx = t.size - 1

        # Move crosshair
        l1, l2, l3 = cross
        l1.setPos(t[idx])
        l2.setPos(t[idx])
        if l3 is not None:
            l3.setPos(t[idx])

        # Readout
        tm_ms = (t[idx] - t[0]) * 1000.0
        v = float(V[idx])
        i = float(I[idx])
        if AE is None:
            ae_txt = "—"
        else:
            ae_txt = f"{float(AE[idx]):.3g}"

        readout.setText(f"t: {tm_ms:.3f} sec | V: {v:.3g} | I: {i:.3g} | AE: {ae_txt}")

    # Throttle mouse-move updates to reduce CPU
    # mouse_proxy = pg.SignalProxy(glw.scene().sigMouseMoved, rateLimit=60, slot=lambda evt: update_crosshair_from_scene(evt[0]))
    # state["scene_proxy"] = mouse_proxy

    def on_mouse_clicked(evt):
        e = evt[0]
        if hasattr(e, "button") and e.button() != QtCore.Qt.LeftButton:
            return
        if e.modifiers() & QtCore.Qt.ControlModifier:
            # Ctrl+click is reserved for the slope tool; keep crosshair unchanged.
            return

        setter = state.get("set_cross_x")
        if setter is None:
            return

        vb = get_active_viewbox()
        if vb is None:
            return
        pt = vb.mapSceneToView(e.scenePos())
        setter(float(pt.x()))

    click_proxy = pg.SignalProxy(
        glw.scene().sigMouseClicked, rateLimit=60, slot=on_mouse_clicked
    )
    state["scene_proxy"] = click_proxy

    def on_key_press(event):
        if event.key() == QtCore.Qt.Key_Escape:
            clear_slope_region()
            event.accept()
            return
        QtWidgets.QMainWindow.keyPressEvent(win, event)

    win.keyPressEvent = on_key_press

    def select_index(i: int):
        i = max(0, min(i, combo.count() - 1))
        combo.setCurrentIndex(i)

    def on_combo_changed(i: int):
        if i < 0 or i >= len(H5_FILES):
            return
        p = H5_FILES[i]
        # Small console info (useful when debugging)
        try:
            print("Opening:", str(p))
            print("File size (MB):", p.stat().st_size / 1024 / 1024)
        except Exception:
            pass
        selected_seg = populate_segments_for_file(p)
        if not selected_seg:
            return
        plot_file(p, selected_seg)

    def on_seg_changed(i: int):
        if combo.count() == 0 or combo_seg.count() == 0:
            return
        p = H5_FILES[combo.currentIndex()]
        seg_name = combo_seg.currentText()
        if not seg_name:
            return
        plot_file(p, seg_name)

    def on_prev():
        select_index(combo.currentIndex() - 1)

    def on_next():
        select_index(combo.currentIndex() + 1)

    def on_home():
        p1 = state.get("p1")
        p2 = state.get("p2")
        p3 = state.get("p3")
        x0 = state.get("x0")
        x1 = state.get("x1")
        if p1 is None or x0 is None or x1 is None:
            return

        # Reset X range to original full extent
        p1.setXRange(x0, x1, padding=0)
        if p2 is not None:
            p2.setXRange(x0, x1, padding=0)
        if p3 is not None:
            p3.setXRange(x0, x1, padding=0)

        cross = state.get("cross")
        t = state.get("t")
        V = state.get("V")
        I = state.get("I")
        AE = state.get("AE")
        if cross is not None and t is not None and t.size > 0:
            l1, l2, l3 = cross
            l1.setPos(x0)
            l2.setPos(x0)
            if l3 is not None:
                l3.setPos(x0)
            ae_txt = "—" if AE is None else f"{float(AE[0]):.3g}"
            readout.setText(
                f"t: {0.0:.3f} sec | V: {float(V[0]):.3g} | I: {float(I[0]):.3g} | AE: {ae_txt}"
            )

    combo.currentIndexChanged.connect(on_combo_changed)
    combo_seg.currentIndexChanged.connect(on_seg_changed)
    btn_prev.clicked.connect(on_prev)
    btn_home.clicked.connect(on_home)
    btn_next.clicked.connect(on_next)

    # Load first file explicitly after signals are connected.
    if combo.count() > 0:
        on_combo_changed(combo.currentIndex())

    win.show()
    app.exec()


if __name__ == "__main__":
    main()
