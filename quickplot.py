import h5py
import numpy as np

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

def load_first_segment(path):
    path = str(path)
    with h5py.File(path, "r") as f:
        # Quick sanity checks (helps detect if you're always loading the same file)
        src = f.attrs.get("source_file", "(missing)")
        nseg = int(np.array(f.attrs.get("nSegments", 0)).squeeze()) if "nSegments" in f.attrs else 0

        # Use a stable ordering (h5py key iteration order can be arbitrary)
        seg_name = sorted(f["segments"].keys())[0]
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
            return float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x))

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
    return t, V, I, AE, seg_name, Fs, src, nseg

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

    # Populate dropdown with filenames
    for p in H5_FILES:
        combo.addItem(p.name)

    top.addWidget(btn_prev)
    top.addWidget(btn_home)
    top.addWidget(btn_next)
    top.addWidget(combo, 1)

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

    # Performance settings
    pg.setConfigOptions(useOpenGL=True, antialias=False)

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

    def clear_plots():
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

    def plot_file(p: Path):
        # Rebuild plots (clear previous widgets/items first)
        clear_plots()

        # Load data
        t, V, I, AE, seg_name, Fs, src, nseg = load_first_segment(p)

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
        win.setWindowTitle(f"EDM Signal Viewer — {p.name} | source={src} | seg={seg_name} | Fs={Fs/1e6:.1f} MHz")

        p1 = glw.addPlot(row=0, col=0)
        p1.setLabel("left", "Voltage (V)")
        p1.showGrid(x=True, y=True, alpha=0.25)
        c1 = p1.plot(
            t, V,
            pen=pg.mkPen(color=(0, 120, 255), width=1.5),
        )  # blue
        # Performance (rendering only): keep full data, but draw efficiently
        c1.setClipToView(True)
        c1.setDownsampling(auto=True, method='peak')
        vmin, vmax = y_range(V)
        p1.setYRange(vmin, vmax, padding=0)

        p2 = glw.addPlot(row=1, col=0)
        p2.setLabel("left", "Current (A)")
        p2.showGrid(x=True, y=True, alpha=0.25)
        c2 = p2.plot(
            t, I,
            pen=pg.mkPen(color=(220, 50, 50), width=1.5),
        )  # red
        # Performance (rendering only)
        c2.setClipToView(True)
        c2.setDownsampling(auto=True, method='peak')
        imin, imax = y_range(I)
        p2.setYRange(imin, imax, padding=0)

        p3 = None
        if AE is not None:
            p3 = glw.addPlot(row=2, col=0)
            p3.setLabel("left", "AE (V)")
            p3.setLabel("bottom", "Time (sec)")
            p3.showGrid(x=True, y=True, alpha=0.25)
            c3 = p3.plot(
                t, AE,
                pen=pg.mkPen(color=(240, 240, 240), width=1.2),
            )  # white
            # Performance (rendering only)
            c3.setClipToView(True)
            c3.setDownsampling(auto=True, method='peak')
            aemin, aemax = y_range(AE)
            p3.setYRange(aemin, aemax, padding=0)
        else:
            p2.setLabel("bottom", "Time (sec)")

        # Link X axes
        p2.setXLink(p1)
        if p3 is not None:
            p3.setXLink(p1)

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
                readout.setText(f"t: {tm_ms:.3f} sec | V: {v:.3g} | I: {i:.3g} | AE: {ae_txt}")
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

        # Map scene position to data coordinates using the top plot's ViewBox
        vb = p1.vb
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

        setter = state.get("set_cross_x")
        p1 = state.get("p1")
        if setter is None or p1 is None:
            return

        vb = p1.vb
        pt = vb.mapSceneToView(e.scenePos())
        setter(float(pt.x()))

    click_proxy = pg.SignalProxy(glw.scene().sigMouseClicked, rateLimit=60, slot=on_mouse_clicked)
    state["scene_proxy"] = click_proxy

    def select_index(i: int):
        i = max(0, min(i, combo.count() - 1))
        combo.setCurrentIndex(i)

    def on_combo_changed(i: int):
        p = H5_FILES[i]
        # Small console info (useful when debugging)
        try:
            print("Opening:", str(p))
            print("File size (MB):", p.stat().st_size / 1024 / 1024)
        except Exception:
            pass
        plot_file(p)

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
            readout.setText(f"t: {0.0:.3f} sec | V: {float(V[0]):.3g} | I: {float(I[0]):.3g} | AE: {ae_txt}")

    combo.currentIndexChanged.connect(on_combo_changed)
    btn_prev.clicked.connect(on_prev)
    btn_home.clicked.connect(on_home)
    btn_next.clicked.connect(on_next)

    # Load first file explicitly
    if combo.count() > 0:
        combo.setCurrentIndex(0)
        plot_file(H5_FILES[0])

    win.show()
    app.exec()

if __name__ == "__main__":
    main()