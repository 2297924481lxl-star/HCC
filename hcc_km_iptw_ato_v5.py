# hcc_km_compare_app.py
# ---------------------------------------------------------
# HCC OS KM Compare App (Qt + Matplotlib + lifelines)
# - Embedded anonymized Excel (drop name/住院号/电话)
# - Based on IPTW-ATO weighted cohort
# - Two modes:
#   (A) Single predictor: POS vs NEG panels (each panel compares Immunotherapy vs No)
#   (B) Multi-filter cohort: Selected cohort vs Remaining cohort panels (each compares Immunotherapy vs No)
# - Each panel includes risk table + weighted Cox HR (Immunotherapy vs No)
# - Follow-up truncated at 60 months (beyond 60 => censored)
# ---------------------------------------------------------

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QMessageBox,
    QSlider, QTabWidget, QSizePolicy, QScrollArea
)

# =========================
# Defaults
# =========================
DEFAULT_SOURCE_EXCEL = r"C:\Users\86180\Desktop\liver cancer\V5\BC期_overlap_weighting_ATO_加权完整数据.xlsx"
DEFAULT_OUTPUT_DIR   = r"C:\Users\86180\Desktop\HCC_KM_EXE_IPTW_ATO"

# Runtime / embedded data
EMBEDDED_REL_PATH = os.path.join("data", "BC期_overlap_weighting_ATO_加权完整数据_V5_anon.xlsx")
ICON_PNG_NAME = "image.png"
ICON_ICO_NAME = "app_icon.ico"

MAX_MO = 60
PRIVACY_DROP_COLS = ["name", "住院号", "电话"]

# Chinese fonts (Windows)
mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# Top-journal friendly palette (Okabe–Ito-ish)
COL_IMMUNE = "#D55E00"   # Immunotherapy: orange/red
COL_NO     = "#0072B2"   # No immunotherapy: blue
COL_GRAY   = "#4D4D4D"


# =========================
# Helpers
# =========================
def coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def resource_path(rel_path: str) -> str:
    """
    Get absolute path to bundled resource (PyInstaller onefile uses sys._MEIPASS)
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)

def drop_privacy_cols(df: pd.DataFrame, drop_cols):
    out = df.copy()
    cols_lower = {str(c).lower(): c for c in out.columns}

    for c in drop_cols:
        if c in out.columns:
            out = out.drop(columns=[c])
        else:
            cl = str(c).lower()
            if cl in cols_lower:
                out = out.drop(columns=[cols_lower[cl]])
    return out

def make_anonymized_excel(src_path: str, out_path: str, drop_cols=None):
    if drop_cols is None:
        drop_cols = PRIVACY_DROP_COLS

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source Excel not found: {src_path}")

    df = pd.read_excel(src_path, engine="openpyxl")
    if isinstance(df, dict):
        first_sheet = next(iter(df.keys()))
        df = df[first_sheet]

    df = drop_privacy_cols(df, drop_cols)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False, engine="openpyxl")
    return out_path

def create_windows_icon_if_needed(work_dir: str):
    """
    Create app_icon.ico from image.png if possible.
    Returns (png_path_or_None, ico_path_or_None).
    """
    png_path = os.path.join(work_dir, ICON_PNG_NAME)
    ico_path = os.path.join(work_dir, ICON_ICO_NAME)

    if not os.path.exists(png_path):
        return None, None

    # If ICO already exists, reuse it
    if os.path.exists(ico_path):
        return png_path, ico_path

    try:
        from PIL import Image
        img = Image.open(png_path)
        img.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        return png_path, ico_path
    except Exception:
        return png_path, None


def qtile_range(x: pd.Series, qlo=0.05, qhi=0.95):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return None
    lo = float(x.quantile(qlo))
    hi = float(x.quantile(qhi))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo == hi:
        lo = float(x.min())
        hi = float(x.max())
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi

def fmt_p(p):
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NA"
    return f"{p:.3g}"

def safe_logrank_p(d_no, d_yes):
    try:
        r = logrank_test(
            d_no["time60"].values, d_yes["time60"].values,
            event_observed_A=d_no["status60"].values,
            event_observed_B=d_yes["status60"].values
        )
        return float(r.p_value)
    except Exception:
        return None

def cox_hr_immunotherapy(d_panel: pd.DataFrame):
    """
    Weighted HR(Immune=1 vs 0) within a cohort, using CoxPHFitter.
    Return (hr, lo, hi, p) or (None,...)
    """
    try:
        x = coerce_num(d_panel["_immune"])
        w = coerce_num(d_panel["_weight"])
        valid = x.isin([0, 1]) & w.notna()
        d = d_panel.loc[valid].copy()
        if len(d) == 0:
            return (None, None, None, None)

        n_yes = int((d["_immune"] == 1).sum())
        n_no  = int((d["_immune"] == 0).sum())
        if n_yes == 0 or n_no == 0:
            return (None, None, None, None)

        cox_df = d[["time60", "status60"]].copy()
        cox_df["Immune"] = d["_immune"].astype(int).values
        cox_df["weight"] = coerce_num(d["_weight"]).values

        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time60", event_col="status60", formula="Immune", weights_col="weight", robust=True)
        s = cph.summary.loc["Immune"]
        hr = float(np.exp(s["coef"]))
        lo = float(np.exp(s["coef lower 95%"]))
        hi = float(np.exp(s["coef upper 95%"]))
        p = float(s["p"])
        return (hr, lo, hi, p)
    except Exception:
        return (None, None, None, None)

def draw_risk_table(ax_table, d_no, d_yes, times):
    """
    Simple risk table like R: rows=groups, cols=time points
    """
    ax_table.axis("off")
    # counts at risk: time >= t
    no_counts  = [int((d_no["time60"]  >= t).sum()) for t in times]
    yes_counts = [int((d_yes["time60"] >= t).sum()) for t in times]

    header = "Time: " + "  ".join([f"{int(t):>3d}" for t in times])
    row_no = "No  : " + "  ".join([f"{c:>3d}" for c in no_counts])
    row_yes= "Yes : " + "  ".join([f"{c:>3d}" for c in yes_counts])

    ax_table.text(0.01, 0.75, header, fontsize=9, family="monospace", color=COL_GRAY)
    ax_table.text(0.01, 0.40, row_no, fontsize=9, family="monospace", color=COL_NO)
    ax_table.text(0.01, 0.10, row_yes, fontsize=9, family="monospace", color=COL_IMMUNE)


# =========================
# Filter slot widget helper
# =========================
class FilterSlot:
    """
    A row of widgets to define one condition:
      - var dropdown (None or a predictor)
      - pick dropdown: choose G1 (positive/high) or G0 (negative/low)
      - optional slider for continuous cutoff + label
    """
    def __init__(self, parent, title: str):
        self.parent = parent
        self.title = title

        self.row = QWidget()
        self.row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QHBoxLayout(self.row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.lbl = QLabel(title)
        self.lbl.setMinimumWidth(28)
        self.lbl.setMaximumWidth(28)

        self.cb_var = QComboBox()
        self.cb_var.setMinimumWidth(220)
        self.cb_var.setMaximumWidth(320)

        self.cb_pick = QComboBox()
        self.cb_pick.addItems(["G1", "G0"])
        self.cb_pick.setToolTip("G1 = positive/high; G0 = negative/low")
        self.cb_pick.setMinimumWidth(56)
        self.cb_pick.setMaximumWidth(62)

        self.lbl_cut = QLabel("cut: NA")
        self.lbl_cut.setStyleSheet("font-weight:bold; color:#222;")
        self.lbl_cut.setMinimumWidth(66)
        self.lbl_cut.setMaximumWidth(74)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(500)
        self.slider.setMinimumWidth(150)

        lay.addWidget(self.lbl)
        lay.addWidget(self.cb_var, stretch=1)
        lay.addWidget(self.cb_pick, stretch=0)
        lay.addWidget(self.lbl_cut, stretch=0)
        lay.addWidget(self.slider, stretch=1)

        self._cut_lo = None
        self._cut_hi = None

        self.cb_var.currentTextChanged.connect(lambda _: self.on_var_changed())
        self.cb_pick.currentIndexChanged.connect(lambda _: self.parent.update_plot())
        self.slider.valueChanged.connect(lambda _: (self.update_cut_label(), self.parent.update_plot()))

        self.set_continuous_visible(False)

    def set_predictor_names(self, names):
        self.cb_var.blockSignals(True)
        self.cb_var.clear()
        self.cb_var.addItem("None (not used)")
        for n in names:
            self.cb_var.addItem(n)
        self.cb_var.blockSignals(False)

    def set_continuous_visible(self, vis: bool):
        # Keep slider visible for compact layout; disable when not applicable.
        self.lbl_cut.setVisible(True)
        self.slider.setVisible(True)
        self.slider.setEnabled(vis)

    def on_var_changed(self):
        self.parent.update_slot_slider(self)
        self.parent.update_plot()

    def is_used(self):
        return self.cb_var.currentText() != "None (not used)"

    def current_name(self):
        return self.cb_var.currentText()

    def pick_g1(self):
        return self.cb_pick.currentIndex() == 0  # 0 => G1, 1 => G0

    def set_cut_range(self, lo, hi):
        self._cut_lo, self._cut_hi = lo, hi

    def get_cut_value(self):
        if self._cut_lo is None or self._cut_hi is None:
            return None
        v = self.slider.value() / 1000.0
        return self._cut_lo + (self._cut_hi - self._cut_lo) * v

    def update_cut_label(self):
        cut = self.get_cut_value()
        if cut is None:
            self.lbl_cut.setText("cut: NA")
        else:
            self.lbl_cut.setText(f"cut: {cut:.2f}")


# =========================
# GUI
# =========================
class KMCompareApp(QMainWindow):
    def __init__(self, use_embedded=True, external_excel=None):
        super().__init__()
        self._use_embedded = use_embedded
        self._external_excel = external_excel
        self.setWindowTitle("HCC OS KM Compare (IPTW-ATO weighted cohort)")
        icon_path = resource_path(ICON_PNG_NAME)
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 860)

        self.df_raw = None
        self.cols = {}
        self.predictors = {}
        self.current_pred = None

        # ---- UI root ----
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # =========================
        # Top controls (common)
        # =========================
        top_box = QGroupBox("Embedded IPTW-ATO weighted cohort")
        top_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        top = QGridLayout(top_box)
        top.setContentsMargins(6, 6, 6, 6)
        top.setHorizontalSpacing(6)
        top.setVerticalSpacing(2)

        self.cb_bclc = QComboBox()
        self.cb_bclc.addItems([
            "BCLC B+C (2+3)",
            "BCLC B (2)",
            "BCLC C (3)",
            "BCLC D (4)",
        ])
        self.cb_bclc.currentTextChanged.connect(lambda _: self.update_everything())

        top.addWidget(QLabel("BCLC filter:"), 0, 0)
        top.addWidget(self.cb_bclc, 0, 1, 1, 1)

        # hidden status label retained for compatibility with the original loading logic
        self.lbl_file = QLabel("")
        self.lbl_file.setVisible(False)

        root.addWidget(top_box)

        # =========================
        # Tabs: single vs multi
        # =========================
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        root.addWidget(self.tabs, 0)

        # ---- Tab A: Single predictor POS vs NEG
        tabA = QWidget()
        layA = QVBoxLayout(tabA)

        boxA = QGroupBox("Single predictor mode")
        boxA.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        gA = QGridLayout(boxA)
        gA.setContentsMargins(6, 6, 6, 6)
        gA.setHorizontalSpacing(5)
        gA.setVerticalSpacing(2)

        self.cb_pred = QComboBox()
        self.cb_pred.currentTextChanged.connect(self.on_pred_changed)

        # continuous slider for Tab A
        self.lbl_cutA = QLabel("cut: NA")
        self.lbl_cutA.setStyleSheet("font-weight:bold; color:#222;")
        self.sliderA = QSlider(Qt.Horizontal)
        self.sliderA.setMinimum(0)
        self.sliderA.setMaximum(1000)
        self.sliderA.setValue(500)
        self.sliderA.valueChanged.connect(lambda _: (self.update_cutA_label(), self.update_plot()))
        self._cutA_lo = None
        self._cutA_hi = None

        self.btn_update = QPushButton("Update plot")
        self.btn_update.clicked.connect(self.update_plot)

        self.cb_pred.setMinimumWidth(250)
        self.sliderA.setMinimumWidth(300)
        self.btn_update.setMinimumWidth(96)

        gA.addWidget(QLabel("Predictor:"), 0, 0)
        gA.addWidget(self.cb_pred, 0, 1, 1, 2)
        gA.addWidget(QLabel("Cutoff:"), 0, 3)
        gA.addWidget(self.lbl_cutA, 0, 4)
        gA.addWidget(self.sliderA, 0, 5, 1, 2)
        gA.addWidget(self.btn_update, 0, 7, 1, 1)
        gA.setColumnStretch(1, 1)
        gA.setColumnStretch(5, 1)

        layA.setContentsMargins(0, 0, 0, 0)
        layA.setSpacing(2)
        layA.addWidget(boxA, 0)
        self.tabs.addTab(tabA, "Single predictor")

        # ---- Tab B: Multi-filter cohort (Selected vs Remaining)
        tabB = QWidget()
        layB = QVBoxLayout(tabB)

        boxB = QGroupBox("Multi-filter cohort mode")
        boxB.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        vB = QVBoxLayout(boxB)
        vB.setContentsMargins(6, 6, 6, 6)
        vB.setSpacing(3)

        # 3 slots
        self.slot1 = FilterSlot(self, "C1")
        self.slot2 = FilterSlot(self, "C2")
        self.slot3 = FilterSlot(self, "C3")
        vB.addWidget(self.slot1.row)
        vB.addWidget(self.slot2.row)
        vB.addWidget(self.slot3.row)

        bottomB = QHBoxLayout()
        bottomB.setContentsMargins(0, 0, 0, 0)
        bottomB.setSpacing(6)

        hint = QLabel("AND logic. Example: CK19 = G1 + Age = G1 + P53 = G1")
        hint.setStyleSheet("color:#666;")
        hint.setWordWrap(True)

        self.btn_update_multi = QPushButton("Update plot")
        self.btn_update_multi.setMinimumWidth(110)
        self.btn_update_multi.clicked.connect(self.update_plot)

        bottomB.addWidget(hint, 1)
        bottomB.addWidget(self.btn_update_multi, 0)

        vB.addLayout(bottomB)

        layB.setContentsMargins(0, 0, 0, 0)
        layB.setSpacing(2)
        layB.addWidget(boxB, 0)
        self.tabs.addTab(tabB, "Multi-filter cohort")

        # ---- Tab C: Software guide
        tabC = QWidget()
        layC = QVBoxLayout(tabC)
        scrollC = QScrollArea()
        scrollC.setWidgetResizable(True)
        scrollC.setFrameShape(QScrollArea.NoFrame)

        guide_wrap = QWidget()
        guide_wrap_lay = QVBoxLayout(guide_wrap)
        guide_wrap_lay.setContentsMargins(0, 0, 0, 0)
        guide_wrap_lay.setSpacing(2)

        guide_box = QGroupBox("Software guide")
        guide_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        guide_layout = QVBoxLayout(guide_box)

        guide_text = QLabel(
            "<b>Purpose</b><br>"
            "This software compares overall survival for <b>immunotherapy (1)</b> versus <b>no immunotherapy (0)</b> "
            "within selected subgroups in an <b>IPTW-ATO weighted cohort</b>.<br>"
            "<b>Note:</b> The built-in dataset has already undergone <b>IPTW-ATO (overlap weighting)</b> adjustment, "
            "and the analysis uses the <code>weight</code> column in the dataset.<br><br>"

            "<b>Weighting variables</b><br>"
            "The embedded cohort was balanced using IPTW-ATO for <b>BCLC stage (B/C)</b>, <b>surgery</b>, "
            "<b>interventional therapy</b>, <b>targeted therapy</b>, and <b>radiotherapy</b>, "
            "with immunotherapy as the exposure variable.<br><br>"

            "<b>Main functions</b><br>"
            "1. <b>Single predictor</b>: choose one variable and display POS/HIGH vs NEG/LOW subgroups; "
            "each panel compares immunotherapy vs no immunotherapy.<br>"
            "2. <b>Multi-filter cohort</b>: combine several conditions with AND logic to define a selected cohort; "
            "the software compares the selected cohort versus the remaining cohort.<br>"
            "3. Continuous variables support an adjustable <b>cutoff slider</b>.<br><br>"

            "<b>Embedded dataset note</b><br>"
            "This release is configured to use the embedded IPTW-ATO weighted dataset only.<br><br>"

            "<b>Developer information</b><br>"
            "Developed by the Xiang Bangde Research Group, Department of Hepatobiliary Surgery, Guangxi Medical University Cancer Hospital.<br>"
            "Correspondence: <code>xiangbangde@gxmu.edu.cn</code>"
        )
        guide_text.setWordWrap(True)
        guide_text.setTextFormat(Qt.RichText)
        guide_text.setStyleSheet("font-size: 10.5px;")
        guide_layout.addWidget(guide_text)

        guide_wrap_lay.addWidget(guide_box, 0)
        guide_wrap_lay.addStretch(1)
        scrollC.setWidget(guide_wrap)

        layC.setContentsMargins(0, 0, 0, 0)
        layC.setSpacing(2)
        layC.addWidget(scrollC, 1)
        self.tabs.addTab(tabC, "Software guide")
        self.tabs.currentChanged.connect(lambda _: self._adjust_tabs_height())
        self._adjust_tabs_height()

        # =========================
        # Plot Canvas (shared for tabs)
        # =========================
        self.fig = Figure(figsize=(12, 7), dpi=110)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, stretch=1)

        # Try load data at startup
        if self._external_excel:
            self.load_external_path(self._external_excel)
        else:
            if self._use_embedded:
                self.load_embedded(show_msg=False)

    def _adjust_tabs_height(self):
        """
        Keep the upper tab panel compact so more space is left for the plots.
        """
        try:
            current = self.tabs.currentWidget()
            if current is None:
                return
            tabbar_h = self.tabs.tabBar().sizeHint().height()
            frame = 10
            # Different tabs need slightly different heights
            idx = self.tabs.currentIndex()
            content_h = current.sizeHint().height()
            if idx == 0:
                target = min(95, max(72, content_h + tabbar_h + frame))
            elif idx == 1:
                target = min(165, max(130, content_h + tabbar_h + frame))
            else:
                target = 185
            self.tabs.setMaximumHeight(target)
            self.tabs.setMinimumHeight(target)
        except Exception:
            pass

    # -------------------------
    # Loading
    # -------------------------
    def load_embedded(self, show_msg=True):
        try:
            path = resource_path(EMBEDDED_REL_PATH)
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            QMessageBox.critical(
                self, "Embedded data load failed",
                f"Could not read embedded data:\n{EMBEDDED_REL_PATH}\n\nError: {e}\n\n"
                f"If you have not generated the embedded anonymized file yet, run:\n"
                f"py hcc_km_iptw_ato.py --make-anon --src \"{DEFAULT_SOURCE_EXCEL}\""
            )
            return
        self._load_df(df, f"Embedded IPTW-ATO weighted dataset: {EMBEDDED_REL_PATH}", show_msg=show_msg)

    def load_external(self):
        start_dir = os.path.dirname(DEFAULT_SOURCE_EXCEL)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel file", start_dir, "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        self.load_external_path(path)

    def load_external_path(self, path):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Could not read Excel:\n{e}\nPath: {path}")
            return
        df = drop_privacy_cols(df, PRIVACY_DROP_COLS)
        self._load_df(df, f"External: {path} (privacy cols dropped in memory)", show_msg=True)

    def _load_df(self, df: pd.DataFrame, source_desc: str, show_msg=True):
        df = drop_privacy_cols(df, PRIVACY_DROP_COLS)

        time_col   = find_col(df, ["Survival_time"])
        status_col = find_col(df, ["Survival_status"])
        bclc_col   = find_col(df, ["BCLC_stage", "BCLC分期", "BCLC"])

        immune_col = find_col(df, ["免疫治疗", "Immunotherapy", "ICI"])
        weight_col = find_col(df, ["weight"])
        if immune_col is None:
            QMessageBox.critical(self, "Missing columns", "Missing required column: immunotherapy (0/1). Please check the Excel column names.")
            return
        if weight_col is None:
            QMessageBox.critical(self, "Missing columns", "Missing required column: weight. Please use the IPTW-ATO weighted dataset.")
            return

        if time_col is None or status_col is None:
            QMessageBox.critical(
                self, "Missing columns",
                f"Missing required columns: Survival_time and/or Survival_status.\n"
                f"Detected: Survival_time={time_col}, Survival_status={status_col}\n"
                f"Please check the Excel column names."
            )
            return

        self.cols = dict(time=time_col, status=status_col, bclc=bclc_col, immune=immune_col, weight=weight_col)

        # optional predictors
        self.cols.update({
            "Gender": find_col(df, ["Gender", "性别"]),
            "Age": find_col(df, ["Age", "年龄"]),
            "CK19": find_col(df, ["CK19"]),
            "CK7": find_col(df, ["CK7"]),
            "KI67": find_col(df, ["KI67", "Ki67"]),
            "P53": find_col(df, ["P53", "p53"]),
            "Edmonson_Grading": find_col(df, ["Edmonson_Grading", "Edmonson分级"]),
            "Tumor_number": find_col(df, ["Tumor_number", "肿瘤个数", "肿瘤数量"]),
            "Gross_thrombus": find_col(df, ["肉眼癌栓", "Gross_thrombus"]),
            "Macrovascular_invasion": find_col(df, ["Macrovascular_invasion"]),
            "HBsAg": find_col(df, ["HBsAg"]),
            "HBV_D(0＜500,1≥500)": find_col(df, ["HBV_D(0＜500,1≥500)", "HBV_D(0<500,1≥500)"]),
            "Cirrhosis": find_col(df, ["Cirrhosis", "肝硬化"]),
            "Tumor_diameter": find_col(df, ["Tumor_diameter", "肿瘤最大直径(cm)"]),
            "Child_Pugh": find_col(df, ["Child_Pugh", "Child-Pugh"]),
            "Extrahepatic_metastasis": find_col(df, ["Extrahepatic_metastasis", "肝外转移"]),
            "TBIL": find_col(df, ["TBIL"]),
            "ALB": find_col(df, ["ALB"]),
            "ALT": find_col(df, ["ALT"]),
            "AST": find_col(df, ["AST"]),
            "WBC": find_col(df, ["WBC"]),
            "platelet": find_col(df, ["platelet", "PLT", "血小板"]),
            "AFP": find_col(df, ["AFP"]),
            "INR": find_col(df, ["INR"]),
            "PT": find_col(df, ["PT"]),
        })

        self.df_raw = df
        if hasattr(self, "lbl_file") and self.lbl_file is not None:
            self.lbl_file.setText(source_desc)
            self.lbl_file.setToolTip(source_desc)

        self.build_predictors()
        self.populate_predictors()

        if show_msg:
            QMessageBox.information(self, "Success", "Weighted dataset loaded (name / hospital ID / phone removed).")

        self.update_everything()

    # -------------------------
    # Predictor specs (用于“阳性/阴性”拆分)
    # -------------------------
    def build_predictors(self):
        C = self.cols
        self.predictors = {}

        def add_binary(display_name, colkey, g1_label="Positive (1)", g0_label="Negative (0)"):
            col = C.get(colkey)
            if not col:
                return
            self.predictors[display_name] = {
                "type": "binary",
                "col": col,
                "g1_label": g1_label,
                "g0_label": g0_label
            }

        def add_cont(display_name, colkey, high_label="High (>cut)", low_label="Low (<=cut)"):
            col = C.get(colkey)
            if not col:
                return
            self.predictors[display_name] = {
                "type": "continuous",
                "col": col,
                "high_label": high_label,
                "low_label": low_label
            }

        def add_custom(display_name, colkey, custom, g1_label, g0_label):
            col = C.get(colkey)
            if not col:
                return
            self.predictors[display_name] = {
                "type": "custom",
                "col": col,
                "custom": custom,
                "g1_label": g1_label,
                "g0_label": g0_label
            }

        # Immunohistochemical markers
        add_cont("P53 (<=cut vs >cut)", "P53", high_label="P53 > cut", low_label="P53 ≤ cut")
        add_binary("CK19 (1 vs 0)", "CK19", g1_label="CK19+ (1)", g0_label="CK19- (0)")
        add_binary("CK7 (1 vs 0)", "CK7", g1_label="CK7+ (1)", g0_label="CK7- (0)")
        add_cont("KI67 (<=cut vs >cut)", "KI67", high_label="KI67 > cut", low_label="KI67 ≤ cut")

        # Clinical and tumor-related variables
        add_cont("Age (<=cut vs >cut)", "Age", high_label="Age > cut", low_label="Age ≤ cut")
        add_binary("Gender (Male=1 vs 0)", "Gender", g1_label="Male (1)", g0_label="Female (0)")
        add_custom("Child_Pugh (1 vs 2–3)", "Child_Pugh", "child_pugh_1_vs_23",
                   g1_label="Child_Pugh=1", g0_label="Child_Pugh=2–3")
        add_binary("Cirrhosis (1 vs 0)", "Cirrhosis", g1_label="Cirrhosis=1", g0_label="Cirrhosis=0")
        add_binary("HBsAg (1 vs 0)", "HBsAg", g1_label="HBsAg=1", g0_label="HBsAg=0")
        add_binary("HBV_D(0<500,1≥500) (1 vs 0)", "HBV_D(0＜500,1≥500)", g1_label="HBV_D≥500 (1)", g0_label="HBV_D<500 (0)")
        add_cont("Tumor_diameter (<=cut vs >cut)", "Tumor_diameter", high_label="Diameter > cut", low_label="Diameter ≤ cut")
        add_custom("Tumor_number (3–4 vs 1–2)", "Tumor_number", "tumor_number_34_vs_12",
                   g1_label="Tumor_number 3–4", g0_label="Tumor_number 1–2")
        add_custom("Edmonson_Grading (3–4 vs 0–2)", "Edmonson_Grading", "edmonson_34_vs_02",
                   g1_label="Edmonson 3–4", g0_label="Edmonson 0–2")
        add_binary("Gross thrombus (1 vs 0)", "Gross_thrombus", g1_label="Gross thrombus (1)", g0_label="No gross thrombus (0)")
        add_binary("Macrovascular_invasion (1 vs 0)", "Macrovascular_invasion", g1_label="Macrovascular=1", g0_label="Macrovascular=0")
        add_binary("Extrahepatic_metastasis (1 vs 0)", "Extrahepatic_metastasis", g1_label="Extrahepatic=1", g0_label="Extrahepatic=0")

        # Blood / laboratory variables
        add_cont("TBIL (<=cut vs >cut)", "TBIL", high_label="TBIL > cut", low_label="TBIL ≤ cut")
        add_cont("ALB (<=cut vs >cut)", "ALB", high_label="ALB > cut", low_label="ALB ≤ cut")
        add_cont("ALT (<=cut vs >cut)", "ALT", high_label="ALT > cut", low_label="ALT ≤ cut")
        add_cont("AST (<=cut vs >cut)", "AST", high_label="AST > cut", low_label="AST ≤ cut")
        add_cont("WBC (<=cut vs >cut)", "WBC", high_label="WBC > cut", low_label="WBC ≤ cut")
        add_cont("platelet (<=cut vs >cut)", "platelet", high_label="platelet > cut", low_label="platelet ≤ cut")
        add_cont("AFP (<=cut vs >cut)", "AFP", high_label="AFP > cut", low_label="AFP ≤ cut")
        add_cont("INR (<=cut vs >cut)", "INR", high_label="INR > cut", low_label="INR ≤ cut")
        add_cont("PT (<=cut vs >cut)", "PT", high_label="PT > cut", low_label="PT ≤ cut")

    def populate_predictors(self):
        names = list(self.predictors.keys())
        self.cb_pred.blockSignals(True)
        self.cb_pred.clear()
        self.cb_pred.addItems(names)
        self.cb_pred.blockSignals(False)

        # also set names to multi-filter slots
        self.slot1.set_predictor_names(names)
        self.slot2.set_predictor_names(names)
        self.slot3.set_predictor_names(names)

        # trigger initial
        self.on_pred_changed(self.cb_pred.currentText())

    def on_pred_changed(self, name):
        self.current_pred = self.predictors.get(name)
        self.update_sliderA_range()
        self.update_plot()

    # -------------------------
    # Context dataframe (BCLC + event + truncation)
    # -------------------------
    def get_context_df(self):
        if self.df_raw is None:
            return None

        d = self.df_raw.copy()

        d["_time"]   = coerce_num(d[self.cols["time"]])
        d["_status"] = coerce_num(d[self.cols["status"]])
        d["_immune"] = coerce_num(d[self.cols["immune"]])
        d["_weight"] = coerce_num(d[self.cols["weight"]])

        d = d.dropna(subset=["_time", "_status", "_immune", "_weight"])
        d = d[d["_immune"].isin([0, 1])].copy()

        # truncate at 60 months (beyond 60 => censored)
        d["time60"] = np.minimum(d["_time"].values, MAX_MO)
        d["status60"] = d["_status"].values.copy()
        d.loc[d["_time"] > MAX_MO, "status60"] = 0


        # BCLC filter
        bclc_col = self.cols.get("bclc")
        sel = self.cb_bclc.currentText()
        if bclc_col is not None:
            d["_bclc"] = coerce_num(d[bclc_col])
            if sel.startswith("BCLC B (2)"):
                d = d[d["_bclc"] == 2]
            elif sel.startswith("BCLC C (3)"):
                d = d[d["_bclc"] == 3]
            elif sel.startswith("BCLC D (4)"):
                d = d[d["_bclc"] == 4]
            elif sel.startswith("BCLC B+C"):
                d = d[d["_bclc"].isin([2, 3])]

        return d

    # -------------------------
    # Slider A (single mode) cutoff
    # -------------------------
    def update_sliderA_range(self):
        if self.current_pred is None:
            self.sliderA.setEnabled(False)
            self.lbl_cutA.setText("cut: NA")
            return

        if self.current_pred["type"] != "continuous":
            self.sliderA.setEnabled(False)
            self.sliderA.setVisible(True)
            self.lbl_cutA.setVisible(True)
            self.lbl_cutA.setText("cut: NA")
            return

        self.sliderA.setVisible(True)
        self.lbl_cutA.setVisible(True)

        d = self.get_context_df()
        if d is None or len(d) == 0:
            self.sliderA.setEnabled(False)
            self.lbl_cutA.setText("cut: NA")
            return

        col = self.current_pred["col"]
        rng = qtile_range(d[col], 0.05, 0.95)
        if rng is None:
            self.sliderA.setEnabled(False)
            self.lbl_cutA.setText("cut: NA")
            return

        self._cutA_lo, self._cutA_hi = rng
        self.sliderA.setEnabled(True)
        self.update_cutA_label()

    def get_cutA_value(self):
        if self._cutA_lo is None or self._cutA_hi is None:
            return None
        v = self.sliderA.value() / 1000.0
        return self._cutA_lo + (self._cutA_hi - self._cutA_lo) * v

    def update_cutA_label(self):
        cut = self.get_cutA_value()
        if cut is None:
            self.lbl_cutA.setText("cut: NA")
        else:
            self.lbl_cutA.setText(f"cut: {cut:.2f}")

    # -------------------------
    # Multi slot slider setup
    # -------------------------
    def update_slot_slider(self, slot: FilterSlot):
        d = self.get_context_df()
        if d is None or len(d) == 0 or (not slot.is_used()):
            slot.set_continuous_visible(False)
            slot.set_cut_range(None, None)
            slot.update_cut_label()
            return

        name = slot.current_name()
        spec = self.predictors.get(name)
        if spec is None:
            slot.set_continuous_visible(False)
            slot.set_cut_range(None, None)
            slot.update_cut_label()
            return

        if spec["type"] != "continuous":
            slot.set_continuous_visible(False)
            slot.set_cut_range(None, None)
            slot.update_cut_label()
            return

        col = spec["col"]
        rng = qtile_range(d[col], 0.05, 0.95)
        if rng is None:
            slot.set_continuous_visible(True)
            slot.set_cut_range(None, None)
            slot.update_cut_label()
            return

        slot.set_continuous_visible(True)
        slot.set_cut_range(rng[0], rng[1])
        slot.update_cut_label()

    # -------------------------
    # Group masks for predictor (single mode)
    # -------------------------
    def mask_g1_g0_from_spec(self, d: pd.DataFrame, spec: dict, cut_value=None):
        """
        Return (d2, g1_mask, g0_mask, g1_label, g0_label)
        where g1=positive/high, g0=negative/low.
        """
        col = spec["col"]
        tp = spec["type"]

        if tp == "binary":
            x = coerce_num(d[col])
            valid = x.isin([0, 1])
            d2 = d.loc[valid].copy()
            x2 = coerce_num(d2[col])
            g1 = (x2 == 1)
            g0 = (x2 == 0)
            return d2, g1, g0, spec["g1_label"], spec["g0_label"]

        if tp == "continuous":
            x = coerce_num(d[col])
            valid = x.notna()
            d2 = d.loc[valid].copy()
            x2 = coerce_num(d2[col])
            if cut_value is None:
                # fallback median
                cut_value = float(x2.median())
            g0 = (x2 <= cut_value)
            g1 = (x2 > cut_value)
            g1_label = spec["high_label"] + f" (cut={cut_value:.2f})"
            g0_label = spec["low_label"]  + f" (cut={cut_value:.2f})"
            return d2, g1, g0, g1_label, g0_label

        if tp == "custom":
            x = coerce_num(d[col])
            valid = x.notna()
            d2 = d.loc[valid].copy()
            x2 = coerce_num(d2[col])

            if spec["custom"] == "child_pugh_1_vs_23":
                g1 = (x2 == 1)
                g0 = x2.isin([2, 3])
            elif spec["custom"] == "edmonson_34_vs_02":
                g1 = x2.isin([3, 4])
                g0 = x2.isin([0, 1, 2])
            elif spec["custom"] == "tumor_number_34_vs_12":
                g1 = x2.isin([3, 4])
                g0 = x2.isin([1, 2])
            else:
                return None

            d2 = d2.loc[g1 | g0].copy()
            x2 = coerce_num(d2[col])
            if spec["custom"] == "child_pugh_1_vs_23":
                g1 = (x2 == 1); g0 = x2.isin([2, 3])
            elif spec["custom"] == "edmonson_34_vs_02":
                g1 = x2.isin([3, 4]); g0 = x2.isin([0, 1, 2])
            elif spec["custom"] == "tumor_number_34_vs_12":
                g1 = x2.isin([3, 4]); g0 = x2.isin([1, 2])

            return d2, g1, g0, spec["g1_label"], spec["g0_label"]

        return None

    # -------------------------
    # Multi-filter masks (tab B)
    # -------------------------
    def mask_from_slot(self, d: pd.DataFrame, slot: FilterSlot):
        """
        Return boolean mask for rows that satisfy this slot condition.
        If slot not used => None.
        """
        if not slot.is_used():
            return None, ""

        name = slot.current_name()
        spec = self.predictors.get(name)
        if spec is None:
            return None, ""

        col = spec["col"]
        tp  = spec["type"]
        want_g1 = slot.pick_g1()

        if tp == "binary":
            x = coerce_num(d[col])
            valid = x.isin([0, 1])
            m = pd.Series(False, index=d.index)
            if want_g1:
                m.loc[valid] = (x.loc[valid] == 1)
                desc = f"{name}: G1"
            else:
                m.loc[valid] = (x.loc[valid] == 0)
                desc = f"{name}: G0"
            return m, desc

        if tp == "continuous":
            x = coerce_num(d[col])
            valid = x.notna()
            cut = slot.get_cut_value()
            if cut is None:
                cut = float(x.dropna().median()) if valid.any() else None
            m = pd.Series(False, index=d.index)
            if cut is None:
                return None, ""
            if want_g1:
                m.loc[valid] = (x.loc[valid] > cut)
                desc = f"{name}: > {cut:.2f}"
            else:
                m.loc[valid] = (x.loc[valid] <= cut)
                desc = f"{name}: ≤ {cut:.2f}"
            return m, desc

        if tp == "custom":
            x = coerce_num(d[col])
            valid = x.notna()
            m = pd.Series(False, index=d.index)
            if not valid.any():
                return None, ""

            if spec["custom"] == "child_pugh_1_vs_23":
                g1 = (x == 1)
                g0 = x.isin([2, 3])
            elif spec["custom"] == "edmonson_34_vs_02":
                g1 = x.isin([3, 4])
                g0 = x.isin([0, 1, 2])
            elif spec["custom"] == "tumor_number_34_vs_12":
                g1 = x.isin([3, 4])
                g0 = x.isin([1, 2])
            else:
                return None, ""

            if want_g1:
                m.loc[valid] = g1.loc[valid]
                desc = f"{name}: G1"
            else:
                m.loc[valid] = g0.loc[valid]
                desc = f"{name}: G0"
            return m, desc

        return None, ""

    # -------------------------
    # Plotting core for one panel (immunotherapy vs no)
    # -------------------------
    def plot_one_panel(self, ax_km, ax_tbl, d_panel: pd.DataFrame, title: str):
        ax_km.clear()
        ax_tbl.clear()

        if d_panel is None or len(d_panel) == 0:
            ax_km.axis("off")
            ax_km.text(0.02, 0.6, "No data in this panel", fontsize=12, color=COL_GRAY)
            ax_km.set_title(title, fontsize=12, fontweight="bold")
            ax_tbl.axis("off")
            return

        # split by immunotherapy
        d_no = d_panel[d_panel["_immune"] == 0].copy()
        d_yes = d_panel[d_panel["_immune"] == 1].copy()

        n_no = len(d_no); n_yes = len(d_yes)
        if n_no == 0 or n_yes == 0:
            ax_km.axis("off")
            ax_km.set_title(title, fontsize=12, fontweight="bold")
            ax_km.text(
                0.02, 0.6,
                f"One group is empty.\nNo={n_no}, Yes={n_yes}\n无法比较免疫 vs 未免疫。",
                fontsize=11, color=COL_GRAY
            )
            ax_tbl.axis("off")
            return

        # Fit KM
        km_no = KaplanMeierFitter()
        km_yes = KaplanMeierFitter()

        km_no.fit(d_no["time60"].values, event_observed=d_no["status60"].values,
                  weights=coerce_num(d_no["_weight"]).values,
                  label=f"No immunotherapy (n={n_no})")
        km_yes.fit(d_yes["time60"].values, event_observed=d_yes["status60"].values,
                   weights=coerce_num(d_yes["_weight"]).values,
                   label=f"Immunotherapy (n={n_yes})")

        km_no.plot_survival_function(ax=ax_km, ci_show=False, color=COL_NO, linewidth=2.2)
        km_yes.plot_survival_function(ax=ax_km, ci_show=False, color=COL_IMMUNE, linewidth=2.2)

        ax_km.set_xlim(0, MAX_MO)
        ax_km.set_xlabel("Time (months)")
        ax_km.set_ylabel("Overall survival probability")
        ax_km.set_title(title, fontsize=12, fontweight="bold")
        ax_km.legend(loc="lower left", fontsize=9, frameon=False)

        # Weighted HR; omit log-rank here because the current workflow is based on IPTW-ATO weights.
        hr, lo, hi, p = cox_hr_immunotherapy(d_panel)
        if hr is None:
            hr_txt = "Weighted HR: NA"
        else:
            hr_txt = f"Weighted HR(Immune vs No)={hr:.2f} ({lo:.2f}–{hi:.2f}) | CoxP={fmt_p(p)}"

        ax_km.text(
            0.99, 0.98, hr_txt,
            transform=ax_km.transAxes, ha="right", va="top",
            fontsize=9.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#999", alpha=0.92),
        )

        # Risk table
        times = [0, 12, 24, 36, 48, 60]
        draw_risk_table(ax_tbl, d_no, d_yes, times)

    # -------------------------
    # Main update
    # -------------------------
    def update_everything(self):
        # update slider ranges for current tab
        self.update_sliderA_range()
        self.update_slot_slider(self.slot1)
        self.update_slot_slider(self.slot2)
        self.update_slot_slider(self.slot3)
        self.update_plot()

    def update_plot(self):
        if self.df_raw is None:
            return

        d = self.get_context_df()
        if d is None or len(d) == 0:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.02, 0.6, "No data after current filters.", fontsize=12, color=COL_GRAY)
            self.canvas.draw()
            return

        self.fig.clear()

        # 2 columns, each column has KM + risk table (2 rows)
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[3.2, 1.0], wspace=0.25, hspace=0.15)
        axL = self.fig.add_subplot(gs[0, 0])
        axR = self.fig.add_subplot(gs[0, 1])
        axLt= self.fig.add_subplot(gs[1, 0])
        axRt= self.fig.add_subplot(gs[1, 1])

        tab_idx = self.tabs.currentIndex()

        # =========================
        # Tab A: Single predictor POS vs NEG
        # =========================
        if tab_idx == 0:
            if self.current_pred is None:
                return

            if self.current_pred["type"] == "continuous":
                self.update_cutA_label()
                cut = self.get_cutA_value()
            else:
                cut = None

            grp = self.mask_g1_g0_from_spec(d, self.current_pred, cut_value=cut)
            if grp is None:
                return
            d2, g1, g0, g1_label, g0_label = grp

            # two panels: POS (G1) and NEG (G0), each compares immune vs no
            d_pos = d2.loc[g1].copy()
            d_neg = d2.loc[g0].copy()

            left_title = f"POS/HIGH group: {g1_label}\n({self.cb_bclc.currentText()})"
            right_title= f"NEG/LOW group: {g0_label}\n({self.cb_bclc.currentText()})"

            self.plot_one_panel(axL, axLt, d_pos, left_title)
            self.plot_one_panel(axR, axRt, d_neg, right_title)

            self.fig.suptitle("Single Predictor Mode: Immunotherapy vs No within POS and NEG (IPTW-ATO weighted)", fontsize=13, fontweight="bold")

        # =========================
        # Tab B: Multi-filter cohort Selected vs Remaining
        # =========================
        else:
            # update slot labels
            self.slot1.update_cut_label()
            self.slot2.update_cut_label()
            self.slot3.update_cut_label()

            m1, d1 = self.mask_from_slot(d, self.slot1)
            m2, d2 = self.mask_from_slot(d, self.slot2)
            m3, d3 = self.mask_from_slot(d, self.slot3)

            masks = []
            descs = []
            for m, ds in [(m1, d1), (m2, d2), (m3, d3)]:
                if m is not None:
                    masks.append(m)
                    if ds:
                        descs.append(ds)

            if len(masks) == 0:
                # no conditions => left=all, right=empty (提示)
                d_sel = d.copy()
                d_rem = d.iloc[0:0].copy()
                desc_txt = "No conditions selected => left=ALL, right=EMPTY"
            else:
                sel_mask = masks[0].copy()
                for mm in masks[1:]:
                    sel_mask = sel_mask & mm
                d_sel = d.loc[sel_mask].copy()
                d_rem = d.loc[~sel_mask].copy()
                desc_txt = " AND ".join(descs)

            left_title = f"Selected cohort (ALL conditions)\n{desc_txt}\n({self.cb_bclc.currentText()})"
            right_title= f"Remaining cohort (others)\n({self.cb_bclc.currentText()})"

            self.plot_one_panel(axL, axLt, d_sel, left_title)
            self.plot_one_panel(axR, axRt, d_rem, right_title)

            self.fig.suptitle("Multi-filter Mode: Immunotherapy vs No in Selected vs Remaining (IPTW-ATO weighted)", fontsize=13, fontweight="bold")

        self.fig.tight_layout(rect=[0, 0.01, 1, 0.95])
        self.canvas.draw()


# =========================
# GUI runner
# =========================
def run_gui(use_embedded=True, external_excel=None):
    app = QApplication(sys.argv)
    icon_path = resource_path(ICON_PNG_NAME)
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    w = KMCompareApp(use_embedded=use_embedded, external_excel=external_excel)
    w.show()
    sys.exit(app.exec())


# =========================
# Build EXE
# =========================
def ensure_packages():
    pkgs = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("matplotlib", "matplotlib"),
        ("lifelines", "lifelines"),
        ("PySide6", "PySide6"),
        ("PyInstaller", "pyinstaller"),
        ("PIL", "Pillow"),
    ]
    for import_name, pip_name in pkgs:
        try:
            __import__(import_name)
        except Exception:
            print(f"[INFO] Installing missing package: {pip_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pip_name])

def build_exe(src_excel: str, out_dir: str):
    ensure_packages()

    script_path = os.path.abspath(__file__)
    work_dir = os.path.dirname(script_path)

    # 1) generate embedded anonymized excel under ./data/
    embedded_rel = os.path.join("data", "BC期_overlap_weighting_ATO_加权完整数据_V5_anon.xlsx")
    embedded_abs = os.path.join(work_dir, embedded_rel)

    print("[STEP] Generating anonymized embedded Excel ...")
    make_anonymized_excel(src_excel, embedded_abs, drop_cols=PRIVACY_DROP_COLS)
    print(f"[OK] Embedded Excel: {embedded_abs}")

    # 2) prepare icon + pyinstaller --add-data
    sep = ";" if os.name == "nt" else ":"
    add_data_args = [f"{embedded_abs}{sep}data"]

    png_icon_path, ico_icon_path = create_windows_icon_if_needed(work_dir)
    if png_icon_path is not None:
        add_data_args.append(f"{png_icon_path}{sep}.")
        print(f"[OK] App window icon resource found: {png_icon_path}")
    else:
        print(f"[WARN] {ICON_PNG_NAME} not found in script folder; build will continue without custom icon.")

    print("[STEP] Building exe via PyInstaller ...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
    ]
    for arg in add_data_args:
        cmd.extend(["--add-data", arg])
    if ico_icon_path is not None:
        cmd.extend(["--icon", ico_icon_path])
        print(f"[OK] Windows exe icon will use: {ico_icon_path}")
    else:
        print("[WARN] app_icon.ico could not be created; exe may use the default icon.")
    cmd.append(script_path)
    print("[CMD]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=work_dir)

    # 3) copy exe
    dist_dir = os.path.join(work_dir, "dist")
    exe_name = os.path.splitext(os.path.basename(script_path))[0] + ".exe"
    exe_src = os.path.join(dist_dir, exe_name)

    os.makedirs(out_dir, exist_ok=True)
    exe_dst = os.path.join(out_dir, "hcc_km_iptw_ato_v5.exe")
    shutil.copy2(exe_src, exe_dst)

    print("[DONE] EXE copied to:", exe_dst)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-anon", action="store_true", help="Only generate anonymized Excel into ./data/")
    parser.add_argument("--build-exe", action="store_true", help="Build EXE with embedded anonymized Excel")
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE_EXCEL, help="Source Excel path")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTPUT_DIR, help="Where to copy the final exe")
    parser.add_argument("--external", type=str, default=None, help="Run GUI with external excel (optional)")
    parser.add_argument("--no-embedded", action="store_true", help="Run GUI without embedded (use external)")
    args = parser.parse_args()

    if args.make_anon:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "BC期_overlap_weighting_ATO_加权完整数据_V5_anon.xlsx")
        print("[STEP] Generating anonymized Excel:", out)
        make_anonymized_excel(args.src, out, drop_cols=PRIVACY_DROP_COLS)
        print("[DONE] Anonymized Excel saved:", out)
        return

    if args.build_exe:
        build_exe(args.src, args.outdir)
        return

    # Run GUI
    if args.external:
        run_gui(use_embedded=False, external_excel=args.external)
    else:
        run_gui(use_embedded=(not args.no_embedded), external_excel=None)


if __name__ == "__main__":
    main()