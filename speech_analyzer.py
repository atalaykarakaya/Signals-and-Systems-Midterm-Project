"""
COE216 - Signals and Systems
Midterm Project: Sound Signal Analysis and Gender Classification
Features: F0 (Autocorrelation), ZCR, Energy, Rule-Based Classifier
GUI: CustomTkinter modern dark interface
"""

import os
import sys
import glob
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import librosa
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

# ─────────────────────────────────────────────
#  SIGNAL PROCESSING CORE
# ─────────────────────────────────────────────

WINDOW_MS = 25          # window length in ms
HOP_MS    = 10          # hop length in ms
F0_MIN    = 60          # Hz – minimum pitch to detect
F0_MAX    = 450         # Hz – maximum pitch to detect

# Rule-based thresholds — grid-searched on Turkish speech literature
# Male:   80–160 Hz avg ~120 Hz
# Female: 155–270 Hz avg ~205 Hz   (overlap with Child 220–270 Hz → ZCR breaks the tie)
# Child:  180–420 Hz avg ~275 Hz   (6-16 yrs, high ZCR due to small vocal tract)
THRESH_MALE_F0_MAX   = 175
THRESH_CHILD_F0_MIN  = 270
THRESH_CHILD_ZCR_MIN = 0.032

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def preprocess(y, sr):
    """Normalize + bandpass 60–450 Hz"""
    b, a = butter_bandpass(60, min(450, sr // 2 - 1), sr)
    y = filtfilt(b, a, y)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def short_term_energy(frame):
    return np.sum(frame ** 2) / len(frame)

def zero_crossing_rate(frame):
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

def autocorrelation_f0(frame, sr):
    """Estimate F0 via autocorrelation, returns Hz or None."""
    n = len(frame)
    # Remove DC
    frame = frame - np.mean(frame)
    # Autocorrelation via FFT
    R = np.real(np.fft.ifft(np.abs(np.fft.fft(frame, n=2 * n)) ** 2))[:n]
    R = R / (R[0] + 1e-9)

    lag_min = int(sr / F0_MAX)
    lag_max = int(sr / F0_MIN)
    if lag_max >= n:
        lag_max = n - 1
    if lag_min >= lag_max:
        return None

    search = R[lag_min:lag_max]
    if len(search) == 0:
        return None
    peak_idx = np.argmax(search)
    if search[peak_idx] < 0.3:   # confidence threshold
        return None
    lag = peak_idx + lag_min
    return sr / lag

def analyze_file(file_path, sr_target=16000):
    """
    Full analysis pipeline for one wav file.
    Returns dict with f0_mean, f0_std, zcr_mean, energy_mean,
    voiced_ratio, f0_values, zcr_values, energy_values, times, y, sr
    """
    try:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)
    except Exception as e:
        return None

    y = preprocess(y, sr)

    win_len = int(sr * WINDOW_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)

    f0_values, zcr_values, energy_values, times = [], [], [], []

    for start in range(0, len(y) - win_len, hop_len):
        frame = y[start:start + win_len]
        t     = (start + win_len / 2) / sr

        e   = short_term_energy(frame)
        zcr = zero_crossing_rate(frame)

        energy_values.append(e)
        zcr_values.append(zcr)
        times.append(t)

    energy_arr = np.array(energy_values)
    zcr_arr    = np.array(zcr_values)

    # Voiced / unvoiced decision
    e_thresh   = np.percentile(energy_arr, 30)
    zcr_thresh = np.percentile(zcr_arr, 70)
    voiced     = (energy_arr > e_thresh) & (zcr_arr < zcr_thresh)

    # F0 only on voiced frames
    f0_per_frame = []
    for i, start in enumerate(range(0, len(y) - win_len, hop_len)):
        if voiced[i]:
            frame = y[start:start + win_len]
            f0    = autocorrelation_f0(frame, sr)
            if f0 is not None:
                f0_per_frame.append(f0)
        else:
            f0_per_frame.append(np.nan)

    f0_arr    = np.array(f0_per_frame)
    valid_f0  = f0_arr[~np.isnan(f0_arr)]

    return {
        "f0_mean"      : float(np.mean(valid_f0))   if len(valid_f0) else np.nan,
        "f0_std"       : float(np.std(valid_f0))    if len(valid_f0) else np.nan,
        "zcr_mean"     : float(np.mean(zcr_arr)),
        "energy_mean"  : float(np.mean(energy_arr)),
        "voiced_ratio" : float(np.mean(voiced)),
        "f0_values"    : f0_arr.tolist(),
        "zcr_values"   : zcr_arr.tolist(),
        "energy_values": energy_arr.tolist(),
        "times"        : times,
        "y"            : y.tolist(),
        "sr"           : sr,
    }

def rule_based_classify(f0_mean, zcr_mean=None):
    if np.isnan(f0_mean):
        return "Unknown"

    if f0_mean <= THRESH_MALE_F0_MAX:          # ≤175 Hz → Male
        return "Male"

    if f0_mean >= THRESH_CHILD_F0_MIN:         # ≥270 Hz
        # ZCR yüksekse Child, düşükse yüksek-perdeli Female
        if zcr_mean is not None and zcr_mean >= THRESH_CHILD_ZCR_MIN:
            return "Child"
        return "Female"

    return "Female"   # 175–270 Hz arası → Female

LABEL_MAP = {"K": "Male", "E": "Female", "C": "Child",
             "M": "Male", "F": "Female",
             "Male": "Male", "Female": "Female", "Child": "Child"}

def normalize_label(raw):
    return LABEL_MAP.get(str(raw).strip(), str(raw).strip())

# ─────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
BG3      = "#21262d"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
ACCENT4  = "#d2a8ff"
TEXT     = "#e6edf3"
TEXT2    = "#8b949e"
BORDER   = "#30363d"

MALE_C   = "#58a6ff"
FEM_C    = "#f778ba"
CHILD_C  = "#56d364"
UNK_C    = "#e3b341"

CLASS_COLORS = {"Male": MALE_C, "Female": FEM_C, "Child": CHILD_C, "Unknown": UNK_C}

MPL_STYLE = {
    "axes.facecolor"  : BG2,
    "figure.facecolor": BG,
    "axes.edgecolor"  : BORDER,
    "axes.labelcolor" : TEXT2,
    "xtick.color"     : TEXT2,
    "ytick.color"     : TEXT2,
    "text.color"      : TEXT,
    "grid.color"      : BORDER,
    "grid.linestyle"  : "--",
    "grid.alpha"      : 0.4,
}
plt.rcParams.update(MPL_STYLE)

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class SpeechAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("COE216 • Speech Analysis & Gender Classification")
        self.root.configure(bg=BG)
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)

        self.dataset_path  = tk.StringVar(value="")
        self.metadata_path = tk.StringVar(value="")
        self.single_wav    = tk.StringVar(value="")

        self.results_df    = None   # full analysis results
        self.current_result= None   # result for single-file analysis

        self._build_ui()

    # ── Layout ────────────────────────────────
    def _build_ui(self):
        # Top header bar
        hdr = tk.Frame(self.root, bg=BG, height=56)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🎙  Speech Analysis & Gender Classification",
                 bg=BG, fg=TEXT, font=("Courier New", 15, "bold")).pack(side="left", padx=24, pady=10)
        tk.Label(hdr, text="COE216 · Signals and Systems · 2025-2026",
                 bg=BG, fg=TEXT2, font=("Courier New", 9)).pack(side="right", padx=24)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook",       background=BG,  borderwidth=0)
        style.configure("Dark.TNotebook.Tab",   background=BG2, foreground=TEXT2,
                        padding=[16, 7], font=("Courier New", 9, "bold"), borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG3)],
                  foreground=[("selected", ACCENT)])

        self.nb = ttk.Notebook(self.root, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=0, pady=0)

        self._tab_single  = tk.Frame(self.nb, bg=BG)
        self._tab_dataset = tk.Frame(self.nb, bg=BG)
        self._tab_stats   = tk.Frame(self.nb, bg=BG)

        self.nb.add(self._tab_single,  text="  Single File Analysis  ")
        self.nb.add(self._tab_dataset, text="  Dataset Batch Analysis  ")
        self.nb.add(self._tab_stats,   text="  Statistics & Confusion Matrix  ")

        self._build_single_tab()
        self._build_dataset_tab()
        self._build_stats_tab()

    # ─────────────────────────────────────────
    #  TAB 1 – Single File
    # ─────────────────────────────────────────
    def _build_single_tab(self):
        p = self._tab_single

        # Left control panel
        ctrl = tk.Frame(p, bg=BG2, width=310)
        ctrl.pack(side="left", fill="y", padx=0)
        ctrl.pack_propagate(False)
        tk.Frame(ctrl, bg=BORDER, width=1).pack(side="right", fill="y")

        self._section(ctrl, "SINGLE FILE ANALYSIS")

        # File picker
        self._label(ctrl, "Select WAV File")
        row = tk.Frame(ctrl, bg=BG2)
        row.pack(fill="x", padx=14, pady=(0, 8))
        tk.Entry(row, textvariable=self.single_wav, bg=BG3, fg=TEXT2,
                 relief="flat", font=("Courier New", 8),
                 insertbackground=TEXT).pack(side="left", fill="x", expand=True)
        self._btn(row, "Browse", self._browse_single, small=True).pack(side="right", padx=(4, 0))

        self._btn(ctrl, "▶  Analyze File", self._run_single_analysis).pack(fill="x", padx=14, pady=(4, 0))

        # Result display
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", padx=14, pady=12)
        self._section(ctrl, "CLASSIFICATION RESULT", pad_top=0)

        self.result_frame = tk.Frame(ctrl, bg=BG2)
        self.result_frame.pack(fill="x", padx=14)
        self.lbl_class  = tk.Label(self.result_frame, text="—", bg=BG2,
                                   fg=TEXT2, font=("Courier New", 28, "bold"))
        self.lbl_class.pack()
        self.lbl_f0     = self._info_row(ctrl, "Avg F0 (Hz)",    "—")
        self.lbl_f0std  = self._info_row(ctrl, "F0 Std (Hz)",    "—")
        self.lbl_zcr    = self._info_row(ctrl, "Avg ZCR",        "—")
        self.lbl_energy = self._info_row(ctrl, "Avg Energy",     "—")
        self.lbl_voiced = self._info_row(ctrl, "Voiced Ratio",   "—")

        # Autocorr vs FFT button
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", padx=14, pady=12)
        self._btn(ctrl, "📊  Autocorr vs FFT Comparison", self._show_autocorr_vs_fft).pack(fill="x", padx=14)
        self._btn(ctrl, "💾  Export Feature CSV", self._export_single).pack(fill="x", padx=14, pady=(6, 0))

        # Right plot area
        plot_area = tk.Frame(p, bg=BG)
        plot_area.pack(side="right", fill="both", expand=True)
        self.fig_single  = Figure(figsize=(9, 6), facecolor=BG)
        self.canvas_single = FigureCanvasTkAgg(self.fig_single, master=plot_area)
        self.canvas_single.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._draw_placeholder(self.fig_single, "Select a WAV file and click Analyze")

    # ─────────────────────────────────────────
    #  TAB 2 – Dataset
    # ─────────────────────────────────────────
    def _build_dataset_tab(self):
        p = self._tab_dataset

        ctrl = tk.Frame(p, bg=BG2, width=310)
        ctrl.pack(side="left", fill="y")
        ctrl.pack_propagate(False)
        tk.Frame(ctrl, bg=BORDER, width=1).pack(side="right", fill="y")

        self._section(ctrl, "DATASET CONFIGURATION")

        self._label(ctrl, "WAV Folder (Dataset/)")
        row = tk.Frame(ctrl, bg=BG2); row.pack(fill="x", padx=14, pady=(0, 8))
        tk.Entry(row, textvariable=self.dataset_path, bg=BG3, fg=TEXT2,
                 relief="flat", font=("Courier New", 8),
                 insertbackground=TEXT).pack(side="left", fill="x", expand=True)
        self._btn(row, "Browse", self._browse_dataset, small=True).pack(side="right", padx=(4, 0))

        self._label(ctrl, "Metadata CSV (semicolon)")
        row2 = tk.Frame(ctrl, bg=BG2); row2.pack(fill="x", padx=14, pady=(0, 8))
        tk.Entry(row2, textvariable=self.metadata_path, bg=BG3, fg=TEXT2,
                 relief="flat", font=("Courier New", 8),
                 insertbackground=TEXT).pack(side="left", fill="x", expand=True)
        self._btn(row2, "Browse", self._browse_meta, small=True).pack(side="right", padx=(4, 0))

        self._btn(ctrl, "⚙  Run Batch Analysis", self._run_batch).pack(fill="x", padx=14, pady=(4, 0))

        # Progress
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", padx=14, pady=10)
        self._label(ctrl, "Progress")
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(ctrl, variable=self.progress_var,
                                             maximum=100, length=270)
        style2 = ttk.Style()
        style2.configure("green.Horizontal.TProgressbar",
                         troughcolor=BG3, background=ACCENT2, borderwidth=0)
        self.progress_bar.configure(style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(padx=14, pady=(0, 4))

        self.lbl_progress = tk.Label(ctrl, text="Idle", bg=BG2, fg=TEXT2,
                                      font=("Courier New", 8))
        self.lbl_progress.pack(padx=14, anchor="w")

        # Quick stats
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", padx=14, pady=10)
        self._section(ctrl, "QUICK STATS", pad_top=0)
        self.lbl_total     = self._info_row(ctrl, "Total Files",   "—")
        self.lbl_acc       = self._info_row(ctrl, "Accuracy",      "—")
        self.lbl_male_acc  = self._info_row(ctrl, "Male Acc.",     "—")
        self.lbl_fem_acc   = self._info_row(ctrl, "Female Acc.",   "—")
        self.lbl_child_acc = self._info_row(ctrl, "Child Acc.",    "—")

        self._btn(ctrl, "💾  Export Results CSV", self._export_results).pack(fill="x", padx=14, pady=(12, 0))

        # Table frame (right)
        right = tk.Frame(p, bg=BG)
        right.pack(fill="both", expand=True)

        # Treeview
        tree_frame = tk.Frame(right, bg=BG)
        tree_frame.pack(fill="both", expand=True, padx=8, pady=8)

        cols = ("File", "True", "Predicted", "F0 (Hz)", "ZCR", "Energy", "Status")
        style3 = ttk.Style()
        style3.configure("Dark.Treeview",
                         background=BG2, foreground=TEXT, fieldbackground=BG2,
                         borderwidth=0, rowheight=22)
        style3.configure("Dark.Treeview.Heading",
                         background=BG3, foreground=ACCENT,
                         font=("Courier New", 8, "bold"), relief="flat")
        style3.map("Dark.Treeview", background=[("selected", BG3)],
                   foreground=[("selected", ACCENT)])

        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                  style="Dark.Treeview")
        widths = [220, 70, 80, 70, 70, 75, 65]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center" if col != "File" else "w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        # Tag colors
        self.tree.tag_configure("correct",   foreground=ACCENT2)
        self.tree.tag_configure("incorrect", foreground=ACCENT3)
        self.tree.tag_configure("unknown",   foreground=UNK_C)

    # ─────────────────────────────────────────
    #  TAB 3 – Stats
    # ─────────────────────────────────────────
    def _build_stats_tab(self):
        p = self._tab_stats

        self.fig_stats  = Figure(figsize=(13, 7), facecolor=BG)
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=p)
        self.canvas_stats.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._draw_placeholder(self.fig_stats, "Run Dataset Batch Analysis to see statistics")

    # ─────────────────────────────────────────
    #  UI Helpers
    # ─────────────────────────────────────────
    def _section(self, parent, text, pad_top=14):
        tk.Label(parent, text=text, bg=BG2, fg=ACCENT,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=14, pady=(pad_top, 4))

    def _label(self, parent, text):
        tk.Label(parent, text=text, bg=BG2, fg=TEXT2,
                 font=("Courier New", 8)).pack(anchor="w", padx=14)

    def _btn(self, parent, text, cmd, small=False):
        f = ("Courier New", 8, "bold") if not small else ("Courier New", 7)
        b = tk.Button(parent, text=text, command=cmd,
                      bg=BG3, fg=ACCENT, activebackground=ACCENT,
                      activeforeground=BG, relief="flat",
                      font=f, padx=6, pady=5 if not small else 3,
                      cursor="hand2", bd=0)
        b.bind("<Enter>", lambda e: b.config(bg=ACCENT, fg=BG))
        b.bind("<Leave>", lambda e: b.config(bg=BG3, fg=ACCENT))
        return b

    def _info_row(self, parent, label, value):
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=label + ":", bg=BG2, fg=TEXT2,
                 font=("Courier New", 8), width=14, anchor="w").pack(side="left")
        lbl = tk.Label(row, text=value, bg=BG2, fg=TEXT,
                       font=("Courier New", 8, "bold"), anchor="w")
        lbl.pack(side="left")
        return lbl

    def _draw_placeholder(self, fig, msg):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", color=TEXT2,
                fontsize=13, fontfamily="Courier New", style="italic")
        ax.axis("off")
        fig.tight_layout()
        if fig == self.fig_single:
            self.canvas_single.draw()
        elif fig == self.fig_stats:
            self.canvas_stats.draw()

    # ─────────────────────────────────────────
    #  File Browsers
    # ─────────────────────────────────────────
    def _browse_single(self):
        f = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All", "*.*")])
        if f: self.single_wav.set(f)

    def _browse_dataset(self):
        d = filedialog.askdirectory()
        if d: self.dataset_path.set(d)

    def _browse_meta(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")])
        if f: self.metadata_path.set(f)

    # ─────────────────────────────────────────
    #  Single File Analysis
    # ─────────────────────────────────────────
    def _run_single_analysis(self):
        path = self.single_wav.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid WAV file.")
            return
        threading.Thread(target=self._single_worker, args=(path,), daemon=True).start()

    def _single_worker(self, path):
        result = analyze_file(path)
        if result is None:
            self.root.after(0, lambda: messagebox.showerror("Error", "Could not analyze file."))
            return
        self.current_result = result
        pred = rule_based_classify(result["f0_mean"], result["zcr_mean"])
        self.root.after(0, lambda: self._update_single_ui(result, pred))

    def _update_single_ui(self, r, pred):
        color = CLASS_COLORS.get(pred, TEXT2)
        self.lbl_class.config(text=pred, fg=color)
        self.lbl_f0.config(    text=f"{r['f0_mean']:.1f}" if not np.isnan(r['f0_mean']) else "N/A")
        self.lbl_f0std.config( text=f"{r['f0_std']:.1f}"  if not np.isnan(r['f0_std'])  else "N/A")
        self.lbl_zcr.config(   text=f"{r['zcr_mean']:.4f}")
        self.lbl_energy.config(text=f"{r['energy_mean']:.4f}")
        self.lbl_voiced.config(text=f"{r['voiced_ratio']*100:.1f}%")
        self._plot_single(r, pred)

    def _plot_single(self, r, pred):
        fig = self.fig_single
        fig.clear()
        color = CLASS_COLORS.get(pred, TEXT2)

        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                               left=0.07, right=0.97, top=0.92, bottom=0.08)

        times      = np.array(r["times"])
        energy     = np.array(r["energy_values"])
        zcr        = np.array(r["zcr_values"])
        f0_vals    = np.array(r["f0_values"])
        y          = np.array(r["y"])
        sr         = r["sr"]
        t_axis     = np.linspace(0, len(y) / sr, len(y))

        # 1. Waveform
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t_axis, y, color=color, lw=0.5, alpha=0.8)
        ax1.set_title(f"Waveform  —  Predicted: {pred}", color=TEXT, fontsize=9)
        ax1.set_xlabel("Time (s)", fontsize=7); ax1.set_ylabel("Amplitude", fontsize=7)
        ax1.grid(True)

        # 2. Short-Term Energy
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(times, energy, color=ACCENT, lw=1)
        ax2.fill_between(times, energy, alpha=0.2, color=ACCENT)
        ax2.set_title("Short-Term Energy", color=TEXT, fontsize=9)
        ax2.set_xlabel("Time (s)", fontsize=7); ax2.set_ylabel("Energy", fontsize=7)
        ax2.grid(True)

        # 3. ZCR
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(times, zcr, color=ACCENT4, lw=1)
        ax3.fill_between(times, zcr, alpha=0.2, color=ACCENT4)
        ax3.set_title("Zero Crossing Rate", color=TEXT, fontsize=9)
        ax3.set_xlabel("Time (s)", fontsize=7); ax3.set_ylabel("ZCR", fontsize=7)
        ax3.grid(True)

        # 4. F0 Trajectory
        ax4 = fig.add_subplot(gs[2, 0])
        valid = ~np.isnan(f0_vals)
        if np.any(valid):
            ax4.scatter(times[valid], f0_vals[valid], color=color, s=4, alpha=0.7, zorder=3)
            ax4.axhline(r["f0_mean"], color=color, lw=1.2, ls="--", label=f"Mean: {r['f0_mean']:.1f} Hz")
            ax4.legend(fontsize=7, loc="upper right")
        ax4.set_title("F0 Trajectory (Voiced Frames)", color=TEXT, fontsize=9)
        ax4.set_xlabel("Time (s)", fontsize=7); ax4.set_ylabel("F0 (Hz)", fontsize=7)
        ax4.set_ylim(50, 500); ax4.grid(True)

        # 5. FFT Spectrum
        ax5 = fig.add_subplot(gs[2, 1])
        N   = min(4096, len(y))
        Yf  = np.abs(fft(y[:N]))[:N // 2]
        freqs = fftfreq(N, 1 / sr)[:N // 2]
        ax5.plot(freqs, Yf, color=ACCENT2, lw=0.7)
        ax5.set_xlim(0, 600)
        if not np.isnan(r["f0_mean"]):
            ax5.axvline(r["f0_mean"], color=color, lw=1.2, ls="--",
                        label=f"F0≈{r['f0_mean']:.1f} Hz")
            ax5.legend(fontsize=7, loc="upper right")
        ax5.set_title("FFT Magnitude Spectrum", color=TEXT, fontsize=9)
        ax5.set_xlabel("Freq (Hz)", fontsize=7); ax5.set_ylabel("|X(f)|", fontsize=7)
        ax5.grid(True)

        self.canvas_single.draw()

    # ─────────────────────────────────────────
    #  Autocorr vs FFT comparison popup
    # ─────────────────────────────────────────
    def _show_autocorr_vs_fft(self):
        if self.current_result is None:
            messagebox.showinfo("Info", "Analyze a file first.")
            return
        r = self.current_result
        y = np.array(r["y"]); sr = r["sr"]

        # pick a voiced frame
        f0_arr = np.array(r["f0_values"])
        valid_idx = np.where(~np.isnan(f0_arr))[0]
        if len(valid_idx) == 0:
            messagebox.showinfo("Info", "No voiced frame found."); return

        idx   = valid_idx[len(valid_idx) // 2]
        win   = int(sr * WINDOW_MS / 1000)
        hop   = int(sr * HOP_MS / 1000)
        start = idx * hop
        frame = y[start:start + win]
        frame = frame - np.mean(frame)

        n   = len(frame)
        R   = np.real(np.fft.ifft(np.abs(np.fft.fft(frame, n=2 * n)) ** 2))[:n]
        R   = R / (R[0] + 1e-9)
        lags= np.arange(n) / sr * 1000  # ms

        N   = min(4096, n)
        Yf  = np.abs(fft(frame, N))[:N // 2]
        freqs = fftfreq(N, 1 / sr)[:N // 2]

        win2 = tk.Toplevel(self.root)
        win2.title("Autocorrelation vs FFT"); win2.configure(bg=BG)
        win2.geometry("900x420")

        fig2 = Figure(figsize=(9, 4), facecolor=BG)
        ax_l = fig2.add_subplot(1, 2, 1)
        ax_r = fig2.add_subplot(1, 2, 2)

        ax_l.plot(lags, R, color=ACCENT, lw=1)
        lag_min = int(sr / F0_MAX); lag_max = int(sr / F0_MIN)
        best = np.argmax(R[lag_min:lag_max]) + lag_min
        ax_l.axvline(best / sr * 1000, color=ACCENT3, lw=1.5, ls="--",
                     label=f"F0≈{sr / best:.1f} Hz")
        ax_l.set_xlim(0, 30)
        ax_l.set_title("Autocorrelation R(τ)", color=TEXT, fontsize=10)
        ax_l.set_xlabel("Lag (ms)", fontsize=8); ax_l.set_ylabel("R(τ)", fontsize=8)
        ax_l.legend(fontsize=8); ax_l.grid(True)

        ax_r.plot(freqs, Yf, color=ACCENT2, lw=0.8)
        ax_r.set_xlim(0, 600)
        if not np.isnan(r["f0_mean"]):
            ax_r.axvline(r["f0_mean"], color=ACCENT3, lw=1.5, ls="--",
                         label=f"F0≈{r['f0_mean']:.1f} Hz")
        ax_r.set_title("FFT Magnitude Spectrum", color=TEXT, fontsize=10)
        ax_r.set_xlabel("Freq (Hz)", fontsize=8); ax_r.set_ylabel("|X(f)|", fontsize=8)
        ax_r.legend(fontsize=8); ax_r.grid(True)

        fig2.tight_layout(pad=1.5)
        c2 = FigureCanvasTkAgg(fig2, master=win2)
        c2.get_tk_widget().pack(fill="both", expand=True)
        c2.draw()

    # ─────────────────────────────────────────
    #  Batch Analysis
    # ─────────────────────────────────────────
    def _run_batch(self):
        ds   = self.dataset_path.get().strip()
        meta = self.metadata_path.get().strip()
        if not ds or not os.path.isdir(ds):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        self.lbl_total.config(text="—"); self.lbl_acc.config(text="—")
        self.lbl_male_acc.config(text="—"); self.lbl_fem_acc.config(text="—")
        self.lbl_child_acc.config(text="—")
        self.progress_var.set(0)

        threading.Thread(target=self._batch_worker, args=(ds, meta), daemon=True).start()

    def _batch_worker(self, ds, meta_path):
        # Find all wav files
        wav_files = glob.glob(os.path.join(ds, "**", "*.wav"), recursive=True)
        if not wav_files:
            wav_files = glob.glob(os.path.join(ds, "*.wav"))
        if not wav_files:
            self.root.after(0, lambda: messagebox.showerror("Error", "No WAV files found."))
            return

        # Load metadata if available
        meta_df    = None
        meta_lookup = {}   # fname_lower -> row  (exact)
        meta_lookup_nogrp = {}  # fname_without_group_prefix_lower -> row  (fallback)

        if meta_path and os.path.exists(meta_path):
            try:
                if meta_path.endswith(".xlsx"):
                    meta_df = pd.read_excel(meta_path)
                else:
                    try:
                        meta_df = pd.read_csv(meta_path, sep=";")
                    except Exception:
                        meta_df = pd.read_csv(meta_path)

                # Build two lookup dicts for fast O(1) matching
                name_col = None
                for c in ("File_Name", "File_Path", "filename", "file_name"):
                    if c in meta_df.columns:
                        name_col = c; break

                if name_col:
                    for _, row in meta_df.iterrows():
                        raw_name = os.path.basename(str(row[name_col]))
                        key_exact = raw_name.lower()
                        # Strip leading group token e.g. "G05_" -> "D01_C_9_..."
                        import re as _re
                        key_nogrp = _re.sub(r'^g\d+_', '', key_exact)
                        meta_lookup[key_exact]       = row
                        meta_lookup_nogrp[key_nogrp] = row

            except Exception as e:
                print(f"Metadata load error: {e}")

        total = len(wav_files)
        rows  = []

        for i, wav in enumerate(wav_files):
            fname = os.path.basename(wav)
            self.root.after(0, lambda f=fname, n=i, t=total:
                self.lbl_progress.config(text=f"[{n+1}/{t}] {f[:35]}"))

            result = analyze_file(wav)
            if result is None:
                pred = "Unknown"; f0 = np.nan; zcr = np.nan; energy = np.nan
            else:
                pred   = rule_based_classify(result["f0_mean"], result["zcr_mean"])
                f0     = result["f0_mean"]
                zcr    = result["zcr_mean"]
                energy = result["energy_mean"]


            # Look up true label
            # Step 1: exact filename match in CSV
            true_label = "?"
            if meta_lookup:
                key_exact = fname.lower()
                meta_row  = meta_lookup.get(key_exact)
                if meta_row is not None:
                    for col in ("Gender", "gender", "Label", "label", "Class"):
                        if col in meta_row.index:
                            true_label = normalize_label(meta_row[col])
                            break

            # Step 2: not in CSV -> parse gender token from filename
            # Standard format: G##_D##_<GENDER>_<AGE>_<Emotion>_C#.wav
            if true_label == "?":
                import re as _re
                m = _re.match(r'g\d+_d\d+_([a-z]+)_', fname.lower())
                if m:
                    true_label = normalize_label(m.group(1).upper())
            rows.append({
                "file"      : fname,
                "true"      : true_label,
                "predicted" : pred,
                "f0"        : f0,
                "zcr"       : zcr,
                "energy"    : energy,
            })

            pct = (i + 1) / total * 100
            self.root.after(0, lambda p=pct: self.progress_var.set(p))

            # Add to treeview
            status = "?" if true_label == "?" else ("✓" if pred == true_label else "✗")
            tag    = "correct" if status == "✓" else ("incorrect" if status == "✗" else "unknown")
            self.root.after(0, lambda r=(fname, true_label, pred,
                                         f"{ f0:.1f}" if not np.isnan(f0) else "N/A",
                                         f"{zcr:.4f}" if not np.isnan(zcr) else "N/A",
                                         f"{energy:.4f}" if not np.isnan(energy) else "N/A",
                                         status), tg=tag:
                self.tree.insert("", "end", values=r, tags=(tg,)))

        self.results_df = pd.DataFrame(rows)
        self.root.after(0, self._finalize_batch)

    def _finalize_batch(self):
        df = self.results_df
        total = len(df)
        self.lbl_total.config(text=str(total))
        self.lbl_progress.config(text="Done ✓")

        labeled = df[df["true"] != "?"]
        if len(labeled):
            acc = (labeled["predicted"] == labeled["true"]).mean()
            self.lbl_acc.config(text=f"{acc*100:.1f}%", fg=ACCENT2)
            for cls, lbl in [("Male", self.lbl_male_acc),
                              ("Female", self.lbl_fem_acc),
                              ("Child", self.lbl_child_acc)]:
                sub = labeled[labeled["true"] == cls]
                if len(sub):
                    a = (sub["predicted"] == sub["true"]).mean()
                    lbl.config(text=f"{a*100:.1f}%", fg=CLASS_COLORS.get(cls, TEXT))
                else:
                    lbl.config(text="N/A")

        self._draw_stats_tab()

    # ─────────────────────────────────────────
    #  Statistics Tab
    # ─────────────────────────────────────────
    def _draw_stats_tab(self):
        df = self.results_df
        if df is None or df.empty:
            return

        fig = self.fig_stats; fig.clear()
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38,
                                left=0.07, right=0.97, top=0.93, bottom=0.1)

        labeled = df[(df["true"] != "?") & (~df["f0"].isna())]
        classes = ["Male", "Female", "Child"]
        c_colors= [MALE_C, FEM_C, CHILD_C]

        # 1. F0 distributions per class
        ax1 = fig.add_subplot(gs[0, 0])
        for cls, col in zip(classes, c_colors):
            vals = labeled[labeled["true"] == cls]["f0"].dropna()
            if len(vals):
                ax1.hist(vals, bins=20, alpha=0.6, color=col, label=cls, edgecolor=BG)
        ax1.axvline(THRESH_MALE_F0_MAX,  color="white", lw=1, ls="--", alpha=0.6)
        ax1.axvline(THRESH_CHILD_F0_MIN, color="white", lw=1, ls="--", alpha=0.6)
        ax1.set_title("F0 Distribution by Class", color=TEXT, fontsize=9)
        ax1.set_xlabel("F0 (Hz)", fontsize=7); ax1.set_ylabel("Count", fontsize=7)
        ax1.legend(fontsize=7); ax1.grid(True)

        # 2. Confusion matrix
        ax2 = fig.add_subplot(gs[0, 1])
        cm  = np.zeros((3, 3), dtype=int)
        for i, true_c in enumerate(classes):
            for j, pred_c in enumerate(classes):
                cm[i, j] = int(((labeled["true"] == true_c) & (labeled["predicted"] == pred_c)).sum())
        im = ax2.imshow(cm, cmap="Blues", aspect="auto")
        ax2.set_xticks(range(3)); ax2.set_xticklabels(classes, fontsize=7, color=TEXT2)
        ax2.set_yticks(range(3)); ax2.set_yticklabels(classes, fontsize=7, color=TEXT2)
        ax2.set_xlabel("Predicted", fontsize=7); ax2.set_ylabel("True", fontsize=7)
        ax2.set_title("Confusion Matrix", color=TEXT, fontsize=9)
        cm_max = cm.max() if cm.max() > 0 else 1
        for ii in range(3):
            for jj in range(3):
                text_color = "white" if cm[ii, jj] < cm_max * 0.6 else "black"
                ax2.text(jj, ii, str(cm[ii, jj]),
                        ha="center", va="center", fontsize=9, color=text_color,
                        fontweight="bold")

        # 3. Accuracy bar
        ax3 = fig.add_subplot(gs[0, 2])
        accs = []
        for cls in classes:
            sub = labeled[labeled["true"] == cls]
            accs.append((sub["predicted"] == sub["true"]).mean() * 100 if len(sub) else 0)
        bars = ax3.bar(classes, accs, color=c_colors, width=0.5, edgecolor=BG)
        for bar, val in zip(bars, accs):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax3.set_ylim(0, 110); ax3.set_title("Classification Accuracy", color=TEXT, fontsize=9)
        ax3.set_ylabel("Accuracy (%)", fontsize=7); ax3.grid(True, axis="y")

        # 4. F0 boxplot
        ax4 = fig.add_subplot(gs[1, 0])
        data_box = [labeled[labeled["true"] == cls]["f0"].dropna().values for cls in classes]
        bp = ax4.boxplot(data_box, patch_artist=True, medianprops={"color": TEXT, "lw": 1.5})
        for patch, col in zip(bp["boxes"], c_colors):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax4.set_xticks(range(1, 4)); ax4.set_xticklabels(classes, fontsize=7)
        ax4.set_title("F0 Boxplot by Class", color=TEXT, fontsize=9)
        ax4.set_ylabel("F0 (Hz)", fontsize=7); ax4.grid(True, axis="y")

        # 5. ZCR distribution
        ax5 = fig.add_subplot(gs[1, 1])
        for cls, col in zip(classes, c_colors):
            vals = labeled[labeled["true"] == cls]["zcr"].dropna()
            if len(vals):
                ax5.hist(vals, bins=20, alpha=0.6, color=col, label=cls, edgecolor=BG)
        ax5.set_title("ZCR Distribution by Class", color=TEXT, fontsize=9)
        ax5.set_xlabel("ZCR", fontsize=7); ax5.legend(fontsize=7); ax5.grid(True)

        # 6. Statistical table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        table_data = [["Class", "N", "Avg F0", "Std F0", "Acc%"]]
        for cls in classes:
            sub  = labeled[labeled["true"] == cls]
            n    = len(sub)
            mu   = sub["f0"].mean()
            std  = sub["f0"].std()
            acc  = (sub["predicted"] == sub["true"]).mean() * 100 if n else 0
            table_data.append([cls, str(n),
                                f"{mu:.1f}" if n else "—",
                                f"{std:.1f}" if n else "—",
                                f"{acc:.1f}%" if n else "—"])
        tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        tbl.scale(1, 1.6)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor(BG3 if row == 0 else BG2)
            cell.set_edgecolor(BORDER)
            cell.set_text_props(color=ACCENT if row == 0 else TEXT)
        ax6.set_title("Statistical Summary", color=TEXT, fontsize=9, pad=12)

        fig.tight_layout()
        self.canvas_stats.draw()
        self.nb.select(self._tab_stats)

    # ─────────────────────────────────────────
    #  Exports
    # ─────────────────────────────────────────
    def _export_single(self):
        if self.current_result is None:
            messagebox.showinfo("Info", "No analysis result yet."); return
        r = self.current_result
        row = {"f0_mean": r["f0_mean"], "f0_std": r["f0_std"],
               "zcr_mean": r["zcr_mean"], "energy_mean": r["energy_mean"],
               "voiced_ratio": r["voiced_ratio"],
               "predicted": rule_based_classify(r["f0_mean"], r["zcr_mean"])}
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if path:
            pd.DataFrame([row]).to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Exported to {path}")

    def _export_results(self):
        if self.results_df is None:
            messagebox.showinfo("Info", "Run batch analysis first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if path:
            self.results_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Exported to {path}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = SpeechAnalyzerApp(root)
    root.mainloop()