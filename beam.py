"""
Cardboard Honeycomb Sandwich U-Beam Stress Analyser
====================================================
Inverted U-section (∩) with uniform distributed load on the top flat face.
Uses sandwich beam theory (Allen 1969) + thin-walled open section shear flow.

Run:  python sandwich_u_beam.py
Requires: numpy, matplotlib  (pip install numpy matplotlib)
tkinter is part of the Python standard library.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

import math as _math

def _corrugated_props(flute, liner_grade, orientation):
    p  = flute["p"]
    a  = flute["a"]
    c  = flute["c"]
    tm = flute["tm"]

    Em = liner_grade["E_medium"]
    El = liner_grade["E_MD"] if orientation == "MD" else liner_grade["E_CD"]
    sigma_allow = liner_grade["sigma_MD"] if orientation == "MD" else liner_grade["sigma_CD"]

    Gc = Em * tm / (2 * c * p)
    Ec = (_math.pi**2 * Em * tm * a) / (2 * c * p**2)

    tau_medium = liner_grade["sigma_medium"] * 0.6
    tau_c = tau_medium * tm / (c * p)

    h_core = 2 * a

    return {
        "Ef_MPa":       El / 1e6,
        "Gc_MPa":       Gc / 1e6,
        "Ec_MPa":       Ec / 1e6,
        "sigma_f_MPa":  sigma_allow / 1e6,
        "tau_c_MPa":    max(tau_c / 1e6, 0.08),
        "h_core_mm":    h_core * 1e3,
        "rho_face":     750,
        "rho_core":     80,
        "_flute_label": flute["label"],
        "_liner_label": liner_grade["label"],
    }

_FLUTES = {
    "A-flute  (5mm board, best cushioning)":
        dict(p=8.7e-3, a=2.4e-3, c=1.56, tm=0.20e-3, label="A-flute"),
    "B-flute  (3mm board, good printability)":
        dict(p=6.5e-3, a=1.2e-3, c=1.32, tm=0.18e-3, label="B-flute"),
    "C-flute  (4mm board, most common)":
        dict(p=7.5e-3, a=1.8e-3, c=1.42, tm=0.20e-3, label="C-flute"),
    "E-flute  (1.5mm board, thin/stiff)":
        dict(p=3.5e-3, a=0.6e-3, c=1.24, tm=0.15e-3, label="E-flute"),
    "BC double-wall  (7mm board, heavy duty)":
        dict(p=7.0e-3, a=1.5e-3, c=1.37, tm=0.20e-3, label="BC double-wall"),
}

_LINERS = {
    "Kraft liner - virgin (strong)": dict(
        E_MD=6500e6, E_CD=2900e6,
        sigma_MD=30e6, sigma_CD=18e6,
        E_medium=3000e6, sigma_medium=15e6,
        label="Kraft virgin"),
    "Testliner - recycled (standard)": dict(
        E_MD=4000e6, E_CD=1400e6,
        sigma_MD=18e6, sigma_CD=10e6,
        E_medium=2500e6, sigma_medium=12e6,
        label="Testliner recycled"),
    "Kraftliner - semi-recycled (mid)": dict(
        E_MD=5000e6, E_CD=2000e6,
        sigma_MD=24e6, sigma_CD=14e6,
        E_medium=2800e6, sigma_medium=13e6,
        label="Kraftliner semi-recycled"),
}

_ORIENTATIONS = {
    "MD - flutes run along beam span  (recommended, stiffer)": "MD",
    "CD - flutes run across beam span (weaker, ~2.7x less stiff)": "CD",
}

MATERIALS = {}
for fname, fdata in _FLUTES.items():
    for lname, ldata in _LINERS.items():
        key = f"{fname}  |  {ldata['label']}"
        MATERIALS[key] = _corrugated_props(fdata, ldata, "MD")
        MATERIALS[key]["_flute_key"] = fname
        MATERIALS[key]["_liner_key"] = lname

MATERIALS["Custom (edit values below)"] = {
    "Ef_MPa": 4000, "Gc_MPa": 23, "Ec_MPa": 50,
    "sigma_f_MPa": 18, "tau_c_MPa": 0.15,
    "rho_face": 750, "rho_core": 80,
}

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyse(params):
    L   = params["L"]
    bf  = params["bf"]
    hw  = params["hw"]
    tf  = params["tf"]
    tc  = params["tc"]
    q   = params["q"]
    Ef  = params["Ef"]
    Gc  = params["Gc"]
    Ec  = params["Ec"]
    sig_f_all = params["sigma_f_allow"]
    tau_c_all = params["tau_c_allow"]

    A_top_faces = bf * (2 * tf)
    A_web_faces = 4 * (tf * hw)

    y_top_centroid = hw + tc + tf
    y_web_centroid = hw / 2.0

    y_bar = (A_top_faces * y_top_centroid + A_web_faces * y_web_centroid) / \
            (A_top_faces + A_web_faces)

    d_top = y_top_centroid - y_bar
    d_web = y_web_centroid - y_bar

    d_local_top = (tc + tf) / 2.0
    I_local_top = 2 * ((bf * tf**3 / 12.0) + (bf * tf * d_local_top**2))
    I_local_web = 4 * (tf * hw**3 / 12.0)

    I_top = I_local_top + A_top_faces * d_top**2
    I_web = I_local_web + A_web_faces * d_web**2

    EI_eff = Ef * (I_top + I_web)

    h_total = hw + 2 * (tc + tf)
    y_top = h_total - y_bar
    y_bot = y_bar

    GA_eff = 2 * Gc * tc * hw

    s = L / 2.0
    M_B = -q * s**2 / 8.0
    R_A  = q*s/2.0 + M_B/s
    R_C  = R_A
    R_B  = q*L - R_A - R_C

    n = 800
    x1 = np.linspace(0, s, n//2, endpoint=False)
    M1 = R_A*x1 - q*x1**2/2.0
    V1 = R_A - q*x1

    x2  = np.linspace(s, L, n//2)
    xr  = L - x2
    M2  =  R_C*xr - q*xr**2/2.0
    V2  = -(R_C - q*xr)

    x = np.concatenate([x1, x2])
    M = np.concatenate([M1, M2])
    V = np.concatenate([V1, V2])

    x_sag   = R_A / q
    M_sag   = R_A*x_sag - q*x_sag**2/2.0
    M_max   = max(abs(M_sag), abs(M_B))
    V_max   = abs(R_B) / 2.0

    C1_L    = -(R_A*s**2/6.0 - q*s**3/24.0)
    w_bend1 = (R_A*x1**3/6.0 - q*x1**4/24.0 + C1_L*x1) / EI_eff

    xr2     = L - x2
    C1_R    = C1_L
    w_bend2 = (R_C*xr2**3/6.0 - q*xr2**4/24.0 + C1_R*xr2) / EI_eff

    w_bend  = np.concatenate([w_bend1, w_bend2])

    w_s1_raw    = np.cumsum((R_A - q*x1) * np.gradient(x1)) / GA_eff
    w_s1_raw[0] = 0.0
    w_shear1    = w_s1_raw - w_s1_raw[-1] * (x1 / s)

    xi2         = x2 - s
    w_s2_raw    = np.cumsum((R_A - q*x2 + R_B) * np.gradient(x2)) / GA_eff
    w_s2_raw   -= w_s2_raw[0]
    w_shear2    = w_s2_raw - w_s2_raw[-1] * (xi2 / s)

    w_shear = np.concatenate([w_shear1, w_shear2])

    w_total = w_bend + w_shear
    w_max   = np.max(np.abs(w_total))

    sigma_f_top = -M * y_top * Ef / EI_eff
    sigma_f_bot =  M * y_bot * Ef / EI_eff

    sigma_f = np.where(np.abs(sigma_f_top) >= np.abs(sigma_f_bot),
                       sigma_f_top, sigma_f_bot)

    tau_c = V / (2 * tc * hw)

    y_web_from_na = abs(y_web_centroid - y_bar)
    sigma_web = M * y_web_from_na * Ef / EI_eff

    sigma_vm_web = np.sqrt(sigma_web**2 + 3 * tau_c**2)
    sigma_vm_flange = np.abs(sigma_f_top)
    sigma_vm_bot = np.abs(sigma_f_bot)
    sigma_vm = np.maximum(np.maximum(sigma_vm_web, sigma_vm_flange), sigma_vm_bot)

    sigma_wrinkle = 0.5 * (Ef * Ec * Gc)**(1/3)

    UR_face = np.max(np.abs(sigma_f)) / sig_f_all
    UR_wrinkle = np.max(np.abs(sigma_f)) / sigma_wrinkle

    return {
        "x": x, "M": M, "V": V,
        "w_bend": w_bend, "w_shear": w_shear, "w_total": w_total,
        "sigma_f": sigma_f,
        "sigma_f_top": sigma_f_top,
        "sigma_f_bot": sigma_f_bot,
        "tau_c": tau_c,
        "sigma_vm": sigma_vm,
        "EI_eff": EI_eff,
        "GA_eff": GA_eff,
        "M_max": M_max,
        "V_max": V_max,
        "w_max": w_max,
        "sigma_wrinkle": sigma_wrinkle,
        "UR_face": UR_face,
        "UR_wrinkle": UR_wrinkle,
        "y_top": y_top,
        "y_bot": y_bot,
        "y_bar": y_bar,
        "h_total": hw + 2*(tf + tc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def draw_section(ax, bf, hw, tf, tc, title="Cross-section (∩)", y_bar=None):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold")

    c_face = "#c8864a"
    c_core = "#f0d9a0"

    total_h = hw + 2*(tf + tc)

    def rect(x0, y0, w, h, fc, ec="#555", lw=0.8):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fc=fc, ec=ec, lw=lw))

    rect(0, hw + 2*tc + tf, bf, tf, c_face)
    rect(0, hw + tc + tf,   bf, tc, c_core)
    rect(0, hw + tc,        bf, tf, c_face)

    rect(0,       0, tf, hw + tc, c_face)
    rect(tf,      0, tc, hw + tc, c_core)
    rect(tf + tc, 0, tf, hw + tc, c_face)

    rect(bf - tf,           0, tf, hw + tc, c_face)
    rect(bf - tf - tc,      0, tc, hw + tc, c_core)
    rect(bf - tf - tc - tf, 0, tf, hw + tc, c_face)

    na = y_bar if y_bar is not None else (hw + 2*(tf + tc)) / 2
    ax.axhline(na, color="blue", ls="--", lw=1.0, label="NA")

    ax.set_xlim(-0.05*bf, 1.05*bf)
    ax.set_ylim(-0.15*total_h, 1.15*total_h)
    ax.set_xlabel("width [m]", fontsize=7)
    ax.set_ylabel("height [m]", fontsize=7)
    ax.tick_params(labelsize=7)

    ax.annotate("Face\n(cardboard)", xy=(0, total_h), xytext=(-0.3*bf, 1.05*total_h),
                fontsize=6, color=c_face, arrowprops=dict(arrowstyle="-", color=c_face, lw=0.6))
    ax.annotate("Core\n(corrugated)", xy=(bf*0.5, hw + tc + tf + tc/2),
                xytext=(bf*0.6, hw + tc + tf + tc/2 + 0.05*total_h),
                fontsize=6, color="#8a6500",
                arrowprops=dict(arrowstyle="-", color="#8a6500", lw=0.6))


def plot_results(res, params):
    x = res["x"]
    L = params["L"]

    fig = plt.figure(figsize=(15, 9), facecolor="#f7f7f5")
    fig.suptitle("Corrugated Cardboard Sandwich  ∩-Beam — Two-Span Continuous (pin-pin-pin)",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    ax_sec  = fig.add_subplot(gs[0, 0])
    ax_M    = fig.add_subplot(gs[0, 1])
    ax_V    = fig.add_subplot(gs[0, 2])
    ax_w    = fig.add_subplot(gs[0, 3])
    ax_sf   = fig.add_subplot(gs[1, 0:2])
    ax_ur   = fig.add_subplot(gs[1, 2:4])   # utilisation ratio bar chart
    ax_vm   = fig.add_subplot(gs[2, 0:3])
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis("off")

    COL = {"M": "#2176ae", "V": "#e07b39", "w": "#2c9e4b",
           "sf": "#7b2d8b", "vm": "#1a1a2e"}

    def _fill(ax, y, col, label, unit):
        ax.fill_between(x, y, alpha=0.25, color=col)
        ax.plot(x, y, color=col, lw=1.8, label=label)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)

    # ── Cross-section sketch ─────────────────────────────────────────────────
    draw_section(ax_sec, params["bf"], params["hw"], params["tf"], params["tc"],
                 y_bar=res["y_bar"])

    # ── Bending moment ───────────────────────────────────────────────────────
    _fill(ax_M, res["M"]/1e3, COL["M"], f"M  (max {res['M_max']/1e3:.2f} kN·m)", "M [kN·m]")
    ax_M.axvline(params["L"]/2, color="#888", ls=":", lw=1.2, label="centre pin")
    ax_M.legend(fontsize=7)
    ax_M.set_title("Bending Moment  (hogging at centre pin)", fontsize=9, fontweight="bold")

    # ── Shear force ──────────────────────────────────────────────────────────
    _fill(ax_V, res["V"]/1e3, COL["V"], f"V  (max {res['V_max']/1e3:.2f} kN)", "V [kN]")
    ax_V.axvline(params["L"]/2, color="#888", ls=":", lw=1.2, label="centre pin")
    ax_V.legend(fontsize=7)
    ax_V.set_title("Shear Force  (discontinuity at centre pin)", fontsize=9, fontweight="bold")

    # ── Deflection ───────────────────────────────────────────────────────────
    ax_w.fill_between(x, -res["w_total"]*1e3, alpha=0.2, color=COL["w"])
    ax_w.plot(x, -res["w_total"]*1e3, color=COL["w"], lw=1.8,
              label=f"w_total  (max {res['w_max']*1e3:.2f} mm)")
    ax_w.plot(x, -res["w_bend"]*1e3,  color=COL["w"], lw=0.9, ls="--", label="w_bending")
    ax_w.plot(x, -res["w_shear"]*1e3, color="#999",   lw=0.9, ls=":",  label="w_shear")
    ax_w.axhline(0, color="#aaa", lw=0.6)
    ax_w.set_xlabel("x [m]", fontsize=8)
    ax_w.set_ylabel("deflection [mm]", fontsize=8)
    ax_w.tick_params(labelsize=7)
    ax_w.legend(fontsize=6)
    ax_w.grid(True, lw=0.3, alpha=0.5)
    ax_w.set_title("Deflection", fontsize=9, fontweight="bold")

    # ── Face bending stress ──────────────────────────────────────────────────
    sf_top_MPa = res["sigma_f_top"] / 1e6
    sf_bot_MPa = res["sigma_f_bot"] / 1e6
    allow_f = params["sigma_f_allow"] / 1e6
    ax_sf.fill_between(x, sf_top_MPa, alpha=0.2, color=COL["sf"])
    ax_sf.plot(x, sf_top_MPa, color=COL["sf"], lw=1.8,
               label=f"sigma top (compr.)  max {np.max(np.abs(sf_top_MPa)):.2f} MPa")
    ax_sf.fill_between(x, sf_bot_MPa, alpha=0.15, color="#e07b39")
    ax_sf.plot(x, sf_bot_MPa, color="#e07b39", lw=1.5, ls="--",
               label=f"sigma bot (tension) max {np.max(sf_bot_MPa):.2f} MPa")
    ax_sf.axhline( allow_f, color="red", ls="--", lw=1.0, label=f"Allowable +/-{allow_f:.1f} MPa")
    ax_sf.axhline(-allow_f, color="red", ls="--", lw=1.0)
    wrinkle = res["sigma_wrinkle"] / 1e6
    ax_sf.axhline(-wrinkle, color="purple", ls=":", lw=1.0, label=f"Wrinkling {wrinkle:.1f} MPa")
    ax_sf.axhline(0, color="#aaa", lw=0.6)
    ax_sf.set_xlabel("x [m]", fontsize=8); ax_sf.set_ylabel("sigma [MPa]", fontsize=8)
    ax_sf.tick_params(labelsize=7); ax_sf.grid(True, lw=0.3, alpha=0.5)
    ax_sf.legend(fontsize=7)
    ax_sf.set_title("Face-Sheet Bending Stress  (true NA, Parallel Axis Theorem)",
                    fontsize=9, fontweight="bold")

    # ── Utilisation ratios bar chart ─────────────────────────────────────────
    UR_f  = res["UR_face"]
    UR_w  = res["UR_wrinkle"]

    ur_labels = ["Face stress", "Face wrinkling"]
    ur_values = [UR_f, UR_w]
    ur_colors = ["#c94040" if v > 1.0 else "#2c9e4b" for v in ur_values]

    bars = ax_ur.barh(ur_labels, ur_values, color=ur_colors, height=0.45, zorder=3)
    ax_ur.axvline(1.0, color="#333", lw=1.8, ls="--", zorder=4)
    ax_ur.set_xlim(0, max(max(ur_values) * 1.3, 1.4))
    ax_ur.set_xlabel("Utilisation Ratio  (limit = 1.0)", fontsize=8)
    ax_ur.tick_params(labelsize=8)
    ax_ur.grid(axis="x", lw=0.3, alpha=0.5, zorder=0)
    ax_ur.set_title("Utilisation Ratios", fontsize=9, fontweight="bold")
    for bar, val in zip(bars, ur_values):
        color = "red" if val > 1.0 else "darkgreen"
        ax_ur.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                   f"{val:.2f}", va="center", fontsize=9, fontweight="bold", color=color)

    # ── Von Mises ─────────────────────────────────────────────────────────────
    vm_MPa = res["sigma_vm"] / 1e6
    ax_vm.fill_between(x, vm_MPa, alpha=0.3, color=COL["vm"])
    ax_vm.plot(x, vm_MPa, color=COL["vm"], lw=2.2,
               label=f"sigma_VM  (max {np.max(vm_MPa):.2f} MPa)")
    ax_vm.axhline(allow_f, color="red", ls="--", lw=1.2,
                  label=f"Face strength {allow_f:.1f} MPa")
    ax_vm.fill_between(x, vm_MPa, allow_f,
                       where=(vm_MPa > allow_f),
                       color="red", alpha=0.3, label="Overstress zone")
    ax_vm.set_xlabel("x [m]", fontsize=8)
    ax_vm.set_ylabel("sigma_VM [MPa]", fontsize=8)
    ax_vm.tick_params(labelsize=7)
    ax_vm.legend(fontsize=8)
    ax_vm.grid(True, lw=0.3, alpha=0.5)
    ax_vm.set_title("Von Mises Stress along beam", fontsize=11, fontweight="bold")

    # ── Info panel ────────────────────────────────────────────────────────────
    col_f = "red" if UR_f > 1 else "green"
    col_w = "red" if UR_w > 1 else "green"

    info_lines = [
        ("--- Section ---", "black"),
        (f"  bf = {params['bf']*1e3:.0f} mm,  hw = {params['hw']*1e3:.0f} mm", "black"),
        (f"  tf = {params['tf']*1e3:.1f} mm,  tc = {params['tc']*1e3:.0f} mm", "black"),
        (f"  NA  = {res['y_bar']*1e3:.1f} mm from web bot", "black"),
        (f"  y_top = {res['y_top']*1e3:.1f} mm  y_bot = {res['y_bot']*1e3:.1f} mm", "black"),
        (f"  EI_eff = {res['EI_eff']:.3g} N·m²", "black"),
        (f"  GA_eff = {res['GA_eff']:.3g} N", "black"),
        ("--- Load ---", "black"),
        (f"  q = {params['q']:.1f} N/m,  L = {params['L']:.2f} m", "black"),
        ("--- Results ---", "black"),
        (f"  w_max = {res['w_max']*1e3:.2f} mm", "black"),
        (f"  L/w   = {L/res['w_max']:.0f}", "black"),
        ("--- Utilisation ---", "black"),
        (f"  UR_face    = {UR_f:.2f}", col_f),
        (f"  UR_wrinkle = {UR_w:.2f}", col_w),
    ]
    y_txt = 0.97
    for txt, col in info_lines:
        bold = "bold" if "---" in txt else "normal"
        ax_info.text(0.04, y_txt, txt, transform=ax_info.transAxes,
                     fontsize=8, color=col, fontweight=bold,
                     verticalalignment="top", fontfamily="monospace")
        y_txt -= 0.068

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sandwich ∩-Beam Analyser")
        self.resizable(False, False)
        self.configure(bg="#f0ede8")
        self._build_ui()

    def _lbl(self, parent, text, row, col, **kw):
        tk.Label(parent, text=text, bg="#f0ede8", **kw).grid(
            row=row, column=col, sticky="e", padx=4, pady=3)

    def _entry(self, parent, default, row, col, width=10):
        v = tk.StringVar(value=str(default))
        e = tk.Entry(parent, textvariable=v, width=width,
                     relief="solid", bd=1, bg="white")
        e.grid(row=row, column=col, sticky="w", padx=4, pady=3)
        return v

    def _build_ui(self):
        PAD = dict(padx=12, pady=6)

        hdr = tk.Frame(self, bg="#2c3e6b")
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Cardboard Honeycomb Sandwich ∩-Beam",
                 bg="#2c3e6b", fg="white", font=("Helvetica", 13, "bold"),
                 pady=8).pack(side="left")

        body = tk.Frame(self, bg="#f0ede8")
        body.pack(fill="both", padx=16, pady=10)

        mat_frame = tk.LabelFrame(body, text="Corrugated Board Selection",
                                  bg="#f0ede8", font=("Helvetica", 9, "bold"))
        mat_frame.grid(row=0, column=0, columnspan=3, sticky="ew", **PAD)

        tk.Label(mat_frame, text="Flute grade:", bg="#f0ede8",
                 font=("Helvetica", 8)).grid(row=0, column=0, sticky="e", padx=4, pady=3)
        self.flute_var = tk.StringVar(value=list(_FLUTES.keys())[2])
        ttk.Combobox(mat_frame, textvariable=self.flute_var,
                     values=list(_FLUTES.keys()), width=42,
                     state="readonly").grid(row=0, column=1, padx=4, pady=3)

        tk.Label(mat_frame, text="Liner paper:", bg="#f0ede8",
                 font=("Helvetica", 8)).grid(row=1, column=0, sticky="e", padx=4, pady=3)
        self.liner_var = tk.StringVar(value=list(_LINERS.keys())[1])
        ttk.Combobox(mat_frame, textvariable=self.liner_var,
                     values=list(_LINERS.keys()), width=42,
                     state="readonly").grid(row=1, column=1, padx=4, pady=3)

        tk.Label(mat_frame, text="Flute orientation:", bg="#f0ede8",
                 font=("Helvetica", 8)).grid(row=2, column=0, sticky="e", padx=4, pady=3)
        self.orient_var = tk.StringVar(value=list(_ORIENTATIONS.keys())[0])
        ttk.Combobox(mat_frame, textvariable=self.orient_var,
                     values=list(_ORIENTATIONS.keys()), width=42,
                     state="readonly").grid(row=2, column=1, padx=4, pady=3)

        tk.Button(mat_frame, text="Load props ->", command=self._on_mat_change,
                  bg="#2c3e6b", fg="white", relief="flat", padx=8
                  ).grid(row=0, column=2, rowspan=3, padx=12, pady=6)

        self.mat_info = tk.StringVar(value="")
        tk.Label(mat_frame, textvariable=self.mat_info, bg="#f0ede8",
                 fg="#555", font=("Courier", 7), justify="left"
                 ).grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=2)

        self.mat_var = tk.StringVar(value="")

        mp = tk.LabelFrame(body, text="Material Properties (editable)",
                           bg="#f0ede8", font=("Helvetica", 9, "bold"))
        mp.grid(row=1, column=0, sticky="nw", **PAD)

        self._lbl(mp, "E_face  [MPa]",   0, 0)
        self._lbl(mp, "G_core  [MPa]",   1, 0)
        self._lbl(mp, "E_core  [MPa]",   2, 0)
        self._lbl(mp, "s_f allow [MPa]", 3, 0)
        self._lbl(mp, "t_c allow [MPa]", 4, 0)

        _f0 = _FLUTES[list(_FLUTES.keys())[2]]
        _l0 = _LINERS[list(_LINERS.keys())[1]]
        _m0 = _corrugated_props(_f0, _l0, "MD")
        self.v_Ef   = self._entry(mp, f"{_m0['Ef_MPa']:.0f}",      0, 1)
        self.v_Gc   = self._entry(mp, f"{_m0['Gc_MPa']:.2f}",      1, 1)
        self.v_Ec   = self._entry(mp, f"{_m0['Ec_MPa']:.1f}",      2, 1)
        self.v_sigf = self._entry(mp, f"{_m0['sigma_f_MPa']:.1f}", 3, 1)
        self.v_tauc = self._entry(mp, f"{_m0['tau_c_MPa']:.3f}",   4, 1)

        gp = tk.LabelFrame(body, text="Section Geometry",
                           bg="#f0ede8", font=("Helvetica", 9, "bold"))
        gp.grid(row=1, column=1, sticky="nw", **PAD)

        self._lbl(gp, "Span L  [m]",           0, 0)
        self._lbl(gp, "Flange width bf [mm]",   1, 0)
        self._lbl(gp, "Web height hw  [mm]",    2, 0)
        self._lbl(gp, "Face thickness tf [mm]",  3, 0)
        self._lbl(gp, "Core thickness tc [mm]",  4, 0)

        self.v_L  = self._entry(gp, "1.20", 0, 1)
        self.v_bf = self._entry(gp, "200",  1, 1)
        self.v_hw = self._entry(gp, "150",  2, 1)
        self.v_tf = self._entry(gp, "1.5",  3, 1)
        self.v_tc = self._entry(gp, "3.6",  4, 1)

        sl_frame = tk.Frame(gp, bg="#f0ede8")
        sl_frame.grid(row=5, column=0, columnspan=2, pady=4)

        tk.Label(sl_frame, text="tf slider [mm]:", bg="#f0ede8", font=("Helvetica", 8)
                 ).grid(row=0, column=0, sticky="e")
        self.sl_tf = tk.Scale(sl_frame, from_=0.5, to=5.0, resolution=0.1,
                               orient="horizontal", length=130,
                               command=lambda v: self.v_tf.set(v),
                               bg="#f0ede8", highlightthickness=0)
        self.sl_tf.set(1.5)
        self.sl_tf.grid(row=0, column=1, padx=4)

        tk.Label(sl_frame, text="tc slider [mm]:", bg="#f0ede8", font=("Helvetica", 8)
                 ).grid(row=1, column=0, sticky="e")
        self.sl_tc = tk.Scale(sl_frame, from_=1.0, to=10.0, resolution=0.1,
                               orient="horizontal", length=130,
                               command=lambda v: self.v_tc.set(v),
                               bg="#f0ede8", highlightthickness=0)
        self.sl_tc.set(3.6)
        self.sl_tc.grid(row=1, column=1, padx=4)

        lp = tk.LabelFrame(body, text="Loading",
                           bg="#f0ede8", font=("Helvetica", 9, "bold"))
        lp.grid(row=1, column=2, sticky="nw", **PAD)

        self._lbl(lp, "UDL  q  [N/m]", 0, 0)
        self.v_q = self._entry(lp, "500", 0, 1)

        tk.Label(lp, text="vvvvvvvvvv\n∩  (inverted U)\n pin-pin-pin (2 spans)",
                 bg="#f7f2e8", relief="solid", bd=1,
                 font=("Courier", 10), fg="#2c3e6b", padx=6, pady=6
                 ).grid(row=1, column=0, columnspan=2, pady=8)

        btn_frame = tk.Frame(body, bg="#f0ede8")
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10)

        tk.Button(btn_frame, text="  Run Analysis & Plot  ",
                  command=self._run,
                  bg="#c0392b", fg="white",
                  font=("Helvetica", 11, "bold"),
                  relief="flat", padx=16, pady=8
                  ).pack()

        self.status = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status, bg="#2c3e6b", fg="#aadcff",
                 anchor="w", font=("Helvetica", 8), pady=3
                 ).pack(fill="x", side="bottom")

    def _on_mat_change(self, event=None):
        flute = _FLUTES[self.flute_var.get()]
        liner = _LINERS[self.liner_var.get()]
        orient = _ORIENTATIONS[self.orient_var.get()]
        m = _corrugated_props(flute, liner, orient)
        self.v_Ef.set(f"{m['Ef_MPa']:.0f}")
        self.v_Gc.set(f"{m['Gc_MPa']:.2f}")
        self.v_Ec.set(f"{m['Ec_MPa']:.1f}")
        self.v_sigf.set(f"{m['sigma_f_MPa']:.1f}")
        self.v_tauc.set(f"{m['tau_c_MPa']:.3f}")
        h = m["h_core_mm"]
        self.mat_info.set(
            f"  Computed from flute geometry:  "
            f"Gc={m['Gc_MPa']:.2f} MPa  |  "
            f"Ec={m['Ec_MPa']:.1f} MPa  |  "
            f"Core height={h:.1f} mm  |  "
            f"s_allow={m['sigma_f_MPa']:.1f} MPa  |  "
            f"t_allow={m['tau_c_MPa']:.3f} MPa"
        )

    def _run(self):
        try:
            params = {
                "L":  float(self.v_L.get()),
                "bf": float(self.v_bf.get()) / 1e3,
                "hw": float(self.v_hw.get()) / 1e3,
                "tf": float(self.v_tf.get()) / 1e3,
                "tc": float(self.v_tc.get()) / 1e3,
                "q":  float(self.v_q.get()),
                "Ef": float(self.v_Ef.get()) * 1e6,
                "Gc": float(self.v_Gc.get()) * 1e6,
                "Ec": float(self.v_Ec.get()) * 1e6,
                "sigma_f_allow": float(self.v_sigf.get()) * 1e6,
                "tau_c_allow":   float(self.v_tauc.get()) * 1e6,
            }
        except ValueError as e:
            self.status.set(f"Input error: {e}")
            return

        self.status.set("Computing...")
        self.update_idletasks()
        try:
            res = analyse(params)
            self.status.set(
                f"Done | w_max={res['w_max']*1e3:.2f} mm | "
                f"UR_face={res['UR_face']:.2f} | UR_wrinkle={res['UR_wrinkle']:.2f}"
            )
            plot_results(res, params)
        except Exception as e:
            self.status.set(f"Error: {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()