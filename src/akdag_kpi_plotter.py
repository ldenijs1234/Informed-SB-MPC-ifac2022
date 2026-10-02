#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "/root/catkin_ws/src/informed_sbmpc/src/sim_results"
D_SAFE_FULLSCALE = 300.0  # Full-scale spatial safety boundary (m)
R_MAX_DEG_S = 3.0         # Full-scale steering limit (~3.0 deg/s for standard vessels)

NOMINAL_WPS = {
    'case01': np.array([[0.0, -2000.0], [0.0, 2500.0]]),
    'case02': np.array([[0.0, -2000.0], [0.0, 2500.0]]),
    'case03': np.array([[0.0, -2000.0], [0.0, 2000.0]]),
    'case04': np.array([[-1800.0, 1400.0], [-700.0, 300.0], [100.0, -50.0], [1200.0, 200.0]]),
    'case05': np.array([[0.0, -2000.0], [0.0, 1000.0], [1500.0, 1000.0]]),
    'case06': np.array([[50.0, -2000.0], [50.0, 2500.0]])
}

class AkdagScenarioKPIEvaluator:
    def __init__(
        self,
        d_safe: float = D_SAFE_FULLSCALE,
        r_max_deg_s: float = R_MAX_DEG_S,
        tolerance_pct: float = 0.02,
        w_cte: float = 0.7,
        w_ctrl: float = 0.3
    ):
        self.d_safe = d_safe
        self.r_max = r_max_deg_s
        self.tolerance_pct = tolerance_pct
        self.d_safe_eff = d_safe * (1.0 - tolerance_pct)
        self.w_cte = w_cte
        self.w_ctrl = w_ctrl

    def calculate_cte(self, x: np.ndarray, y: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        errors = []
        for xi, yi in zip(x, y):
            p = np.array([xi, yi])
            dists = []
            for j in range(len(polyline) - 1):
                a, b = polyline[j], polyline[j + 1]
                ab = b - a
                ab_len_sq = np.dot(ab, ab)
                if ab_len_sq == 0.0:
                    dists.append(np.linalg.norm(p - a))
                else:
                    t_proj = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
                    proj = a + t_proj * ab
                    dists.append(np.linalg.norm(p - proj))
            errors.append(min(dists))
        return np.array(errors)

    def evaluate_single_run(self, t: np.ndarray, os_pos: np.ndarray, os_psi: np.ndarray, ts_pos: np.ndarray, nominal_wps: np.ndarray) -> dict:
        t_clean, unique_indices = np.unique(t, return_index=True)
        os_pos = os_pos[unique_indices]
        os_psi = os_psi[unique_indices]
        ts_pos = ts_pos[unique_indices]
        t = t_clean

        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        duration = max(t[-1] - t[0], 1.0)

        # 1. Proximity Risk (d_safe / r_min)
        ranges = np.linalg.norm(ts_pos - os_pos, axis=1)
        r_min = float(np.min(ranges))
        r_prox = (self.d_safe / r_min) if r_min > 0 else 10.0
        breached_prox = r_min < self.d_safe_eff

        # 2. Rudder Rate Risk (|r_max| / limit)
        dt = np.gradient(t)
        dt[dt == 0] = 1.0
        r_yaw_deg = np.rad2deg(np.gradient(np.unwrap(os_psi), t))
        max_turn_rate = float(np.max(np.abs(r_yaw_deg)))
        r_rudder = (max_turn_rate / self.r_max) if self.r_max > 0 else 0.0
        breached_rudder = max_turn_rate > (self.r_max * (1.0 + self.tolerance_pct))

        # Combined 50/50 Safety Risk Factor
        r_safety = 0.5 * r_prox + 0.5 * r_rudder
        breached = breached_prox or breached_rudder

        # Control Effort & Mission Tracking
        r_dot = np.gradient(np.deg2rad(r_yaw_deg), t)
        j_ctrl = float(trapz_fn(r_dot**2, t)) / duration if len(t) > 2 else 0.0

        e_cte = self.calculate_cte(os_pos[:, 0], os_pos[:, 1], nominal_wps)
        j_cte = float(trapz_fn(np.abs(e_cte), t)) / duration if len(t) > 1 else 0.0

        return {
            "r_min": r_min,
            "max_r": max_turn_rate,
            "r_prox": r_prox,
            "r_rudder": r_rudder,
            "r_safety": r_safety,
            "safety_factor": r_safety,
            "breached": breached,
            "j_ctrl": j_ctrl,
            "j_cte": j_cte
        }

    def evaluate_comparison(self, raw_ra: dict, raw_is: dict) -> dict:
        if raw_is["breached"]:
            return {"status": "IS_BREACH", "norm_ctrl": np.inf, "norm_cte": np.inf, "j_total_is": np.inf, "delta_j": np.inf, "delta_j_pct": -100.0}

        if raw_ra["breached"] and not raw_is["breached"]:
            return {"status": "SAFETY_ENHANCEMENT", "norm_ctrl": 0.0, "norm_cte": 0.0, "j_total_is": 0.0, "delta_j": -1.0, "delta_j_pct": 100.0}

        norm_ctrl = (raw_is["j_ctrl"] / raw_ra["j_ctrl"]) if raw_ra["j_ctrl"] > 0 else 1.0
        norm_cte = (raw_is["j_cte"] / raw_ra["j_cte"]) if raw_ra["j_cte"] > 0 else 1.0

        j_total_is = self.w_ctrl * norm_ctrl + self.w_cte * norm_cte
        delta_j = j_total_is - 1.0
        delta_j_pct = (1.0 - j_total_is) * 100.0

        return {
            "status": "SAFE_COMPARISON",
            "norm_ctrl": norm_ctrl,
            "norm_cte": norm_cte,
            "j_total_is": j_total_is,
            "delta_j": delta_j,
            "delta_j_pct": delta_j_pct
        }

def compute_all_benchmarks():
    evaluator = AkdagScenarioKPIEvaluator()
    cases = [f"case{i:02d}" for i in range(1, 7)]
    rows = []

    for case in cases:
        ra_path = os.path.join(RESULTS_DIR, f"{case}_RA_metrics.csv")
        is_path = os.path.join(RESULTS_DIR, f"{case}_IS_metrics.csv")

        if not (os.path.exists(ra_path) and os.path.exists(is_path)):
            continue

        df_ra = pd.read_csv(ra_path)
        df_is = pd.read_csv(is_path)

        df_ra = df_ra.loc[~((df_ra['ship_2_x'] == 0.0) & (df_ra['ship_2_y'] == 0.0))].copy()
        df_is = df_is.loc[~((df_is['ship_2_x'] == 0.0) & (df_is['ship_2_y'] == 0.0))].copy()

        nom_wps = NOMINAL_WPS.get(case, np.array([[0.0, -2000.0], [0.0, 2500.0]]))

        kpi_ra = evaluator.evaluate_single_run(
            df_ra['time'].to_numpy(),
            df_ra[['ship_1_x', 'ship_1_y']].to_numpy(),
            df_ra['ship_1_psi'].to_numpy(),
            df_ra[['ship_2_x', 'ship_2_y']].to_numpy(),
            nom_wps
        )

        kpi_is = evaluator.evaluate_single_run(
            df_is['time'].to_numpy(),
            df_is[['ship_1_x', 'ship_1_y']].to_numpy(),
            df_is['ship_1_psi'].to_numpy(),
            df_is[['ship_2_x', 'ship_2_y']].to_numpy(),
            nom_wps
        )

        comp = evaluator.evaluate_comparison(kpi_ra, kpi_is)

        rows.append({
            'Scenario': case.upper(),
            'SF_RA': kpi_ra['r_safety'],
            'SF_IS': kpi_is['r_safety'],
            'SF_Clr_RA': kpi_ra['r_prox'],
            'SF_Clr_IS': kpi_is['r_prox'],
            'SF_Str_RA': kpi_ra['r_rudder'],
            'SF_Str_IS': kpi_is['r_rudder'],
            'Delta_J_pct': comp['delta_j_pct'],
            'Status': comp['status']
        })

    if not rows:
        print("[ERROR] No matched simulation pairs found.")
        return

    summary_df = pd.DataFrame(rows)
    print("\n================== KPI EVALUATION SUMMARY ==================")
    print(summary_df.round(3).to_string(index=False))

    plot_safety_and_performance(summary_df)

def plot_safety_and_performance(df: pd.DataFrame):
    scenarios = df['Scenario'].tolist()
    x = np.arange(len(scenarios))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Risk Assessment (R_safety)
    ax1 = axes[0]
    ax1.set_ylim(-0.05, 1.15)

    # Risk Background Bands matching your original plot style
    ax1.axhspan(0.00, 0.35, color='#F0FFF0', alpha=0.6, label='_nolegend_') # Low
    ax1.axhspan(0.35, 0.70, color='#FFFFE0', alpha=0.6, label='_nolegend_') # Moderate
    ax1.axhspan(0.70, 1.00, color='#FFF0F5', alpha=0.6, label='_nolegend_') # High
    ax1.axhspan(1.00, 1.15, color='#B22222', alpha=0.25, hatch='//', label='_nolegend_') # Breach

    ax1.text(len(scenarios)-0.6, 0.17, "LOW RISK", color='#2E8B57', fontweight='bold', fontsize=8)
    ax1.text(len(scenarios)-0.6, 0.52, "MODERATE RISK", color='#B8860B', fontweight='bold', fontsize=8)
    ax1.text(len(scenarios)-0.6, 0.85, "HIGH RISK", color='#A52A2A', fontweight='bold', fontsize=8)
    ax1.text(len(scenarios)-0.6, 1.05, "BREACH", color='#800000', fontweight='bold', fontsize=8)

    # Bars
    ax1.bar(x - width/2, df['SF_RA'], width, label='Reactive Baseline (RA)', color='#4C72B0')
    ax1.bar(x + width/2, df['SF_IS'], width, label='Informed Routing (IS)', color='#55A868')
    ax1.axhline(1.0, color='red', linestyle=':', linewidth=1.2, label=r'Breach Threshold ($R_{\mathrm{safety}} \ge 1.0$)')

    ax1.set_ylabel(r"Safety Risk Factor $R_{\mathrm{safety}} = 0.5 \cdot R_{\mathrm{prox}} + 0.5 \cdot R_{\mathrm{rudder}}$")
    ax1.set_title("Safety Assessment: Proximity & Rudder Rate Risk", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8)

    # Subplot 2: Operational Efficiency Differential Delta J %
    ax2 = axes[1]
    colors = ['#55A868' if v >= 0 else '#C44E52' for v in df['Delta_J_pct']]
    bars = ax2.bar(x, df['Delta_J_pct'], width=0.5, color=colors)
    ax2.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
    ax2.set_ylabel(r"Improvement over Baseline $\Delta J_{\%} = (1 - J_{\mathrm{total}}) \times 100\%$")
    ax2.set_title("Operational Efficiency Differential (CTE + Dynamic Control)", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.grid(True, linestyle=':', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax2.annotate(f"{h:+.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
                     ha='center', va=va, fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_file = os.path.join(RESULTS_DIR, "safety_factor_and_efficiency_kpis.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"\n[SUCCESS] Generated dual KPI chart: {out_file}")

if __name__ == "__main__":
    compute_all_benchmarks()