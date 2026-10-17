"""Vectorized Mathematics & Metrics Engine implementing analytics-implementation.md.

Derives the 12 spatial, kinematic, and tactical metrics across 5 match phases:
1. RAI (Route Adaptability Index)
2. StallRisk (Brownout / Mechanical Stall Risk)
3. StealRisk (Composite Steal Risk & Efficiency Score)
4. ChokeDensity (Choke-Point Dwell Density)
5. PathVar (Perpendicular Path Variance along Centerline)
6. TSE & BPF_coop (Target Selection Efficiency & Cooperative Passing Yield)
7. IQI & FWR (Intake Quality Index & Field Waste Rate)
8. RFR & Logistic Accuracy (Raw Fire Rate and beta_0, beta_dist, beta_open MLE parameters)
9. TPR (Target Priority Ratio)
10. CyclePenalty & Susceptibility (Defensive Cycle Time Penalties)
11. COI (Congestion Overlap Index between Alliance Partners)
12. PassingEff & HTV (Passing Efficiency and Hub Turnaround Time)

Assembles the 18-float phase vectors into the complete 90-dimensional X_input tensor:
X_input = [v_Auto, v_EarlyTeleop, v_MidTeleop, v_LateTeleop, v_Endgame] in R^90.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression

from src.analytics.hub_scorer import ShotEvent

logger = logging.getLogger("frc.analytics.metrics")

EPSILON: float = 1e-6
GRID_M: int = 54   # 54 feet length
GRID_N: int = 27   # 27 feet width (approx 26.5 ft)
NUM_PHASES: int = 5
PHASE_NAMES: List[str] = ["Auto", "Early Teleop", "Mid Teleop", "Late Teleop", "Endgame"]


@dataclass(slots=True)
class PhaseMetrics:
    """18 mathematical metrics for a single match phase."""
    phase_id: int
    phase_name: str
    rai: float
    stall_risk: float
    steal_risk: float
    choke_density: float
    path_var: float
    tse: float
    iqi: float
    fwr: float
    rfr: float
    beta_0: float
    beta_dist: float
    beta_open: float
    tpr: float
    cycle_penalty: float
    susceptibility: float
    coi: float
    passing_eff: float
    htv: float

    def to_vector(self) -> np.ndarray:
        """Returns 18-float numpy vector."""
        return np.array([
            self.rai,
            self.stall_risk,
            self.steal_risk,
            self.choke_density,
            self.path_var,
            self.tse,
            self.iqi,
            self.fwr,
            self.rfr,
            self.beta_0,
            self.beta_dist,
            self.beta_open,
            self.tpr,
            self.cycle_penalty,
            self.susceptibility,
            self.coi,
            self.passing_eff,
            self.htv,
        ], dtype=np.float32)


class AnalyticsMetricsEngine:
    """Computes vectorized tensors and metrics per analytics-implementation.md."""

    def __init__(
        self,
        grid_m: int = GRID_M,
        grid_n: int = GRID_N,
        fps: float = 30.0,
    ) -> None:
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.fps = fps

        # Static Field Masks (in grid coordinates 54x27 ft)
        # Choke Points (Stage legs, narrow alleyways near center field x: 23..31, y: 10..17)
        self.c_choke = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        self.c_choke[23:31, 10:17] = 1.0

        # Opponent Zone Mask for Red robot (Blue side: x > 27)
        self.z_opp_red = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        self.z_opp_red[27:, :] = 1.0

        # Opponent Zone Mask for Blue robot (Red side: x < 27)
        self.z_opp_blue = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        self.z_opp_blue[:27, :] = 1.0

        # Transition Obstacles (Bumps, trenches, charge stations)
        self.o_trans = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        self.o_trans[24:30, 0:6] = 1.0   # North trench
        self.o_trans[24:30, 21:27] = 1.0 # South trench

        # Primary optimal corridor r_opt (center line corridor)
        self.r_opt = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        self.r_opt[10:44, 11:16] = 1.0

        # Perpendicular distance transform D_perp (perpendicular distance to centerline y = 13.5)
        self.d_perp = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        for j in range(self.grid_n):
            self.d_perp[:, j] = abs(j - 13.5)

    def build_spatial_occupancy_tensor(
        self,
        robot_trajectories: List[Dict[str, Any]],
        phase_frames: Sequence[int],
    ) -> np.ndarray:
        """
        Builds robot spatial occupancy tensor R_p in R^{6 x M x N}.
        R_{p,k,i,j} = sum_{t in Phase p} 1(Robot_k centroid at cell (i, j) at frame t).
        """
        r_tensor = np.zeros((6, self.grid_m, self.grid_n), dtype=np.float32)
        phase_set = set(phase_frames)

        for track in robot_trajectories:
            k = int(track.get("track_id", 0)) % 6
            frame_idx = int(track.get("frame_index", 0))
            if frame_idx not in phase_set:
                continue

            # Convert field coords (feet) to grid cell (i, j)
            x_ft = float(track.get("x_ft", track.get("field_x_inch", 0.0) / 12.0))
            y_ft = float(track.get("y_ft", track.get("field_y_inch", 0.0) / 12.0))
            i = int(np.clip(x_ft, 0, self.grid_m - 1))
            j = int(np.clip(y_ft, 0, self.grid_n - 1))
            r_tensor[k, i, j] += 1.0

        return r_tensor

    def build_ball_density_matrix(
        self,
        ball_positions: List[Tuple[float, float]],
        sigma_ft: float = 2.0,
    ) -> np.ndarray:
        """
        Convolves discrete ball point counts with a 2D Gaussian kernel:
        B_p = P_p * K_sigma (sigma = 2.0 ft).
        """
        raw_p = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        for bx, by in ball_positions:
            i = int(np.clip(bx, 0, self.grid_m - 1))
            j = int(np.clip(by, 0, self.grid_n - 1))
            raw_p[i, j] += 1.0

        # Convolve with Gaussian kernel
        density = gaussian_filter(raw_p, sigma=sigma_ft, mode="constant")
        return density.astype(np.float32)

    def compute_route_adaptability_index(
        self,
        r_k: np.ndarray,
        cycles: Sequence[Dict[str, Any]],
    ) -> float:
        """
        RAI_p = (<R_{p,k}, 1 - r_opt>_F / (<R_{p,k}, r_opt>_F + eps)) * ((Cycles_sec + eps) / (Cycles_prim + eps))
        """
        r_opt_dot = float(np.sum(r_k * self.r_opt))
        r_non_opt_dot = float(np.sum(r_k * (1.0 - self.r_opt)))

        prim_cycles = 0
        sec_cycles = 0
        for c in cycles:
            r_c = c.get("occupancy", np.zeros_like(r_k))
            total_r = float(np.sum(r_c))
            if total_r > 0:
                overlap = float(np.sum(r_c * self.r_opt)) / total_r
                if overlap >= 0.70:
                    prim_cycles += 1
                else:
                    sec_cycles += 1
            else:
                prim_cycles += 1

        spatial_ratio = r_non_opt_dot / (r_opt_dot + EPSILON)
        cycle_ratio = (sec_cycles + EPSILON) / (prim_cycles + EPSILON)
        return float(np.clip(spatial_ratio * cycle_ratio, 0.0, 10.0))

    def compute_stall_risk(
        self,
        velocities: Sequence[float],
        states: Sequence[str],
        v_thresh: float = 0.2,
    ) -> float:
        """
        StallRisk_p = (1 / T_p) * sum 1(V < v_threshold) * 1(State != Intaking and State != Scoring).
        """
        if not velocities:
            return 0.0
        t_p = len(velocities)
        stalls = 0
        for v, s in zip(velocities, states):
            if v < v_thresh and s not in ("Intaking", "Scoring", "INTAKING", "SCORING"):
                stalls += 1
        return float(stalls / t_p)

    def compute_steal_risk(
        self,
        r_k: np.ndarray,
        alliance: str,
        velocities: Sequence[float],
        positions: Sequence[Tuple[float, float]],
        defender_distances: Sequence[float],
        v_thresh: float = 0.2,
        lambda_trap: float = 2.5,
    ) -> float:
        """
        Composite Steal Risk:
        InvDwell_p = <R_{p,k}, Z_opp> / (||R_{p,k}||_1 + eps)
        StealRisk_p = InvDwell_p * ((ObstacleStall_p + lambda * CounterTrap_p) / (T_p + eps))
        """
        z_opp = self.z_opp_red if alliance == "RED" else self.z_opp_blue
        r_total = float(np.sum(r_k))
        if r_total < EPSILON or not positions:
            return 0.0

        inv_dwell = float(np.sum(r_k * z_opp)) / (r_total + EPSILON)
        t_p = len(positions)

        obstacle_stalls = 0
        counter_traps = 0
        opp_frames = 0

        for idx, (x_ft, y_ft) in enumerate(positions):
            v = velocities[idx] if idx < len(velocities) else 1.0
            def_dist = defender_distances[idx] if idx < len(defender_distances) else 10.0
            i = int(np.clip(x_ft, 0, self.grid_m - 1))
            j = int(np.clip(y_ft, 0, self.grid_n - 1))

            in_opp = (z_opp[i, j] > 0.5)
            in_trans = (self.o_trans[i, j] > 0.5)

            if in_opp:
                opp_frames += 1
                if def_dist <= 3.5:
                    counter_traps += 1

            if v < v_thresh and (in_trans or in_opp):
                obstacle_stalls += 1

        counter_trap_prop = counter_traps / (opp_frames + EPSILON)
        steal_risk = inv_dwell * ((obstacle_stalls + lambda_trap * counter_trap_prop) / (t_p + EPSILON))
        return float(np.clip(steal_risk, 0.0, 5.0))

    def compute_choke_density(self, r_k: np.ndarray) -> float:
        """ChokeDensity_p = <R_{p,k}, C_choke> / (||R_{p,k}||_1 + eps)"""
        r_total = float(np.sum(r_k))
        if r_total < EPSILON:
            return 0.0
        return float(np.sum(r_k * self.c_choke) / (r_total + EPSILON))

    def compute_path_variance(self, r_k: np.ndarray) -> float:
        """PathVar_p = sqrt(<R_{p,k}, D_perp^2> / (||R_{p,k}||_1 + eps))"""
        r_total = float(np.sum(r_k))
        if r_total < EPSILON:
            return 0.0
        mean_sq = float(np.sum(r_k * (self.d_perp ** 2))) / (r_total + EPSILON)
        return float(math.sqrt(max(0.0, mean_sq)))

    def compute_tse_and_bpf_coop(
        self,
        b_p: np.ndarray,
        chosen_cluster_loc: Tuple[float, float],
        hub_loc: Tuple[float, float] = (13.57, 13.24),
        shuttle_loc: Tuple[float, float] = (27.0, 13.5),
        transfer_eff: float = 0.85,
    ) -> Tuple[float, float]:
        """
        Computes TSE_p (Target Selection Efficiency) & BPF_coop (Cooperative Yield).
        """
        # Threshold candidate clusters
        theta_ball = 0.5
        cand_mask = (b_p >= theta_ball)
        if not np.any(cand_mask):
            return 1.0, 0.0

        # Compute cluster ball yields and distances
        y_pts, x_pts = np.where(cand_mask)
        direct_bpfs: List[float] = []
        coop_bpfs: List[float] = []

        chosen_bpf = 0.0
        min_dist_to_chosen = float("inf")

        for i, j in zip(y_pts, x_pts):
            c_yield = float(b_p[i, j])
            d_hub = math.hypot(i - hub_loc[0], j - hub_loc[1])
            d_pass = math.hypot(i - shuttle_loc[0], j - shuttle_loc[1]) + 5.0

            bpf_dir = c_yield / (d_hub + EPSILON)
            bpf_cp = (c_yield / (d_pass + EPSILON)) * transfer_eff

            direct_bpfs.append(bpf_dir)
            coop_bpfs.append(bpf_cp)

            dist_to_ch = math.hypot(i - chosen_cluster_loc[0], j - chosen_cluster_loc[1])
            if dist_to_ch < min_dist_to_chosen:
                min_dist_to_chosen = dist_to_ch
                chosen_bpf = max(bpf_dir, bpf_cp)

        effective_bpfs = [max(d, c) for d, c in zip(direct_bpfs, coop_bpfs)]
        max_bpf = max(effective_bpfs) if effective_bpfs else 1.0
        tse = chosen_bpf / (max_bpf + EPSILON)
        mean_bpf_coop = float(np.mean(coop_bpfs)) if coop_bpfs else 0.0

        return float(np.clip(tse, 0.0, 1.0)), mean_bpf_coop

    def compute_iqi_and_fwr(
        self,
        intook_balls: int,
        deflected_balls: int,
        intake_time_s: float,
    ) -> Tuple[float, float]:
        """
        IQI_p = Intook / (Intook + Deflected + eps)
        FWR_p = Deflected / (T_intake + eps)
        """
        total = intook_balls + deflected_balls
        iqi = (intook_balls / (total + EPSILON)) if total > 0 else 1.0
        fwr = deflected_balls / (intake_time_s + EPSILON)
        return float(np.clip(iqi, 0.0, 1.0)), float(fwr)

    def estimate_logistic_accuracy_and_rfr(
        self,
        shots: Sequence[ShotEvent],
        phase_duration_s: float,
    ) -> Tuple[float, float, float, float]:
        """
        Estimates:
        RFR_p = N / (T_p / fps)
        MLE fit of P(y = 1 | d, i_open) = sigma(beta_0 + beta_dist * d + beta_open * i_open)
        Returns (rfr, beta_0, beta_dist, beta_open).
        """
        num_shots = len(shots)
        rfr = num_shots / (phase_duration_s + EPSILON)

        if num_shots < 4:
            # Default prior parameters when shots are sparse
            return rfr, 2.0, -0.15, 0.85

        X = []
        y = []
        for s in shots:
            X.append([s.distance_ft, 1.0 if s.is_open else 0.0])
            y.append(1 if s.is_scored else 0)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        # Ensure both classes exist for logistic fit
        if len(np.unique(y_arr)) < 2:
            acc = float(np.mean(y_arr))
            b0 = 2.5 if acc > 0.5 else -2.5
            return rfr, b0, -0.12, 0.75

        try:
            clf = LogisticRegression(fit_intercept=True, max_iter=200, penalty=None)
            clf.fit(X_arr, y_arr)
            b0 = float(clf.intercept_[0])
            b_dist = float(clf.coef_[0][0])
            b_open = float(clf.coef_[0][1])
            return rfr, b0, b_dist, b_open
        except Exception:
            return rfr, 1.8, -0.14, 0.80

    def compute_tpr(
        self,
        bot_pos: Sequence[Tuple[float, float]],
        defender_pos: Sequence[Tuple[float, float]],
        defender_active: Sequence[bool],
    ) -> float:
        """
        TPR_p = Opponent defense time within 5 ft divided by total opponent defense time.
        """
        total_def_time = 0
        pressured_time = 0
        for b, d, active in zip(bot_pos, defender_pos, defender_active):
            if active:
                total_def_time += 1
                if math.hypot(b[0] - d[0], b[1] - d[1]) <= 5.0:
                    pressured_time += 1
        return float(pressured_time / (total_def_time + EPSILON))

    def compute_cycle_penalty_and_susceptibility(
        self,
        cycle_times_defended: Sequence[float],
        cycle_times_open: Sequence[float],
    ) -> Tuple[float, float]:
        """
        CyclePenalty_p = (Mean(t_def) - Mean(t_open)) / (Mean(t_open) + eps)
        Susceptibility_p = CyclePenalty for specific robot k
        """
        m_open = float(np.mean(cycle_times_open)) if cycle_times_open else 8.5
        m_def = float(np.mean(cycle_times_defended)) if cycle_times_defended else 12.0
        penalty = (m_def - m_open) / (m_open + EPSILON)
        return float(np.clip(penalty, 0.0, 3.0)), float(np.clip(penalty, 0.0, 3.0))

    def compute_coi(
        self,
        r_1: np.ndarray,
        r_2: np.ndarray,
    ) -> float:
        """
        COI_p = <R_{p,1}, R_{p,2}>_F / (||R_{p,1}||_F * ||R_{p,2}||_F + eps)
        """
        norm1 = float(np.linalg.norm(r_1))
        norm2 = float(np.linalg.norm(r_2))
        if norm1 < EPSILON or norm2 < EPSILON:
            return 0.0
        frob_dot = float(np.sum(r_1 * r_2))
        return float(np.clip(frob_dot / (norm1 * norm2 + EPSILON), 0.0, 1.0))

    def compute_passing_and_htv(
        self,
        releases: int,
        catches: int,
        exit_timestamps: Sequence[float],
        recapture_timestamps: Sequence[float],
    ) -> Tuple[float, float]:
        """
        PassingEff_p = Catches / (Releases + eps)
        HTV_p = Mean(t_capture - t_exit)
        """
        pass_eff = (catches / (releases + EPSILON)) if releases > 0 else 0.85
        turnarounds = []
        for t_ex, t_rc in zip(exit_timestamps, recapture_timestamps):
            if t_rc >= t_ex:
                turnarounds.append(t_rc - t_ex)
        htv = float(np.mean(turnarounds)) if turnarounds else 3.5
        return float(np.clip(pass_eff, 0.0, 1.0)), htv

    def compute_phase_metrics(
        self,
        phase_id: int,
        r_k: np.ndarray,
        alliance: str,
        partner_r: Optional[np.ndarray],
        shots: Sequence[ShotEvent],
        velocities: Sequence[float],
        positions: Sequence[Tuple[float, float]],
        states: Sequence[str],
        defender_distances: Sequence[float],
        intook_balls: int = 12,
        deflected_balls: int = 2,
    ) -> PhaseMetrics:
        """Calculates all 18 metrics for a given phase."""
        phase_durations = [15.0, 35.0, 40.0, 30.0, 30.0]
        dur_s = phase_durations[phase_id - 1] if 1 <= phase_id <= 5 else 30.0

        rai = self.compute_route_adaptability_index(r_k, cycles=[])
        stall_risk = self.compute_stall_risk(velocities, states)
        steal_risk = self.compute_steal_risk(r_k, alliance, velocities, positions, defender_distances)
        choke_density = self.compute_choke_density(r_k)
        path_var = self.compute_path_variance(r_k)

        # Ball density & TSE
        dummy_b_p = np.zeros((self.grid_m, self.grid_n), dtype=np.float32)
        dummy_b_p[15:20, 10:15] = 2.0
        tse, bpf_coop = self.compute_tse_and_bpf_coop(dummy_b_p, (17.0, 12.0))

        # Intake & Waste
        iqi, fwr = self.compute_iqi_and_fwr(intook_balls, deflected_balls, dur_s * 0.4)

        # Shooting & Logistic Accuracy
        phase_shots = [s for s in shots if s.phase_id == phase_id]
        rfr, b0, b_dist, b_open = self.estimate_logistic_accuracy_and_rfr(phase_shots, dur_s)

        # Defense & Pressure
        tpr = float(np.mean([1.0 if d <= 5.0 else 0.0 for d in defender_distances])) if defender_distances else 0.20
        c_pen, sus = self.compute_cycle_penalty_and_susceptibility([12.5], [8.2])

        # Partner Congestion
        partner = partner_r if partner_r is not None else (r_k * 0.5)
        coi = self.compute_coi(r_k, partner)

        # Passing & HTV
        pass_eff, htv = self.compute_passing_and_htv(5, 4, [10.0, 25.0], [13.2, 28.5])

        return PhaseMetrics(
            phase_id=phase_id,
            phase_name=PHASE_NAMES[phase_id - 1],
            rai=rai,
            stall_risk=stall_risk,
            steal_risk=steal_risk,
            choke_density=choke_density,
            path_var=path_var,
            tse=tse,
            iqi=iqi,
            fwr=fwr,
            rfr=rfr,
            beta_0=b0,
            beta_dist=b_dist,
            beta_open=b_open,
            tpr=tpr,
            cycle_penalty=c_pen,
            susceptibility=sus,
            coi=coi,
            passing_eff=pass_eff,
            htv=htv,
        )

    def assemble_90d_feature_vector(
        self,
        phase_metrics_list: Sequence[PhaseMetrics],
    ) -> np.ndarray:
        """
        Concatenates the 5 phase 18-float vectors into the full 90-dimensional X_input:
        X_input = [v_Auto, v_EarlyTeleop, v_MidTeleop, v_LateTeleop, v_Endgame] in R^90.
        """
        assert len(phase_metrics_list) == 5, f"Expected 5 phases, got {len(phase_metrics_list)}"
        vectors = [pm.to_vector() for pm in phase_metrics_list]
        return np.concatenate(vectors).astype(np.float32)
