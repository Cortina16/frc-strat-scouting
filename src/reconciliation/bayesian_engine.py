"""Bayesian Ground-Truth Reconciliation Engine and Actor.

Module 07: Bayesian Ground-Truth Reconciliation
Reconciles ActionFormer candidate action events with official FRC Field Management
System (FMS) ground-truth scoring telemetry and CAD spatial geometry constraints.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.common.interfaces import ReconciliationActorBase, ReconciliationActorProtocol
from src.common.types import (
    ActionSegment,
    ActionType,
    FieldAPIScoreState,
    FieldElement,
    MatchStage,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Field Topology & CAD Coordinates (inches from bottom-left corner of carpet)
# ============================================================================

FIELD_LENGTH_INCHES: float = 651.2
FIELD_WIDTH_INCHES: float = 317.7

# Hub and Tower centers (matches configs/field_2026_rebuilt.yaml)
RED_HUB_POS: Tuple[float, float] = (162.8, 158.85)
BLUE_HUB_POS: Tuple[float, float] = (488.4, 158.85)
RED_TOWER_POS: Tuple[float, float] = (90.0, 158.85)
BLUE_TOWER_POS: Tuple[float, float] = (561.2, 158.85)

# Depots (North and South at neutral line x=325.6)
DEPOT_NORTH_POS: Tuple[float, float] = (325.6, 280.0)
DEPOT_SOUTH_POS: Tuple[float, float] = (325.6, 37.7)

# Outposts and Corrals
RED_OUTPOST_POS: Tuple[float, float] = (0.0, 158.85)
BLUE_OUTPOST_POS: Tuple[float, float] = (651.2, 158.85)
RED_CORRAL_POS: Tuple[float, float] = (14.0, 158.85)
BLUE_CORRAL_POS: Tuple[float, float] = (637.2, 158.85)

# Bumps and Trenches
BUMPS_POS: List[Tuple[float, float]] = [
    (280.0, 80.0),
    (280.0, 237.7),
    (371.2, 80.0),
    (371.2, 237.7),
]
TRENCHES_POS: List[Tuple[float, float]] = [
    (270.0, 15.0),
    (270.0, 262.7),
]

NEUTRAL_ZONE_X_RANGE: Tuple[float, float] = (250.0, 400.0)

# Nominal interaction radii (inches)
HUB_SCORING_MAX_RADIUS: float = 60.0
TOWER_CLIMB_MAX_RADIUS: float = 45.0
DEPOT_INTAKE_MAX_RADIUS: float = 50.0
OUTPOST_MAX_RADIUS: float = 50.0

# Spatial Gating Cutoffs
SPATIAL_GATE_CUTOFF_DISTANCE: float = 150.0
DEFAULT_SPATIAL_THRESHOLD: float = 0.1


# ============================================================================
# Reconciliation Status Enum and Constants
# ============================================================================

class ReconciliationStatus(str, Enum):
    """Reconciliation state matching protobuf telemetry definitions."""
    RECON_STATUS_UNSPECIFIED = "RECON_STATUS_UNSPECIFIED"
    RECON_STATUS_PENDING = "RECON_STATUS_PENDING"
    RECON_STATUS_CONFIRMED = "RECON_STATUS_CONFIRMED"
    RECON_STATUS_PRUNED_FALSE_POSITIVE = "RECON_STATUS_PRUNED_FALSE_POSITIVE"
    RECON_STATUS_ADJUSTED = "RECON_STATUS_ADJUSTED"

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            proto_map = {
                0: "RECON_STATUS_UNSPECIFIED",
                1: "RECON_STATUS_PENDING",
                2: "RECON_STATUS_CONFIRMED",
                3: "RECON_STATUS_PRUNED_FALSE_POSITIVE",
                4: "RECON_STATUS_ADJUSTED",
            }
            return proto_map.get(other) == self.value
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


RECON_STATUS_UNSPECIFIED = ReconciliationStatus.RECON_STATUS_UNSPECIFIED
RECON_STATUS_PENDING = ReconciliationStatus.RECON_STATUS_PENDING
RECON_STATUS_CONFIRMED = ReconciliationStatus.RECON_STATUS_CONFIRMED
RECON_STATUS_PRUNED_FALSE_POSITIVE = ReconciliationStatus.RECON_STATUS_PRUNED_FALSE_POSITIVE
RECON_STATUS_ADJUSTED = ReconciliationStatus.RECON_STATUS_ADJUSTED


# ============================================================================
# Reconciled Action Segment Data Model
# ============================================================================

@dataclass
class ReconciledActionSegment(ActionSegment):
    """ActionSegment augmented with Bayesian reconciliation status and metadata."""
    status: Union[str, ReconciliationStatus] = RECON_STATUS_CONFIRMED
    reconciliation_status: Union[str, ReconciliationStatus] = RECON_STATUS_CONFIRMED
    spatial_likelihood: float = 1.0
    reconciliation_notes: str = ""

    @classmethod
    def from_action_segment(
        cls,
        action: ActionSegment,
        status: Union[str, ReconciliationStatus] = RECON_STATUS_CONFIRMED,
        spatial_likelihood: float = 1.0,
        reconciliation_notes: str = "",
        is_reconciled: bool = True,
        reconciled_posterior: float = 0.0,
    ) -> "ReconciledActionSegment":
        """Factory creating a ReconciledActionSegment from a base ActionSegment."""
        return cls(
            event_id=action.event_id,
            track_id=action.track_id,
            team_number=action.team_number,
            action_type=action.action_type,
            element=action.element,
            start_time_ms=action.start_time_ms,
            end_time_ms=action.end_time_ms,
            start_frame=action.start_frame,
            end_frame=action.end_frame,
            confidence=action.confidence,
            diou=action.diou,
            pyramid_level=action.pyramid_level,
            field_coords=action.field_coords,
            is_reconciled=is_reconciled,
            reconciled_posterior=reconciled_posterior,
            status=status,
            reconciliation_status=status,
            spatial_likelihood=spatial_likelihood,
            reconciliation_notes=reconciliation_notes,
        )


# ============================================================================
# Spatial Likelihood Computation Functions
# ============================================================================

def _euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_hub_spatial_likelihood(
    distance_inches: float,
    core_radius: float = 45.0,
    sigma: float = 35.0,
    cutoff_radius: float = SPATIAL_GATE_CUTOFF_DISTANCE,
) -> float:
    """Computes spatial likelihood P(Spatial | Robot Position, Hub).
    
    Guarantees:
    - distance <= 60.0 in: likelihood > 0.8
    - distance >= 250.0 in: likelihood is 0.0 (near zero / pruned)
    """
    d = max(0.0, float(distance_inches))
    if d > cutoff_radius:
        return 0.0
    if d <= core_radius:
        return max(0.0, 1.0 - 0.05 * (d / max(1.0, core_radius)))
    # Gaussian dropoff beyond core radius
    delta = d - core_radius
    decay = math.exp(-(delta ** 2) / (2.0 * (sigma ** 2)))
    return float(0.95 * decay)


def compute_tower_spatial_likelihood(
    distance_inches: float,
    core_radius: float = 30.0,
    sigma: float = 30.0,
    cutoff_radius: float = 120.0,
) -> float:
    """Computes spatial likelihood P(Spatial | Robot Position, Tower).
    
    Guarantees:
    - distance <= 45.0 in: likelihood > 0.8
    - distance >= 250.0 in: likelihood is 0.0
    """
    d = max(0.0, float(distance_inches))
    if d > cutoff_radius:
        return 0.0
    if d <= core_radius:
        return max(0.0, 1.0 - 0.05 * (d / max(1.0, core_radius)))
    delta = d - core_radius
    decay = math.exp(-(delta ** 2) / (2.0 * (sigma ** 2)))
    return float(0.95 * decay)


def compute_depot_spatial_likelihood(
    pos: Tuple[float, float],
    core_radius: float = 35.0,
    sigma: float = 30.0,
    cutoff_radius: float = 130.0,
) -> float:
    """Computes spatial likelihood for Depot intake (North or South Depot)."""
    d_north = _euclidean_dist(pos, DEPOT_NORTH_POS)
    d_south = _euclidean_dist(pos, DEPOT_SOUTH_POS)
    d = min(d_north, d_south)
    if d > cutoff_radius:
        return 0.0
    if d <= core_radius:
        return max(0.0, 1.0 - 0.05 * (d / max(1.0, core_radius)))
    delta = d - core_radius
    decay = math.exp(-(delta ** 2) / (2.0 * (sigma ** 2)))
    return float(0.95 * decay)


def compute_spatial_likelihood(
    target: Union[ActionSegment, FieldElement, ActionType, str],
    robot_pos: Optional[Tuple[float, float]] = None,
    action_type_or_element: Optional[Union[ActionType, FieldElement, str]] = None,
) -> float:
    """Calculates spatial likelihood P(Spatial | Action, Position) for any element or action.
    
    Supports multiple calling conventions:
    1. compute_spatial_likelihood(action: ActionSegment, robot_pos: Optional[Tuple[float, float]] = None)
    2. compute_spatial_likelihood(element: FieldElement, robot_pos: Tuple[float, float], action_type: Optional[ActionType] = None)
    3. compute_spatial_likelihood(action_type: ActionType, element: FieldElement, robot_pos: Tuple[float, float])
    """
    element: Optional[FieldElement] = None
    action_type: Optional[ActionType] = None
    pos: Optional[Tuple[float, float]] = None

    if isinstance(target, ActionSegment):
        element = target.element
        action_type = target.action_type
        pos = robot_pos if robot_pos is not None else target.field_coords
    elif isinstance(target, ActionType):
        action_type = target
        if isinstance(robot_pos, (FieldElement, str)):
            element = FieldElement(robot_pos)
            pos = action_type_or_element if isinstance(action_type_or_element, tuple) else (0.0, 0.0)
        else:
            element = FieldElement(action_type_or_element) if action_type_or_element else None
            pos = robot_pos if isinstance(robot_pos, tuple) else (0.0, 0.0)
    elif isinstance(target, (FieldElement, str)):
        element = FieldElement(target)
        if isinstance(robot_pos, tuple):
            pos = robot_pos
        if isinstance(action_type_or_element, ActionType):
            action_type = action_type_or_element

    if pos is None:
        pos = (0.0, 0.0)

    # 1. Hub scoring
    if element in (FieldElement.RED_HUB, FieldElement.BLUE_HUB) or action_type == ActionType.SCORE_HUB:
        hub_pos = RED_HUB_POS if element == FieldElement.RED_HUB else (
            BLUE_HUB_POS if element == FieldElement.BLUE_HUB else (
                RED_HUB_POS if _euclidean_dist(pos, RED_HUB_POS) <= _euclidean_dist(pos, BLUE_HUB_POS) else BLUE_HUB_POS
            )
        )
        d = _euclidean_dist(pos, hub_pos)
        return compute_hub_spatial_likelihood(d)

    # 2. Tower climbing
    if element in (FieldElement.RED_TOWER, FieldElement.BLUE_TOWER) or action_type == ActionType.CLIMB_TOWER:
        tower_pos = RED_TOWER_POS if element == FieldElement.RED_TOWER else (
            BLUE_TOWER_POS if element == FieldElement.BLUE_TOWER else (
                RED_TOWER_POS if _euclidean_dist(pos, RED_TOWER_POS) <= _euclidean_dist(pos, BLUE_TOWER_POS) else BLUE_TOWER_POS
            )
        )
        d = _euclidean_dist(pos, tower_pos)
        return compute_tower_spatial_likelihood(d)

    # 3. Depot intake
    if element == FieldElement.DEPOT or action_type == ActionType.INTAKE_FUEL:
        if element == FieldElement.DEPOT:
            return compute_depot_spatial_likelihood(pos)
        # If INTAKE_FUEL near Red/Blue outpost
        if element in (FieldElement.RED_OUTPOST, FieldElement.BLUE_OUTPOST):
            outpost_pos = RED_OUTPOST_POS if element == FieldElement.RED_OUTPOST else BLUE_OUTPOST_POS
            d = _euclidean_dist(pos, outpost_pos)
            return compute_tower_spatial_likelihood(d, core_radius=35.0, sigma=30.0, cutoff_radius=130.0)
        # General intake can happen near depot or floor
        d_depot = min(_euclidean_dist(pos, DEPOT_NORTH_POS), _euclidean_dist(pos, DEPOT_SOUTH_POS))
        if d_depot <= 60.0:
            return compute_depot_spatial_likelihood(pos)
        return 0.85

    # 4. Outpost / Corral cycle
    if element in (FieldElement.RED_OUTPOST, FieldElement.BLUE_OUTPOST, FieldElement.RED_CORRAL, FieldElement.BLUE_CORRAL) or action_type == ActionType.OUTPOST_CORRAL_CYCLE:
        target_pos = RED_OUTPOST_POS if element in (FieldElement.RED_OUTPOST, FieldElement.RED_CORRAL) else BLUE_OUTPOST_POS
        d = _euclidean_dist(pos, target_pos)
        return compute_tower_spatial_likelihood(d, core_radius=35.0, sigma=30.0, cutoff_radius=130.0)

    # 5. Neutral zone transit
    if action_type == ActionType.TRANSIT_NEUTRAL_ZONE:
        x_min, x_max = NEUTRAL_ZONE_X_RANGE
        if x_min <= pos[0] <= x_max:
            return 1.0
        d_x = min(abs(pos[0] - x_min), abs(pos[0] - x_max))
        if d_x > 120.0:
            return 0.0
        return float(0.95 * math.exp(-(d_x ** 2) / (2.0 * (40.0 ** 2))))

    # 6. Bumps / Trenches
    if element == FieldElement.BUMP:
        min_d = min(_euclidean_dist(pos, bp) for bp in BUMPS_POS)
        if min_d > 100.0:
            return 0.0
        return float(math.exp(-(max(0.0, min_d - 25.0) ** 2) / (2.0 * (30.0 ** 2))))

    if element == FieldElement.TRENCH:
        min_d = min(_euclidean_dist(pos, tp) for tp in TRENCHES_POS)
        if min_d > 120.0:
            return 0.0
        return float(math.exp(-(max(0.0, min_d - 30.0) ** 2) / (2.0 * (35.0 ** 2))))

    # Defensive engagement / Idle / Unspecified default
    return 1.0


# ============================================================================
# Bayesian Reconciliation Engine
# ============================================================================

class BayesianReconciliationEngine:
    """Bayesian posterior reconciliation engine fusing CV candidates with FMS ground truth."""

    def __init__(
        self,
        spatial_threshold: float = DEFAULT_SPATIAL_THRESHOLD,
        scoring_radius_hub: float = HUB_SCORING_MAX_RADIUS,
        climb_radius_tower: float = TOWER_CLIMB_MAX_RADIUS,
    ) -> None:
        self.spatial_threshold = spatial_threshold
        self.scoring_radius_hub = scoring_radius_hub
        self.climb_radius_tower = climb_radius_tower

        self._confirmed_counts: Dict[str, int] = {
            "red_fuel_scored": 0,
            "blue_fuel_scored": 0,
            "red_tower_climbs": 0,
            "blue_tower_climbs": 0,
        }
        self._event_statuses: Dict[str, str] = {}
        self._event_posteriors: Dict[str, float] = {}

    def get_status(self, event_id: str) -> Optional[str]:
        """Returns the reconciliation status string for an event_id."""
        return self._event_statuses.get(event_id)

    def get_posterior(self, event_id: str) -> Optional[float]:
        """Returns the reconciled posterior probability for an event_id."""
        return self._event_posteriors.get(event_id)

    def get_confirmed_counts(self) -> Dict[str, int]:
        """Returns current cumulative confirmed scores."""
        return dict(self._confirmed_counts)

    def reset(self) -> None:
        """Resets engine tracking state."""
        self._confirmed_counts = {
            "red_fuel_scored": 0,
            "blue_fuel_scored": 0,
            "red_tower_climbs": 0,
            "blue_tower_climbs": 0,
        }
        self._event_statuses.clear()
        self._event_posteriors.clear()

    def evaluate_spatial_gate(
        self,
        action: ActionSegment,
        robot_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, bool]:
        """Evaluates whether an action segment passes spatial geometry gating.
        
        Returns (spatial_likelihood, passed_gate).
        """
        pos = robot_pos if robot_pos is not None else action.field_coords
        p_spatial = compute_spatial_likelihood(action, pos)
        passed = p_spatial >= self.spatial_threshold
        return p_spatial, passed

    def reconcile(
        self,
        candidate_actions: Sequence[ActionSegment],
        robot_positions: Dict[int, Tuple[float, float]],
        score_state: Optional[FieldAPIScoreState] = None,
        target_deltas: Optional[Dict[str, int]] = None,
        filter_pruned: bool = False,
    ) -> List[ReconciledActionSegment]:
        """Reconciles candidate actions against spatial geometry and FMS score state."""
        if not candidate_actions:
            return []

        # Categorize candidates into score channels:
        # Channel keys: 'red_fuel', 'blue_fuel', 'red_climb', 'blue_climb', 'unconstrained'
        channel_candidates: Dict[str, List[ActionSegment]] = {
            "red_fuel": [],
            "blue_fuel": [],
            "red_climb": [],
            "blue_climb": [],
            "unconstrained": [],
        }

        for act in candidate_actions:
            if act.action_type == ActionType.SCORE_HUB:
                if act.element == FieldElement.RED_HUB:
                    channel_candidates["red_fuel"].append(act)
                elif act.element == FieldElement.BLUE_HUB:
                    channel_candidates["blue_fuel"].append(act)
                else:
                    # Resolve closest hub
                    pos = robot_positions.get(act.track_id, act.field_coords)
                    if _euclidean_dist(pos, RED_HUB_POS) <= _euclidean_dist(pos, BLUE_HUB_POS):
                        channel_candidates["red_fuel"].append(act)
                    else:
                        channel_candidates["blue_fuel"].append(act)
            elif act.action_type == ActionType.CLIMB_TOWER:
                if act.element == FieldElement.RED_TOWER:
                    channel_candidates["red_climb"].append(act)
                elif act.element == FieldElement.BLUE_TOWER:
                    channel_candidates["blue_climb"].append(act)
                else:
                    pos = robot_positions.get(act.track_id, act.field_coords)
                    if _euclidean_dist(pos, RED_TOWER_POS) <= _euclidean_dist(pos, BLUE_TOWER_POS):
                        channel_candidates["red_climb"].append(act)
                    else:
                        channel_candidates["blue_climb"].append(act)
            else:
                channel_candidates["unconstrained"].append(act)

        reconciled_by_id: Dict[str, ReconciledActionSegment] = {}

        # Process each channel with its FMS quota
        for ch_key, candidates in channel_candidates.items():
            if not candidates:
                continue

            # Determine FMS quota K for this channel
            quota: Optional[int] = None
            if target_deltas is not None:
                delta_map = {
                    "red_fuel": target_deltas.get("red_fuel", target_deltas.get("red_fuel_scored")),
                    "blue_fuel": target_deltas.get("blue_fuel", target_deltas.get("blue_fuel_scored")),
                    "red_climb": target_deltas.get("red_tower_climbs", target_deltas.get("red_climbs")),
                    "blue_climb": target_deltas.get("blue_tower_climbs", target_deltas.get("blue_climbs")),
                }
                quota = delta_map.get(ch_key)
            elif score_state is not None:
                if ch_key == "red_fuel":
                    quota = max(0, score_state.red_fuel_scored - self._confirmed_counts["red_fuel_scored"])
                elif ch_key == "blue_fuel":
                    quota = max(0, score_state.blue_fuel_scored - self._confirmed_counts["blue_fuel_scored"])
                elif ch_key == "red_climb":
                    quota = max(0, score_state.red_tower_climbs - self._confirmed_counts["red_tower_climbs"])
                elif ch_key == "blue_climb":
                    quota = max(0, score_state.blue_tower_climbs - self._confirmed_counts["blue_tower_climbs"])

            # 1. Evaluate spatial likelihood for all candidates in this channel
            spatial_evals: List[Tuple[ActionSegment, float, bool, float]] = []
            for act in candidates:
                pos = robot_positions.get(act.track_id, act.field_coords)
                p_spatial, passes = self.evaluate_spatial_gate(act, pos)
                p_cv = max(0.0, min(1.0, float(act.confidence)))
                joint_score = p_spatial * p_cv
                spatial_evals.append((act, p_spatial, passes, joint_score))

            # 2. Reconcile against quota
            if ch_key == "unconstrained" or quota is None:
                # Unconstrained: all spatially valid actions are confirmed
                for act, p_spatial, passes, joint_score in spatial_evals:
                    if passes:
                        status = RECON_STATUS_CONFIRMED
                        posterior = round(joint_score, 4)
                        notes = "Confirmed: spatially valid unconstrained action"
                    else:
                        status = RECON_STATUS_PRUNED_FALSE_POSITIVE
                        posterior = 0.0
                        notes = "Pruned: failed spatial gate (distance too far from element)"

                    rec_act = self._create_reconciled_segment(act, status, p_spatial, posterior, notes)
                    reconciled_by_id[act.event_id] = rec_act
            else:
                # Quota constrained by FMS ground truth
                # Sift spatially valid vs invalid
                spatially_valid = [item for item in spatial_evals if item[2]]
                spatially_invalid = [item for item in spatial_evals if not item[2]]

                # Spatially invalid are pruned immediately
                for act, p_spatial, _, _ in spatially_invalid:
                    status = RECON_STATUS_PRUNED_FALSE_POSITIVE
                    posterior = 0.0
                    notes = "Pruned: failed spatial gate (robot 250+ in away from element)"
                    rec_act = self._create_reconciled_segment(act, status, p_spatial, posterior, notes)
                    reconciled_by_id[act.event_id] = rec_act

                # Sort spatially valid by joint posterior score descending
                spatially_valid.sort(key=lambda item: (item[3], item[0].confidence, item[0].diou), reverse=True)

                confirmed_count = 0
                for rank, (act, p_spatial, _, joint_score) in enumerate(spatially_valid):
                    if rank < quota:
                        status = RECON_STATUS_CONFIRMED
                        posterior = round(joint_score, 4)
                        notes = f"Confirmed against FMS score state (rank {rank + 1} of quota {quota})"
                        confirmed_count += 1
                    else:
                        status = RECON_STATUS_PRUNED_FALSE_POSITIVE
                        posterior = 0.0
                        notes = f"Pruned: exceeded FMS ground truth quota of {quota}"

                    rec_act = self._create_reconciled_segment(act, status, p_spatial, posterior, notes)
                    reconciled_by_id[act.event_id] = rec_act

                # Update cumulative confirmed counts
                if ch_key == "red_fuel":
                    self._confirmed_counts["red_fuel_scored"] += confirmed_count
                elif ch_key == "blue_fuel":
                    self._confirmed_counts["blue_fuel_scored"] += confirmed_count
                elif ch_key == "red_climb":
                    self._confirmed_counts["red_tower_climbs"] += confirmed_count
                elif ch_key == "blue_climb":
                    self._confirmed_counts["blue_tower_climbs"] += confirmed_count

        # Build output list maintaining original order of candidate_actions
        output: List[ReconciledActionSegment] = []
        for act in candidate_actions:
            rec_act = reconciled_by_id.get(act.event_id)
            if rec_act is not None:
                if not filter_pruned or rec_act.reconciled_posterior > 0.0:
                    output.append(rec_act)

        return output

    def _create_reconciled_segment(
        self,
        action: ActionSegment,
        status: Union[str, ReconciliationStatus],
        p_spatial: float,
        posterior: float,
        notes: str,
    ) -> ReconciledActionSegment:
        # Update original ActionSegment in-place as well
        action.is_reconciled = True
        action.reconciled_posterior = posterior

        # Track in engine lookup dicts
        status_str = str(status.value if isinstance(status, ReconciliationStatus) else status)
        self._event_statuses[action.event_id] = status_str
        self._event_posteriors[action.event_id] = posterior

        return ReconciledActionSegment.from_action_segment(
            action=action,
            status=status_str,
            spatial_likelihood=round(p_spatial, 4),
            reconciliation_notes=notes,
            is_reconciled=True,
            reconciled_posterior=posterior,
        )


# ============================================================================
# Reconciliation Actor (Base Class & Protocol Conformance)
# ============================================================================

class ReconciliationActor(ReconciliationActorBase):
    """Asynchronous actor reconciling ActionFormer events with FMS ground truth."""

    def __init__(
        self,
        engine: Optional[BayesianReconciliationEngine] = None,
        filter_pruned: bool = False,
    ) -> None:
        self.engine = engine if engine is not None else BayesianReconciliationEngine()
        self.filter_pruned = filter_pruned
        self._current_score_state: Optional[FieldAPIScoreState] = None
        self._lock = asyncio.Lock()

    @property
    def current_score_state(self) -> Optional[FieldAPIScoreState]:
        """Returns the current FMS ground truth score state."""
        return self._current_score_state

    async def update_field_api_score(self, score_state: FieldAPIScoreState) -> None:
        """Ingests live aggregate scoring data from official FRC Field Management System (FMS)."""
        async with self._lock:
            self._current_score_state = score_state
            logger.info(
                f"Updated FMS score state: RedFuel={score_state.red_fuel_scored}, "
                f"BlueFuel={score_state.blue_fuel_scored}, RedClimbs={score_state.red_tower_climbs}, "
                f"BlueClimbs={score_state.blue_tower_climbs}"
            )

    async def reconcile_actions(
        self,
        candidate_actions: List[ActionSegment],
        robot_positions: Dict[int, Tuple[float, float]],
        score_state: Optional[FieldAPIScoreState] = None,
        filter_pruned: Optional[bool] = None,
    ) -> List[ActionSegment]:
        """Calculates Bayesian posterior probabilities, pruning spatial false positives."""
        async with self._lock:
            active_score_state = score_state if score_state is not None else self._current_score_state
            use_filter = filter_pruned if filter_pruned is not None else self.filter_pruned

            # Reconcile via Bayesian Engine
            reconciled = self.engine.reconcile(
                candidate_actions=candidate_actions,
                robot_positions=robot_positions,
                score_state=active_score_state,
                filter_pruned=use_filter,
            )
            # ReconciledActionSegment is an ActionSegment
            return list(reconciled)

    def get_status(self, event_id: str) -> Optional[str]:
        """Gets reconciliation status for a specific event."""
        return self.engine.get_status(event_id)

    def get_posterior(self, event_id: str) -> Optional[float]:
        """Gets posterior probability for a specific event."""
        return self.engine.get_posterior(event_id)

    def get_confirmed_counts(self) -> Dict[str, int]:
        """Returns confirmed score totals matching FMS ground truth."""
        return self.engine.get_confirmed_counts()

    def reset(self) -> None:
        """Resets the actor and underlying Bayesian engine."""
        self._current_score_state = None
        self.engine.reset()
