"""Unit tests for Module 07: Bayesian Ground-Truth Reconciliation.

Verifies:
1. Interface contracts: ReconciliationActor conforms to ReconciliationActorBase and ReconciliationActorProtocol.
2. Spatial gating: Evaluates robot Euclidean distance to CAD geometry. Likelihood > 0.8 within 60", near 0 at 250".
3. Bayesian score reconciliation under conflict: ActionFormer predicts 5 Red Hub scores, FMS reports 3;
   top 3 confirmed, 2 pruned false positives, total matches FMS exactly.
4. Tower climb and depot intake reconciliation with strict Red vs Blue alliance isolation.
5. FRC Field API client polling, updates, and simulator streaming.
"""

import asyncio
from typing import Dict, List, Tuple
import pytest

from src.common.interfaces import ReconciliationActorBase, ReconciliationActorProtocol
from src.common.types import (
    ActionSegment,
    ActionType,
    FieldAPIScoreState,
    FieldElement,
    MatchStage,
)
from src.reconciliation import (
    BLUE_HUB_POS,
    BLUE_TOWER_POS,
    DEPOT_NORTH_POS,
    DEPOT_SOUTH_POS,
    HUB_SCORING_MAX_RADIUS,
    RECON_STATUS_CONFIRMED,
    RECON_STATUS_PRUNED_FALSE_POSITIVE,
    RED_HUB_POS,
    RED_TOWER_POS,
    BayesianReconciliationEngine,
    FMSSimulator,
    FRCFieldAPIClient,
    ReconciledActionSegment,
    ReconciliationActor,
    ReconciliationStatus,
    compute_depot_spatial_likelihood,
    compute_hub_spatial_likelihood,
    compute_spatial_likelihood,
    compute_tower_spatial_likelihood,
)


def _make_candidate(
    event_id: str,
    action_type: ActionType,
    element: FieldElement,
    track_id: int = 1,
    team_number: int = 254,
    confidence: float = 0.90,
    field_coords: Tuple[float, float] = (162.8, 158.85),
    start_time_ms: int = 1000,
    end_time_ms: int = 3000,
    diou: float = 0.85,
) -> ActionSegment:
    """Helper to construct an ActionSegment fixture."""
    return ActionSegment(
        event_id=event_id,
        track_id=track_id,
        team_number=team_number,
        action_type=action_type,
        element=element,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        start_frame=int(start_time_ms * 30 / 1000),
        end_frame=int(end_time_ms * 30 / 1000),
        confidence=confidence,
        diou=diou,
        pyramid_level=1,
        field_coords=field_coords,
        is_reconciled=False,
        reconciled_posterior=0.0,
    )


# ============================================================================
# 1. Interface Contract Validation
# ============================================================================

def test_actor_interface_contracts():
    """Assert ReconciliationActor inherits ReconciliationActorBase and satisfies ReconciliationActorProtocol."""
    actor = ReconciliationActor()

    # Verify ABC inheritance
    assert isinstance(actor, ReconciliationActorBase), "ReconciliationActor must inherit ReconciliationActorBase"

    # Verify runtime checkable Protocol conformance
    assert isinstance(actor, ReconciliationActorProtocol), "ReconciliationActor must satisfy ReconciliationActorProtocol"

    # Verify required asynchronous interface methods exist and are callable
    assert hasattr(actor, "update_field_api_score")
    assert asyncio.iscoroutinefunction(actor.update_field_api_score)
    assert hasattr(actor, "reconcile_actions")
    assert asyncio.iscoroutinefunction(actor.reconcile_actions)


# ============================================================================
# 2. Spatial Gating Validation
# ============================================================================

def test_spatial_gating_hub_within_60_inches():
    """Robot within 60\" of Hub: spatial likelihood must be high (> 0.8)."""
    # Exact center of Red Hub
    p_center = compute_hub_spatial_likelihood(0.0)
    assert p_center == 1.0, f"Expected 1.0 at center, got {p_center}"

    # 30 inches away (well within 60" radius)
    p_30 = compute_hub_spatial_likelihood(30.0)
    assert p_30 > 0.8, f"Expected > 0.8 at 30\", got {p_30}"

    # Exactly 60 inches away
    p_60 = compute_hub_spatial_likelihood(60.0)
    assert p_60 > 0.8, f"Expected > 0.8 at 60\" nominal radius, got {p_60}"

    # Generic compute_spatial_likelihood within 60" of Red Hub
    # Red Hub is at (162.8, 158.85). Robot at (162.8 + 40.0, 158.85) -> distance = 40.0"
    pos_40 = (RED_HUB_POS[0] + 40.0, RED_HUB_POS[1])
    p_hub_40 = compute_spatial_likelihood(FieldElement.RED_HUB, pos_40, ActionType.SCORE_HUB)
    assert p_hub_40 > 0.8, f"Expected > 0.8 at 40\" from Red Hub, got {p_hub_40}"


def test_spatial_gating_hub_at_250_inches_prunes_false_positive():
    """Robot 250\" away from Hub: spatial likelihood is near 0, action is pruned / marked false positive."""
    # 250 inches away
    p_250 = compute_hub_spatial_likelihood(250.0)
    assert p_250 < 1e-4, f"Expected near zero at 250\", got {p_250}"
    assert p_250 == 0.0, f"Expected strictly 0.0 with cutoff, got {p_250}"

    # Using full coordinate check: Red Hub (162.8, 158.85), Robot at (412.8, 158.85) (250" away)
    pos_far = (RED_HUB_POS[0] + 250.0, RED_HUB_POS[1])
    candidate = _make_candidate(
        event_id="evt_far_hub",
        action_type=ActionType.SCORE_HUB,
        element=FieldElement.RED_HUB,
        field_coords=pos_far,
        confidence=0.92,
    )

    p_spatial = compute_spatial_likelihood(candidate, pos_far)
    assert p_spatial < 1e-4, f"Spatial likelihood at 250\" should be near 0, got {p_spatial}"

    engine = BayesianReconciliationEngine()
    reconciled = engine.reconcile(
        candidate_actions=[candidate],
        robot_positions={candidate.track_id: pos_far},
        score_state=FieldAPIScoreState(
            timestamp_ms=1000,
            match_stage=MatchStage.TELEOP,
            red_fuel_scored=10,
            blue_fuel_scored=10,
            red_tower_climbs=0,
            blue_tower_climbs=0,
        ),
    )

    assert len(reconciled) == 1
    rec_event = reconciled[0]
    # Assert spatial gate marks event as RECON_STATUS_PRUNED_FALSE_POSITIVE
    assert rec_event.status == RECON_STATUS_PRUNED_FALSE_POSITIVE
    assert rec_event.status == "RECON_STATUS_PRUNED_FALSE_POSITIVE"
    assert rec_event.is_reconciled is True
    assert rec_event.reconciled_posterior == 0.0


def test_spatial_gating_blue_hub_and_towers():
    """Verify spatial likelihoods on Blue Hub, Red Tower, and Blue Tower."""
    # Blue Hub at (488.4, 158.85)
    pos_blue_near = (BLUE_HUB_POS[0] - 30.0, BLUE_HUB_POS[1])
    p_blue_near = compute_spatial_likelihood(FieldElement.BLUE_HUB, pos_blue_near, ActionType.SCORE_HUB)
    assert p_blue_near > 0.8

    pos_blue_far = (BLUE_HUB_POS[0] - 250.0, BLUE_HUB_POS[1])
    p_blue_far = compute_spatial_likelihood(FieldElement.BLUE_HUB, pos_blue_far, ActionType.SCORE_HUB)
    assert p_blue_far < 1e-4

    # Red Tower at (90.0, 158.85), climb radius 45"
    pos_tower_near = (RED_TOWER_POS[0] + 25.0, RED_TOWER_POS[1])
    p_tower_near = compute_spatial_likelihood(FieldElement.RED_TOWER, pos_tower_near, ActionType.CLIMB_TOWER)
    assert p_tower_near > 0.8

    pos_tower_far = (RED_TOWER_POS[0] + 250.0, RED_TOWER_POS[1])
    p_tower_far = compute_spatial_likelihood(FieldElement.RED_TOWER, pos_tower_far, ActionType.CLIMB_TOWER)
    assert p_tower_far < 1e-4


# ============================================================================
# 3. Bayesian Score Reconciliation Under Conflict
# ============================================================================

@pytest.mark.asyncio
async def test_bayesian_score_reconciliation_under_conflict():
    """ActionFormer predicts 5 Red Hub scores, but FMS reports delta of 3.
    
    Requirements:
    - Assert that exactly 3 events with highest posterior probabilities are confirmed
      (is_reconciled=True, reconciled_posterior > 0).
    - Exactly 2 events are pruned/rejected (reconciled_posterior == 0.0).
    - Total confirmed reconciled score matches FMS state exactly.
    """
    actor = ReconciliationActor()

    # Robot positions near Red Hub (all within 35 inches of Red Hub)
    hub_x, hub_y = RED_HUB_POS
    robot_positions: Dict[int, Tuple[float, float]] = {
        1: (hub_x + 10.0, hub_y),
        2: (hub_x - 15.0, hub_y),
        3: (hub_x, hub_y + 20.0),
        4: (hub_x, hub_y - 25.0),
        5: (hub_x + 30.0, hub_y),
    }

    # 5 Candidate actions with distinct confidences
    # Sorted true ranking by joint likelihood will follow confidences since all are close to Hub
    candidates: List[ActionSegment] = [
        _make_candidate("red_fuel_1", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=1, confidence=0.96),
        _make_candidate("red_fuel_2", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=2, confidence=0.91),
        _make_candidate("red_fuel_3", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=3, confidence=0.87),
        _make_candidate("red_fuel_4", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=4, confidence=0.78),
        _make_candidate("red_fuel_5", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=5, confidence=0.70),
    ]

    # FMS ground-truth reports only 3 scores for Red Hub
    fms_state = FieldAPIScoreState(
        timestamp_ms=15000,
        match_stage=MatchStage.TELEOP,
        red_fuel_scored=3,
        blue_fuel_scored=0,
        red_tower_climbs=0,
        blue_tower_climbs=0,
    )

    await actor.update_field_api_score(fms_state)
    reconciled = await actor.reconcile_actions(candidates, robot_positions)

    assert len(reconciled) == 5

    # Separate confirmed vs pruned
    confirmed = [act for act in reconciled if act.reconciled_posterior > 0.0]
    pruned = [act for act in reconciled if act.reconciled_posterior == 0.0]

    # Assert exactly 3 events confirmed, exactly 2 pruned
    assert len(confirmed) == 3, f"Expected exactly 3 confirmed events, got {len(confirmed)}"
    assert len(pruned) == 2, f"Expected exactly 2 pruned events, got {len(pruned)}"

    # Check top 3 confirmed events are the ones with highest confidence
    confirmed_ids = {act.event_id for act in confirmed}
    expected_top_3 = {"red_fuel_1", "red_fuel_2", "red_fuel_3"}
    assert confirmed_ids == expected_top_3, f"Expected {expected_top_3}, got {confirmed_ids}"

    # Check that confirmed events have is_reconciled=True and status=RECON_STATUS_CONFIRMED
    for act in confirmed:
        assert act.is_reconciled is True
        assert act.reconciled_posterior > 0.0
        assert act.status == RECON_STATUS_CONFIRMED

    # Check that pruned events have is_reconciled=True and status=RECON_STATUS_PRUNED_FALSE_POSITIVE
    for act in pruned:
        assert act.is_reconciled is True
        assert act.reconciled_posterior == 0.0
        assert act.status == RECON_STATUS_PRUNED_FALSE_POSITIVE

    # Assert total confirmed reconciled score matches FMS state exactly
    confirmed_counts = actor.get_confirmed_counts()
    assert confirmed_counts["red_fuel_scored"] == fms_state.red_fuel_scored == 3


@pytest.mark.asyncio
async def test_spatial_gate_and_score_conflict_combined():
    """Conflict scenario where 1 of the 5 candidates fails spatial gate (250\" away),
    and FMS reports delta of 3 out of the remaining 4 candidates.
    """
    actor = ReconciliationActor()

    hub_x, hub_y = RED_HUB_POS
    robot_positions: Dict[int, Tuple[float, float]] = {
        1: (hub_x + 10.0, hub_y),           # valid (~10 in)
        2: (hub_x + 250.0, hub_y),          # INVALID (250 in away!)
        3: (hub_x, hub_y + 15.0),           # valid (~15 in)
        4: (hub_x, hub_y - 20.0),           # valid (~20 in)
        5: (hub_x + 25.0, hub_y),           # valid (~25 in)
    }

    # Candidate 2 has highest CV confidence (0.99) but is 250" away!
    candidates: List[ActionSegment] = [
        _make_candidate("cand_1", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=1, confidence=0.88),
        _make_candidate("cand_2_far", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=2, confidence=0.99),
        _make_candidate("cand_3", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=3, confidence=0.85),
        _make_candidate("cand_4", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=4, confidence=0.80),
        _make_candidate("cand_5", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=5, confidence=0.72),
    ]

    fms_state = FieldAPIScoreState(
        timestamp_ms=20000,
        match_stage=MatchStage.TELEOP,
        red_fuel_scored=3,
        blue_fuel_scored=0,
        red_tower_climbs=0,
        blue_tower_climbs=0,
    )

    reconciled = await actor.reconcile_actions(candidates, robot_positions, score_state=fms_state)

    # cand_2_far must be pruned due to spatial gate regardless of its 0.99 CV confidence
    rec_cand_2 = next(c for c in reconciled if c.event_id == "cand_2_far")
    assert rec_cand_2.status == RECON_STATUS_PRUNED_FALSE_POSITIVE
    assert rec_cand_2.reconciled_posterior == 0.0

    confirmed = [c for c in reconciled if c.reconciled_posterior > 0.0]
    assert len(confirmed) == 3
    confirmed_ids = {c.event_id for c in confirmed}
    # Remaining valid sorted by confidence: cand_1 (0.88), cand_3 (0.85), cand_4 (0.80)
    assert confirmed_ids == {"cand_1", "cand_3", "cand_4"}

    # cand_5 is pruned due to FMS quota exhaustion
    rec_cand_5 = next(c for c in reconciled if c.event_id == "cand_5")
    assert rec_cand_5.status == RECON_STATUS_PRUNED_FALSE_POSITIVE
    assert rec_cand_5.reconciled_posterior == 0.0


# ============================================================================
# 4. Tower Climb & Depot Intake Reconciliation & Alliance Isolation
# ============================================================================

@pytest.mark.asyncio
async def test_tower_climb_and_alliance_score_isolation():
    """Reconcile tower climb actions against FMS tower climb state.
    
    Ingests multiple alliance events (Red vs Blue) and verifies alliance score isolation.
    """
    actor = ReconciliationActor()

    # Robot positions
    robot_positions: Dict[int, Tuple[float, float]] = {
        # Red alliance robots near Red Hub and Red Tower
        10: (RED_HUB_POS[0] + 15.0, RED_HUB_POS[1]),
        11: (RED_HUB_POS[0] - 15.0, RED_HUB_POS[1]),
        12: (RED_TOWER_POS[0] + 20.0, RED_TOWER_POS[1]),
        # Blue alliance robots near Blue Hub and Blue Tower
        20: (BLUE_HUB_POS[0] + 15.0, BLUE_HUB_POS[1]),
        21: (BLUE_HUB_POS[0] - 15.0, BLUE_HUB_POS[1]),
        22: (BLUE_TOWER_POS[0] - 20.0, BLUE_TOWER_POS[1]),
    }

    # Red candidate actions
    red_fuel_cands = [
        _make_candidate("r_fuel_1", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=10, team_number=254, confidence=0.92),
        _make_candidate("r_fuel_2", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=11, team_number=1323, confidence=0.88),
        _make_candidate("r_fuel_3", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=10, team_number=254, confidence=0.75),
    ]
    red_climb_cands = [
        _make_candidate("r_climb_1", ActionType.CLIMB_TOWER, FieldElement.RED_TOWER, track_id=12, team_number=971, confidence=0.94),
        _make_candidate("r_climb_2", ActionType.CLIMB_TOWER, FieldElement.RED_TOWER, track_id=11, team_number=1323, confidence=0.72),
    ]

    # Blue candidate actions
    blue_fuel_cands = [
        _make_candidate("b_fuel_1", ActionType.SCORE_HUB, FieldElement.BLUE_HUB, track_id=20, team_number=1678, confidence=0.95),
        _make_candidate("b_fuel_2", ActionType.SCORE_HUB, FieldElement.BLUE_HUB, track_id=21, team_number=2910, confidence=0.90),
        _make_candidate("b_fuel_3", ActionType.SCORE_HUB, FieldElement.BLUE_HUB, track_id=20, team_number=1678, confidence=0.85),
        _make_candidate("b_fuel_4", ActionType.SCORE_HUB, FieldElement.BLUE_HUB, track_id=21, team_number=2910, confidence=0.70),
    ]
    blue_climb_cands = [
        _make_candidate("b_climb_1", ActionType.CLIMB_TOWER, FieldElement.BLUE_TOWER, track_id=22, team_number=4414, confidence=0.91),
    ]

    all_candidates = red_fuel_cands + red_climb_cands + blue_fuel_cands + blue_climb_cands

    # FMS Ground truth:
    # Red: 2 fuel scores, 1 tower climb
    # Blue: 3 fuel scores, 1 tower climb
    fms_state = FieldAPIScoreState(
        timestamp_ms=30000,
        match_stage=MatchStage.TELEOP,
        red_fuel_scored=2,
        blue_fuel_scored=3,
        red_tower_climbs=1,
        blue_tower_climbs=1,
    )

    reconciled = await actor.reconcile_actions(all_candidates, robot_positions, score_state=fms_state)

    # Validate Red Hub: exactly 2 confirmed, 1 pruned
    red_hub_confirmed = [a for a in reconciled if a.element == FieldElement.RED_HUB and a.reconciled_posterior > 0.0]
    red_hub_pruned = [a for a in reconciled if a.element == FieldElement.RED_HUB and a.reconciled_posterior == 0.0]
    assert len(red_hub_confirmed) == 2
    assert len(red_hub_pruned) == 1
    assert {a.event_id for a in red_hub_confirmed} == {"r_fuel_1", "r_fuel_2"}

    # Validate Red Tower: exactly 1 confirmed, 1 pruned
    red_climb_confirmed = [a for a in reconciled if a.element == FieldElement.RED_TOWER and a.reconciled_posterior > 0.0]
    red_climb_pruned = [a for a in reconciled if a.element == FieldElement.RED_TOWER and a.reconciled_posterior == 0.0]
    assert len(red_climb_confirmed) == 1
    assert len(red_climb_pruned) == 1
    assert red_climb_confirmed[0].event_id == "r_climb_1"

    # Validate Blue Hub: exactly 3 confirmed, 1 pruned
    blue_hub_confirmed = [a for a in reconciled if a.element == FieldElement.BLUE_HUB and a.reconciled_posterior > 0.0]
    blue_hub_pruned = [a for a in reconciled if a.element == FieldElement.BLUE_HUB and a.reconciled_posterior == 0.0]
    assert len(blue_hub_confirmed) == 3
    assert len(blue_hub_pruned) == 1
    assert {a.event_id for a in blue_hub_confirmed} == {"b_fuel_1", "b_fuel_2", "b_fuel_3"}

    # Validate Blue Tower: exactly 1 confirmed, 0 pruned
    blue_climb_confirmed = [a for a in reconciled if a.element == FieldElement.BLUE_TOWER and a.reconciled_posterior > 0.0]
    assert len(blue_climb_confirmed) == 1
    assert blue_climb_confirmed[0].event_id == "b_climb_1"

    # Verify complete confirmed counts match FMS exactly
    counts = actor.get_confirmed_counts()
    assert counts["red_fuel_scored"] == 2
    assert counts["blue_fuel_scored"] == 3
    assert counts["red_tower_climbs"] == 1
    assert counts["blue_tower_climbs"] == 1


@pytest.mark.asyncio
async def test_depot_intake_spatial_reconciliation():
    """Verify depot intake actions reconcile according to depot geometry."""
    actor = ReconciliationActor()

    # Robot 1 is within 20 inches of North Depot (325.6, 280.0)
    pos_depot_valid = (DEPOT_NORTH_POS[0] + 15.0, DEPOT_NORTH_POS[1])
    # Robot 2 is far away at (50.0, 50.0) (> 200 in away from either depot)
    pos_depot_invalid = (50.0, 50.0)

    cand_valid = _make_candidate(
        "intake_valid",
        ActionType.INTAKE_FUEL,
        FieldElement.DEPOT,
        track_id=1,
        confidence=0.88,
        field_coords=pos_depot_valid,
    )
    cand_invalid = _make_candidate(
        "intake_invalid",
        ActionType.INTAKE_FUEL,
        FieldElement.DEPOT,
        track_id=2,
        confidence=0.91,
        field_coords=pos_depot_invalid,
    )

    reconciled = await actor.reconcile_actions(
        [cand_valid, cand_invalid],
        robot_positions={1: pos_depot_valid, 2: pos_depot_invalid},
    )

    rec_valid = next(a for a in reconciled if a.event_id == "intake_valid")
    rec_invalid = next(a for a in reconciled if a.event_id == "intake_invalid")

    assert rec_valid.is_reconciled is True
    assert rec_valid.reconciled_posterior > 0.8
    assert rec_valid.status == RECON_STATUS_CONFIRMED

    assert rec_invalid.is_reconciled is True
    assert rec_invalid.reconciled_posterior == 0.0
    assert rec_invalid.status == RECON_STATUS_PRUNED_FALSE_POSITIVE


# ============================================================================
# 5. FRC Field API Client and Simulator Tests
# ============================================================================

@pytest.mark.asyncio
async def test_frc_api_client_polling_and_simulation():
    """Verify FRCFieldAPIClient tracks state, increments scores, and supports simulated streaming."""
    client = FRCFieldAPIClient(event_key="2026test", match_id="qm1")

    # Initial state verification
    state = client.get_state()
    assert state.red_fuel_scored == 0
    assert state.blue_fuel_scored == 0
    assert state.match_stage == MatchStage.PRE_MATCH

    # Score update listener
    notifications: List[FieldAPIScoreState] = []
    client.subscribe(lambda s: notifications.append(s))

    # Test increments
    await client.increment_score(red_fuel_delta=3, match_stage=MatchStage.AUTONOMOUS)
    assert client.get_state().red_fuel_scored == 3
    assert client.get_state().match_stage == MatchStage.AUTONOMOUS

    await client.increment_score(blue_fuel_delta=5, red_tower_climbs_delta=1, match_stage=MatchStage.TELEOP)
    assert client.get_state().red_fuel_scored == 3
    assert client.get_state().blue_fuel_scored == 5
    assert client.get_state().red_tower_climbs == 1
    assert client.get_state().match_stage == MatchStage.TELEOP

    assert len(notifications) == 2

    # Deltas calculation
    deltas = client.compute_deltas()
    assert deltas["red_fuel_delta"] == 3
    assert deltas["blue_fuel_delta"] == 5
    assert deltas["red_tower_climbs_delta"] == 1

    # Simulator execution
    sim = FMSSimulator(client)
    await sim.run_match_script(
        events=[
            {"delay_s": 0.01, "red_fuel_delta": 2, "blue_fuel_delta": 1},
            {"delay_s": 0.01, "blue_climbs_delta": 1, "match_stage": MatchStage.ENDGAME},
        ],
        time_dilation=10.0,
    )

    final_state = client.get_state()
    assert final_state.red_fuel_scored == 5
    assert final_state.blue_fuel_scored == 6
    assert final_state.blue_tower_climbs == 1
    assert final_state.match_stage == MatchStage.ENDGAME

    # Stream polling
    polled = []
    async for s in client.poll_score_stream(interval_s=0.01, max_polls=3):
        polled.append(s)
    assert len(polled) == 3


@pytest.mark.asyncio
async def test_filter_pruned_mode_toggle():
    """Verify filter_pruned mode returns only confirmed events when enabled."""
    actor = ReconciliationActor(filter_pruned=True)

    hub_x, hub_y = RED_HUB_POS
    candidates = [
        _make_candidate("c1", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=1, confidence=0.90),
        _make_candidate("c2", ActionType.SCORE_HUB, FieldElement.RED_HUB, track_id=1, confidence=0.70),
    ]

    fms = FieldAPIScoreState(
        timestamp_ms=1000,
        match_stage=MatchStage.TELEOP,
        red_fuel_scored=1,
        blue_fuel_scored=0,
        red_tower_climbs=0,
        blue_tower_climbs=0,
    )

    # With filter_pruned=True, only 1 confirmed event returned
    reconciled_filtered = await actor.reconcile_actions(
        candidates,
        {1: (hub_x, hub_y)},
        score_state=fms,
        filter_pruned=True,
    )
    assert len(reconciled_filtered) == 1
    assert reconciled_filtered[0].event_id == "c1"

    # With filter_pruned=False, both returned (1 confirmed, 1 pruned)
    actor.reset()
    reconciled_all = await actor.reconcile_actions(
        candidates,
        {1: (hub_x, hub_y)},
        score_state=fms,
        filter_pruned=False,
    )
    assert len(reconciled_all) == 2
