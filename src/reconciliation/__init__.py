"""Reconciliation Package.

Module 07: Bayesian Ground-Truth Reconciliation
Exports components for Bayesian spatial-temporal action gating and FRC FMS score reconciliation.
"""

from src.reconciliation.bayesian_engine import (
    BLUE_HUB_POS,
    BLUE_OUTPOST_POS,
    BLUE_TOWER_POS,
    BUMPS_POS,
    DEPOT_NORTH_POS,
    DEPOT_SOUTH_POS,
    HUB_SCORING_MAX_RADIUS,
    NEUTRAL_ZONE_X_RANGE,
    RECON_STATUS_ADJUSTED,
    RECON_STATUS_CONFIRMED,
    RECON_STATUS_PENDING,
    RECON_STATUS_PRUNED_FALSE_POSITIVE,
    RECON_STATUS_UNSPECIFIED,
    RED_HUB_POS,
    RED_OUTPOST_POS,
    RED_TOWER_POS,
    TOWER_CLIMB_MAX_RADIUS,
    TRENCHES_POS,
    BayesianReconciliationEngine,
    ReconciledActionSegment,
    ReconciliationActor,
    ReconciliationStatus,
    compute_depot_spatial_likelihood,
    compute_hub_spatial_likelihood,
    compute_spatial_likelihood,
    compute_tower_spatial_likelihood,
)
from src.reconciliation.frc_api_client import (
    FMSSimulator,
    FRCFieldAPIClient,
)

__all__ = [
    "ReconciliationActor",
    "BayesianReconciliationEngine",
    "FRCFieldAPIClient",
    "FMSSimulator",
    "ReconciledActionSegment",
    "ReconciliationStatus",
    "RECON_STATUS_UNSPECIFIED",
    "RECON_STATUS_PENDING",
    "RECON_STATUS_CONFIRMED",
    "RECON_STATUS_PRUNED_FALSE_POSITIVE",
    "RECON_STATUS_ADJUSTED",
    "compute_spatial_likelihood",
    "compute_hub_spatial_likelihood",
    "compute_tower_spatial_likelihood",
    "compute_depot_spatial_likelihood",
    "RED_HUB_POS",
    "BLUE_HUB_POS",
    "RED_TOWER_POS",
    "BLUE_TOWER_POS",
    "DEPOT_NORTH_POS",
    "DEPOT_SOUTH_POS",
    "RED_OUTPOST_POS",
    "BLUE_OUTPOST_POS",
    "BUMPS_POS",
    "TRENCHES_POS",
    "NEUTRAL_ZONE_X_RANGE",
    "HUB_SCORING_MAX_RADIUS",
    "TOWER_CLIMB_MAX_RADIUS",
]
