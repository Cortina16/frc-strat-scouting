"""Analytics package for FRC scouting pipeline.

Module 08: DuckDB Analytical Engine & Spatio-Temporal Queries.
Exports in-process DuckDB storage actor, DuckDB analytical store, and specialized
spatio-temporal query helpers.
"""

from src.analytics.duckdb_store import (
    BLUE_HUB_POS,
    BLUE_TOWER_POS,
    DEPOT_NORTH_POS,
    DEPOT_SOUTH_POS,
    HUB_INTERACTION_RADIUS,
    NEUTRAL_ZONE_X_MAX,
    NEUTRAL_ZONE_X_MIN,
    RED_HUB_POS,
    RED_TOWER_POS,
    SOURCE_BLUE_X_MIN,
    SOURCE_RED_X_MAX,
    TRENCH_NORTH_Y_MIN,
    TRENCH_SOUTH_Y_MAX,
    TRENCH_X_MAX,
    TRENCH_X_MIN,
    DuckDBStorageActor,
    DuckDBStore,
    ExtendedActionSegment,
    ExtendedRobotTrackletState,
    create_tables,
    export_table_to_parquet,
    query_asof_join,
    query_cycle_times,
    query_match_summary,
    query_velocity_degradation,
)

__all__ = [
    "DuckDBStorageActor",
    "DuckDBStore",
    "ExtendedRobotTrackletState",
    "ExtendedActionSegment",
    "create_tables",
    "query_asof_join",
    "query_cycle_times",
    "query_velocity_degradation",
    "query_match_summary",
    "export_table_to_parquet",
    "RED_HUB_POS",
    "BLUE_HUB_POS",
    "RED_TOWER_POS",
    "BLUE_TOWER_POS",
    "DEPOT_NORTH_POS",
    "DEPOT_SOUTH_POS",
    "HUB_INTERACTION_RADIUS",
    "SOURCE_RED_X_MAX",
    "SOURCE_BLUE_X_MIN",
    "NEUTRAL_ZONE_X_MIN",
    "NEUTRAL_ZONE_X_MAX",
    "TRENCH_X_MIN",
    "TRENCH_X_MAX",
    "TRENCH_NORTH_Y_MIN",
    "TRENCH_SOUTH_Y_MAX",
]
