"""FRC Field Management API Client and Score Simulator.

Module 07: Bayesian Ground-Truth Reconciliation
Provides polling client and simulator for official FRC Field Management System (FMS)
and The Blue Alliance (TBA) API scoring states.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from src.common.types import FieldAPIScoreState, MatchStage

logger = logging.getLogger(__name__)


class FRCFieldAPIClient:
    """Client and simulator for official FRC Field Management System (FMS) / TBA API.
    
    Tracks aggregate scoring state across match stages, supporting asynchronous
    polling, real-time score injection, and synthetic score stream generation.
    """

    def __init__(
        self,
        event_key: str = "2026cmp",
        match_id: str = "qm1",
        base_url: Optional[str] = None,
        auth_key: Optional[str] = None,
        initial_state: Optional[FieldAPIScoreState] = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.event_key = event_key
        self.match_id = match_id
        self.base_url = base_url
        self.auth_key = auth_key
        self.poll_interval_s = poll_interval_s

        if initial_state is not None:
            self._current_state = initial_state
        else:
            self._current_state = FieldAPIScoreState(
                timestamp_ms=int(time.time() * 1000),
                match_stage=MatchStage.PRE_MATCH,
                red_fuel_scored=0,
                blue_fuel_scored=0,
                red_tower_climbs=0,
                blue_tower_climbs=0,
            )

        self._history: List[FieldAPIScoreState] = [self._current_state]
        self._subscribers: List[Callable[[FieldAPIScoreState], Any]] = []
        self._lock = asyncio.Lock()

    @property
    def current_state(self) -> FieldAPIScoreState:
        """Returns the current score state."""
        return self._current_state

    def get_state(self) -> FieldAPIScoreState:
        """Synchronous accessor for the current score state."""
        return self._current_state

    def set_state(self, state: FieldAPIScoreState) -> None:
        """Synchronously sets the current score state and appends to history."""
        self._current_state = state
        self._history.append(state)
        self._notify_subscribers_sync(state)

    def get_history(self) -> List[FieldAPIScoreState]:
        """Returns the list of historical score states."""
        return list(self._history)

    def subscribe(self, callback: Callable[[FieldAPIScoreState], Any]) -> None:
        """Registers a callback to receive score state updates."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[FieldAPIScoreState], Any]) -> None:
        """Unregisters an existing callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _notify_subscribers_sync(self, state: FieldAPIScoreState) -> None:
        for cb in self._subscribers:
            try:
                res = cb(state)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.warning(f"Error in score update subscriber: {e}")

    async def _notify_subscribers_async(self, state: FieldAPIScoreState) -> None:
        for cb in self._subscribers:
            try:
                res = cb(state)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.warning(f"Error in score update subscriber: {e}")

    async def fetch_score_state(self) -> FieldAPIScoreState:
        """Fetches the latest official FMS score state.
        
        If configured with a remote HTTP base_url, queries the external endpoint.
        Otherwise, returns the latest tracked state.
        """
        if self.base_url:
            try:
                import urllib.request
                import json

                url = f"{self.base_url.rstrip('/')}/match/{self.event_key}_{self.match_id}"
                req = urllib.request.Request(url)
                if self.auth_key:
                    req.add_header("X-TBA-Auth-Key", self.auth_key)

                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, urllib.request.urlopen, req)
                data = json.loads(response.read().decode("utf-8"))

                score_breakdown = data.get("score_breakdown", {})
                red = score_breakdown.get("red", {})
                blue = score_breakdown.get("blue", {})

                state = FieldAPIScoreState(
                    timestamp_ms=int(time.time() * 1000),
                    match_stage=MatchStage(data.get("stage", MatchStage.TELEOP)),
                    red_fuel_scored=int(red.get("fuelScored", 0)),
                    blue_fuel_scored=int(blue.get("fuelScored", 0)),
                    red_tower_climbs=int(red.get("towerClimbs", 0)),
                    blue_tower_climbs=int(blue.get("towerClimbs", 0)),
                )
                async with self._lock:
                    self._current_state = state
                    self._history.append(state)
                await self._notify_subscribers_async(state)
                return state
            except Exception as e:
                logger.debug(f"HTTP fetch failed or not reachable, using local state: {e}")

        async with self._lock:
            return self._current_state

    async def get_latest_score(self) -> FieldAPIScoreState:
        """Alias for fetch_score_state."""
        return await self.fetch_score_state()

    async def update_score(
        self,
        red_fuel: Optional[int] = None,
        blue_fuel: Optional[int] = None,
        red_climbs: Optional[int] = None,
        blue_climbs: Optional[int] = None,
        match_stage: Optional[MatchStage] = None,
        timestamp_ms: Optional[int] = None,
    ) -> FieldAPIScoreState:
        """Updates the score state with absolute values and notifies subscribers."""
        async with self._lock:
            ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
            stage = match_stage if match_stage is not None else self._current_state.match_stage
            new_state = FieldAPIScoreState(
                timestamp_ms=ts,
                match_stage=stage,
                red_fuel_scored=red_fuel if red_fuel is not None else self._current_state.red_fuel_scored,
                blue_fuel_scored=blue_fuel if blue_fuel is not None else self._current_state.blue_fuel_scored,
                red_tower_climbs=red_climbs if red_climbs is not None else self._current_state.red_tower_climbs,
                blue_tower_climbs=blue_climbs if blue_climbs is not None else self._current_state.blue_tower_climbs,
            )
            self._current_state = new_state
            self._history.append(new_state)

        await self._notify_subscribers_async(new_state)
        return new_state

    async def increment_score(
        self,
        red_fuel_delta: int = 0,
        blue_fuel_delta: int = 0,
        red_tower_climbs_delta: int = 0,
        blue_tower_climbs_delta: int = 0,
        match_stage: Optional[MatchStage] = None,
        timestamp_ms: Optional[int] = None,
    ) -> FieldAPIScoreState:
        """Increments scores by relative deltas and notifies subscribers."""
        async with self._lock:
            ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
            stage = match_stage if match_stage is not None else self._current_state.match_stage
            new_state = FieldAPIScoreState(
                timestamp_ms=ts,
                match_stage=stage,
                red_fuel_scored=max(0, self._current_state.red_fuel_scored + red_fuel_delta),
                blue_fuel_scored=max(0, self._current_state.blue_fuel_scored + blue_fuel_delta),
                red_tower_climbs=max(0, self._current_state.red_tower_climbs + red_tower_climbs_delta),
                blue_tower_climbs=max(0, self._current_state.blue_tower_climbs + blue_tower_climbs_delta),
            )
            self._current_state = new_state
            self._history.append(new_state)

        await self._notify_subscribers_async(new_state)
        return new_state

    async def poll_score_stream(
        self,
        interval_s: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
        max_polls: Optional[int] = None,
    ) -> AsyncIterator[FieldAPIScoreState]:
        """Asynchronously streams polled score updates at the specified interval."""
        dt = interval_s if interval_s is not None else self.poll_interval_s
        count = 0
        while stop_event is None or not stop_event.is_set():
            if max_polls is not None and count >= max_polls:
                break
            state = await self.fetch_score_state()
            yield state
            count += 1
            await asyncio.sleep(dt)

    def compute_deltas(
        self,
        since_state: Optional[FieldAPIScoreState] = None,
    ) -> Dict[str, int]:
        """Computes score increments from a previous state to current state."""
        baseline = since_state or (self._history[0] if self._history else self._current_state)
        return {
            "red_fuel_delta": self._current_state.red_fuel_scored - baseline.red_fuel_scored,
            "blue_fuel_delta": self._current_state.blue_fuel_scored - baseline.blue_fuel_scored,
            "red_tower_climbs_delta": self._current_state.red_tower_climbs - baseline.red_tower_climbs,
            "blue_tower_climbs_delta": self._current_state.blue_tower_climbs - baseline.blue_tower_climbs,
        }

    def reset(
        self,
        initial_stage: MatchStage = MatchStage.PRE_MATCH,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """Resets the client state to zero scores."""
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        self._current_state = FieldAPIScoreState(
            timestamp_ms=ts,
            match_stage=initial_stage,
            red_fuel_scored=0,
            blue_fuel_scored=0,
            red_tower_climbs=0,
            blue_tower_climbs=0,
        )
        self._history = [self._current_state]


class FMSSimulator:
    """Simulator helper to orchestrate automated match score increments."""

    def __init__(self, client: FRCFieldAPIClient) -> None:
        self.client = client

    async def run_match_script(
        self,
        events: List[Dict[str, Any]],
        time_dilation: float = 1.0,
    ) -> None:
        """Executes a sequence of simulated match score events."""
        for event in events:
            delay = event.get("delay_s", 0.0) / max(1e-4, time_dilation)
            if delay > 0:
                await asyncio.sleep(delay)
            await self.client.increment_score(
                red_fuel_delta=event.get("red_fuel_delta", 0),
                blue_fuel_delta=event.get("blue_fuel_delta", 0),
                red_tower_climbs_delta=event.get("red_climbs_delta", 0),
                blue_tower_climbs_delta=event.get("blue_climbs_delta", 0),
                match_stage=event.get("match_stage"),
            )
