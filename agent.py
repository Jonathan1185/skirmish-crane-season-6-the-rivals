"""Hybrid Crane agent with an optional trained-policy upgrade.

The reliable baseline always remains available: it advances toward the enemy side,
then applies unit-specific archer, cavalry, and footman tactics.
When ``torch``, ``encoder.py``, ``imitation_model.py``, and
``imitation-policy.pt`` are supplied beside this file, the supplied PyTorch policy
also scores the legal target after that role-specific movement is fixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sandbox.crane import action, me, paths, tile, units, visible, zone
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


# The copied trained-policy agent relies on these project artifacts, which are not
# part of this repository yet. Keep the submission importable until they are added.
try:
    import torch

    from encoder import (
        describe_target,
        encode_entities,
        target_entity_indices,
        target_options,
    )
    from imitation_model import load_policy

    _POLICY_IMPORT_ERROR: Exception | None = None
    _POLICY_READY = True
except (ImportError, OSError) as error:
    torch = None  # type: ignore[assignment]
    _POLICY_IMPORT_ERROR = error
    _POLICY_READY = False


def _option_tensor(rows: list[list[float]]) -> Any:
    """Add the batch dimension expected by the copied policy network."""
    if torch is None:  # Defensive: this function is only used with a loaded policy.
        raise RuntimeError("The trained-policy dependencies are unavailable")
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


class Agent:
    """Use the trained policy when present, otherwise play a safe starter strategy."""

    # ``waste`` is the hazardous, damaging feature in the current Crane rules.  Including
    # ``trap`` also keeps the strategy safe if a map uses that more direct feature name.
    TRAP_FEATURES = frozenset({"trap", "waste"})

    def __init__(self) -> None:
        self.model: Any | None = None
        self.policy_error: Exception | None = _POLICY_IMPORT_ERROR
        policy_file = Path(__file__).with_name("imitation-policy.pt")

        if _POLICY_READY and policy_file.is_file():
            try:
                # The policy agent uses one CPU thread to keep per-turn inference fast.
                torch.set_num_threads(1)
                self.model = load_policy(policy_file)
            except (OSError, RuntimeError, ValueError) as error:
                # A failed optional model must not make the whole team unable to act.
                self.policy_error = error

    def reset(self, seed: int, observation: SkirmishObservation) -> None:
        """Called once before a match; this hybrid agent has no episode state."""

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        """Choose a role-specific, trap-safe movement plan for this unit."""
        unit_type = me.unit_type(observation)
        if unit_type == "archer":
            role_order = self._archer_order(observation)
        elif unit_type == "cavalry":
            role_order = self._cavalry_order(observation)
        else:
            role_order = self._footman_order(observation)

        if self.model is not None:
            try:
                return self._policy_target_for_path(observation, role_order["path"])
            except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as error:
                # Keep the deterministic role order if the optional checkpoint is incompatible.
                self.model = None
                self.policy_error = error
        return role_order

    def _policy_target_for_path(self, observation: SkirmishObservation, path_id: int) -> SkirmishAction:
        """Let the optional model name a target for a role-selected destination."""
        if torch is None or self.model is None:
            raise RuntimeError("No trained policy is loaded")

        entities, enemy_indices = encode_entities(observation)
        entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
        context = self.model.encode(_option_tensor(entities), entity_mask)

        targets = target_options(observation)
        if not targets:
            raise ValueError("The observation must contain legal target options")

        end = tile.at_path_end(me.position(observation), path_id)
        target_rows = _option_tensor(
            [describe_target(observation, end, target) for target in targets]
        )
        pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
        target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
        target_scores = self.model.score_targets(context, target_rows, pointers, target_mask)
        return {"path": path_id, "target": targets[int(target_scores[0].argmax())]}

    def _starter_action(self, observation: SkirmishObservation) -> SkirmishAction:
        """Original starter behavior, retained as the dependency-free fallback."""
        enemies = visible.enemies(observation)
        if not enemies:
            forward = me.direction(observation)
            if forward in action.legal_steps(observation) and self._path_avoids_traps(observation, forward):
                return action.move(forward)
            return action.stay()

        here = me.position(observation)
        nearest = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        step = self._step_toward(observation, nearest["position"])
        if step == 0:
            return action.stay(nearest["unit_id"], observation)
        return action.move(step, nearest["unit_id"], observation)

    def _archer_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Fire from range, retreating instead of trading blows at close quarters."""
        enemies = visible.enemies(observation)
        if not enemies:
            return self._move_toward(observation, self._strategic_goal(observation))

        here = me.position(observation)
        target = min(enemies, key=lambda enemy: (enemy["hit_points"], tile.distance(here, enemy["position"])))
        attack_range = units.STATS["archer"].attack_range
        preferred_range = max(2, attack_range - 1)

        def score(path_id: int) -> tuple[int, int, int]:
            distance = tile.distance(tile.at_path_end(here, path_id), target["position"])
            # Reaching firing range is essential; within it, preserve as much separation
            # as possible, with the preferred range breaking otherwise equal options.
            return (
                int(distance <= attack_range),
                -abs(distance - preferred_range),
                min(tile.distance(tile.at_path_end(here, path_id), enemy["position"]) for enemy in enemies),
            )

        return self._order_for_path(observation, max(self._safe_paths(observation), key=score), target["unit_id"])

    def _cavalry_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Seize capture zones first; otherwise approach enemies through a flank tile."""
        here = me.position(observation)
        enemies = visible.enemies(observation)
        target_id = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))["unit_id"] if enemies else None

        # Cavalry's speed is most valuable for claiming and keeping the nearest objective.
        areas = zone.zones(observation)
        if areas:
            if zone.at(observation, here) is not None:
                return action.stay(target_id, observation) if target_id else action.stay()
            goal = min(areas, key=lambda area: tile.distance(here, area["center"]))["center"]
            return self._move_toward(observation, goal, target_id)

        if not enemies:
            return self._move_toward(observation, tile.at_center(observation))

        target = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        allies = visible.allies(observation)
        flank_tiles = [
            position
            for position in tile.neighbors(target["position"]).values()
            if self._is_safe_tile(observation, position)
        ]
        if flank_tiles:
            # Prefer the side opposite nearby teammates, so cavalry does not pile into
            # the footman's lane and instead approaches around the enemy formation.
            goal = max(
                flank_tiles,
                key=lambda position: min(
                    (tile.distance(position, ally["position"]) for ally in allies), default=0
                ),
            )
            return self._move_toward(observation, goal, target["unit_id"])
        return self._move_toward(observation, target["position"], target["unit_id"])

    def _footman_order(self, observation: SkirmishObservation) -> SkirmishAction:
        """Hold objectives and screen the closest archer from approaching enemies."""
        here = me.position(observation)
        enemies = visible.enemies(observation)
        target_id = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))["unit_id"] if enemies else None

        # A footman already occupying a zone anchors it for the team.
        if zone.at(observation, here) is not None:
            return action.stay(target_id, observation) if target_id else action.stay()

        archers = [ally for ally in visible.allies(observation) if ally["type"] == "archer"]
        if archers and enemies:
            archer = min(archers, key=lambda ally: tile.distance(here, ally["position"]))
            threat = min(enemies, key=lambda enemy: tile.distance(enemy["position"], archer["position"]))

            def screen_score(path_id: int) -> tuple[int, int]:
                end = tile.at_path_end(here, path_id)
                # Stay close enough to protect the archer while moving into the threat's path.
                return (-tile.distance(end, archer["position"]), -tile.distance(end, threat["position"]))

            return self._order_for_path(observation, max(self._safe_paths(observation), key=screen_score), threat["unit_id"])

        areas = zone.zones(observation)
        if areas:
            goal = min(areas, key=lambda area: tile.distance(here, area["center"]))["center"]
            return self._move_toward(observation, goal, target_id)
        if enemies:
            return self._move_toward(observation, min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))["position"], target_id)
        return self._move_toward(observation, self._strategic_goal(observation))

    def _strategic_goal(self, observation: SkirmishObservation) -> AxialPosition:
        """Prefer the nearest zone, falling back to the field centre when none exist."""
        areas = zone.zones(observation)
        here = me.position(observation)
        return min(areas, key=lambda area: tile.distance(here, area["center"]))["center"] if areas else tile.at_center(observation)

    def _move_toward(
        self, observation: SkirmishObservation, goal: AxialPosition, target_id: str | None = None
    ) -> SkirmishAction:
        here = me.position(observation)
        path_id = min(
            self._safe_paths(observation),
            key=lambda path: (tile.distance(tile.at_path_end(here, path), goal), -len(paths.decode(path)), path),
        )
        return self._order_for_path(observation, path_id, target_id)

    @staticmethod
    def _order_for_path(
        observation: SkirmishObservation, path_id: int, target_id: str | None = None
    ) -> SkirmishAction:
        return action.move(path_id, target_id, observation) if target_id else action.move(path_id)

    def _safe_paths(self, observation: SkirmishObservation) -> list[int]:
        """Return legal paths that never enter a trap, always retaining a stay order."""
        return [path_id for path_id in action.legal_paths(observation) if self._path_avoids_traps(observation, path_id)]

    def _is_safe_tile(self, observation: SkirmishObservation, position: AxialPosition) -> bool:
        ground = tile.terrain_at(observation, position)
        return ground["terrain"] not in {"trap", "void", "water"} and ground["feature"] not in self.TRAP_FEATURES

    def _step_toward(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the legal single step that most closes the gap to ``goal``."""
        here = me.position(observation)
        best_step = 0
        best_distance = tile.distance(here, goal)
        for step in action.legal_steps(observation):
            if not self._path_avoids_traps(observation, step):
                continue
            step_distance = tile.distance(tile.at_path_end(here, step), goal)
            if step_distance < best_distance:
                best_step, best_distance = step, step_distance
        return best_step

    def _path_avoids_traps(self, observation: SkirmishObservation, path_id: int) -> bool:
        """Return whether every tile entered along a legal path is non-hazardous."""
        position = me.position(observation)
        for direction in paths.decode(path_id):
            position = tile.at_path_end(position, direction)
            ground = tile.terrain_at(observation, position)
            if ground["terrain"] == "trap" or ground["feature"] in self.TRAP_FEATURES:
                return False
        return True
