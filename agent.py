"""Hybrid Crane agent with an optional trained-policy upgrade.

The reliable baseline always remains available: it advances toward the enemy side,
then takes the legal one-step move that best closes on the nearest visible enemy.
When ``torch``, ``encoder.py``, ``imitation_model.py``, and
``imitation-policy.pt`` are supplied beside this file, the same agent instead lets
the supplied PyTorch policy choose a reachable destination and legal target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sandbox.crane import action, me, paths, tile, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


# The copied trained-policy agent relies on these project artifacts, which are not
# part of this repository yet. Keep the submission importable until they are added.
try:
    import torch

    from encoder import (
        describe_move,
        describe_target,
        encode_entities,
        move_options,
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
        """Choose a trained-policy order when possible, with a legal tactical fallback."""
        if self.model is not None:
            try:
                return self._policy_action(observation)
            except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as error:
                # If a model/checkpoint is incompatible with the match schema, retain
                # a valid move for this turn and disable repeated failing inferences.
                self.model = None
                self.policy_error = error

        return self._starter_action(observation)

    def _policy_action(self, observation: SkirmishObservation) -> SkirmishAction:
        """The destination-and-target inference from the supplied PyTorch agent."""
        if torch is None or self.model is None:
            raise RuntimeError("No trained policy is loaded")

        entities, enemy_indices = encode_entities(observation)
        entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
        context = self.model.encode(_option_tensor(entities), entity_mask)

        # A trap is legal to enter according to the engine's action mask, but every role
        # treats it as strategically impassable. Filter full paths, not just endpoints,
        # so cavalry cannot cross one on the way to an otherwise safe destination.
        moves = [move for move in move_options(observation) if self._path_avoids_traps(observation, move[0])]
        targets = target_options(observation)
        if not moves or not targets:
            raise ValueError("The observation must contain legal movement and target options")

        move_rows = _option_tensor([describe_move(observation, end) for _, end in moves])
        move_mask = torch.ones((1, len(moves)), dtype=torch.bool)
        move_scores = self.model.score_moves(context, entity_mask, move_rows, move_mask)
        path, end = moves[int(move_scores[0].argmax())]

        target_rows = _option_tensor(
            [describe_target(observation, end, target) for target in targets]
        )
        pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
        target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
        target_scores = self.model.score_targets(context, target_rows, pointers, target_mask)
        return {"path": path, "target": targets[int(target_scores[0].argmax())]}

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
