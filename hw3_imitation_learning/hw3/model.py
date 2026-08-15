"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, d_model: int = 128, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = [
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        return self.mlp(state).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred_action_chunk = self.forward(state)
        return nn.functional.mse_loss(pred_action_chunk, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    #   ee_xyz(3) | gripper(1) | cube_red_xyz(3) | cube_green_xyz(3) | cube_blue_xyz(3) | goal_onehot(3) | bin_pos(3)
    _ROBOT_DIM = 4    # ee_xyz + gripper
    _CUBE_DIM = 3     # per-cube xyz
    _N_CUBES = 3
    _GOAL_ONEHOT_DIM = 3
    _BIN_DIM = 3

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        depth: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        input_dim = (
            self._ROBOT_DIM
            + self._CUBE_DIM
            + self._CUBE_DIM * self._N_CUBES
            + 3
            + 3
            + self._BIN_DIM
            + self._GOAL_ONEHOT_DIM
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.goal_enc = nn.Sequential(
            nn.Linear(self._GOAL_ONEHOT_DIM + self._BIN_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        self.film_generators = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            ))
            self.film_generators.append(nn.Linear(d_model, d_model * 2))

        self.action_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, chunk_size * action_dim),
        )

    def _structure_input(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse raw state and build structured features + goal embedding.

        Returns (structured_input, goal_for_film).
        """
        robot = state[:, :self._ROBOT_DIM]
        ee_xyz = state[:, :3]

        cube_start = self._ROBOT_DIM
        cubes = state[:, cube_start:cube_start + self._CUBE_DIM * self._N_CUBES]
        cubes_3 = cubes.view(-1, self._N_CUBES, self._CUBE_DIM)

        goal_start = cube_start + self._CUBE_DIM * self._N_CUBES
        goal_onehot = state[:, goal_start:goal_start + self._GOAL_ONEHOT_DIM]
        bin_pos = state[:, goal_start + self._GOAL_ONEHOT_DIM:]

        target_selection = goal_onehot.unsqueeze(-1)
        target_cube = (cubes_3 * target_selection).sum(dim=1)

        mask = 1.0 - goal_onehot
        distractors = cubes_3 * mask.unsqueeze(-1)
        distractors_flat = distractors.view(-1, self._N_CUBES * self._CUBE_DIM)

        # inductive bias
        ee_to_target = target_cube - ee_xyz
        target_to_bin = bin_pos - target_cube

        structured = torch.cat([
            robot,
            target_cube,
            distractors_flat,
            ee_to_target,
            target_to_bin,
            bin_pos,
            goal_onehot,
        ], dim=-1)

        goal_info = torch.cat([goal_onehot, bin_pos], dim=-1)

        return structured, goal_info

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        structured, goal_info = self._structure_input(state)

        h = self.input_proj(structured)
        h_goal = self.goal_enc(goal_info)

        # FiLM-conditioned residual blocks
        for block, film_gen in zip(self.blocks, self.film_generators):
            film_params = film_gen(h_goal)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = gamma + 1.0  # init at 1 to avoid killing signal from the beginning
            residual = block(h)
            h = h + torch.relu(gamma * residual + beta)

        return self.action_head(h).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred_action_chunk = self.forward(state)
        return nn.functional.mse_loss(pred_action_chunk, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    # TODO,
    chunk_size: int = 16,
    d_model: int = 128,
    depth: int = 2,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=0.1,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
