from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _make_activation(name: str) -> nn.Module:
    activation_name = name.lower()
    if activation_name == "elu":
        return nn.ELU(inplace=True)
    if activation_name == "relu":
        return nn.ReLU(inplace=True)
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str = "elu") -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_make_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def build_amp_state(root_pos: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
    if root_pos.shape[-1] >= 3:
        base_height = root_pos[..., 2:3]
    else:
        base_height = root_pos[..., :1]
    return torch.cat((base_height, dof_pos, dof_vel), dim=-1)


def build_amp_transition(current_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    return torch.cat((current_state, next_state), dim=-1)


class AMPDiscriminator(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (512, 256, 128), activation: str = "elu"):
        super().__init__()
        self.state_dim = int(state_dim)
        self.network = build_mlp(self.state_dim * 2, hidden_dims, 1, activation=activation)

    def forward(self, transition: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(transition, (tuple, list)):
            transition = build_amp_transition(transition[0], transition[1])
        logits = self.network(transition)
        return logits.squeeze(-1)

    def score(self, transition: torch.Tensor | Sequence[torch.Tensor], detach: bool = False) -> torch.Tensor:
        score = torch.sigmoid(self.forward(transition))
        return score.detach() if detach else score


@dataclass
class AmpSampleBatch:
    current_state: torch.Tensor
    next_state: torch.Tensor

    @property
    def transition(self) -> torch.Tensor:
        return build_amp_transition(self.current_state, self.next_state)


class AmpExpertBuffer:
    def __init__(
        self,
        motion_file: str | Path,
        motion_fps: float | None = None,
        device: str | torch.device | None = None,
    ):
        self.device = _as_device(device)
        self.motion_file = Path(motion_file)
        if not self.motion_file.is_file():
            raise FileNotFoundError(f"AMP reference file not found: {self.motion_file}")

        loaded = np.load(self.motion_file, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        elif hasattr(loaded, "item") and not isinstance(loaded, dict):
            loaded = loaded.item()

        self._data = loaded
        self.motion_fps = float(loaded.get("fps", motion_fps if motion_fps is not None else 0.0))
        if self.motion_fps <= 0.0:
            raise ValueError("AMP reference data must provide fps or motion_fps must be specified.")

        dof_pos_key = "dof_pos" if "dof_pos" in loaded else "joint_pos"
        dof_vel_key = "dof_vel" if "dof_vel" in loaded else "joint_vel"
        root_pos_key = "root_pos" if "root_pos" in loaded else "body_pos_w"
        root_rot_key = "root_rot" if "root_rot" in loaded else "body_quat_w"

        self.dof_pos = torch.as_tensor(loaded[dof_pos_key], dtype=torch.float32, device=self.device)
        self.dof_vel = torch.as_tensor(loaded[dof_vel_key], dtype=torch.float32, device=self.device)
        
        root_pos_tensor = torch.as_tensor(loaded[root_pos_key], dtype=torch.float32, device=self.device)
        root_rot_tensor = torch.as_tensor(loaded[root_rot_key], dtype=torch.float32, device=self.device)
        
        # If it contains all bodies (frames, num_bodies, 3), extract only the root (index 0)
        if root_pos_tensor.ndim == 3:
            root_pos_tensor = root_pos_tensor[:, 0, :]
        if root_rot_tensor.ndim == 3:
            root_rot_tensor = root_rot_tensor[:, 0, :]
            
        self.root_pos = root_pos_tensor
        self.root_rot = root_rot_tensor

        if self.dof_pos.ndim != 2 or self.dof_vel.ndim != 2:
            raise ValueError("dof_pos and dof_vel must have shape [frames, dof].")
        if self.dof_pos.shape != self.dof_vel.shape:
            raise ValueError("dof_pos and dof_vel must have the same shape.")
        if self.root_pos.shape[0] != self.dof_pos.shape[0]:
            raise ValueError("root_pos must have the same frame count as dof_pos.")

        self.num_frames = int(self.dof_pos.shape[0])
        if self.num_frames < 2:
            raise ValueError("AMP reference data must contain at least two frames.")

        self.state_dim = int(self.dof_pos.shape[1] * 2 + 1)
        self.transition_dim = self.state_dim * 2

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.motion_fps

    def to(self, device: str | torch.device) -> "AmpExpertBuffer":
        device = _as_device(device)
        self.device = device
        self.dof_pos = self.dof_pos.to(device)
        self.dof_vel = self.dof_vel.to(device)
        self.root_pos = self.root_pos.to(device)
        self.root_rot = self.root_rot.to(device)
        return self

    def _frame_at(self, frame_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx0 = torch.floor(frame_index).long().clamp(0, self.num_frames - 1)
        idx1 = (idx0 + 1).clamp(max=self.num_frames - 1)
        blend = (frame_index - idx0.to(frame_index.dtype)).unsqueeze(-1)
        root_pos = self.root_pos[idx0] * (1.0 - blend) + self.root_pos[idx1] * blend
        dof_pos = self.dof_pos[idx0] * (1.0 - blend) + self.dof_pos[idx1] * blend
        dof_vel = self.dof_vel[idx0] * (1.0 - blend) + self.dof_vel[idx1] * blend
        return root_pos, dof_pos, dof_vel

    def state_at_time(self, time_s: torch.Tensor) -> torch.Tensor:
        frame_index = torch.clamp(time_s, 0.0, self.duration) * self.motion_fps
        root_pos, dof_pos, dof_vel = self._frame_at(frame_index)
        return build_amp_state(root_pos, dof_pos, dof_vel)

    def transition_at_time(self, time_s: torch.Tensor, step_dt: float) -> AmpSampleBatch:
        current_state = self.state_at_time(time_s)
        next_state = self.state_at_time(time_s + step_dt)
        return AmpSampleBatch(current_state=current_state, next_state=next_state)

    def sample(self, batch_size: int, step_dt: float | None = None) -> AmpSampleBatch:
        if step_dt is None:
            step_dt = 1.0 / self.motion_fps
        max_start = max(self.duration - step_dt, 0.0)
        time_s = torch.rand(batch_size, device=self.device) * max_start
        return self.transition_at_time(time_s, step_dt)

    def sample_transition(self, batch_size: int, step_dt: float | None = None) -> torch.Tensor:
        return self.sample(batch_size, step_dt=step_dt).transition


def amp_discriminator_loss(
    discriminator: AMPDiscriminator,
    expert_transition: torch.Tensor | Sequence[torch.Tensor],
    policy_transition: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    expert_logits = discriminator(expert_transition)
    policy_logits = discriminator(policy_transition)
    expert_labels = torch.ones_like(expert_logits)
    policy_labels = torch.zeros_like(policy_logits)
    logits = torch.cat((expert_logits, policy_logits), dim=0)
    labels = torch.cat((expert_labels, policy_labels), dim=0)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def amp_style_reward(
    discriminator: AMPDiscriminator,
    current_state: torch.Tensor | Sequence[torch.Tensor],
    next_state: torch.Tensor | None = None,
) -> torch.Tensor:
    if next_state is None:
        transition = current_state
    else:
        transition = build_amp_transition(current_state, next_state)
    with torch.no_grad():
        style_score = discriminator.score(transition, detach=False)
    return torch.exp(-2.0 * torch.square(style_score - 1.0))


def amp_style_reward_term(env, asset_cfg=None):
    """Reward term callable for integration with isaaclab RewardTermCfg.

    This function builds the AMP state vector for the robot (base height, joint pos, joint vel),
    checks dimensionality for 29-DoF (will raise if mismatch), and computes style reward using
    a discriminator attached on the env as `env.amp_discriminator`.

    It also keeps a short FIFO of recent policy transitions in `env.amp_recent_transitions` so
    the training loop can sample policy transitions for discriminator updates.
    """
    # lazy imports / type agnostic
    try:
        from isaaclab.managers import SceneEntityCfg
    except Exception:
        SceneEntityCfg = None

    asset = env.scene[asset_cfg.name] if asset_cfg is not None else env.scene["robot"]

    # construct current state: base height (z), joint positions, joint velocities
    root_pos = asset.data.root_pos_w[:, :3]
    base_height = root_pos[:, 2:3]
    dof_pos = asset.data.joint_pos
    dof_vel = asset.data.joint_vel

    # dimension checks
    num_joints = dof_pos.shape[1]
    if num_joints != 29:
        raise RuntimeError(f"AMP expects 29 joint DOF for G1, found {num_joints}.")

    current_state = torch.cat((base_height, dof_pos, dof_vel), dim=-1)

    # initialize env buffers if needed
    if not hasattr(env, "amp_prev_state") or env.amp_prev_state is None:
        env.amp_prev_state = current_state.detach().cpu()
        # init recent transitions buffer
        env.amp_recent_transitions = []
        # return zero reward on first call
        return torch.zeros(env.num_envs, device=env.device)

    prev_state = env.amp_prev_state.to(current_state.device)

    # build transition tensor
    transition = build_amp_transition(prev_state.to(current_state.device), current_state)

    # push to recent transitions buffer (store on CPU to avoid GPU memory growth)
    try:
        cpu_transition = transition.detach().cpu()
        if not hasattr(env, "amp_recent_transitions") or env.amp_recent_transitions is None:
            env.amp_recent_transitions = []
        env.amp_recent_transitions.append(cpu_transition)
        
        # Calculate dynamic buffer cap to avoid massive memory leaks
        # E.g. limit to around 256,000 transitions total across all envs
        num_envs_current = getattr(env, "num_envs", 1)
        max_buffer_len = max(1, 256000 // num_envs_current)
        
        # keep reasonable cap
        if len(env.amp_recent_transitions) > max_buffer_len:
            env.amp_recent_transitions.pop(0)
    except Exception:
        # best-effort, ignore buffer if push fails
        pass

    # compute style reward using attached discriminator if available
    if not hasattr(env, "amp_discriminator") or env.amp_discriminator is None:
        env.amp_prev_state = current_state.detach().cpu()
        return torch.zeros(env.num_envs, device=env.device)

    disc = env.amp_discriminator
    # move transition to discriminator device
    device = next(disc.parameters()).device if any(True for _ in disc.parameters()) else env.device
    transition_dev = transition.to(device)
    with torch.no_grad():
        score = torch.sigmoid(disc(transition_dev))
        style_reward = torch.exp(-2.0 * torch.square(score - 1.0))

    # scale by env.amp_style_scale if present, else default to 0.0 to avoid accidental reward injection
    scale = getattr(env, "amp_style_scale", 0.0)
    style_reward = style_reward * float(scale)

    # update prev_state
    env.amp_prev_state = current_state.detach().cpu()

    return style_reward.to(env.device).view(-1)
