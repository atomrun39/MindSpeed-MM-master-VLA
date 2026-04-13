from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from mindspeed_mm.models.action.flow_matching_modules.action_encoder import ActionEncoder
from mindspeed_mm.models.action.flow_matching_modules.cross_attention_dit import DiT


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)
    return default


def _to_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "items"):
        return dict(config.items())
    # Support argparse/pydantic/custom config objects that expose attributes but no items()
    if hasattr(config, "__dict__"):
        return {
            k: v for k, v in vars(config).items()
            if not k.startswith("_")
        }
    return {}


def _pick_num_heads(inner_dim: int, preferred: int = 16) -> int:
    for heads in [preferred, 12, 10, 8, 6, 5, 4, 3, 2, 1]:
        if inner_dim % heads == 0:
            return heads
    return 1


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class FlowmatchingActionHead(nn.Module):
    def __init__(self, config, text_hidden_size: int):
        super().__init__()
        self.config = config
        self.action_dim = int(_cfg_get(config, "action_dim", 7))
        self.action_horizon = int(_cfg_get(config, "action_horizon", _cfg_get(config, "num_queries", 8)))
        self.state_dim = int(_cfg_get(config, "state_dim", 0) or 0)
        self.hidden_layout = str(_cfg_get(config, "hidden_layout", "sbh")).lower()
        self.num_inference_timesteps = int(_cfg_get(config, "num_inference_timesteps", 10))
        self.noise_beta_alpha = float(_cfg_get(config, "noise_beta_alpha", 1.5))
        self.noise_beta_beta = float(_cfg_get(config, "noise_beta_beta", 1.0))
        self.noise_s = float(_cfg_get(config, "noise_s", 0.999))
        self.num_timestep_buckets = int(_cfg_get(config, "num_timestep_buckets", 1000))
        self.input_embedding_dim = int(_cfg_get(config, "input_embedding_dim", _cfg_get(config, "hidden_size", text_hidden_size)))
        self.hidden_size = int(_cfg_get(config, "hidden_size", self.input_embedding_dim))
        self.add_pos_embed = bool(_cfg_get(config, "add_pos_embed", True))
        self.max_seq_len = int(_cfg_get(config, "max_seq_len", 1024))
        self.num_target_vision_tokens = int(_cfg_get(config, "num_target_vision_tokens", 32))

        diffusion_cfg = _to_dict(_cfg_get(config, "diffusion_model_cfg", {}))
        if "num_attention_heads" not in diffusion_cfg:
            diffusion_cfg["num_attention_heads"] = _pick_num_heads(self.input_embedding_dim, 16)
        if "attention_head_dim" not in diffusion_cfg:
            diffusion_cfg["attention_head_dim"] = self.input_embedding_dim // diffusion_cfg["num_attention_heads"]
        if "output_dim" not in diffusion_cfg:
            diffusion_cfg["output_dim"] = self.hidden_size
        if "num_layers" not in diffusion_cfg:
            diffusion_cfg["num_layers"] = 8
        if "dropout" not in diffusion_cfg:
            diffusion_cfg["dropout"] = 0.1
        if "cross_attention_dim" not in diffusion_cfg:
            diffusion_cfg["cross_attention_dim"] = text_hidden_size
        if "interleave_self_attention" not in diffusion_cfg:
            diffusion_cfg["interleave_self_attention"] = False

        self.model = DiT(**diffusion_cfg)
        use_vl_proj = bool(_cfg_get(config, "use_vl_proj", True))
        if (not use_vl_proj) or (text_hidden_size == self.model.inner_dim):
            self.vl_proj = nn.Identity()
        else:
            self.vl_proj = nn.Linear(text_hidden_size, self.model.inner_dim, bias=False)
        self.state_encoder = (
            MLP(self.state_dim, self.hidden_size, self.model.inner_dim)
            if self.state_dim > 0
            else None
        )
        self.action_decoder = MLP(self.hidden_size, self.hidden_size, self.action_dim)
        self.action_encoder = ActionEncoder(action_dim=self.action_dim, hidden_size=self.model.inner_dim)
        self.future_tokens = nn.Embedding(self.num_target_vision_tokens, self.model.inner_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(self.max_seq_len, self.model.inner_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None
        self.beta_dist = Beta(self.noise_beta_alpha, self.noise_beta_beta)

    def _to_batch_first(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states as 3D tensor, got shape {tuple(hidden_states.shape)}")
        if self.hidden_layout == "sbh":
            return hidden_states.transpose(0, 1).contiguous()
        return hidden_states

    def _prepare_state(self, state: Optional[torch.Tensor], dtype: torch.dtype, device: torch.device):
        if state is None or self.state_encoder is None:
            return None
        if state.ndim == 3:
            state = state[:, -1, :]
        return state.to(device=device, dtype=dtype)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def _build_sa_embs(
        self,
        actions: torch.Tensor,
        t_discretized: torch.Tensor,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, t_discretized)
        if self.position_embedding is not None:
            pos_ids = torch.arange(action_features.shape[1], device=action_features.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state).unsqueeze(1)
            return torch.cat([state_features, future_tokens, action_features], dim=1)
        return torch.cat([future_tokens, action_features], dim=1)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        repeated_diffusion_steps: int = 1,
    ) -> torch.Tensor:
        vl_embs = self.vl_proj(self._to_batch_first(hidden_states))
        actions = actions.to(device=vl_embs.device, dtype=vl_embs.dtype)
        if repeated_diffusion_steps > 1:
            actions = actions.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs = vl_embs.repeat(repeated_diffusion_steps, 1, 1)
            if state is not None:
                state = state.repeat(repeated_diffusion_steps, 1, 1) if state.ndim == 3 else state.repeat(repeated_diffusion_steps, 1)
        state = self._prepare_state(state, dtype=vl_embs.dtype, device=vl_embs.device)
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], vl_embs.device, actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        sa_embs = self._build_sa_embs(noisy_trajectory, t_discretized, vl_embs, state=state)
        model_out = self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=t_discretized)
        pred = self.action_decoder(model_out)
        pred_actions = pred[:, -actions.shape[1]:]
        return F.mse_loss(pred_actions, velocity)

    @torch.no_grad()
    def predict_action(self, hidden_states: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        vl_embs = self.vl_proj(self._to_batch_first(hidden_states))
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        dtype = vl_embs.dtype
        state = self._prepare_state(state, dtype=dtype, device=device)
        actions = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device, dtype=dtype)
        steps = max(1, self.num_inference_timesteps)
        dt = 1.0 / steps
        for step in range(steps):
            t_cont = step / float(steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((batch_size,), t_discretized, device=device, dtype=torch.long)
            sa_embs = self._build_sa_embs(actions, timesteps_tensor, vl_embs, state=state)
            model_out = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_out)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity
        return actions

    def forward(self, hidden_states: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.predict_action(hidden_states=hidden_states, state=state)
