"""A tiny TRM-style latent reasoning policy/value model.

The model is intentionally implemented with NumPy rather than requiring
PyTorch.  It has a shared latent transition unrolled for several reasoning
steps before producing its policy and value predictions.  The training method
implements the corresponding policy-gradient plus value-regression backward
pass, and Adam keeps it stable on a laptop CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelConfig:
    input_size: int = 16
    hidden_size: int = 64
    action_size: int = 4
    reasoning_steps: int = 4
    learning_rate: float = 0.002
    value_loss_weight: float = 0.45
    gradient_clip: float = 2.0
    seed: int = 7


class LatentReasoningPolicy:
    """Shared-latent recurrent policy/value network.

    Given a board observation ``x`` this computes an embedding ``h``, then
    repeatedly applies ``z <- tanh([z, h] W_rec + b_rec)``.  Reusing the same
    transition at each step is the useful TRM property here: the agent spends
    a few internal iterations refining a state before acting.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        if self.config.reasoning_steps < 1:
            raise ValueError("reasoning_steps must be at least 1")
        self.rng = np.random.default_rng(self.config.seed)
        i, h, a = self.config.input_size, self.config.hidden_size, self.config.action_size
        self.params: dict[str, np.ndarray] = {
            "w_in": self._xavier(i, h),
            "b_in": np.zeros(h, dtype=np.float32),
            "w_init": self._xavier(h, h),
            "b_init": np.zeros(h, dtype=np.float32),
            "w_rec": self._xavier(h * 2, h),
            "b_rec": np.zeros(h, dtype=np.float32),
            "w_policy": self._xavier(h, a),
            "b_policy": np.zeros(a, dtype=np.float32),
            "w_value": self._xavier(h, 1),
            "b_value": np.zeros(1, dtype=np.float32),
        }
        self._m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._step = 0

    def _xavier(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

    def predict(self, observations: np.ndarray | list[list[float]] | list[float]) -> tuple[np.ndarray, np.ndarray]:
        probs, values, _ = self._forward(self._batch(observations), with_cache=False)
        return probs, values[:, 0]

    def latent(self, observation: np.ndarray | list[float]) -> np.ndarray:
        _, _, cache = self._forward(self._batch(observation), with_cache=True)
        return cache["z"][-1][0].copy()

    def train_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        *,
        advantages: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Run one policy-gradient update and return meaningful diagnostics."""
        x = self._batch(observations)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        returns = np.asarray(returns, dtype=np.float32).reshape(-1)
        if len(x) != len(actions) or len(x) != len(returns) or not len(x):
            raise ValueError("observations, actions and returns must have the same non-zero length")
        probs, values, cache = self._forward(x, with_cache=True)
        values_flat = values[:, 0]
        raw_advantages = returns - values_flat
        adv = np.asarray(advantages, dtype=np.float32).reshape(-1) if advantages is not None else raw_advantages
        if len(adv) != len(x):
            raise ValueError("advantages must match the batch length")
        # Normalizing advantages is particularly helpful while the value head
        # is warming up.  Keep a tiny floor to avoid a zero update on repeats.
        adv_mean = float(np.mean(adv))
        adv_std = float(np.std(adv))
        if adv_std > 1e-6:
            adv = (adv - adv_mean) / (adv_std + 1e-8)

        n = float(len(x))
        chosen = np.clip(probs[np.arange(len(x)), actions], 1e-8, 1.0)
        policy_loss = float(np.mean(-np.log(chosen) * adv))
        value_error = values_flat - returns
        value_loss = float(np.mean(value_error * value_error) * 0.5)
        entropy = float(np.mean(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)))

        dlogits = probs.copy()
        dlogits[np.arange(len(x)), actions] -= 1.0
        dlogits *= (adv / n)[:, None]
        dvalue = (value_error * self.config.value_loss_weight / n)[:, None]
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        z_final = cache["z"][-1]
        grads["w_policy"] = z_final.T @ dlogits
        grads["b_policy"] = np.sum(dlogits, axis=0)
        grads["w_value"] = z_final.T @ dvalue
        grads["b_value"] = np.sum(dvalue, axis=0)
        dz = dlogits @ self.params["w_policy"].T + dvalue @ self.params["w_value"].T
        dh = np.zeros_like(cache["h"])
        # z[0] is the initial latent; z[t+1] is the result of transition t.
        for t in range(self.config.reasoning_steps - 1, -1, -1):
            z_next = cache["z"][t + 1]
            q = cache["q"][t]
            da = dz * (1.0 - z_next * z_next)
            grads["w_rec"] += q.T @ da
            grads["b_rec"] += np.sum(da, axis=0)
            dq = da @ self.params["w_rec"].T
            dz = dq[:, : self.config.hidden_size]
            dh += dq[:, self.config.hidden_size :]
        z_initial = cache["z"][0]
        da_initial = dz * (1.0 - z_initial * z_initial)
        grads["w_init"] = cache["h"].T @ da_initial
        grads["b_init"] = np.sum(da_initial, axis=0)
        dh += da_initial @ self.params["w_init"].T
        da_h = dh * (1.0 - cache["h"] * cache["h"])
        grads["w_in"] = x.T @ da_h
        grads["b_in"] = np.sum(da_h, axis=0)
        grad_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
        if grad_norm > self.config.gradient_clip:
            scale = self.config.gradient_clip / (grad_norm + 1e-8)
            for name in grads:
                grads[name] *= scale
        self._adam_step(grads)
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "mean_value": float(np.mean(values_flat)),
            "mean_return": float(np.mean(returns)),
            "advantage_std": adv_std,
            "gradient_norm": grad_norm,
        }

    def _adam_step(self, grads: dict[str, np.ndarray]) -> None:
        self._step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, gradient in grads.items():
            self._m[name] = beta1 * self._m[name] + (1.0 - beta1) * gradient
            self._v[name] = beta2 * self._v[name] + (1.0 - beta2) * (gradient * gradient)
            m_hat = self._m[name] / (1.0 - beta1**self._step)
            v_hat = self._v[name] / (1.0 - beta2**self._step)
            self.params[name] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def _forward(self, x: np.ndarray, with_cache: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        p = self.params
        h = np.tanh(x @ p["w_in"] + p["b_in"])
        z_initial = np.tanh(h @ p["w_init"] + p["b_init"])
        z_values = [z_initial]
        q_values: list[np.ndarray] = []
        z = z_initial
        for _ in range(self.config.reasoning_steps):
            q = np.concatenate((z, h), axis=1)
            z = np.tanh(q @ p["w_rec"] + p["b_rec"])
            q_values.append(q)
            z_values.append(z)
        logits = z @ p["w_policy"] + p["b_policy"]
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits, -40.0, 40.0))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        values = z @ p["w_value"] + p["b_value"]
        cache = {"h": h, "z": z_values, "q": q_values} if with_cache else {}
        return probs, values, cache

    def _batch(self, observations: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
        x = np.asarray(observations, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.config.input_size:
            raise ValueError(f"expected observations shaped (batch, {self.config.input_size}), got {x.shape}")
        return x

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "step": self._step,
            "params": {name: value.copy() for name, value in self.params.items()},
            "adam_m": {name: value.copy() for name, value in self._m.items()},
            "adam_v": {name: value.copy() for name, value in self._v.items()},
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        state = self.state_dict()
        arrays: dict[str, np.ndarray] = {f"param::{k}": v for k, v in state["params"].items()}
        arrays.update({f"adam_m::{k}": v for k, v in state["adam_m"].items()})
        arrays.update({f"adam_v::{k}": v for k, v in state["adam_v"].items()})
        arrays["metadata"] = np.asarray(json.dumps({"config": state["config"], "step": state["step"]}))
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LatentReasoningPolicy":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            model = cls(ModelConfig(**metadata["config"]))
            for name in model.params:
                model.params[name][...] = data[f"param::{name}"]
                model._m[name][...] = data[f"adam_m::{name}"]
                model._v[name][...] = data[f"adam_v::{name}"]
            model._step = int(metadata.get("step", 0))
        return model

