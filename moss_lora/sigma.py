"""Training noise schedule.

MOSS-Image trains with a truncated logit-normal sigma distribution plus a uniform
mixture (technical report, eq. 4 / table 6): Z ~ N(mu, s^2), S = 1 - sigmoid(Z),
F(x) = 1 - Phi((log((1-x)/x) - mu) / s), discretised on the 1000-step grid.
"""

from __future__ import annotations

import math

import torch

TOTAL = 1000


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class SigmaSampler:
    def __init__(self, cfg, total: int = TOTAL, generator: torch.Generator | None = None):
        self.cfg = cfg
        self.total = total
        self.generator = generator
        self.table = None
        if cfg.mode == "logit_normal":
            self.table = self._build_table(cfg.mu, cfg.sigma, cfg.cutoff, cfg.uniform_mix, total)

    @staticmethod
    def _build_table(mu, sigma, cutoff, uniform_mix, total):
        a = cutoff if cutoff and 0 < cutoff <= 1 else 1.0
        cdf = torch.zeros(total + 1, dtype=torch.float64)
        for j in range(1, total + 1):
            x = a * j / total
            z = (math.log((1 - x) / x) - mu) / sigma
            cdf[j] = 1.0 - _phi(z)
        cdf = cdf / cdf[-1]
        probs = cdf[1:] - cdf[:-1]
        probs = (1.0 - uniform_mix) * probs + uniform_mix / total
        probs = probs / probs.sum()
        return torch.cumsum(probs, 0)

    def sample(self, n: int, device, generator=None) -> torch.Tensor:
        mode = self.cfg.mode
        generator = generator if generator is not None else self.generator
        if mode == "uniform":
            return torch.rand(n, generator=generator, device="cpu").to(device)
        if mode == "shifted":
            base = torch.rand(n, generator=generator, device="cpu").double()
            shift = self.cfg.shift
            values = shift * base / (1 + (shift - 1) * base)
            return values.float().to(device)
        # logit-normal on the discrete grid
        u = torch.rand(n, generator=generator, device="cpu").double()
        index = torch.searchsorted(self.table, u).clamp(1, self.total)
        return (index.double() / self.total).float().to(device)
