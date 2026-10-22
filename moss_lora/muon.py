"""Muon optimiser (momentum + Newton-Schulz orthogonalisation) for LoRA matrices.

The reference pretraining uses Muon for the 2-D matrices; here it only ever sees the
LoRA A/B factors, so the orthogonalisation is cheap. Memory is one momentum buffer
per parameter instead of AdamW's two moments.
"""

from __future__ import annotations

import torch


def _ns5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration approximating the orthogonal polar factor."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + b * (xx @ x) + c * (xx @ xx @ x)
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(param)
                buf = state["momentum"]
                buf.mul_(mu).add_(grad)
                update = grad.add(buf, alpha=mu) if group["nesterov"] else buf
                if param.ndim == 2 and min(param.shape) > 1:
                    update = _ns5(update, group["ns_steps"])
                    # match AdamW-style update magnitude across shapes
                    update = update * (0.2 * max(param.shape) ** 0.5)
                else:
                    update = update / (update.norm() + 1e-7) * (0.2 * max(param.shape) ** 0.5)
                if group["weight_decay"]:
                    param.mul_(1 - lr * group["weight_decay"])
                param.add_(update, alpha=-lr)
        return loss
