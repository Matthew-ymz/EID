"""Optional shared conditional neural TM and common-intervention marginalization.

PyTorch and nflows are imported only when this backend is selected. Experiment
dependency discovery and domain feature dictionaries stay with the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from .core import positive_integer, samples
from .transport_map import fit_polynomial_triangular_transport_map_density


def _torch():
    try:
        import nflows  # noqa: F401
        import torch
    except ImportError as exc:
        raise ImportError(
            "tm_neural requires optional dependencies: pip install 'eid-toolbox[neural]'."
        ) from exc
    return torch


def _context(transform, values):
    result = samples(values if transform is None else transform(values), "context_transform output")
    if len(result) != len(values):
        raise ValueError("context_transform must preserve the sample axis.")
    return result


def _sobol(intervention, count, seed):
    d = getattr(intervention, "sampling_dimension", intervention.dimension)
    u = qmc.Sobol(d, scramble=True, seed=seed).random_base2(math.ceil(math.log2(count)))[:count]
    return intervention.from_unit(u)


class ConditionalNeuralTM:
    def __init__(
        self,
        x,
        y,
        *,
        context_transform=None,
        epochs=100,
        hidden_features=48,
        layers=2,
        batch_size=256,
        learning_rate=8e-4,
        weight_decay=1e-6,
        train_fraction=0.8,
        seed=0,
        target_min_scale=1e-6,
        context_selection=True,
        threads=2,
    ):
        torch = _torch()
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        for name, value in [
            ("epochs", epochs),
            ("hidden_features", hidden_features),
            ("layers", layers),
            ("batch_size", batch_size),
            ("threads", threads),
        ]:
            positive_integer(value, name)
        for name, value in [("learning_rate", learning_rate), ("target_min_scale", target_min_scale)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
            raise ValueError("Neural TM train_fraction must be between zero and one.")
        if not isinstance(context_selection, (bool, np.bool_)):
            raise TypeError("context_selection must be a boolean.")
        split = int(len(y) * train_fraction)
        if split < 4 or len(y) - split < 4:
            raise ValueError("Neural TM needs at least four training and four validation samples.")
        self.context_transform = context_transform
        cx = _context(context_transform, x)
        self.context_dimension = cx.shape[1]
        self.mean = y[:split].mean(axis=0)
        self.scale = np.maximum(y[:split].std(axis=0), target_min_scale)
        standardized = (y - self.mean) / self.scale
        ct, yt = torch.as_tensor(cx, dtype=torch.float32), torch.as_tensor(standardized, dtype=torch.float32)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                transforms = []
                for _ in range(layers):
                    transforms.extend(
                        [
                            MaskedAffineAutoregressiveTransform(
                                features=y.shape[1],
                                hidden_features=hidden_features,
                                context_features=cx.shape[1],
                                num_blocks=2,
                                use_residual_blocks=True,
                                random_mask=False,
                            ),
                            ReversePermutation(features=y.shape[1]),
                        ]
                    )
                self.flow = Flow(CompositeTransform(transforms), StandardNormal([y.shape[1]]))
                optimizer = torch.optim.Adam(
                    self.flow.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                rng = np.random.default_rng(seed + 71)
                best, state, best_epoch = math.inf, None, 0
                for epoch in range(epochs):
                    order = rng.permutation(split)
                    self.flow.train()
                    for start in range(0, split, batch_size):
                        idx = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
                        loss = -self.flow.log_prob(yt[idx], context=ct[idx]).mean()
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite neural TM training loss.")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    self.flow.eval()
                    with torch.no_grad():
                        validation = float(-self.flow.log_prob(yt[split:], context=ct[split:]).mean())
                    if math.isfinite(validation) and validation < best:
                        best, best_epoch = validation, epoch + 1
                        state = {k: v.detach().clone() for k, v in self.flow.state_dict().items()}
                if state is None:
                    raise ValueError("Neural TM did not produce a finite validation likelihood.")
                self.flow.load_state_dict(state)
                self.flow.eval()
                with torch.no_grad():
                    conditional_log = self.flow.log_prob(yt[split:], context=ct[split:]).numpy()
        finally:
            torch.set_num_threads(previous_threads)
        # The manuscript's context gate selects a channel, not a projection of Syn.
        baseline = fit_polynomial_triangular_transport_map_density(standardized[:split], degree=3, ridge=0.05)
        gains = conditional_log - baseline.log_prob(standardized[split:])
        sem = float(gains.std(ddof=1) / np.sqrt(len(gains)))
        self.active = bool(gains.mean() > 2 * sem) if context_selection else True
        self.diagnostics = {
            "train_count": split,
            "validation_count": len(y) - split,
            "best_epoch": best_epoch,
            "heldout_conditional_nll_nats": best,
            "conditional_gain_nats": float(gains.mean()),
            "conditional_gain_sem_nats": sem,
            "context_active": self.active,
            "context_selection": "two paired SEM against degree-3 target density"
            if context_selection
            else "disabled",
            "context_dimension": cx.shape[1],
            "target_dimension": y.shape[1],
            "target_scaling": "per-dimension training statistics",
            "target_min_scale": target_min_scale,
            "architecture": "conditional masked affine autoregressive flow",
            "layers": layers,
            "hidden_features": hidden_features,
            "epochs": epochs,
            "seed": seed,
        }

    def log_prob(self, y, x):
        torch = _torch()
        cx = _context(self.context_transform, x)
        if cx.shape[1] != self.context_dimension:
            raise ValueError("context_transform changed its output dimension after fitting.")
        with torch.no_grad():
            result = self.flow.log_prob(
                torch.as_tensor((y - self.mean) / self.scale, dtype=torch.float32),
                context=torch.as_tensor(cx, dtype=torch.float32),
            ).numpy()
        return result.astype(float) - np.log(self.scale).sum()

    def sample(self, x, seed):
        torch = _torch()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            y = (
                self.flow.sample(
                    1, context=torch.as_tensor(_context(self.context_transform, x), dtype=torch.float32)
                )
                .squeeze(1)
                .numpy()
            )
        return y * self.scale + self.mean


class NeuralEIOracle:
    """Every subset marginal comes from the same fitted conditional channel.

    components is an optional list of {'sources': group IDs, 'targets': column
    IDs, 'context_transform': callable}. It declares a factorized learned-channel
    approximation; independence/component discovery is never inferred here.
    """

    def __init__(
        self,
        x,
        y,
        intervention,
        groups,
        *,
        evaluation_count=512,
        marginal_samples=256,
        integration_batch_size=16384,
        components=None,
        seed=0,
        context_transform=None,
        **fit_options,
    ):
        for name, value in [
            ("evaluation_count", evaluation_count),
            ("marginal_samples", marginal_samples),
            ("integration_batch_size", integration_batch_size),
        ]:
            positive_integer(value, name, 4 if name == "evaluation_count" else 1)
        definitions = (
            list(components)
            if components is not None
            else [
                {
                    "sources": list(range(len(groups))),
                    "targets": list(range(y.shape[1])),
                    "context_transform": context_transform,
                }
            ]
        )
        if components is not None and context_transform is not None:
            raise ValueError("Set context_transform on individual neural components.")
        if not definitions or any(
            not isinstance(c, Mapping)
            or not {"sources", "targets"} <= c.keys()
            or c.keys() - {"sources", "targets", "context_transform"}
            for c in definitions
        ):
            raise ValueError(
                "Each neural component needs sources, targets, and optionally context_transform."
            )
        for field, expected in [("sources", len(groups)), ("targets", y.shape[1])]:
            values = [i for c in definitions for i in c[field]]
            if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in values) or sorted(
                values
            ) != list(range(expected)):
                raise ValueError(
                    f"Neural component {field} must partition all registered {field} exactly once."
                )
        self.parts, self.cache, self.estimate_diagnostics = [], {}, {}
        evaluation_x = _sobol(intervention, evaluation_count, seed + 6000)
        base = _sobol(intervention, evaluation_count * marginal_samples, seed + 6001)
        self.integration_batch_size = integration_batch_size
        self.marginal_samples = marginal_samples
        for index, definition in enumerate(definitions):
            sg, tg = tuple(definition["sources"]), tuple(definition["targets"])
            if not sg or not tg:
                raise ValueError("Neural components must have nonempty source and target groups.")
            columns = [i for g in sg for i in groups[g]]
            channel = ConditionalNeuralTM(
                x[:, columns],
                y[:, tg],
                seed=seed + 5000 + index,
                context_transform=definition.get("context_transform"),
                **fit_options,
            )
            ex = evaluation_x[:, columns]
            ey = channel.sample(ex, seed + 6002 + index) if channel.active else None
            local_base = base[:, columns].reshape(evaluation_count, marginal_samples, len(columns))
            positions = {g: tuple(columns.index(i) for i in groups[g]) for g in sg}
            part = {"channel": channel, "x": ex, "y": ey, "base": local_base, "positions": positions}
            part["log_py"] = self._marginal(part, ()) if channel.active else None
            self.parts.append(part)
        self.metadata = {
            "evaluation_count": evaluation_count,
            "marginal_samples": marginal_samples,
            "integration": "common scrambled Sobol intervention draws",
            "evaluation_distribution": "declared intervention times fitted conditional neural channel",
            "resampling_bias_correction": "none",
            "component_factorization": components is not None,
            "component_definitions": definitions,
            "fits": [p["channel"].diagnostics for p in self.parts],
        }

    def _marginal(self, part, group_ids):
        channel, x, y = part["channel"], part["x"], part["y"]
        columns = [i for g in group_ids for i in part["positions"][g]]
        if len(columns) == x.shape[1]:
            return channel.log_prob(y, x)
        chunk = max(1, self.integration_batch_size // self.marginal_samples)
        rows = []
        for start in range(0, len(x), chunk):
            stop = min(start + chunk, len(x))
            completed = part["base"][start:stop].copy()
            if columns:
                completed[:, :, columns] = x[start:stop, None, columns]
            tiled_y = np.repeat(y[start:stop], self.marginal_samples, axis=0)
            logs = channel.log_prob(tiled_y, completed.reshape(-1, x.shape[1])).reshape(
                -1, self.marginal_samples
            )
            rows.append(logsumexp(logs, axis=1) - np.log(self.marginal_samples))
        return np.concatenate(rows)

    def ei(self, coalition):
        key = tuple(coalition)
        if key not in self.cache:
            terms = np.zeros(len(self.parts[0]["x"]))
            for part in self.parts:
                selected = tuple(g for g in key if g in part["positions"])
                if selected and part["channel"].active:
                    terms += self._marginal(part, selected) - part["log_py"]
            if not np.isfinite(terms).all():
                raise ValueError("Non-finite neural marginal information estimate.")
            self.cache[key] = float(terms.mean())
            self.estimate_diagnostics[key] = {
                "evaluation_count": len(terms),
                "pointwise_sem_nats": float(terms.std(ddof=1) / np.sqrt(len(terms))),
            }
        return self.cache[key]
