"""Clean-room lightweight forecasting backbones for retrieval-augmented TSAD.

All backbones expose the same interface::

    latent = model.encode(x)       # x: [B, L, C], latent: [B, C, D]
    forecast = model.decode(latent)  # [B, H, C]
    forecast = model(x)

The implementations intentionally depend only on PyTorch.  They are compact
surrogates for three broad model families rather than copies of any external
project implementation.
"""

from __future__ import annotations

import copy
import math
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class ForecastBackbone(nn.Module):
    """Base class enforcing the common multivariate forecasting contract."""

    def __init__(
        self,
        channels: int,
        context: int,
        horizon: int,
        d_model: int,
        revin: bool = True,
    ) -> None:
        super().__init__()
        if min(channels, context, horizon, d_model) <= 0:
            raise ValueError("channels, context, horizon, and d_model must be positive")
        self.channels = int(channels)
        self.context = int(context)
        self.horizon = int(horizon)
        self.d_model = int(d_model)
        self.revin = bool(revin)
        self.revin_eps = 1e-5
        # A shared channel-independent decoder keeps the small models comparable.
        self.forecast_head = nn.Linear(self.d_model, self.horizon)

    def _check_input(self, x: Tensor) -> tuple[int, int, int]:
        if x.ndim != 3:
            raise ValueError(f"expected x with shape [B, L, C], got {tuple(x.shape)}")
        batch, length, channels = x.shape
        if length != self.context:
            raise ValueError(f"expected context length {self.context}, got {length}")
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        return batch, length, channels

    def _check_latent(self, latent: Tensor) -> tuple[int, int, int]:
        if latent.ndim != 3:
            raise ValueError(
                f"expected latent with shape [B, C, D], got {tuple(latent.shape)}"
            )
        batch, channels, width = latent.shape
        if channels != self.channels or width != self.d_model:
            raise ValueError(
                "latent shape mismatch: expected "
                f"[B, {self.channels}, {self.d_model}], got {tuple(latent.shape)}"
            )
        return batch, channels, width

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def normalise_input(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Instance-normalise each window/channel using context statistics only."""

        self._check_input(x)
        if not self.revin:
            mean = torch.zeros_like(x[:, :1, :])
            scale = torch.ones_like(x[:, :1, :])
            return x, (mean, scale)
        mean = x.mean(dim=1, keepdim=True).detach()
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        scale = torch.sqrt(variance + self.revin_eps).detach()
        return (x - mean) / scale, (mean, scale)

    def encode_with_stats(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        normalised, stats = self.normalise_input(x)
        return self.encode(normalised), stats

    def decode(self, latent: Tensor) -> Tensor:
        """Decode channel latents into a [batch, horizon, channel] forecast."""
        self._check_latent(latent)
        return self.forecast_head(latent).transpose(1, 2).contiguous()

    def decode_with_stats(
        self, latent: Tensor, stats: tuple[Tensor, Tensor]
    ) -> Tensor:
        forecast = self.decode(latent)
        mean, scale = stats
        return forecast * scale + mean

    def forward(self, x: Tensor) -> Tensor:
        latent, stats = self.encode_with_stats(x)
        return self.decode_with_stats(latent, stats)


def _compatible_nheads(d_model: int, preferred: int = 4) -> int:
    """Choose a small attention head count that exactly divides ``d_model``."""
    for heads in range(min(preferred, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class PatchTransformer(ForecastBackbone):
    """Channel-independent patch encoder with a two-layer Transformer.

    Each variable is treated as an independent member of the batch, so neither
    the patch embedding nor attention parameters depend on the channel count.
    """

    def __init__(
        self,
        channels: int,
        context: int,
        horizon: int,
        d_model: int = 48,
        patch_length: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(channels, context, horizon, d_model)
        if patch_length <= 0 or stride <= 0:
            raise ValueError("patch_length and stride must be positive")
        self.patch_length = min(int(patch_length), self.context)
        self.stride = min(int(stride), self.patch_length)
        self.num_patches = max(
            1, math.ceil((self.context - self.patch_length) / self.stride) + 1
        )
        covered = (self.num_patches - 1) * self.stride + self.patch_length
        self.pad_right = covered - self.context

        self.patch_embedding = nn.Linear(self.patch_length, self.d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=_compatible_nheads(self.d_model),
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.token_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        nn.init.trunc_normal_(self.position, std=0.02)

    def encode(self, x: Tensor) -> Tensor:
        batch, _, channels = self._check_input(x)
        # [B, L, C] -> [B*C, L]; variables share one temporal encoder.
        series = x.transpose(1, 2).reshape(batch * channels, self.context)
        if self.pad_right:
            series = F.pad(series, (0, self.pad_right), mode="replicate")
        patches = series.unfold(-1, self.patch_length, self.stride)
        if patches.shape[1] != self.num_patches:
            raise RuntimeError("internal patch-count mismatch")
        tokens = self.patch_embedding(patches) + self.position
        tokens = self.encoder(tokens)
        latent = self.output_norm(self.token_projection(tokens))
        return latent.reshape(batch, channels, self.d_model)


class _PeriodInception(nn.Module):
    """Small 2-D inception block for a cycle-by-phase representation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(1, d_model, kernel_size=(1, 3), padding=(0, 1)),
                nn.Conv2d(1, d_model, kernel_size=(3, 1), padding=(1, 0)),
                nn.Conv2d(1, d_model, kernel_size=(3, 3), padding=(1, 1)),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_mean = torch.stack([branch(x) for branch in self.branches], dim=0).mean(0)
        return F.gelu(branch_mean)


class PeriodConvNet(ForecastBackbone):
    """FFT-weighted multi-period encoder with lightweight 2-D convolutions.

    The period candidates are fixed from the context length while their mixing
    weights are computed independently for every sample.  This avoids the
    undesirable situation where one sample's forecast changes merely because
    other samples were placed in the same minibatch.
    """

    def __init__(
        self,
        channels: int,
        context: int,
        horizon: int,
        d_model: int = 48,
        top_k: int = 3,
    ) -> None:
        super().__init__(channels, context, horizon, d_model)
        if context < 2:
            raise ValueError("PeriodConvNet requires context >= 2")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = int(top_k)
        maximum_frequency = max(1, min(self.context // 4, self.context // 2))
        candidate_frequencies = torch.logspace(
            0.0,
            math.log10(float(maximum_frequency)),
            steps=self.top_k,
        ).round().to(torch.long)
        candidate_frequencies = torch.unique(candidate_frequencies, sorted=True)
        # Very short contexts can collapse rounded log-spaced candidates.  Add
        # unused bins deterministically until the requested count is reached.
        if candidate_frequencies.numel() < self.top_k:
            present = set(candidate_frequencies.tolist())
            additions = [
                value
                for value in range(1, self.context // 2 + 1)
                if value not in present
            ][: self.top_k - candidate_frequencies.numel()]
            candidate_frequencies = torch.cat(
                [candidate_frequencies, torch.tensor(additions, dtype=torch.long)]
            ).sort().values
        self.register_buffer(
            "candidate_frequencies", candidate_frequencies[: self.top_k], persistent=True
        )
        self.period_conv = _PeriodInception(self.d_model)
        # Raw statistics preserve level and trend information lost by pooling.
        self.statistics_projection = nn.Sequential(
            nn.Linear(4, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def encode(self, x: Tensor) -> Tensor:
        batch, length, channels = self._check_input(x)
        series = x.transpose(1, 2).contiguous()  # [B, C, L]
        spectrum = torch.fft.rfft(series, dim=-1)
        amplitude = spectrum.abs()
        # Fixed candidate bins make the representation batch invariant; each
        # sample/channel still receives data-dependent FFT mixing weights.
        frequency_indices = self.candidate_frequencies.clamp_max(amplitude.shape[-1] - 1)
        sample_weights = amplitude.index_select(-1, frequency_indices).softmax(dim=-1)

        flat_series = series.reshape(batch * channels, 1, length)
        period_latents: list[Tensor] = []
        for frequency_index in frequency_indices:
            frequency = int(frequency_index.item())
            period = max(2, int(round(length / frequency)))
            padded_length = math.ceil(length / period) * period
            if padded_length > length:
                periodic_input = F.pad(
                    flat_series,
                    (0, padded_length - length),
                    mode="replicate",
                )
            else:
                periodic_input = flat_series
            # Rows are cycles and columns are positions within a cycle.
            periodic_input = periodic_input.reshape(
                batch * channels, 1, padded_length // period, period
            )
            feature_map = self.period_conv(periodic_input)
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
            period_latents.append(pooled.reshape(batch, channels, self.d_model))

        stacked = torch.stack(period_latents, dim=2)  # [B, C, K, D]
        periodic_latent = (stacked * sample_weights.unsqueeze(-1)).sum(dim=2)

        statistics = torch.stack(
            [
                series.mean(dim=-1),
                series.std(dim=-1, unbiased=False),
                series[..., -1],
                series[..., -1] - series[..., 0],
            ],
            dim=-1,
        )
        return self.output_norm(periodic_latent + self.statistics_projection(statistics))


class SelectiveSSM(ForecastBackbone):
    """Pure-PyTorch input-dependent gated diagonal state-space encoder.

    This is deliberately a small, transparent recurrent SSM.  Its discretized
    diagonal decay, input gate, and readout gate all depend on the current input;
    no custom CUDA kernels or external SSM package are required.
    """

    def __init__(
        self,
        channels: int,
        context: int,
        horizon: int,
        d_model: int = 48,
    ) -> None:
        super().__init__(channels, context, horizon, d_model)
        self.input_projection = nn.Linear(1, self.d_model)
        self.delta_projection = nn.Linear(self.d_model, self.d_model)
        self.input_gate = nn.Linear(self.d_model, self.d_model)
        self.input_value = nn.Linear(self.d_model, self.d_model)
        self.output_gate = nn.Linear(self.d_model, self.d_model)
        # Softplus turns these into stable positive decay rates.
        self.log_decay = nn.Parameter(torch.linspace(-3.0, 0.0, self.d_model))
        self.output_norm = nn.LayerNorm(self.d_model)

        # Start with moderate memory and a slight preference for the SSM state.
        nn.init.constant_(self.delta_projection.bias, -1.0)
        nn.init.constant_(self.input_gate.bias, 0.0)
        nn.init.constant_(self.output_gate.bias, 0.5)

    def encode(self, x: Tensor) -> Tensor:
        batch, length, channels = self._check_input(x)
        series = x.transpose(1, 2).reshape(batch * channels, length, 1)
        embedded = self.input_projection(series)
        step_size = F.softplus(self.delta_projection(embedded)) + 1e-4
        decay_rate = F.softplus(self.log_decay).view(1, 1, self.d_model)
        decay = torch.exp(-step_size * decay_rate)
        write_gate = torch.sigmoid(self.input_gate(embedded))
        candidate = torch.tanh(self.input_value(embedded))
        read_gate = torch.sigmoid(self.output_gate(embedded))

        # Only the final state is consumed.  Expanding the diagonal recurrence
        # into its closed-form suffix products removes the Python time-step
        # loop while preserving the exact selective-SSM update:
        #   s_T = sum_i b_i * product_{j=i+1..T} a_j.
        # Computing products in log space is stable for long contexts; very
        # old contributions may underflow to zero, which is also their true
        # limiting behaviour under a positive decay rate.
        log_decay = torch.log(decay.clamp_min(torch.finfo(decay.dtype).tiny))
        inclusive_suffix = torch.flip(
            torch.cumsum(torch.flip(log_decay, dims=(1,)), dim=1), dims=(1,)
        )
        exclusive_suffix = inclusive_suffix - log_decay
        suffix_product = torch.exp(exclusive_suffix)
        write = (1.0 - decay) * write_gate * candidate
        state = (write * suffix_product).sum(dim=1)
        final_read_gate = read_gate[:, -1]
        observation = (
            final_read_gate * state
            + (1.0 - final_read_gate) * embedded[:, -1]
        )
        latent = self.output_norm(observation)
        return latent.reshape(batch, channels, self.d_model)


class LoRALinear(nn.Module):
    """Frozen linear map with a trainable low-rank weight increment.

    The effective mapping is ``W x + (alpha / rank) B A x``.  ``B`` is zero
    initialized, so replacing a fitted ``nn.Linear`` with this module preserves
    the forecast exactly at initialization.  Unlike the former post-hoc latent
    bottleneck, this is weight-level LoRA and therefore adapts the feature maps
    that the retrieval selector and gate consume.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        # Keep the configured rank even for narrow projections.  A low-rank
        # factorisation with r > min(in, out) is still well-defined, while
        # silently truncating r changes alpha/r and can amplify updates by up
        # to an order of magnitude in one-dimensional SSM projections.
        self.rank = int(rank)
        self.alpha = float(self.rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.dropout = nn.Dropout(float(dropout))
        self.adapter_scale = 1.0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: Tensor) -> Tensor:
        frozen = F.linear(value, self.weight, self.bias)
        if self.adapter_scale == 0.0:
            return frozen
        update = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return frozen + update * self.scaling * float(self.adapter_scale)

    def adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.lora_A, self.lora_B


class LoRAWeightParametrization(nn.Module):
    """Low-rank parametrization for weights consumed by functional kernels.

    PyTorch's ``MultiheadAttention`` reads packed Q/K/V and output projection
    weights directly, so merely replacing ``out_proj`` with a module would not
    execute an adapter ``forward`` method.  A registered parametrization makes
    those kernels consume the effective ``W + BA`` weight itself.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: Optional[float],
        weight_shape: Optional[tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        # Do not derive the LoRA scale from a layer-specific truncated rank.
        # All injected layers must share the configured alpha/r convention;
        # otherwise narrow/flattened convolution weights receive much larger
        # updates than attention and feed-forward weights.
        self.rank = int(rank)
        self.alpha = float(self.rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, int(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(int(out_features), self.rank))
        self.dropout = nn.Identity()
        self.adapter_scale = 1.0
        self.weight_shape = (
            (int(out_features), int(in_features))
            if weight_shape is None
            else tuple(int(value) for value in weight_shape)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: Tensor) -> Tensor:
        if self.adapter_scale == 0.0:
            return weight
        update = (self.lora_B @ self.lora_A).reshape(self.weight_shape)
        return weight + update * self.scaling * float(self.adapter_scale)

    def adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.lora_A, self.lora_B


def _module_at_path(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _lora_target_names(base: ForecastBackbone) -> tuple[str, ...]:
    """Return explicit, architecture-aware LoRA targets.

    The shared forecast head is intentionally excluded: the paper trains that
    head as an ordinary prediction head, while LoRA is applied to frozen
    backbone mappings.  MultiheadAttention's packed projections are also
    excluded because its functional kernel bypasses a replacement module's
    ``forward`` method.
    """

    if isinstance(base, PatchTransformer):
        targets = ["patch_embedding", "token_projection.1"]
        for index in range(len(base.encoder.layers)):
            targets.extend(
                (
                    f"encoder.layers.{index}.linear1",
                    f"encoder.layers.{index}.linear2",
                )
            )
        return tuple(targets)
    if isinstance(base, PeriodConvNet):
        return ("statistics_projection.0", "statistics_projection.2")
    if isinstance(base, SelectiveSSM):
        return (
            "input_projection",
            "delta_projection",
            "input_gate",
            "input_value",
            "output_gate",
        )
    raise TypeError(f"unsupported backbone type for LoRA: {type(base).__name__}")


def inject_lora(
    base: ForecastBackbone,
    rank: int,
    *,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
) -> tuple[nn.Module, ...]:
    """Replace the selected fitted backbone linears and return the adapters."""

    adapters: list[nn.Module] = []
    for path in _lora_target_names(base):
        parent, leaf = _module_at_path(base, path)
        if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            original = parent[int(leaf)]
        else:
            original = getattr(parent, leaf)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"LoRA target {path!r} is not an nn.Linear")
        replacement = LoRALinear(
            original,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(leaf)] = replacement
        else:
            setattr(parent, leaf, replacement)
        adapters.append(replacement)
    if isinstance(base, PatchTransformer):
        for layer in base.encoder.layers:
            attention = layer.self_attn
            in_projection = LoRAWeightParametrization(
                attention.in_proj_weight.shape[0],
                attention.in_proj_weight.shape[1],
                rank,
                alpha,
            )
            parametrize.register_parametrization(
                attention, "in_proj_weight", in_projection
            )
            attention.parametrizations.in_proj_weight.original.requires_grad_(False)
            out_projection = LoRAWeightParametrization(
                attention.out_proj.weight.shape[0],
                attention.out_proj.weight.shape[1],
                rank,
                alpha,
            )
            parametrize.register_parametrization(
                attention.out_proj, "weight", out_projection
            )
            attention.out_proj.parametrizations.weight.original.requires_grad_(False)
            adapters.extend((in_projection, out_projection))
    if isinstance(base, PeriodConvNet):
        for convolution in base.period_conv.branches:
            weight_shape = tuple(convolution.weight.shape)
            convolution_adapter = LoRAWeightParametrization(
                weight_shape[0],
                math.prod(weight_shape[1:]),
                rank,
                alpha,
                weight_shape=weight_shape,
            )
            parametrize.register_parametrization(
                convolution, "weight", convolution_adapter
            )
            convolution.parametrizations.weight.original.requires_grad_(False)
            adapters.append(convolution_adapter)
    if not adapters:
        raise RuntimeError("no LoRA targets were injected")
    return tuple(adapters)


def _broadcast_auxiliary(feature: Tensor, target: Tensor, name: str) -> Tensor:
    """Broadcast a retrieval statistic to forecast shape [B, H, C]."""
    value = torch.as_tensor(feature, dtype=target.dtype, device=target.device)
    batch, horizon, channels = target.shape
    if value.ndim == 0:
        value = value.reshape(1, 1, 1)
    elif value.ndim == 1:
        if value.shape[0] == batch:
            value = value[:, None, None]
        elif value.shape[0] == channels:
            value = value[None, None, :]
        else:
            raise ValueError(f"cannot broadcast {name} shape {tuple(value.shape)}")
    elif value.ndim == 2:
        if value.shape == (batch, channels):
            value = value[:, None, :]
        elif value.shape == (batch, horizon):
            value = value[:, :, None]
        elif value.shape == (horizon, channels):
            value = value[None, :, :]
        elif value.shape == (batch, 1):
            value = value[:, :, None]
        else:
            raise ValueError(f"cannot broadcast {name} shape {tuple(value.shape)}")
    elif value.ndim == 3:
        if value.shape == (batch, channels, 1):
            value = value.transpose(1, 2)
    else:
        raise ValueError(f"{name} must have at most 3 dimensions")
    try:
        return torch.broadcast_to(value, target.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"cannot broadcast {name} shape {tuple(value.shape)} to {tuple(target.shape)}"
        ) from exc


class SelectiveRetrievalFusion(nn.Module):
    """Candidate-level selective retrieval fusion.

    The memory bank retains K aligned normal futures.  A query-conditioned
    selector assigns candidate weights, while separate conditional and analogue
    branches are mixed by eta and a conservative residual gate gamma.  This is
    intentionally compact, but it preserves the candidate structure that the
    paper's SRF mechanism relies on.
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        hidden: int = 32,
        initial_gate: float = 0.10,
    ) -> None:
        super().__init__()
        if min(d_model, horizon, hidden) <= 0:
            raise ValueError("d_model, horizon, and hidden must be positive")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.tail_encoder = nn.Sequential(
            nn.Linear(horizon, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.selector = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.conditional_head = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, horizon),
        )
        control_width = 2 * hidden + 2
        self.eta_network = nn.Sequential(
            nn.Linear(control_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.gate_network = nn.Sequential(
            nn.Linear(control_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        final_gate = self.gate_network[-1]
        if not isinstance(final_gate, nn.Linear):
            raise TypeError("internal gate network construction failed")
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(
            final_gate.bias, math.log(initial_gate / (1.0 - initial_gate))
        )
        final_conditional = self.conditional_head[-1]
        if isinstance(final_conditional, nn.Linear):
            nn.init.zeros_(final_conditional.weight)
            nn.init.zeros_(final_conditional.bias)

    @staticmethod
    def _safe_log(value: Tensor) -> Tensor:
        finite = torch.nan_to_num(value, nan=0.0, posinf=1.0e12, neginf=0.0)
        return torch.log1p(finite.clamp(min=0.0, max=1.0e12))

    def forward(
        self,
        native_forecast: Tensor,
        candidate_futures: Tensor,
        distance: Tensor,
        divergence: Tensor,
        query_latent: Tensor,
        return_aux: bool = False,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[Tensor, Tensor, dict[str, Tensor]]
    ):
        if native_forecast.ndim != 3:
            raise ValueError("native_forecast must have shape [B,H,C]")
        batch, horizon, channels = native_forecast.shape
        if horizon != self.horizon:
            raise ValueError(f"expected horizon {self.horizon}, got {horizon}")
        if candidate_futures.ndim == 3:
            candidate_futures = candidate_futures[:, None, :, :]
        if candidate_futures.ndim != 4:
            raise ValueError("candidate_futures must have shape [B,K,H,C]")
        if candidate_futures.shape[0] != batch or candidate_futures.shape[2:] != (
            horizon,
            channels,
        ):
            raise ValueError(
                "candidate_futures shape mismatch: got "
                f"{tuple(candidate_futures.shape)} for native {tuple(native_forecast.shape)}"
            )
        candidates = torch.nan_to_num(
            candidate_futures.to(
                device=native_forecast.device, dtype=native_forecast.dtype
            ),
            nan=0.0,
            posinf=1.0e12,
            neginf=-1.0e12,
        )
        candidate_count = candidates.shape[1]
        if candidate_count <= 0:
            raise ValueError("candidate_futures must contain at least one candidate")
        if distance.ndim == 2:
            distance = distance[:, None, :]
        if distance.shape != (batch, candidate_count, channels):
            raise ValueError(
                f"distance must have shape {(batch, candidate_count, channels)}, "
                f"got {tuple(distance.shape)}"
            )

        # A missing-neighbour slot is encoded upstream with float32 max.  It is
        # finite, so an isfinite-only mask silently treated it as a genuine
        # reference.  Test validity in the source dtype before any AMP cast:
        # float16 cannot represent this sentinel and would turn it into inf.
        distance = torch.as_tensor(distance, device=native_forecast.device)
        distance_by_channel = distance.permute(0, 2, 1)
        sentinel_limit = 0.5 * torch.finfo(torch.float32).max
        valid_candidate = torch.isfinite(distance_by_channel) & (
            distance_by_channel < sentinel_limit
        )
        valid_reference = valid_candidate.any(dim=-1, keepdim=True)

        # [B,K,H,C] -> [B,C,K,H], preserving every genuine candidate until
        # selection.  Replacing invalid futures before the tail encoder avoids
        # NaNs/overflows from a placeholder future even though its final weight
        # is zero.
        candidate_by_channel = candidates.permute(0, 3, 1, 2).contiguous()
        native_by_channel = native_forecast.permute(0, 2, 1)
        candidate_by_channel = torch.where(
            valid_candidate.unsqueeze(-1),
            candidate_by_channel,
            native_by_channel[:, :, None, :],
        )
        tail = self.tail_encoder(candidate_by_channel)
        query = self.query_projection(query_latent)
        query_expanded = query[:, :, None, :].expand(-1, -1, candidate_count, -1)
        clean_distance = torch.where(
            valid_candidate,
            distance_by_channel,
            torch.zeros_like(distance_by_channel),
        ).to(dtype=native_forecast.dtype)
        log_distance = self._safe_log(clean_distance).unsqueeze(-1)
        disagreement = (
            candidate_by_channel
            - native_by_channel[:, :, None, :]
        ).abs().mean(dim=-1, keepdim=True)
        selector_input = torch.cat(
            [query_expanded, tail, log_distance, self._safe_log(disagreement)], dim=-1
        )
        selector_logits = self.selector(selector_input).squeeze(-1) - log_distance.squeeze(-1)
        # Multiplying by the mask after softmax and renormalising is deliberate:
        # using one large negative logit alone gives a uniform distribution when
        # every candidate is invalid.  Here an all-invalid row is exactly zero.
        selector_logits = selector_logits.masked_fill(
            ~valid_candidate, torch.finfo(selector_logits.dtype).min
        )
        weights = torch.softmax(selector_logits, dim=-1)
        weights = weights * valid_candidate.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        weighted_tail = (weights.unsqueeze(-1) * tail).sum(dim=2)
        selected_analogue = (
            weights.unsqueeze(-1) * candidate_by_channel
        ).sum(dim=2).permute(0, 2, 1).contiguous()
        forecast_validity = valid_reference.transpose(1, 2)
        analogue = torch.where(
            forecast_validity, selected_analogue, native_forecast
        )

        joint = torch.cat([query, weighted_tail], dim=-1)
        conditional_delta = self.conditional_head(joint).transpose(1, 2)
        conditional = torch.where(
            forecast_validity,
            native_forecast + conditional_delta,
            native_forecast,
        )
        masked_log_distance = log_distance.squeeze(-1).masked_fill(
            ~valid_candidate, torch.finfo(log_distance.dtype).max
        )
        nearest_distance = masked_log_distance.amin(dim=-1, keepdim=True)
        nearest_distance = torch.where(
            valid_reference, nearest_distance, torch.zeros_like(nearest_distance)
        )
        divergence_map = _broadcast_auxiliary(
            divergence, native_forecast, "divergence"
        ).permute(0, 2, 1)
        divergence_summary = self._safe_log(divergence_map).mean(dim=-1, keepdim=True)
        control = torch.cat(
            [query, weighted_tail, nearest_distance, divergence_summary], dim=-1
        )
        raw_eta = torch.sigmoid(self.eta_network(control)).transpose(1, 2)
        raw_gamma = torch.sigmoid(self.gate_network(control)).transpose(1, 2)
        eta = torch.where(
            forecast_validity, raw_eta, torch.zeros_like(raw_eta)
        )
        correction = eta * (conditional - native_forecast) + (1.0 - eta) * (
            analogue - native_forecast
        )
        ungated_forecast = torch.where(
            forecast_validity,
            native_forecast + correction,
            native_forecast,
        )
        gamma = torch.where(
            forecast_validity, raw_gamma, torch.zeros_like(raw_gamma)
        )
        fused = native_forecast + gamma * (ungated_forecast - native_forecast)
        gate = gamma.expand(-1, horizon, -1)
        if not bool(torch.isfinite(fused).all()) or not bool(torch.isfinite(gate).all()):
            raise RuntimeError("retrieval fusion produced non-finite output")
        if return_aux:
            auxiliary = {
                "bypass_forecast": native_forecast,
                "analogue": analogue,
                "conditional_forecast": conditional,
                "ungated_forecast": ungated_forecast,
                "eta": eta.expand(-1, horizon, -1),
                "gamma": gate,
                # ``gate`` is retained as a semantic alias for existing
                # prediction/visualisation consumers.
                "gate": gate,
                "selector_weights": weights,
                "valid_reference": forecast_validity,
            }
            return fused, gate, auxiliary
        return fused, gate


class ConditionModel(nn.Module):
    """Frozen backbone with condition-specific LoRA and/or retrieval modules.

    Valid conditions are ``native``, ``lora``, ``srf``, and ``joint``.  The
    backbone remains frozen in every condition; train it before wrapping it.
    ``configure_trainable`` is intentionally strict so optimizer parameter
    groups cannot silently include the wrong experimental component.
    """

    VALID_CONDITIONS = frozenset({"native", "lora", "srf", "joint"})

    def __init__(
        self,
        base: ForecastBackbone,
        condition: str = "native",
        rank: int = 4,
        fusion_hidden: int = 16,
        lora_alpha: Optional[float] = None,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, ForecastBackbone):
            raise TypeError("base must be a ForecastBackbone")
        self.base = base
        # Preserve an immutable Native prediction-head path.  Non-Native
        # conditions may train base.forecast_head, but a validation-selected
        # blend of zero must still reproduce the fitted Native model exactly.
        self.register_buffer(
            "native_head_weight", base.forecast_head.weight.detach().clone()
        )
        native_bias = base.forecast_head.bias
        self.register_buffer(
            "native_head_bias",
            None if native_bias is None else native_bias.detach().clone(),
        )
        adapters = inject_lora(
            base,
            rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        # Store plain references rather than registering every adapter twice;
        # the modules are already registered at their backbone target paths.
        self.__dict__["_lora_layers"] = adapters
        self.lora_blend = 1.0
        self.fusion = SelectiveRetrievalFusion(
            base.d_model,
            base.horizon,
            hidden=fusion_hidden,
            initial_gate=0.10,
        )
        # Joint collaborative fine-tuning may update the live fusion/head after
        # its SRF warm start.  Keep an immutable stage-one branch so blend=0 is
        # still the exact paired SRF-Only model rather than a moving target.
        self.add_module("srf_reference_fusion", None)
        self.register_buffer("srf_reference_head_weight", None)
        self.register_buffer("srf_reference_head_bias", None)
        self.condition = "native"
        self.configure_trainable(condition)

    def lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for layer in self._lora_layers
            for parameter in layer.adapter_parameters()
        ]

    def lora_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.lora_parameters()))

    def set_lora_blend(self, value: float) -> "ConditionModel":
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("LoRA blend must lie in [0, 1]")
        self.lora_blend = value
        return self

    def capture_srf_reference(self) -> "ConditionModel":
        """Snapshot the fitted SRF/head branch for exact Joint fallback."""

        reference = copy.deepcopy(self.fusion)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        reference.eval()
        self.srf_reference_fusion = reference
        self.srf_reference_head_weight = (
            self.base.forecast_head.weight.detach().clone()
        )
        bias = self.base.forecast_head.bias
        self.srf_reference_head_bias = (
            None if bias is None else bias.detach().clone()
        )
        return self

    @property
    def has_srf_reference(self) -> bool:
        return self.srf_reference_fusion is not None

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Materialise an optional Joint snapshot before strict restoration."""

        has_reference_state = any(
            key.startswith("srf_reference_fusion.")
            or key.startswith("srf_reference_head_")
            for key in state_dict
        )
        if has_reference_state and not self.has_srf_reference:
            self.capture_srf_reference()
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:  # pragma: no cover - older PyTorch has no assign keyword
            return super().load_state_dict(state_dict, strict=strict)

    def configure_trainable(
        self,
        condition: Optional[str] = None,
        *,
        phase: Optional[str] = None,
    ) -> "ConditionModel":
        """Freeze all weights, then enable one explicit adaptation phase.

        ``phase='srf'`` is used to warm-start Joint from the exact SRF training
        path.  ``phase='lora'`` then freezes that solution and adapts only the
        low-rank increments, preventing the destructive three-way optimizer
        interference observed in the former from-scratch Joint training.
        """
        if condition is None:
            condition = self.condition
        condition = condition.lower().strip()
        if condition not in self.VALID_CONDITIONS:
            choices = ", ".join(sorted(self.VALID_CONDITIONS))
            raise ValueError(f"unknown condition {condition!r}; choose one of {choices}")
        self.condition = condition
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            # A fitted base model may still carry gradients from its last
            # optimiser step.  Clearing them here makes a freshly trained base
            # and a checkpoint-loaded base follow the same adaptation path.
            parameter.grad = None
        resolved_phase = condition if phase is None else phase.lower().strip()
        if resolved_phase not in {"native", "lora", "srf", "joint"}:
            raise ValueError("phase must be native, lora, srf, or joint")
        if resolved_phase in {"lora", "joint"}:
            for parameter in self.lora_parameters():
                parameter.requires_grad_(True)
        if resolved_phase in {"srf", "joint"}:
            for parameter in self.fusion.parameters():
                parameter.requires_grad_(True)
        if resolved_phase in {"lora", "srf", "joint"} and not (
            condition == "joint" and resolved_phase == "lora"
        ):
            # The accepted manuscript trains the prediction head together with
            # the condition module.  A LoRA-only Joint phase freezes the
            # already fitted head; phase='joint' enables the live head while an
            # immutable SRF snapshot preserves the exact blend-zero reference.
            for parameter in self.base.forecast_head.parameters():
                parameter.requires_grad_(True)
        self.base.eval()
        return self

    def train(self, mode: bool = True) -> "ConditionModel":
        # Frozen base dropout must not add condition-dependent evaluation noise.
        super().train(mode)
        self.base.eval()
        if self.srf_reference_fusion is not None:
            self.srf_reference_fusion.eval()
        for layer in self._lora_layers:
            layer.dropout.train(mode)
        return self

    def _encode_at_scale(self, normalised: Tensor, scale: float) -> Tensor:
        previous = [layer.adapter_scale for layer in self._lora_layers]
        try:
            for layer in self._lora_layers:
                layer.adapter_scale = float(scale)
            return self.base.encode(normalised)
        finally:
            for layer, value in zip(self._lora_layers, previous):
                layer.adapter_scale = value

    def _decode_native_head(
        self, latent: Tensor, stats: tuple[Tensor, Tensor]
    ) -> Tensor:
        self.base._check_latent(latent)
        forecast = F.linear(
            latent, self.native_head_weight, self.native_head_bias
        ).transpose(1, 2).contiguous()
        mean, scale = stats
        return forecast * scale + mean

    def _decode_current_head(
        self, latent: Tensor, stats: tuple[Tensor, Tensor]
    ) -> Tensor:
        return self.base.decode_with_stats(latent, stats)

    def _decode_srf_reference_head(
        self, latent: Tensor, stats: tuple[Tensor, Tensor]
    ) -> Tensor:
        if self.srf_reference_head_weight is None:
            return self._decode_current_head(latent, stats)
        self.base._check_latent(latent)
        forecast = F.linear(
            latent,
            self.srf_reference_head_weight,
            self.srf_reference_head_bias,
        ).transpose(1, 2).contiguous()
        mean, scale = stats
        return forecast * scale + mean

    @staticmethod
    def _non_retrieval_aux(forecast: Tensor) -> dict[str, Tensor]:
        """Return a shape-stable auxiliary record for a bypass-only model."""

        batch, horizon, channels = forecast.shape
        zero_forecast = torch.zeros_like(forecast)
        return {
            "bypass_forecast": forecast,
            "analogue": forecast,
            "conditional_forecast": forecast,
            "ungated_forecast": forecast,
            "eta": zero_forecast,
            "gamma": zero_forecast,
            "gate": zero_forecast,
            "selector_weights": forecast.new_zeros((batch, channels, 0)),
            "valid_reference": torch.zeros(
                (batch, 1, channels), dtype=torch.bool, device=forecast.device
            ),
        }

    @staticmethod
    def _blend_retrieval_aux(
        reference: dict[str, Tensor],
        adapted: dict[str, Tensor],
        blend: float,
    ) -> dict[str, Tensor]:
        """Blend diagnostics along the same Joint path as the prediction."""

        if reference.keys() != adapted.keys():
            raise RuntimeError("Joint retrieval auxiliary keys do not match")
        result: dict[str, Tensor] = {}
        for name, reference_value in reference.items():
            adapted_value = adapted[name]
            if reference_value.shape != adapted_value.shape:
                raise RuntimeError(
                    f"Joint retrieval auxiliary shape mismatch for {name!r}"
                )
            if reference_value.dtype == torch.bool:
                result[name] = reference_value | adapted_value
            else:
                result[name] = reference_value + float(blend) * (
                    adapted_value - reference_value
                )
        return result

    def forward(
        self,
        x: Tensor,
        analog_future: Optional[Tensor] = None,
        distance: Optional[Tensor] = None,
        divergence: Optional[Tensor] = None,
        return_gate: bool = False,
        return_aux: bool = False,
    ) -> (
        Tensor
        | tuple[Tensor, Optional[Tensor]]
        | tuple[Tensor, dict[str, Tensor]]
    ):
        """Forecast a batch, optionally returning retrieval diagnostics.

        The legacy ``return_gate=True`` contract remains ``(forecast, gate)``.
        The generic training/inference contract uses ``return_aux=True`` and
        returns ``(forecast, auxiliary_mapping)``; that mapping also contains
        ``gate``.  If both flags are true, ``return_aux`` takes precedence so
        callers never receive an ambiguous three-tuple.
        """

        normalised, stats = self.base.normalise_input(x)

        # A full-strength adapter does not consume the Native encoding.  Keep
        # the existing dual path for 0 < blend < 1 and the exact Native/SRF
        # safety path for blend == 0, but avoid an otherwise redundant backbone
        # forward during ordinary LoRA/Joint adaptation and inference.
        if self.lora_blend == 1.0 and self.condition in {"lora", "joint"}:
            adapted_latent = self._encode_at_scale(normalised, 1.0)
            adapted_base = self._decode_current_head(adapted_latent, stats)
            if self.condition == "lora":
                if return_aux:
                    return adapted_base, self._non_retrieval_aux(adapted_base)
                if return_gate:
                    return adapted_base, None
                return adapted_base

            if analog_future is None or distance is None or divergence is None:
                raise ValueError(
                    "SRF conditions require analog_future, distance, and divergence"
                )
            analog_future = analog_future.to(
                device=adapted_base.device, dtype=adapted_base.dtype
            )
            if return_aux:
                forecast, _gate, auxiliary = self.fusion(
                    adapted_base,
                    analog_future,
                    distance,
                    divergence,
                    adapted_latent,
                    return_aux=True,
                )
                return forecast, auxiliary
            forecast, gate = self.fusion(
                adapted_base,
                analog_future,
                distance,
                divergence,
                adapted_latent,
            )
            if return_gate:
                return forecast, gate
            return forecast

        native_latent = self._encode_at_scale(normalised, 0.0)
        native_forecast = self._decode_native_head(native_latent, stats)

        gate: Optional[Tensor] = None
        auxiliary: Optional[dict[str, Tensor]] = None
        forecast = native_forecast
        if self.condition == "lora":
            if self.lora_blend > 0.0:
                adapted_latent = self._encode_at_scale(normalised, 1.0)
                adapted_forecast = self._decode_current_head(adapted_latent, stats)
                forecast = native_forecast + self.lora_blend * (
                    adapted_forecast - native_forecast
                )
        elif self.condition == "srf":
            if analog_future is None or distance is None or divergence is None:
                raise ValueError(
                    "SRF conditions require analog_future, distance, and divergence"
                )
            analog_future = analog_future.to(
                device=native_forecast.device, dtype=native_forecast.dtype
            )
            srf_base = self._decode_current_head(native_latent, stats)
            if return_aux:
                forecast, gate, auxiliary = self.fusion(
                    srf_base,
                    analog_future,
                    distance,
                    divergence,
                    native_latent,
                    return_aux=True,
                )
            else:
                forecast, gate = self.fusion(
                    srf_base,
                    analog_future,
                    distance,
                    divergence,
                    native_latent,
                )
        elif self.condition == "joint":
            if analog_future is None or distance is None or divergence is None:
                raise ValueError(
                    "SRF conditions require analog_future, distance, and divergence"
                )
            analog_future = analog_future.to(
                device=native_forecast.device, dtype=native_forecast.dtype
            )
            reference_forecast: Optional[Tensor] = None
            reference_gate: Optional[Tensor] = None
            reference_auxiliary: Optional[dict[str, Tensor]] = None
            if self.lora_blend < 1.0:
                reference_base = self._decode_srf_reference_head(
                    native_latent, stats
                )
                reference_fusion = (
                    self.srf_reference_fusion
                    if self.srf_reference_fusion is not None
                    else self.fusion
                )
                if return_aux:
                    (
                        reference_forecast,
                        reference_gate,
                        reference_auxiliary,
                    ) = reference_fusion(
                        reference_base,
                        analog_future,
                        distance,
                        divergence,
                        native_latent,
                        return_aux=True,
                    )
                else:
                    reference_forecast, reference_gate = reference_fusion(
                        reference_base,
                        analog_future,
                        distance,
                        divergence,
                        native_latent,
                    )
            if self.lora_blend > 0.0:
                adapted_latent = self._encode_at_scale(normalised, 1.0)
                adapted_base = self._decode_current_head(adapted_latent, stats)
                adapted_auxiliary: Optional[dict[str, Tensor]] = None
                if return_aux:
                    adapted_forecast, adapted_gate, adapted_auxiliary = self.fusion(
                        adapted_base,
                        analog_future,
                        distance,
                        divergence,
                        adapted_latent,
                        return_aux=True,
                    )
                else:
                    adapted_forecast, adapted_gate = self.fusion(
                        adapted_base,
                        analog_future,
                        distance,
                        divergence,
                        adapted_latent,
                    )
                if reference_forecast is None:
                    forecast, gate = adapted_forecast, adapted_gate
                    auxiliary = adapted_auxiliary
                else:
                    forecast = reference_forecast + self.lora_blend * (
                        adapted_forecast - reference_forecast
                    )
                    if reference_gate is None:
                        gate = adapted_gate
                    else:
                        gate = reference_gate + self.lora_blend * (
                            adapted_gate - reference_gate
                        )
                    if return_aux:
                        if reference_auxiliary is None or adapted_auxiliary is None:
                            raise RuntimeError(
                                "Joint auxiliary path was not fully evaluated"
                            )
                        auxiliary = self._blend_retrieval_aux(
                            reference_auxiliary,
                            adapted_auxiliary,
                            self.lora_blend,
                        )
            else:
                if reference_forecast is None:
                    raise RuntimeError("Joint reference path was not evaluated")
                forecast, gate = reference_forecast, reference_gate
                auxiliary = reference_auxiliary
        if return_aux:
            if auxiliary is None:
                auxiliary = self._non_retrieval_aux(forecast)
            return forecast, auxiliary
        if return_gate:
            return forecast, gate
        return forecast


def build_backbone(
    name: str,
    channels: int,
    context: int,
    horizon: int,
    d_model: int = 48,
) -> ForecastBackbone:
    """Construct one of the three clean-room backbone families."""
    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"patch", "patch_transformer", "patchtransformer", "patchtst"}:
        return PatchTransformer(channels, context, horizon, d_model=d_model)
    if normalized in {"period", "period_conv", "periodconvnet", "timesnet"}:
        return PeriodConvNet(channels, context, horizon, d_model=d_model)
    if normalized in {"ssm", "selective_ssm", "selectivessm", "mamba", "mambassm"}:
        return SelectiveSSM(channels, context, horizon, d_model=d_model)
    raise ValueError(
        f"unknown backbone {name!r}; expected patch_transformer, period_conv, or selective_ssm"
    )


def _self_test() -> None:
    torch.manual_seed(7)
    batch, context, channels, horizon, width = 2, 48, 5, 8, 24
    x = torch.randn(batch, context, channels)
    analog = torch.randn(batch, horizon, channels)
    distance = torch.rand(batch, channels)
    divergence = torch.rand(batch, 1, channels)

    for name in ("patch_transformer", "period_conv", "selective_ssm"):
        base = build_backbone(name, channels, context, horizon, d_model=width)
        latent = base.encode(x)
        forecast = base.decode(latent)
        assert latent.shape == (batch, channels, width)
        assert forecast.shape == (batch, horizon, channels)
        assert base(x).shape == forecast.shape
        assert torch.isfinite(forecast).all()

        for condition in ("native", "lora", "srf", "joint"):
            conditioned = ConditionModel(copy.deepcopy(base), condition=condition, rank=4)
            kwargs = {}
            if condition in {"srf", "joint"}:
                kwargs = {
                    "analog_future": analog,
                    "distance": distance,
                    "divergence": divergence,
                }
            output, gate = conditioned(x, return_gate=True, **kwargs)
            assert output.shape == (batch, horizon, channels)
            auxiliary_output, auxiliary = conditioned(
                x, return_aux=True, **kwargs
            )
            assert torch.allclose(output, auxiliary_output, rtol=0.0, atol=1e-6)
            required_auxiliary = {
                "bypass_forecast",
                "analogue",
                "ungated_forecast",
                "gamma",
                "selector_weights",
                "valid_reference",
            }
            assert required_auxiliary.issubset(auxiliary)
            assert auxiliary["bypass_forecast"].shape == output.shape
            assert auxiliary["analogue"].shape == output.shape
            assert auxiliary["ungated_forecast"].shape == output.shape
            assert auxiliary["gamma"].shape == output.shape
            assert auxiliary["valid_reference"].shape == (batch, 1, channels)
            assert auxiliary["valid_reference"].dtype == torch.bool
            trainable_names = {
                parameter_name
                for parameter_name, parameter in conditioned.named_parameters()
                if parameter.requires_grad
            }
            if condition == "native":
                assert not trainable_names
                assert gate is None
                assert auxiliary["selector_weights"].shape == (batch, channels, 0)
            if condition == "lora":
                assert any(name.endswith("lora_A") for name in trainable_names)
                assert any(name.endswith("lora_B") for name in trainable_names)
                assert any(name.startswith("base.forecast_head.") for name in trainable_names)
                assert gate is None
                assert auxiliary["selector_weights"].shape == (batch, channels, 0)
            if condition == "srf":
                assert any(name.startswith("fusion.") for name in trainable_names)
                assert any(name.startswith("base.forecast_head.") for name in trainable_names)
                assert auxiliary["selector_weights"].shape == (batch, channels, 1)
            if condition == "joint":
                assert any(name.endswith("lora_A") for name in trainable_names)
                assert any(name.startswith("fusion.") for name in trainable_names)
                assert auxiliary["selector_weights"].shape == (batch, channels, 1)
            if gate is not None:
                assert gate.shape == output.shape
                assert torch.allclose(gate.mean(), torch.tensor(0.10), atol=1e-6)
                assert torch.allclose(auxiliary["gate"], gate, rtol=0.0, atol=1e-6)

            # At initialization, B=0 makes LoRA exactly identity; both lora
            # blend=0 and blend=1 therefore reproduce Native bit-for-bit.
            if condition == "lora":
                conditioned.set_lora_blend(0.0)
                fallback = conditioned(x)
                conditioned.set_lora_blend(1.0)
                adapted = conditioned(x)
                assert torch.allclose(fallback, adapted, rtol=0.0, atol=1e-6)

    # The memory builder pads missing neighbours with float32 max, which is
    # finite.  Exercise full-row, per-channel, inf, and NaN invalid references
    # directly so a future masking regression cannot create a fake analogue.
    candidate_count = 3
    native = torch.randn(batch, horizon, channels)
    candidates = torch.randn(batch, candidate_count, horizon, channels)
    query = torch.randn(batch, channels, width)
    sentinel = torch.finfo(torch.float32).max
    candidate_distance = torch.rand(batch, candidate_count, channels)
    candidate_distance[0, 0, :] = sentinel
    candidate_distance[0, 1, :] = float("inf")
    candidate_distance[0, 2, :] = float("nan")
    candidate_distance[1, :, 0] = sentinel
    candidate_distance[1, 0, 1] = sentinel
    candidate_divergence = torch.rand(batch, 1, channels)
    fusion = SelectiveRetrievalFusion(width, horizon, hidden=12)
    fused, invalid_gate, auxiliary = fusion(
        native,
        candidates,
        candidate_distance,
        candidate_divergence,
        query,
        return_aux=True,
    )
    valid_reference = auxiliary["valid_reference"]
    assert not bool(valid_reference[0].any())
    assert not bool(valid_reference[1, :, 0].any())
    assert bool(valid_reference[1, :, 1:].all())
    invalid_forecast = ~valid_reference.expand_as(native)
    for name in (
        "analogue",
        "conditional_forecast",
        "ungated_forecast",
    ):
        assert torch.equal(
            auxiliary[name].masked_select(invalid_forecast),
            native.masked_select(invalid_forecast),
        )
    assert torch.equal(
        fused.masked_select(invalid_forecast),
        native.masked_select(invalid_forecast),
    )
    assert torch.count_nonzero(invalid_gate.masked_select(invalid_forecast)) == 0
    assert torch.count_nonzero(auxiliary["gamma"].masked_select(invalid_forecast)) == 0
    invalid_selector_row = ~valid_reference.transpose(1, 2)
    assert torch.count_nonzero(
        auxiliary["selector_weights"].masked_select(
            invalid_selector_row.expand_as(auxiliary["selector_weights"])
        )
    ) == 0
    assert auxiliary["selector_weights"][1, 1, 0].item() == 0.0
    valid_weight_sums = auxiliary["selector_weights"].sum(dim=-1, keepdim=True)
    assert torch.allclose(
        valid_weight_sums.masked_select(valid_reference.transpose(1, 2)),
        torch.ones_like(
            valid_weight_sums.masked_select(valid_reference.transpose(1, 2))
        ),
        rtol=0.0,
        atol=1e-6,
    )

    print("models.py self-test passed")


if __name__ == "__main__":
    _self_test()
