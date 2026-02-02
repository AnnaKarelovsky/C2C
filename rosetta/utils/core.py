"""
Core utilities for Cache-to-Cache (C2C) operations.

Includes:
- Sharer/mask conversion utilities for KV-cache projection
- Token-level metric computations (entropy, perplexity, etc.)
- Model prefill utilities for analysis
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def sharers_to_mask(sharer_indices: List[int]) -> int:
    """
    Convert a list of sharer indices to a bitmask.
    
    Args:
        sharer_indices: List of 1-based sharer indices (e.g., [1, 2, 3])
        
    Returns:
        Bitmask integer (e.g., [1, 2] -> 3, [1, 3] -> 5, [1, 2, 3] -> 7)
    
    Example:
        >>> sharers_to_mask([1])      # 001 = 1
        1
        >>> sharers_to_mask([2])      # 010 = 2
        2
        >>> sharers_to_mask([1, 2])   # 011 = 3
        3
        >>> sharers_to_mask([1, 3])   # 101 = 5
        5
    """
    mask = 0
    for idx in sharer_indices:
        mask |= (1 << (idx - 1))
    return mask


def mask_to_sharers(mask: int) -> List[int]:
    """
    Convert a bitmask to a list of sharer indices.
    
    Args:
        mask: Bitmask integer
        
    Returns:
        List of 1-based sharer indices
    
    Example:
        >>> mask_to_sharers(1)   # 001 -> [1]
        [1]
        >>> mask_to_sharers(3)   # 011 -> [1, 2]
        [1, 2]
        >>> mask_to_sharers(5)   # 101 -> [1, 3]
        [1, 3]
        >>> mask_to_sharers(7)   # 111 -> [1, 2, 3]
        [1, 2, 3]
    """
    if mask <= 0:
        return []
    sharers = []
    idx = 1
    while mask:
        if mask & 1:
            sharers.append(idx)
        mask >>= 1
        idx += 1
    return sharers


def all_sharers_mask(num_sharers: int) -> int:
    """
    Get bitmask that selects all sharers.
    
    Args:
        num_sharers: Number of sharers
        
    Returns:
        Bitmask with all bits set (e.g., 3 sharers -> 7 = 111)
    """
    return (1 << num_sharers) - 1


def format_sharer_mask(mask: int) -> str:
    """
    Format a sharer mask as a human-readable string.
    
    Args:
        mask: Bitmask integer (-1=no projection, 0=self projection, >0=sharer bitmask)
        
    Returns:
        Formatted string like "sharers [1, 2]" or "no projection"
    """
    if mask < 0:
        return "no projection"
    if mask == 0:
        return "self projection"
    sharers = mask_to_sharers(mask)
    return f"sharers {sharers}"


# =============================================================================
# Token-Level Metrics
# =============================================================================


class TokenMetric(ABC):
    """Abstract base class for per-token metrics computed from model outputs.

    Subclasses implement specific metrics like entropy, perplexity, etc.
    All metrics take logits and input_ids, returning per-token values.

    Metrics can be "shifted" or "non-shifted":
    - Non-shifted (is_shifted=False): Output length equals input length (seq_len).
      Example: entropy[i] = uncertainty at position i.
    - Shifted (is_shifted=True): Output length is seq_len-1.
      Example: perplexity[i] = how surprised we are by token at position i+1.
      The metric at index i corresponds to predicting token i+1 from context 0:i.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name for reporting and identification."""
        ...

    @property
    def is_shifted(self) -> bool:
        """Whether this metric is shifted for next-token prediction.

        If True, the metric has length seq_len-1 and value[i] corresponds
        to predicting the token at position i+1. Used for alignment in
        visualization and analysis.

        Default: False (same length as input sequence).
        """
        return False

    @abstractmethod
    def compute(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute metric for each token position.

        Args:
            logits: Model output logits [seq_len, vocab_size] or [batch, seq_len, vocab_size].
            input_ids: Input token IDs [seq_len] or [batch, seq_len].

        Returns:
            Per-token metric values with same batch dimensions as input.
            Length is seq_len if is_shifted=False, seq_len-1 if is_shifted=True.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class EntropyMetric(TokenMetric):
    """Per-token entropy of the probability distribution.

    Entropy H(p) = -sum(p * log(p)) measures the uncertainty in the
    model's predictions. Higher entropy means more uncertainty.
    """

    @property
    def name(self) -> str:
        return "entropy"

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # H(p) = -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Avoid -inf * 0 = nan by using where
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class PerplexityMetric(TokenMetric):
    """Per-token perplexity (exp of cross-entropy loss).

    For each position i, computes exp(-log(p(token_{i+1} | token_{0:i}))).
    This is the perplexity of predicting the next token.

    Note: Returns values for positions 0 to seq_len-2 (predicting tokens 1 to seq_len-1).
    The last position has no next token to predict.
    """

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Shift for next-token prediction
        # logits[i] predicts token at position i+1
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        # Cross-entropy loss per token
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)

        # Perplexity = exp(CE)
        return torch.exp(ce_loss)


class NegLogProbMetric(TokenMetric):
    """Negative log probability of each token.

    For position i, returns -log(p(token_{i+1} | token_{0:i})).
    Lower values indicate the model is more confident about the token.

    Note: Returns values for positions 0 to seq_len-2.
    """

    @property
    def name(self) -> str:
        return "neg_log_prob"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return -token_log_probs


class TopKAccuracyMetric(TokenMetric):
    """Whether the true next token is in the top-k predictions.

    Returns 1.0 if the actual next token is among the top-k predictions,
    0.0 otherwise. Useful for measuring prediction quality.
    """

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return f"top{self.k}_accuracy"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        # Get top-k predictions
        _, top_k_indices = torch.topk(shift_logits, k=self.k, dim=-1)

        # Check if true label is in top-k
        labels_expanded = shift_labels.unsqueeze(-1).expand_as(top_k_indices)
        in_top_k = (top_k_indices == labels_expanded).any(dim=-1)

        return in_top_k.float()


class RankMetric(TokenMetric):
    """Rank of the true next token in the probability distribution.

    Returns the 1-based rank of the actual next token. Rank 1 means
    the model's top prediction was correct.
    """

    @property
    def name(self) -> str:
        return "rank"

    @property
    def is_shifted(self) -> bool:
        return True

    def compute(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]

        # Sort indices by logit value (descending)
        sorted_indices = torch.argsort(shift_logits, dim=-1, descending=True)

        # Find rank of each true label
        # Create position indices
        ranks = torch.zeros_like(shift_labels, dtype=torch.float)
        for i in range(shift_logits.size(-1)):
            matches = sorted_indices[..., i] == shift_labels
            ranks[matches] = i + 1  # 1-based rank

        return ranks


# Default metrics for convenience
DEFAULT_METRICS = [EntropyMetric(), PerplexityMetric(), NegLogProbMetric()]


def get_metric_by_name(name: str) -> TokenMetric:
    """Get a metric instance by name.

    Args:
        name: Metric name ("entropy", "perplexity", "neg_log_prob", "rank", "topN_accuracy").

    Returns:
        TokenMetric instance.
    """
    name = name.lower()
    if name == "entropy":
        return EntropyMetric()
    elif name == "perplexity":
        return PerplexityMetric()
    elif name == "neg_log_prob":
        return NegLogProbMetric()
    elif name == "rank":
        return RankMetric()
    elif name.startswith("top") and name.endswith("_accuracy"):
        k = int(name[3:-9])
        return TopKAccuracyMetric(k=k)
    else:
        raise ValueError(f"Unknown metric: {name}")


# =============================================================================
# Model Prefill and Metric Computation
# =============================================================================


def prefill_and_compute_metrics(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    metrics: Optional[List[TokenMetric]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Prefill model with input sequence and compute metrics.

    This runs a forward pass through the model to get logits for all
    positions, then computes the requested metrics.

    Args:
        model: HuggingFace model (or RosettaModel).
        input_ids: Token IDs [seq_len] or [batch, seq_len].
        metrics: List of metric objects to compute. Defaults to entropy, perplexity, neg_log_prob.
        attention_mask: Optional attention mask.
        device: Device to move inputs to. If None, lets the model handle device
            placement (works with device_map="auto" for multi-GPU setups).

    Returns:
        Dict mapping metric name to per-token values tensor.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # Ensure batch dimension
    was_1d = input_ids.dim() == 1
    if was_1d:
        input_ids = input_ids.unsqueeze(0)

    # Only move to device if explicitly specified
    # For device_map="auto", the model handles placement internally
    if device is not None:
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    logits = outputs.logits
    if was_1d:
        logits = logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    # Compute metrics
    results = {}
    for metric in metrics:
        values = metric.compute(logits, input_ids)
        results[metric.name] = values

    return results


def compute_metrics_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    metrics: Optional[List[TokenMetric]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute metrics from pre-computed logits.

    Useful when you already have logits from a forward pass.

    Args:
        logits: Model output logits [seq_len, vocab_size] or [batch, seq_len, vocab_size].
        input_ids: Input token IDs [seq_len] or [batch, seq_len].
        metrics: List of metric objects. Defaults to entropy, perplexity, neg_log_prob.

    Returns:
        Dict mapping metric name to per-token values tensor.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    return {m.name: m.compute(logits, input_ids) for m in metrics}


def align_metrics(
    metrics_dict: Dict[str, torch.Tensor],
    metric_objects: Optional[List[TokenMetric]] = None,
    mode: str = "truncate",
) -> Dict[str, torch.Tensor]:
    """Align all metrics to the same length.

    Different metrics may have different lengths:
    - Non-shifted metrics (entropy): length seq_len
    - Shifted metrics (perplexity, neg_log_prob): length seq_len-1

    This function aligns them to a common length for easier comparison.

    Args:
        metrics_dict: Dict mapping metric name to per-token values tensor.
        metric_objects: Optional list of TokenMetric objects to get is_shifted info.
            If not provided, infers from tensor lengths (shortest is assumed shifted).
        mode: Alignment mode:
            - "truncate": Truncate longer metrics to match shorter ones (removes first token).
            - "pad_nan": Pad shorter metrics with NaN at position 0.
            - "pad_zero": Pad shorter metrics with 0 at position 0.

    Returns:
        Dict with all metrics aligned to the same length.
    """
    if not metrics_dict:
        return metrics_dict

    # Get lengths
    lengths = {name: len(values) for name, values in metrics_dict.items()}
    min_len = min(lengths.values())
    max_len = max(lengths.values())

    if min_len == max_len:
        return metrics_dict  # Already aligned

    # Determine which metrics are shifted
    if metric_objects is not None:
        shifted = {m.name: m.is_shifted for m in metric_objects if m.name in metrics_dict}
    else:
        # Infer: shorter length = shifted
        shifted = {name: length == min_len for name, length in lengths.items()}

    aligned = {}
    for name, values in metrics_dict.items():
        if mode == "truncate":
            # Truncate non-shifted metrics from the start (remove position 0)
            if not shifted.get(name, False) and len(values) > min_len:
                aligned[name] = values[1:]  # Remove first element
            else:
                aligned[name] = values
        elif mode == "pad_nan":
            # Pad shifted metrics with NaN at the start
            if shifted.get(name, True) and len(values) < max_len:
                pad = torch.full((1,), float("nan"), dtype=values.dtype, device=values.device)
                aligned[name] = torch.cat([pad, values])
            else:
                aligned[name] = values
        elif mode == "pad_zero":
            # Pad shifted metrics with 0 at the start
            if shifted.get(name, True) and len(values) < max_len:
                pad = torch.zeros((1,), dtype=values.dtype, device=values.device)
                aligned[name] = torch.cat([pad, values])
            else:
                aligned[name] = values
        else:
            raise ValueError(f"Unknown alignment mode: {mode}")

    return aligned


def prefill_and_compute_metrics_aligned(
    model: "PreTrainedModel",
    input_ids: torch.Tensor,
    metrics: Optional[List[TokenMetric]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    align_mode: str = "truncate",
) -> Dict[str, torch.Tensor]:
    """Prefill model and compute aligned metrics.

    Convenience function that computes metrics and aligns them to the same length.

    Args:
        model: HuggingFace model.
        input_ids: Token IDs [seq_len] or [batch, seq_len].
        metrics: List of metric objects. Defaults to entropy, perplexity, neg_log_prob.
        attention_mask: Optional attention mask.
        device: Device to run on.
        align_mode: Alignment mode ("truncate", "pad_nan", "pad_zero").

    Returns:
        Dict with aligned metrics (all same length).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    results = prefill_and_compute_metrics(
        model=model,
        input_ids=input_ids,
        metrics=metrics,
        attention_mask=attention_mask,
        device=device,
    )

    return align_metrics(results, metric_objects=metrics, mode=align_mode)
