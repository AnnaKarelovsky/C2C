"""CacheOptimizeModel — HF model wrapper with learnable KV cache prefixes.

Wraps a pretrained model, freezes its parameters, and provides a `register()`
method to turn KV cache outputs of specific input prefixes into learnable
nn.Parameters with RoPE stripped (position-free storage).
"""

import hashlib
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from rosetta.optimize.rope_utils import (
    RoPEConfig,
    apply_rope,
    extract_rope_config,
    inverse_rope,
)
from rosetta.optimize.utils import tool_meta_key


def _get_full_attention_layers(model: PreTrainedModel) -> List[int]:
    """Return indices of full-attention layers (non-sliding-window).

    Models without ``layer_types`` in config are treated as all full-attention.
    """
    layer_types = getattr(model.config, "layer_types", None)
    num_layers = model.config.num_hidden_layers
    if layer_types is None:
        return list(range(num_layers))
    return [i for i, lt in enumerate(layer_types) if lt == "full_attention"]


class CacheOptimizeModel(nn.Module):
    """Model wrapper that makes KV cache prefixes into learnable parameters.

    After wrapping, the underlying model is frozen. Call `register(input_ids)`
    to run a frozen forward pass, extract the KV cache, strip RoPE from keys,
    and store them as trainable nn.Parameters discoverable by optimizers.

    Only full-attention layers are registered as learnable parameters.
    Sliding-window attention layers are skipped because their KV cache only
    covers a local window and will be populated by the frozen prefill during
    ``prepare()``.

    Args:
        model: A HuggingFace PreTrainedModel (e.g., AutoModelForCausalLM).
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.rope_config: RoPEConfig = extract_rope_config(model)
        self.full_attention_layers: List[int] = _get_full_attention_layers(model)
        self._registry: Dict[str, dict] = {}
        self._tool_metas: Dict[str, dict] = {}
        self._param_counter: int = 0

    def kv_parameters(self):
        """Yield learned KV cache parameters."""
        for entry in self._registry.values():
            yield getattr(self, entry["key_param"])
            yield getattr(self, entry["val_param"])

    def _segment_positions(
        self,
        start: int,
        end: int,
        cache_len: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute absolute RoPE positions for a registered segment.

        Args:
            start: Segment start index in ``input_ids``.
            end: Segment end index in ``input_ids``.
            cache_len: Number of tokens already in ``past_key_values``.
                Used to offset positions when ``input_ids`` excludes
                the cached prefix.  Ignored if ``position_ids`` is given.
            position_ids: Optional ``(B, L)`` explicit position ids
                for ``input_ids``.  If provided, absolute positions are
                read directly from ``position_ids[0, start:end]``.

        Returns:
            1-D tensor of absolute positions, shape ``(end - start,)``.
        """
        if position_ids is not None:
            return position_ids[0, start:end]
        return torch.arange(cache_len + start, cache_len + end)

    def _hash_input_ids(self, input_ids: torch.Tensor) -> str:
        """Compute a deterministic hash for input_ids.

        Args:
            input_ids: 1D (N,) or 2D (1, N) tensor of token ids.

        Returns:
            16-char hex string (64-bit hash, sufficient for small registries).
        """
        flat = input_ids.view(-1)
        return hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()[:16]

    def register(
        self,
        input_ids: torch.Tensor,
        prefix: Optional[torch.Tensor] = None,
    ) -> str:
        """Register an input prefix as a learnable KV cache segment.

        Runs a frozen forward pass to obtain the KV cache, strips RoPE from
        keys, and stores both keys and values as trainable nn.Parameters.

        An optional ``prefix`` can be provided to give the model prior context
        during prefill (e.g., a system prompt).  The prefix is prepended to
        ``input_ids`` for the forward pass so that the segment's KV cache is
        conditioned on it, but only the segment's own cache entries are
        extracted and stored.  The prefix does **not** affect the hash.

        Args:
            input_ids: 1D (N,) or 2D (1, N) tensor of token ids for the
                segment to register.
            prefix: Optional 1D (P,) or 2D (1, P) tensor of token ids placed
                before ``input_ids`` during prefill.  Not included in the hash.

        Returns:
            Hash key identifying this registered segment.
        """
        # Normalize shape
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.shape[0] == 1, (
            f"register() expects batch=1, got {input_ids.shape[0]}"
        )

        # Normalize prefix shape
        if prefix is not None:
            if prefix.ndim == 1:
                prefix = prefix.unsqueeze(0)
            assert prefix.shape[0] == 1, (
                f"prefix expects batch=1, got {prefix.shape[0]}"
            )

        # Check for duplicate (hash only the segment, not the prefix)
        key = self._hash_input_ids(input_ids)
        if key in self._registry:
            return key

        # Build the full input sequence: [prefix, input_ids]
        seg_len = input_ids.shape[1]
        if prefix is not None:
            prefix_len = prefix.shape[1]
            full_ids = torch.cat([prefix, input_ids], dim=1)
        else:
            prefix_len = 0
            full_ids = input_ids

        # Move to model device and run frozen forward pass
        full_ids = full_ids.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=full_ids, use_cache=True)
        cache = outputs.past_key_values

        # Extract segment entries from full-attention layers only.
        # Sliding-window layers are skipped — their KV will come from
        # the frozen prefill during prepare().
        # Move to CPU since device_map="auto" may spread layers across GPUs.
        fa_layers = self.full_attention_layers
        seg_keys = torch.stack(
            [cache.layers[i].keys[:, :, -seg_len:, :].cpu()
             for i in fa_layers]
        )
        seg_values = torch.stack(
            [cache.layers[i].values[:, :, -seg_len:, :].cpu()
             for i in fa_layers]
        )

        # Strip RoPE from keys using the segment's actual positions
        positions = torch.arange(prefix_len, prefix_len + seg_len)
        position_free_keys = inverse_rope(
            seg_keys.float(), positions, self.rope_config
        )
        position_free_keys = position_free_keys.to(seg_keys.dtype)

        # Register as nn.Parameters
        idx = self._param_counter
        self._param_counter += 1
        key_param = nn.Parameter(position_free_keys.clone().detach())
        val_param = nn.Parameter(seg_values.clone().detach())
        key_name = f"kv_key_{idx}"
        val_name = f"kv_val_{idx}"
        self.register_parameter(key_name, key_param)
        self.register_parameter(val_name, val_param)

        # Store in registry
        self._registry[key] = {
            "key_param": key_name,
            "val_param": val_name,
            "input_ids": input_ids.clone().cpu(),
            "length": seg_len,
        }

        return key

    def prepare(
        self,
        kv_cache_indices: List[tuple],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        for_generate: bool = False,
        **kwargs,
    ) -> dict:
        """Build a mixed KV cache (frozen prefill + learned segments).

        Prefills ``input_ids[:, :prefill_end]`` through the frozen model, then
        replaces registered segments' KV entries in full-attention layers with
        the learned parameters (with RoPE re-applied).  SWA layers keep their
        frozen prefill cache unchanged.

        Args:
            kv_cache_indices: List of ``(start, end)`` pairs that index into
                ``input_ids``.  Each slice ``input_ids[0, start:end]`` must
                have been previously registered via :meth:`register`.
            input_ids: ``(B, L)`` token ids for the full sequence.
            attention_mask: Optional ``(B, L)`` attention mask.
            position_ids: Optional ``(B, L)`` absolute position ids for
                ``input_ids``.  If provided, RoPE positions for registered
                segments are read from here.  Otherwise, positions are
                inferred as ``cache_len + index``.
            labels: Optional ``(B, L)`` labels for loss computation.
            for_generate: If ``True``, return kwargs suitable for
                ``model.generate(**result)`` instead of ``model(**result)``.
                Returns full ``input_ids`` (not sliced) and an
                ``attention_mask``, since HF generate slices off cached
                tokens internally.
            **kwargs: Extra kwargs forwarded to model (e.g. ``past_key_values``).

        Returns:
            Dict of kwargs suitable for ``self.forward(**result)``
            (default) or ``model.generate(**result)`` (if ``for_generate``).
        """
        B, full_len = input_ids.shape

        # ------------------------------------------------------------------
        # 1. VALIDATE kv_cache_indices against registry
        # ------------------------------------------------------------------
        resolved_segments = []  # (start, end, registry_entry)
        for start, end in kv_cache_indices:
            seg_ids = input_ids[0, start:end]
            seg_hash = self._hash_input_ids(seg_ids)
            if seg_hash not in self._registry:
                raise ValueError(
                    f"Segment input_ids[0, {start}:{end}] not registered "
                    f"(hash={seg_hash}). Call register() first."
                )
            # Batch consistency: all samples must have identical tokens
            if B > 1:
                for b in range(1, B):
                    if not torch.equal(input_ids[b, start:end], seg_ids):
                        raise ValueError(
                            f"Batch element {b} differs from element 0 at "
                            f"positions [{start}:{end}]."
                        )
            resolved_segments.append((start, end, self._registry[seg_hash]))

        prefill_end = max(end for _, end, _ in resolved_segments)

        # ------------------------------------------------------------------
        # 2. CHECK EXISTING CACHE (incremental support)
        # ------------------------------------------------------------------
        past_kv = kwargs.pop("past_key_values", None)
        if past_kv is not None and len(past_kv.layers) > 0:
            # Use a full-attention layer to check cache length
            fa_idx = self.full_attention_layers[0]
            cache_len = past_kv.get_seq_length(fa_idx)
        else:
            cache_len = 0

        if cache_len >= prefill_end:
            # Cache already covers everything
            if for_generate:
                return self._build_generate_kwargs(input_ids, past_kv, B, full_len, attention_mask)
            remaining_ids = input_ids[:, prefill_end:]
            result = {
                "input_ids": remaining_ids,
                "past_key_values": past_kv,
                "use_cache": True,
                **kwargs,
            }
            if attention_mask is not None:
                result["attention_mask"] = attention_mask
            if position_ids is not None:
                result["position_ids"] = position_ids[:, prefill_end:]
            if labels is not None:
                result["labels"] = labels[:, prefill_end:]
            return result

        # ------------------------------------------------------------------
        # 3. FROZEN PREFILL on input_ids[:, cache_len:prefill_end]
        # ------------------------------------------------------------------
        prefill_ids = input_ids[:, cache_len:prefill_end].to(self.model.device)
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=prefill_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        cache = prefill_out.past_key_values

        # ------------------------------------------------------------------
        # 4. REPLACE REGISTERED KV IN FULL-ATTENTION LAYERS
        # ------------------------------------------------------------------
        # Sort segments by start position for ordered splicing
        sorted_segments = sorted(resolved_segments, key=lambda x: x[0])

        for fa_pos, layer_idx in enumerate(self.full_attention_layers):
            layer = cache.layers[layer_idx]
            device = layer.keys.device

            # Clone+detach frozen cache to break grad from model internals
            frozen_keys = layer.keys.clone().detach()   # (B, H, prefill_end, D)
            frozen_vals = layer.values.clone().detach()  # (B, H, prefill_end, D)

            # Build spliced keys via torch.cat
            key_parts = []
            val_parts = []
            prev_end = 0

            for start, end, entry in sorted_segments:
                # Frozen gap before this segment
                if start > prev_end:
                    key_parts.append(frozen_keys[:, :, prev_end:start, :])
                    val_parts.append(frozen_vals[:, :, prev_end:start, :])

                # Learned key: apply RoPE at the correct positions
                key_param = getattr(self, entry["key_param"])  # (L_full, 1, H, N, D)
                val_param = getattr(self, entry["val_param"])  # (L_full, 1, H, N, D)

                positions = self._segment_positions(
                    start, end, cache_len, position_ids
                )
                learned_k = apply_rope(
                    key_param[fa_pos].float(), positions, self.rope_config
                ).to(dtype=frozen_keys.dtype, device=device)
                # fa_pos indexes into the L_full dim → (1, H, N, D)
                learned_v = val_param[fa_pos].to(
                    dtype=frozen_vals.dtype, device=device
                )

                # Expand batch dim if needed: (1, H, N, D) → (B, H, N, D)
                if B > 1:
                    learned_k = learned_k.expand(B, -1, -1, -1)
                    learned_v = learned_v.expand(B, -1, -1, -1)

                key_parts.append(learned_k)
                val_parts.append(learned_v)
                prev_end = end

            # Trailing frozen gap after last segment
            if prev_end < prefill_end:
                key_parts.append(frozen_keys[:, :, prev_end:, :])
                val_parts.append(frozen_vals[:, :, prev_end:, :])

            # Splice via differentiable torch.cat
            spliced_keys = torch.cat(key_parts, dim=-2)
            spliced_vals = torch.cat(val_parts, dim=-2)

            # Assign back — direct attribute assignment preserves autograd
            layer.keys = spliced_keys
            layer.values = spliced_vals

        # SWA layers: untouched — keep their frozen prefill cache as-is

        # ------------------------------------------------------------------
        # 5. BUILD OUTPUT KWARGS
        # ------------------------------------------------------------------
        if for_generate:
            return self._build_generate_kwargs(input_ids, cache, B, full_len, attention_mask)

        remaining_ids = input_ids[:, prefill_end:]
        result = {
            "input_ids": remaining_ids,
            "past_key_values": cache,
            "use_cache": True,
            **kwargs,
        }
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
        if position_ids is not None:
            result["position_ids"] = position_ids[:, prefill_end:]
        if labels is not None:
            result["labels"] = labels[:, prefill_end:]

        return result

    @staticmethod
    def _build_generate_kwargs(
        input_ids: torch.Tensor,
        past_key_values,
        B: int,
        full_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build kwargs for ``model.generate()``."""
        if attention_mask is None:
            attention_mask = torch.ones(B, full_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    # ------------------------------------------------------------------
    # High-level tool API — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _char_to_token_boundaries(
        tokenizer,
        token_ids: List[int],
        char_start: int,
        char_end: int,
    ) -> tuple:
        """Map character-level boundaries to token-level boundaries.

        Decodes each token individually, accumulating character lengths,
        to find the first token whose cumulative decoded length exceeds
        ``char_start`` (→ ``token_start``) and the first whose cumulative
        length reaches ``char_end`` (→ ``token_end``).

        Args:
            tokenizer: HuggingFace tokenizer.
            token_ids: List of token ids.
            char_start: Character offset where the target substring starts.
            char_end: Character offset where the target substring ends.

        Returns:
            ``(token_start, token_end)`` — half-open token index range.
        """
        cumlen = 0
        token_start = None
        token_end = None
        for i, tid in enumerate(token_ids):
            cumlen += len(tokenizer.decode([tid]))
            if token_start is None and cumlen > char_start:
                token_start = i
            if cumlen >= char_end:
                token_end = i + 1
                break
        if token_start is None:
            token_start = len(token_ids)
        if token_end is None:
            token_end = len(token_ids)
        return token_start, token_end

    @staticmethod
    def _find_tool_char_boundaries_json(
        full_text: str,
        tools: List[dict],
    ) -> List[tuple]:
        """Find tool boundaries in JSON-based templates (Qwen, etc.).

        Each tool is rendered as a JSON object with ``"name": "tool_name"``.
        Locates each object by anchoring on the name, walking back to the
        outermost ``{``, and forward via brace-counting to the matching ``}``.

        Searches sequentially (starting after the previous tool's end) to
        avoid matching a tool name in another tool's description.

        Args:
            full_text: Decoded template text.
            tools: List of tool schemas.

        Returns:
            List of ``(char_start, char_end)`` tuples, one per tool.
        """
        boundaries = []
        search_start = 0

        for tool in tools:
            name = tool["function"]["name"]
            anchor = f'"name": "{name}"'
            anchor_pos = full_text.find(anchor, search_start)
            if anchor_pos == -1:
                raise ValueError(
                    f"Tool '{name}' anchor not found in template text "
                    f"(searched from pos {search_start})"
                )
            # Walk back to the outermost { for this tool entry
            brace_pos = full_text.rfind("{", search_start, anchor_pos)
            type_anchor = '{"type"'
            obj_start = full_text.rfind(
                type_anchor, search_start, anchor_pos
            )
            if obj_start == -1:
                obj_start = brace_pos
            # Brace-count forward to find the matching closing }
            depth = 0
            char_end = obj_start
            for ci in range(obj_start, len(full_text)):
                if full_text[ci] == "{":
                    depth += 1
                elif full_text[ci] == "}":
                    depth -= 1
                    if depth == 0:
                        char_end = ci + 1
                        break
            boundaries.append((obj_start, char_end))
            search_start = char_end

        return boundaries

    @staticmethod
    def _find_tool_char_boundaries_gptoss(
        full_text: str,
        tools: List[dict],
    ) -> List[tuple]:
        """Find tool boundaries in GPT-OSS TypeScript-style templates.

        Each tool is rendered as::

            // tool description
            type tool_name = (_: { param: type }) => any;

        Locates each tool by anchoring on ``type tool_name =``, walking
        back to the preceding ``//`` comment line, and forward to the ``;``.

        Args:
            full_text: Decoded template text.
            tools: List of tool schemas.

        Returns:
            List of ``(char_start, char_end)`` tuples, one per tool.
        """
        boundaries = []
        search_start = 0

        for tool in tools:
            name = tool["function"]["name"]
            anchor = f"type {name} ="
            anchor_pos = full_text.find(anchor, search_start)
            if anchor_pos == -1:
                raise ValueError(
                    f"Tool '{name}' anchor not found in template text "
                    f"(searched from pos {search_start})"
                )
            # Expand backward to preceding // comment line
            line_start = full_text.rfind("\n", search_start, anchor_pos)
            if line_start == -1:
                line_start = search_start
            else:
                comment_start = full_text.rfind(
                    "//", search_start, anchor_pos
                )
                if comment_start != -1 and comment_start > line_start:
                    nl_before = full_text.rfind(
                        "\n", search_start, comment_start
                    )
                    if nl_before != -1:
                        line_start = nl_before + 1
                    else:
                        line_start = comment_start
                else:
                    line_start = line_start + 1
            # Expand forward to semicolon
            semi_pos = full_text.find(";", anchor_pos)
            if semi_pos == -1:
                semi_pos = len(full_text)
            char_end = semi_pos + 1
            boundaries.append((line_start, char_end))
            search_start = char_end

        return boundaries

    @staticmethod
    def _find_tool_char_boundaries(
        full_text: str,
        tools: List[dict],
    ) -> List[tuple]:
        """Find each tool's character range in decoded template text.

        Auto-detects the template format and dispatches to the appropriate
        parser:

        - **GPT-OSS** (``namespace functions`` in text) → TypeScript-style
        - **Default** (everything else, including Qwen ``<tools>``) → JSON

        Args:
            full_text: Decoded output of ``apply_chat_template``.
            tools: List of tool schemas (OpenAI function-calling format).

        Returns:
            List of ``(char_start, char_end)`` tuples, one per tool.
        """
        if "namespace functions" in full_text:
            return CacheOptimizeModel._find_tool_char_boundaries_gptoss(
                full_text, tools
            )
        return CacheOptimizeModel._find_tool_char_boundaries_json(
            full_text, tools
        )

    # ------------------------------------------------------------------
    # High-level tool API
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_meta_key(
        tools: List[dict],
        system_message: Optional[dict] = None,
    ) -> str:
        """Compute a deterministic hash key for a (system_message, tools) pair.

        Args:
            tools: List of tool schemas.
            system_message: Optional system message dict.

        Returns:
            16-char hex string.
        """
        return tool_meta_key(tools, system_message)

    def register_tools(
        self,
        tokenizer,
        tools: List[dict],
        system_message: Optional[dict] = None,
        **template_kwargs,
    ) -> str:
        """Register each tool as an independent learnable KV cache segment.

        Registers M tools as M independent KV parameter pairs.  At runtime,
        ``prepare_chat()`` passes the correct per-tool ``kv_cache_indices``
        to :meth:`prepare`.

        Each tool's segment is extracted from the all-tools template so that
        token IDs match exactly at runtime.  The prefix for registration
        (which only affects parameter initialization) is everything before
        the tool's segment in the all-tools template.

        Multiple (system_message, tools) combinations can be registered.
        The correct metadata is looked up automatically in
        :meth:`prepare_chat`.

        Args:
            tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
            tools: List of tool schemas (OpenAI function-calling format).
            system_message: Optional ``{"role": "system", "content": ...}``
                dict that precedes the tool descriptions.
            **template_kwargs: Extra kwargs forwarded to
                ``tokenizer.apply_chat_template`` (e.g. ``enable_thinking``
                for Qwen3 models).  Must be identical to those passed to
                :meth:`prepare_chat`.

        Returns:
            Meta key identifying this (system_message, tools) registration.
        """
        meta_key = self._tool_meta_key(tools, system_message)
        if meta_key in self._tool_metas:
            return meta_key

        msgs = [system_message] if system_message else []

        # Render all-tools template — the source of truth for token IDs
        all_tool_ids = tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            **template_kwargs,
        )
        all_tool_text = tokenizer.decode(all_tool_ids)

        # Find each tool's char and token boundaries
        all_char_bounds = self._find_tool_char_boundaries(all_tool_text, tools)

        per_tool_entries = []
        for i, (char_start, char_end) in enumerate(all_char_bounds):
            tok_start, tok_end = self._char_to_token_boundaries(
                tokenizer, all_tool_ids, char_start, char_end
            )

            # Register segment with everything before it as prefix
            prefix_t = torch.tensor([all_tool_ids[:tok_start]])
            segment_t = torch.tensor([all_tool_ids[tok_start:tok_end]])
            seg_hash = self.register(segment_t, prefix=prefix_t)

            per_tool_entries.append({
                "tool_name": tools[i]["function"]["name"],
                "token_start": tok_start,
                "token_end": tok_end,
                "segment_len": tok_end - tok_start,
                "hash": seg_hash,
            })

        self._tool_metas[meta_key] = {
            "per_tool": per_tool_entries,
            "tool_names": [t["function"]["name"] for t in tools],
        }

        return meta_key

    def prepare_chat(
        self,
        tokenizer,
        messages: List[dict],
        tools: List[dict],
        labels: Optional[torch.Tensor] = None,
        add_generation_prompt: bool = True,
        template_kwargs: Optional[dict] = None,
        for_generate: bool = False,
        **kwargs,
    ) -> dict:
        """Prepare a chat conversation for forward pass with learnable tool KV.

        Renders the full chat template, computes ``kv_cache_indices`` from
        the stored tool metadata, and delegates to :meth:`prepare`.

        The correct metadata is looked up by hashing the system message
        (extracted from ``messages``) and ``tools``.

        Args:
            tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
            messages: Conversation messages (system, user, assistant, ...).
            tools: List of tool schemas (must match :meth:`register_tools`).
            labels: Optional ``(B, L)`` labels for loss computation.
            add_generation_prompt: Whether to append a generation prompt.
                Use ``True`` (default) for inference, ``False`` for training
                on completed conversations.
            template_kwargs: Extra kwargs forwarded to
                ``tokenizer.apply_chat_template`` (e.g. ``enable_thinking``
                for Qwen3 models).  Must be identical to those passed to
                :meth:`register_tools`.
            for_generate: If ``True``, return kwargs suitable for
                ``model.generate(**result)``.  See :meth:`prepare`.
            **kwargs: Extra kwargs forwarded to :meth:`prepare`.

        Returns:
            Dict of kwargs suitable for ``self.forward(**result)``
            (default) or ``model.generate(**result)`` (if ``for_generate``).
        """
        template_kwargs = template_kwargs or {}

        # Extract system message from messages
        system_message = None
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg
                break

        meta_key = self._tool_meta_key(tools, system_message)
        if meta_key not in self._tool_metas:
            raise RuntimeError(
                f"No tools registered for this (system_message, tools) "
                f"combination (key={meta_key}). "
                f"Call register_tools() first."
            )
        meta = self._tool_metas[meta_key]

        full_ids = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        full_ids_t = torch.tensor([full_ids])

        kv_cache_indices = [
            (entry["token_start"], entry["token_end"])
            for entry in meta["per_tool"]
        ]
        return self.prepare(
            kv_cache_indices=kv_cache_indices,
            input_ids=full_ids_t,
            labels=labels,
            for_generate=for_generate,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str) -> None:
        """Save learned KV parameters and metadata.

        Creates two files in *save_dir*:

        - ``kv_params.pt`` — learned KV parameter tensors.
        - ``kv_config.json`` — registry and tool metadata (human-readable).

        After loading with :meth:`load_pretrained`, :meth:`prepare_chat`
        and :meth:`prepare` work immediately without re-registration.
        """
        os.makedirs(save_dir, exist_ok=True)

        # KV parameter tensors
        kv_state = {
            k: v
            for k, v in self.state_dict().items()
            if k.startswith("kv_key_") or k.startswith("kv_val_")
        }
        torch.save(kv_state, os.path.join(save_dir, "kv_params.pt"))

        # Metadata (JSON-serializable)
        config = {
            "tool_metas": self._tool_metas,
            "registry": {
                h: {
                    "key_param": e["key_param"],
                    "val_param": e["val_param"],
                    "length": e["length"],
                    "input_ids": e["input_ids"].squeeze(0).tolist(),
                }
                for h, e in self._registry.items()
            },
            "param_counter": self._param_counter,
        }
        with open(os.path.join(save_dir, "kv_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_pretrained(self, save_dir: str) -> None:
        """Load learned KV parameters and metadata.

        Restores the registry, tool metadata, and trained KV parameters.
        After loading, :meth:`prepare_chat` and :meth:`prepare` work
        immediately without calling :meth:`register_tools`.

        Args:
            save_dir: Directory previously written by :meth:`save_pretrained`.
        """
        with open(os.path.join(save_dir, "kv_config.json")) as f:
            config = json.load(f)

        kv_state = torch.load(
            os.path.join(save_dir, "kv_params.pt"),
            map_location="cpu",
            weights_only=True,
        )

        # Clear existing registrations
        for entry in self._registry.values():
            for pname in (entry["key_param"], entry["val_param"]):
                if pname in self._parameters:
                    del self._parameters[pname]

        # Restore metadata
        self._tool_metas = config["tool_metas"]
        self._param_counter = config["param_counter"]

        # Restore registry and parameters
        self._registry = {}
        for hash_key, meta in config["registry"].items():
            key_name = meta["key_param"]
            val_name = meta["val_param"]

            self.register_parameter(
                key_name, nn.Parameter(kv_state[key_name])
            )
            self.register_parameter(
                val_name, nn.Parameter(kv_state[val_name])
            )

            self._registry[hash_key] = {
                "key_param": key_name,
                "val_param": val_name,
                "length": meta["length"],
                "input_ids": torch.tensor(meta["input_ids"]).unsqueeze(0),
            }

    def forward(self, **kwargs):
        """Delegate to the wrapped model."""
        return self.model(**kwargs)
