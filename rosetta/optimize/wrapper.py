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
        self._registry: Dict[str, dict] = {}  # always keyed by token_hash
        self._json_to_token: Dict[str, str] = {}  # json_hash → token_hash
        self._param_counter: int = 0

    def kv_parameters(self):
        """Yield learned KV cache parameters."""
        for entry in self._registry.values():
            yield getattr(self, entry["key_param"])
            yield getattr(self, entry["val_param"])

    def set_trainable_tools(self, tool_names: Optional[List[str]]) -> None:
        """Toggle which tools' KV params are trainable.

        Only the named tools will have ``requires_grad=True``; all others
        are frozen.  Pass ``None`` to make all tools trainable again.

        Safe to call between micro-batches within a gradient-accumulation
        window — already-accumulated ``.grad`` tensors are preserved;
        ``requires_grad_(False)`` only prevents future backward passes
        from writing to ``.grad``.

        Args:
            tool_names: List of tool names whose KV params should be
                trainable, or ``None`` to unfreeze everything.
        """
        active = set(tool_names) if tool_names is not None else None
        for entry in self._registry.values():
            trainable = active is None or entry.get("tool_name") in active
            getattr(self, entry["key_param"]).requires_grad_(trainable)
            getattr(self, entry["val_param"]).requires_grad_(trainable)

    @property
    def registered_tools(self) -> Dict[str, dict]:
        """Return registered tools as ``{tool_name: tool_schema}``."""
        return {
            entry["tool_name"]: entry["tool_schema"]
            for entry in self._registry.values()
            if "tool_name" in entry
        }

    def get_registry_entry(self, hash_key: str, *, by: str = "json") -> dict:
        """Look up a registry entry by explicit hash type.

        Args:
            hash_key: Hash string to look up.
            by: ``"json"`` (default) resolves via ``_json_to_token`` then
                ``_registry``; ``"token"`` looks up ``_registry`` directly.

        Returns:
            The registry entry dict.

        Raises:
            KeyError: If hash_key cannot be resolved to any entry.
        """
        if by == "token":
            if hash_key in self._registry:
                return self._registry[hash_key]
            raise KeyError(f"Token hash {hash_key!r} not in registry")
        if by == "json":
            token_hash = self._json_to_token.get(hash_key)
            if token_hash is not None and token_hash in self._registry:
                return self._registry[token_hash]
            raise KeyError(f"JSON hash {hash_key!r} not in json_to_token mapping")
        raise ValueError(f"by must be 'json' or 'token', got {by!r}")

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

    @staticmethod
    def _hash_tool_schema(schema: dict) -> str:
        """Compute a position-independent hash for a tool schema.

        Uses the canonical JSON serialization (sorted keys) so that the
        same tool schema always produces the same hash regardless of its
        position in the tool list or tokenization context.

        Args:
            schema: Tool schema dict (OpenAI function-calling format).

        Returns:
            16-char hex string.
        """
        canonical = json.dumps(schema, sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()[:16]

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
            Token-hash key identifying this registered segment.
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
        # Keep on model device so params stay on GPU (avoids CPU↔GPU
        # transfers every forward/backward pass).
        fa_layers = self.full_attention_layers
        param_device = self.model.device
        seg_keys = torch.stack(
            [cache.layers[i].keys[:, :, -seg_len:, :].to(param_device)
             for i in fa_layers]
        )
        seg_values = torch.stack(
            [cache.layers[i].values[:, :, -seg_len:, :].to(param_device)
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
        hash_list: Optional[List[str]] = None,
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
            hash_list: Optional list of token-hash strings, one per entry in
                ``kv_cache_indices``.  When provided, these hashes are used
                directly for registry lookup instead of hashing
                ``input_ids``.  This is used by :meth:`prepare_chat` for
                cross-domain tool matching where the runtime tokens may
                differ from the registered tokens.
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
        for idx, (start, end) in enumerate(kv_cache_indices):
            if hash_list is not None:
                seg_hash = hash_list[idx]
            else:
                seg_hash = self._hash_input_ids(input_ids[0, start:end])
            if seg_hash not in self._registry:
                raise ValueError(
                    f"Segment [{start}:{end}] not registered "
                    f"(hash={seg_hash}). Call register() first."
                )
            # Batch consistency: all samples must have identical tokens
            if B > 1:
                seg_ids = input_ids[0, start:end]
                for b in range(1, B):
                    if not torch.equal(input_ids[b, start:end], seg_ids):
                        raise ValueError(
                            f"Batch element {b} differs from element 0 at "
                            f"positions [{start}:{end}]."
                        )
            resolved_segments.append((start, end, self._registry[seg_hash]))

        prefill_end = max(end for _, end, _ in resolved_segments)

        # Move tensors to model device
        device = self.model.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

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

        # Precompute RoPE-applied learned keys for all segments across all
        # full-attention layers at once (1 kernel per segment instead of
        # N_layers kernels).
        precomputed_keys = []
        for start, end, entry in sorted_segments:
            key_param = getattr(self, entry["key_param"])  # (L_full, 1, H, N, D)
            positions = self._segment_positions(
                start, end, cache_len, position_ids
            )
            learned_k = apply_rope(
                key_param.float(), positions, self.rope_config
            )
            precomputed_keys.append(learned_k)

        for fa_pos, layer_idx in enumerate(self.full_attention_layers):
            layer = cache.layers[layer_idx]
            device = layer.keys.device

            # Detach frozen cache — frozen prefill runs under no_grad so
            # these have no grad_fn; only learned segments carry gradients.
            frozen_keys = layer.keys.detach()   # (B, H, prefill_end, D)
            frozen_vals = layer.values.detach()  # (B, H, prefill_end, D)

            # Build spliced keys via torch.cat
            key_parts = []
            val_parts = []
            prev_end = 0

            for seg_idx, (start, end, entry) in enumerate(sorted_segments):
                # Frozen gap before this segment
                if start > prev_end:
                    key_parts.append(frozen_keys[:, :, prev_end:start, :])
                    val_parts.append(frozen_vals[:, :, prev_end:start, :])

                # Index precomputed learned key for this layer
                val_param = getattr(self, entry["val_param"])
                learned_k = precomputed_keys[seg_idx][fa_pos].to(
                    dtype=frozen_keys.dtype, device=device
                )
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
            attention_mask = torch.ones(
                B, full_len, dtype=torch.long, device=input_ids.device,
            )
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

    def register_tools(
        self,
        tokenizer,
        tools: List[dict],
        system_message: Optional[dict] = None,
        **template_kwargs,
    ) -> List[dict]:
        """Register each tool as an independent learnable KV cache segment.

        Each tool's segment is extracted from the all-tools template so that
        token IDs match exactly at runtime.  The prefix for registration
        (which only affects parameter initialization) is everything before
        the tool's segment in the all-tools template.

        Already-registered tools (matched by segment hash) are included in
        the return list but their metadata is not updated.

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
            List of per-tool info dicts, each with keys ``tool_name``,
            ``token_start``, ``token_end``, ``hash``.
        """
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

        per_tool_info = []
        for i, (char_start, char_end) in enumerate(all_char_bounds):
            tok_start, tok_end = self._char_to_token_boundaries(
                tokenizer, all_tool_ids, char_start, char_end
            )
            tool_name = tools[i]["function"]["name"]

            # Register segment with everything before it as prefix
            prefix_t = torch.tensor([all_tool_ids[:tok_start]])
            segment_t = torch.tensor([all_tool_ids[tok_start:tok_end]])
            token_hash = self.register(segment_t, prefix=prefix_t)

            # Map json_hash → token_hash (position-independent lookup)
            json_hash = self._hash_tool_schema(tools[i])
            self._json_to_token[json_hash] = token_hash

            # Store tool metadata in the registry entry (keyed by token_hash)
            entry = self._registry[token_hash]
            if "tool_name" not in entry:
                entry["tool_name"] = tool_name
                entry["tool_schema"] = tools[i]

            per_tool_info.append({
                "tool_name": tool_name,
                "token_start": tok_start,
                "token_end": tok_end,
                "hash": json_hash,
            })

        return per_tool_info

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

        Renders the full chat template, finds registered tool segments by
        recomputing their token boundaries, and delegates to :meth:`prepare`.

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

        full_ids = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        full_ids_t = torch.tensor([full_ids])

        # Find tool boundaries and resolve token_hashes via json_hash
        full_text = tokenizer.decode(full_ids)
        char_bounds = self._find_tool_char_boundaries(full_text, tools)

        kv_cache_indices = []
        hash_list = []
        for i, (char_start, char_end) in enumerate(char_bounds):
            tok_start, tok_end = self._char_to_token_boundaries(
                tokenizer, full_ids, char_start, char_end
            )
            json_hash = self._hash_tool_schema(tools[i])
            token_hash = self._json_to_token.get(json_hash)
            if token_hash is None or token_hash not in self._registry:
                raise RuntimeError(
                    f"Tool '{tools[i]['function']['name']}' not registered "
                    f"(hash={json_hash}). Call register_tools() first."
                )
            kv_cache_indices.append((tok_start, tok_end))
            hash_list.append(token_hash)

        return self.prepare(
            kv_cache_indices, full_ids_t,
            labels=labels, for_generate=for_generate,
            hash_list=hash_list,
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
        # Use json_hash as on-disk key for tool entries (position-independent);
        # keep token-hash for non-tool entries.
        registry_out = {}
        for h, e in self._registry.items():
            if e.get("tool_schema"):
                disk_key = self._hash_tool_schema(e["tool_schema"])
            else:
                disk_key = h
            registry_out[disk_key] = {
                "key_param": e["key_param"],
                "val_param": e["val_param"],
                "length": e["length"],
                "input_ids": e["input_ids"].squeeze(0).tolist(),
                "tool_name": e.get("tool_name"),
                "tool_schema": e.get("tool_schema"),
            }
        config = {
            "registry": registry_out,
            "param_counter": self._param_counter,
        }
        with open(os.path.join(save_dir, "kv_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_pretrained(self, save_dir: str) -> None:
        """Load learned KV parameters and metadata.

        Restores the registry (including tool metadata) and trained KV
        parameters.  After loading, :meth:`prepare_chat` and :meth:`prepare`
        work immediately without calling :meth:`register_tools`.

        Args:
            save_dir: Directory previously written by :meth:`save_pretrained`.
        """
        with open(os.path.join(save_dir, "kv_config.json")) as f:
            config = json.load(f)

        kv_state = torch.load(
            os.path.join(save_dir, "kv_params.pt"),
            map_location=self.model.device,
            weights_only=True,
        )

        # Clear existing registrations
        for entry in self._registry.values():
            for pname in (entry["key_param"], entry["val_param"]):
                if pname in self._parameters:
                    del self._parameters[pname]

        self._param_counter = config["param_counter"]

        # Restore registry and parameters.
        # Registry is always keyed by token_hash (computed from stored input_ids).
        # _json_to_token is rebuilt from tool_schema so prepare_chat() works.
        self._registry = {}
        self._json_to_token = {}
        for hash_key, meta in config["registry"].items():
            key_name = meta["key_param"]
            val_name = meta["val_param"]

            self.register_parameter(
                key_name, nn.Parameter(kv_state[key_name])
            )
            self.register_parameter(
                val_name, nn.Parameter(kv_state[val_name])
            )

            entry = {
                "key_param": key_name,
                "val_param": val_name,
                "length": meta["length"],
                "input_ids": torch.tensor(meta["input_ids"]).unsqueeze(0),
            }
            if meta.get("tool_name") is not None:
                entry["tool_name"] = meta["tool_name"]
                entry["tool_schema"] = meta["tool_schema"]

            # Always key by token_hash so prepare() finds entries directly
            token_key = self._hash_input_ids(entry["input_ids"])
            self._registry[token_key] = entry

            # Rebuild json_hash → token_hash mapping for prepare_chat()
            if meta.get("tool_schema") is not None:
                json_hash = self._hash_tool_schema(meta["tool_schema"])
                self._json_to_token[json_hash] = token_key

    def get_opt_kv(self) -> dict:
        """Return ``{hash: (K, V)}`` for all registered KV segments.

        Each value is a tuple of two tensors with shape
        ``(num_full_attn_layers, 1, num_kv_heads, seg_len, head_dim)``.
        The format matches what ``/v1/update_opt_kv`` expects.
        """
        result = {}
        for hash_key, entry in self._registry.items():
            k = getattr(self, entry["key_param"]).data
            v = getattr(self, entry["val_param"]).data
            result[hash_key] = (k, v)
        return result

    def forward(self, **kwargs):
        """Delegate to the wrapped model."""
        return self.model(**kwargs)
