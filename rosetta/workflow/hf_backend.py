from __future__ import annotations

"""HFBackend — local HuggingFace model backend for run_with_tools().

Wraps HF ``model.generate()`` and returns ``ChatCompletion`` objects,
making it a drop-in replacement for CAMEL's ``BaseModelBackend`` in
:func:`~rosetta.workflow.singletool.run_with_tools` and
:func:`~rosetta.workflow.camel_utils.model_run_sync`.

Usage::

    from rosetta.workflow.hf_backend import HFBackend

    backend = HFBackend(model, tokenizer)
    response = backend.run(messages, tools=tool_schemas)
    # response is a ChatCompletion with .choices[0].message

With CacheOptimizeModel::

    from rosetta.workflow.hf_backend import CacheOptBackend
    from rosetta.optimize.wrapper import CacheOptimizeModel

    cache_model = CacheOptimizeModel(model)
    cache_model.register_tools(tokenizer, tools, system_msg)  # one-time setup
    backend = CacheOptBackend(cache_model, tokenizer)
    # prepare_chat() is called automatically on each run()
"""

import json
import re
import time
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from rosetta.optimize.wrapper import CacheOptimizeModel

import torch
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from camel.types import ChatCompletion

from rosetta.workflow.camel_utils import _parse_oss_completions_output


def _try_parse_json(s: Any) -> Any:
    """Parse a JSON string to dict/list if possible, else return as-is."""
    if isinstance(s, str) and s.strip()[:1] in ("{", "["):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass
    return s


def _fixup_tool_call_sections(
    rendered: str,
    original_args: List[str],
    tool_names: List[str],
) -> str:
    """Replace template tool-call rendering with model's original format.

    The GPT-OSS chat template renders assistant tool calls as::

        <|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>

    But the model generates them as::

        <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>ARGS<|call|>

    This function corrects three differences:
      1. Header order (``to=functions.NAME`` position)
      2. ``<|constrain|>`` special token (missing in template)
      3. JSON formatting (``tojson`` re-formats spacing)
    """
    for name, raw_arg in zip(tool_names, original_args):
        try:
            compact = json.dumps(
                json.loads(raw_arg), ensure_ascii=False, separators=(", ", ": ")
            )
        except (json.JSONDecodeError, TypeError):
            continue

        # Template format
        template_section = (
            f"<|start|>assistant to=functions.{name}"
            f"<|channel|>commentary json<|message|>{compact}<|call|>"
        )
        # Model's original format
        model_section = (
            f"<|start|>assistant<|channel|>commentary to=functions.{name} "
            f"<|constrain|>json<|message|>{raw_arg}<|call|>"
        )

        rendered = rendered.replace(template_section, model_section, 1)

    return rendered


def _parse_qwen3_output(text: str) -> tuple[str, str, Optional[list]]:
    """Parse raw Qwen3 model output into structured response.

    Qwen3 uses:
    - ``<think>REASONING</think>`` for thinking
    - ``<tool_call>{"name": "NAME", "arguments": {...}}</tool_call>`` for tool calls
    - Everything else is content
    - ``<|im_end|>`` may appear at the end (stripped)

    Args:
        text: Raw decoded text from model.generate().

    Returns:
        Tuple of (content, reasoning, tool_calls).
    """
    content = ""
    reasoning = ""
    tool_calls = None

    # Strip trailing special tokens
    text = re.sub(r"<\|im_end\|>$", "", text).strip()
    text = re.sub(r"<\|endoftext\|>$", "", text).strip()

    # Extract <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        text = text[: think_match.start()] + text[think_match.end() :]

    # Extract all <tool_call>...</tool_call>
    tc_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for m in tc_pattern.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        if tool_calls is None:
            tool_calls = []
        tool_calls.append(
            {
                "id": f"call_{_uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )

    # Remove tool_call blocks from text to get content
    content = tc_pattern.sub("", text).strip()

    return content, reasoning, tool_calls


def _detect_model_format(tokenizer) -> str:
    """Detect model output format from tokenizer vocabulary.

    Returns:
        ``"gpt-oss"`` if the tokenizer has GPT-OSS special tokens,
        ``"qwen3"`` otherwise (default).
    """
    channel_id = tokenizer.convert_tokens_to_ids("<|channel|>")
    if channel_id != tokenizer.unk_token_id:
        return "gpt-oss"
    return "qwen3"


class HFBackend:
    """Local HuggingFace model backend returning ChatCompletion objects.

    Args:
        model: A HuggingFace ``PreTrainedModel`` (e.g. ``AutoModelForCausalLM``).
        tokenizer: Matching HuggingFace tokenizer with ``apply_chat_template``.
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: Whether to use sampling (vs greedy).
        temperature: Sampling temperature (only used when ``do_sample=True``).
        output_parser: Callable that parses raw generated text into
            ``(content, reasoning, tool_calls)``.  Defaults to
            :func:`_parse_oss_completions_output` (GPT-OSS format).
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        output_parser: Optional[Callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self._model_format = _detect_model_format(tokenizer)
        if output_parser is not None:
            self.output_parser = output_parser
        elif self._model_format == "gpt-oss":
            self.output_parser = _parse_oss_completions_output
        else:
            self.output_parser = _parse_qwen3_output

    def _preprocess_messages(
        self, messages: List[dict]
    ) -> Tuple[List[dict], List[str], List[str]]:
        """Prepare messages for ``apply_chat_template``.

        For GPT-OSS models, applies three fixes (mirrors
        ``oss_tokenizer.preprocess_messages``):
          1. ``reasoning_content`` → ``thinking`` (template field name).
          2. Tool result ``content``: parse JSON strings → dicts so the
             template's ``|tojson`` filter produces clean JSON instead of
             double-escaping a string.
          3. Tool-call ``arguments``: parse JSON strings → dicts (same
             reason), and collect the originals for post-render fixup via
             :func:`_fixup_tool_call_sections`.

        For other models (Qwen3, etc.), strips internal fields and drops
        ``reasoning_content`` (reasoning is conveyed via ``<think>`` tags
        in the content field, not a separate template variable).

        Returns:
            ``(processed_messages, original_arg_strings, tool_names)``
        """
        processed = []
        original_args: List[str] = []
        tool_names: List[str] = []
        is_oss = self._model_format == "gpt-oss"

        for msg in messages:
            m = dict(msg)

            if is_oss:
                # GPT-OSS Fix 1: reasoning_content → thinking
                reasoning = m.pop("_reasoning", None) or m.pop(
                    "reasoning_content", None
                )
                if reasoning:
                    m["thinking"] = reasoning

                # GPT-OSS Fix 2: parse tool result content
                if msg.get("role") == "tool" and "content" in msg:
                    m["content"] = _try_parse_json(msg["content"])

                # GPT-OSS Fix 3: parse tool-call arguments
                if msg.get("tool_calls"):
                    new_calls = []
                    for tc in msg["tool_calls"]:
                        tc_copy = dict(tc)
                        func = dict(tc_copy.get("function", {}))
                        raw_args = func.get("arguments", "")
                        if isinstance(raw_args, str):
                            original_args.append(raw_args)
                            tool_names.append(func.get("name", ""))
                            parsed = _try_parse_json(raw_args)
                            if parsed is not raw_args:
                                func["arguments"] = parsed
                        tc_copy["function"] = func
                        new_calls.append(tc_copy)
                    m["tool_calls"] = new_calls
            else:
                # Non-OSS: drop internal/API-only fields the template
                # doesn't understand.
                m.pop("_reasoning", None)
                m.pop("reasoning_content", None)

            processed.append(m)

        return processed, original_args, tool_names

    def _get_stop_token_ids(self) -> List[int]:
        """Collect stop token IDs from the model's generation config.

        Includes ``eos_token_id`` plus GPT-OSS special tokens
        ``<|call|>`` and ``<|return|>`` if they exist in the tokenizer.
        """
        stop_ids = []
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            eos = getattr(gen_cfg, "eos_token_id", None)
            if eos is not None:
                if isinstance(eos, list):
                    stop_ids.extend(eos)
                else:
                    stop_ids.append(eos)
        elif self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)

        # Add GPT-OSS specific stop tokens if they exist
        for special in ("<|call|>", "<|return|>"):
            tok_id = self.tokenizer.convert_tokens_to_ids(special)
            # convert_tokens_to_ids returns unk_token_id if not found
            if tok_id != self.tokenizer.unk_token_id and tok_id not in stop_ids:
                stop_ids.append(tok_id)

        return stop_ids

    def _generate(
        self,
        input_ids: torch.Tensor,
        gen_kwargs: dict,
        *,
        processed: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Run ``model.generate()``.

        Subclasses can override this to inject custom KV caches or other
        generation-time modifications.

        Args:
            input_ids: Encoded prompt tensor of shape ``(1, seq_len)``.
            gen_kwargs: Keyword arguments for ``model.generate()``.
            processed: Preprocessed messages (available for subclass use).
            tools: Tool schemas (available for subclass use).

        Returns:
            Full output tensor from ``model.generate()`` (prompt + generated).
        """
        with torch.no_grad():
            return self.model.generate(
                input_ids.to(self.model.device), **gen_kwargs
            )

    def run(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> ChatCompletion:
        """Generate a response and return a ChatCompletion.

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Optional list of tool schemas (OpenAI function-calling format).

        Returns:
            ChatCompletion compatible with ``model_run_sync`` and
            ``run_with_tools``.
        """
        # 1. Preprocess messages (fix escaping for tojson in template)
        processed, original_args, tool_names = self._preprocess_messages(messages)

        # 2. Render chat template → string, fixup args, then encode
        template_kwargs = {}
        if tools:
            template_kwargs["tools"] = tools
        rendered = self.tokenizer.apply_chat_template(
            processed,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if self._model_format == "gpt-oss":
            rendered = _fixup_tool_call_sections(
                rendered, original_args, tool_names
            )
        full_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        input_ids = torch.tensor([full_ids])
        prompt_len = len(full_ids)

        # 3. Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        stop_ids = self._get_stop_token_ids()
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids

        # 4. Generate (subclass hook)
        output = self._generate(
            input_ids, gen_kwargs, processed=processed, tools=tools
        )

        # 5. Decode generated tokens (skip_special_tokens=False for parser)
        generated_ids = output[0, prompt_len:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        completion_tokens = len(generated_ids)

        # 6. Parse raw text
        content, reasoning, tool_calls = self.output_parser(raw_text)

        # 7. Build ChatCompletion
        message = ChatCompletionMessage(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls if tool_calls else None,
        )
        if reasoning:
            message.reasoning_content = reasoning

        choice_obj = Choice(
            index=0,
            message=message,
            finish_reason="stop",
            logprobs=None,
        )

        usage = CompletionUsage(
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            total_tokens=prompt_len + completion_tokens,
        )

        return ChatCompletion(
            id=f"hf_{_uuid.uuid4().hex[:12]}",
            choices=[choice_obj],
            created=int(time.time()),
            model="hf-local",
            object="chat.completion",
            usage=usage,
        )


class CacheOptBackend(HFBackend):
    """HFBackend with learnable KV-cache optimization via CacheOptimizeModel.

    Extends :class:`HFBackend` to inject a pre-computed KV cache (with
    learned tool-description parameters) during generation.

    Tools must be registered on the ``CacheOptimizeModel`` **before**
    creating this backend (via ``model.register_tools(...)``).

    Args:
        model: A :class:`~rosetta.optimize.wrapper.CacheOptimizeModel`
            with tools already registered.  The underlying HF model is
            taken from ``model.model``.
        tokenizer: Matching HuggingFace tokenizer.
        **kwargs: Forwarded to :class:`HFBackend` (``max_new_tokens``,
            ``do_sample``, ``temperature``, ``output_parser``).
    """

    def __init__(self, model: CacheOptimizeModel, tokenizer, **kwargs):
        self.opt_model = model
        super().__init__(model.model, tokenizer, **kwargs)

    def _generate(
        self,
        input_ids: torch.Tensor,
        gen_kwargs: dict,
        *,
        processed: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Generate with learnable KV cache when tools are provided."""
        if tools and processed is not None:
            result = self.opt_model.prepare_chat(
                self.tokenizer, processed, tools
            )
            gen_kwargs = {**gen_kwargs, "past_key_values": result["past_key_values"]}

        with torch.no_grad():
            return self.model.generate(
                input_ids.to(self.model.device), **gen_kwargs
            )
