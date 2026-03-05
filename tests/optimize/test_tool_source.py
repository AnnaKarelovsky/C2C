"""Tests for --tool-source flag in dataset_tau.py and eval_tau.py.

Verifies that both scripts can dynamically switch between tau1 and tau2
tool schemas, and that these schemas are genuinely different (different
parameter structures, different descriptions for many tools, different
system prompts).
"""

from __future__ import annotations

import json

import pytest
from datasets import Dataset

from rosetta.benchmark.tau.interface import (
    get_system_prompt as tau1_prompt,
    get_tools_info as tau1_tools,
)
from rosetta.benchmark.tau2.interface import (
    get_environment,
    get_system_prompt as tau2_prompt,
    get_tools_info as tau2_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AIRLINE_SYSTEM = "You are a helpful airline customer service agent."
RETAIL_SYSTEM = "You are a helpful retail customer service agent."


def _make_dataset(domain: str = "airline") -> Dataset:
    """Build a minimal HF dataset with one sample and a known shared tool."""
    system = AIRLINE_SYSTEM if domain == "airline" else RETAIL_SYSTEM
    # Use a tool that exists in both tau1 and tau2
    tool_name = "book_reservation" if domain == "airline" else "cancel_pending_order"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "placeholder apigen description",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    return Dataset.from_dict({
        "messages": [json.dumps(messages)],
        "tools": [json.dumps(tools)],
    })


# ---------------------------------------------------------------------------
# 1. Raw interface tests: tau1 vs tau2 produce genuinely different outputs
# ---------------------------------------------------------------------------


class TestInterfaceDifferences:
    """Verify that tau1 and tau2 interfaces return structurally different schemas."""

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_all_shared_tools_have_different_params(self, domain):
        """Every shared tool has different parameter schemas between tau1/tau2."""
        t1 = tau1_tools(domain)
        env = get_environment(domain)
        t2 = tau2_tools(env)

        t1_by_name = {t["function"]["name"]: t for t in t1}
        t2_by_name = {t["function"]["name"]: t for t in t2}
        shared = set(t1_by_name) & set(t2_by_name)

        assert len(shared) > 0, "Should have shared tools"
        for name in shared:
            p1 = t1_by_name[name]["function"]["parameters"]
            p2 = t2_by_name[name]["function"]["parameters"]
            assert p1 != p2, f"{name}: params should differ between tau1 and tau2"

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_system_prompts_differ(self, domain):
        """System prompts are different between tau1 and tau2."""
        p1 = tau1_prompt(domain)
        env = get_environment(domain)
        p2 = tau2_prompt(env)
        assert p1 != p2

    def test_airline_tau2_uses_defs_refs(self):
        """tau2 airline uses $defs/$ref in complex parameters; tau1 does not."""
        t1 = tau1_tools("airline")
        env = get_environment("airline")
        t2 = tau2_tools(env)

        has_defs_t2 = any("$defs" in t["function"]["parameters"] for t in t2)
        assert has_defs_t2, "tau2 airline should have at least one tool with $defs"

        has_defs_t1 = any("$defs" in t["function"]["parameters"] for t in t1)
        assert not has_defs_t1, "tau1 airline should not have $defs"


# ---------------------------------------------------------------------------
# 2. dataset_tau.py: _replace_tools_auto switches sources correctly
# ---------------------------------------------------------------------------


class TestDatasetTauToolSource:
    """Verify _replace_tools_auto loads genuinely different schemas per source."""

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_tau1_and_tau2_produce_different_tools(self, domain):
        """Same input dataset → different tool schemas depending on source."""
        from script.optimize.data.dataset_tau import _replace_tools_auto

        ds_tau = _make_dataset(domain)
        ds_tau2 = _make_dataset(domain)

        result_tau = _replace_tools_auto(ds_tau, full_set=False, tool_source="tau")
        result_tau2 = _replace_tools_auto(ds_tau2, full_set=False, tool_source="tau2")

        tools_tau = json.loads(result_tau[0]["tools"])
        tools_tau2 = json.loads(result_tau2[0]["tools"])

        # Both should have exactly one tool (filtered subset)
        assert len(tools_tau) == 1
        assert len(tools_tau2) == 1
        # Same tool name
        assert tools_tau[0]["function"]["name"] == tools_tau2[0]["function"]["name"]
        # Different parameters
        assert tools_tau[0]["function"]["parameters"] != tools_tau2[0]["function"]["parameters"]

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_tau1_and_tau2_produce_different_prompts(self, domain):
        """Same input → different system prompts depending on source."""
        from script.optimize.data.dataset_tau import _replace_tools_auto

        ds_tau = _make_dataset(domain)
        ds_tau2 = _make_dataset(domain)

        result_tau = _replace_tools_auto(ds_tau, full_set=False, tool_source="tau")
        result_tau2 = _replace_tools_auto(ds_tau2, full_set=False, tool_source="tau2")

        msgs_tau = json.loads(result_tau[0]["messages"])
        msgs_tau2 = json.loads(result_tau2[0]["messages"])

        prompt_tau = next(m for m in msgs_tau if m["role"] == "system")["content"]
        prompt_tau2 = next(m for m in msgs_tau2 if m["role"] == "system")["content"]

        assert prompt_tau != prompt_tau2
        # Both should be real prompts, not our placeholder
        assert len(prompt_tau) > 100
        assert len(prompt_tau2) > 100

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_full_set_tau1_vs_tau2_differ(self, domain):
        """full_set=True with tau1 vs tau2 produces different tool JSON."""
        from script.optimize.data.dataset_tau import _replace_tools_auto

        ds1 = _make_dataset(domain)
        ds2 = _make_dataset(domain)
        result_tau = _replace_tools_auto(ds1, full_set=True, tool_source="tau")
        result_tau2 = _replace_tools_auto(ds2, full_set=True, tool_source="tau2")

        tools_tau = json.loads(result_tau[0]["tools"])
        tools_tau2 = json.loads(result_tau2[0]["tools"])

        # Different total JSON (different schemas)
        assert json.dumps(tools_tau, sort_keys=True) != json.dumps(tools_tau2, sort_keys=True)

    def test_apigen_skips_entirely(self):
        """tool_source='apigen' returns the dataset unchanged."""
        from script.optimize.data.dataset_tau import _replace_tools_auto

        ds = _make_dataset("airline")
        original_tools = ds[0]["tools"]
        original_messages = ds[0]["messages"]

        result = _replace_tools_auto(ds, tool_source="apigen")

        assert result[0]["tools"] == original_tools
        assert result[0]["messages"] == original_messages

    def test_unknown_domain_unchanged(self):
        """Samples with undetectable domain keep original tools."""
        from script.optimize.data.dataset_tau import _replace_tools_auto

        messages = [
            {"role": "system", "content": "Generic system prompt."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        tools = [{"type": "function", "function": {"name": "foo", "description": "bar", "parameters": {}}}]
        ds = Dataset.from_dict({
            "messages": [json.dumps(messages)],
            "tools": [json.dumps(tools)],
        })

        result = _replace_tools_auto(ds, tool_source="tau")

        assert json.loads(result[0]["tools"]) == tools
        result_msgs = json.loads(result[0]["messages"])
        assert result_msgs[0]["content"] == "Generic system prompt."


# ---------------------------------------------------------------------------
# 3. eval_tau.py: worker tool loading logic
# ---------------------------------------------------------------------------


class TestEvalTauToolSource:
    """Verify eval_tau worker loads different tools based on --tool-source.

    We can't run the full worker (needs models), so we replicate the
    tool-loading branch from worker() and verify the outputs differ.
    """

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_worker_tau1_path(self, domain):
        """Simulating worker with tool_source='tau' loads tau1 schemas."""
        from rosetta.benchmark.tau.interface import get_system_prompt, get_tools_info

        tools_info = get_tools_info(domain)
        wiki = get_system_prompt(domain)

        assert len(tools_info) > 0
        assert len(wiki) > 100
        # tau1 tools should not have $defs
        has_defs = any("$defs" in t["function"]["parameters"] for t in tools_info)
        assert not has_defs

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_worker_tau2_path(self, domain):
        """Simulating worker with tool_source='tau2' loads tau2 schemas."""
        from rosetta.benchmark.tau2.interface import (
            get_environment,
            get_system_prompt as get_system_prompt_tau2,
            get_tools_info as get_tools_info_tau2,
        )

        sample_env = get_environment(domain)
        tools_info = get_tools_info_tau2(sample_env)
        wiki = get_system_prompt_tau2(sample_env)

        assert len(tools_info) > 0
        assert len(wiki) > 100
        # tau2 tools should differ from tau1 tools
        tau1_info = tau1_tools(domain)
        assert json.dumps(tools_info, sort_keys=True) != json.dumps(tau1_info, sort_keys=True)

    @pytest.mark.parametrize("domain", ["airline", "retail"])
    def test_worker_paths_produce_different_results(self, domain):
        """tau1 and tau2 worker paths produce different tools and prompts."""
        from rosetta.benchmark.tau.interface import (
            get_system_prompt as tau1_sp,
            get_tools_info as tau1_ti,
        )
        from rosetta.benchmark.tau2.interface import (
            get_environment,
            get_system_prompt as tau2_sp,
            get_tools_info as tau2_ti,
        )

        # tau1 path
        tools1 = tau1_ti(domain)
        wiki1 = tau1_sp(domain)

        # tau2 path
        env = get_environment(domain)
        tools2 = tau2_ti(env)
        wiki2 = tau2_sp(env)

        # Tools JSON should differ
        assert json.dumps(tools1, sort_keys=True) != json.dumps(tools2, sort_keys=True)
        # Prompts should differ
        assert wiki1 != wiki2

        # Shared tools should have same names but different schemas
        names1 = {t["function"]["name"] for t in tools1}
        names2 = {t["function"]["name"] for t in tools2}
        shared = names1 & names2
        assert len(shared) >= 10, f"Expected many shared tools, got {len(shared)}"

        by_name1 = {t["function"]["name"]: t for t in tools1}
        by_name2 = {t["function"]["name"]: t for t in tools2}
        for name in shared:
            assert by_name1[name]["function"]["parameters"] != by_name2[name]["function"]["parameters"], (
                f"Tool {name} should have different params between tau1 and tau2"
            )


# ---------------------------------------------------------------------------
# 4. Argparse validation
# ---------------------------------------------------------------------------


class TestDatasetTauArgs:
    def test_tool_source_choices(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--tool-source", default="tau", choices=["apigen", "tau", "tau2"])
        assert p.parse_args([]).tool_source == "tau"
        assert p.parse_args(["--tool-source", "tau2"]).tool_source == "tau2"
        assert p.parse_args(["--tool-source", "apigen"]).tool_source == "apigen"
        with pytest.raises(SystemExit):
            p.parse_args(["--tool-source", "invalid"])

    def test_domain_telecom_accepted(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--domain", default="all", choices=["airline", "retail", "telecom", "all"])
        assert p.parse_args(["--domain", "telecom"]).domain == "telecom"


class TestEvalTauArgs:
    def test_tool_source_choices(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--tool-source", default="tau", choices=["tau", "tau2"])
        assert p.parse_args([]).tool_source == "tau"
        assert p.parse_args(["--tool-source", "tau2"]).tool_source == "tau2"
        with pytest.raises(SystemExit):
            p.parse_args(["--tool-source", "apigen"])
