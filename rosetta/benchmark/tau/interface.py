"""Tau-bench tool wrapper for C2C evaluation pipeline.

Wraps tau-bench stateful tool classes as FunctionTool-compatible objects
that work with our singletool.py evaluation pipeline.

Usage:
    from rosetta.benchmark.tau.interface import load_tau_tools, load_tau_tasks, get_system_prompt

    tools = load_tau_tools("airline")
    tools = load_tau_tools("retail", tool_names=["cancel_pending_order", "get_user_details"])
    tasks = load_tau_tasks("airline", split="test")
    system_prompt = get_system_prompt("airline")
"""

import importlib
import importlib.util
import os
import sys
import types as builtin_types
from typing import Any, Dict, List, Optional

from camel.toolkits import FunctionTool


VALID_DOMAINS = ("airline", "retail")
_REF_DIR = os.path.join(os.path.dirname(__file__), "ref", "func_tools")


def _load_module_from_file(name: str, filepath: str, is_package: bool = False):
    """Load a Python module directly from file, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(
        name, filepath,
        submodule_search_locations=[os.path.dirname(filepath)] if is_package else None,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_tau_bench_shims():
    """Patch sys.modules so `from tau_bench.envs.tool import Tool` etc. resolve
    to our local ref copies.

    Strategy: use spec_from_file_location to load individual .py files directly,
    avoiding the package __init__.py files that have problematic tau_bench imports.
    We register tau_bench.* aliases in sys.modules BEFORE executing any ref code,
    then load ref files in dependency order.
    """
    if "tau_bench" in sys.modules:
        return

    # --- Phase 1: Register tau_bench namespace and types ---
    tau_bench = builtin_types.ModuleType("tau_bench")
    tau_bench.__path__ = []
    sys.modules["tau_bench"] = tau_bench

    from rosetta.benchmark.tau import types as tau_types
    sys.modules["tau_bench.types"] = tau_types

    envs = builtin_types.ModuleType("tau_bench.envs")
    envs.__path__ = []
    sys.modules["tau_bench.envs"] = envs

    # Pre-register domain namespaces
    for domain in VALID_DOMAINS:
        ns = f"tau_bench.envs.{domain}"
        mod = builtin_types.ModuleType(ns)
        mod.__path__ = [os.path.join(_REF_DIR, domain)]
        sys.modules[ns] = mod

    # --- Phase 2: Load ref modules directly from files (bypassing __init__.py) ---
    # ref/tool.py — base Tool class (no external deps)
    ref_tool = _load_module_from_file(
        "tau_bench.envs.tool",
        os.path.join(_REF_DIR, "tool.py"),
    )

    # ref/user.py — depends on litellm (optional for us)
    try:
        ref_user = _load_module_from_file(
            "tau_bench.envs.user",
            os.path.join(_REF_DIR, "user.py"),
        )
    except ImportError:
        ref_user = builtin_types.ModuleType("tau_bench.envs.user")
        sys.modules["tau_bench.envs.user"] = ref_user

    # ref/base.py — depends on tool, user, types (all now available)
    try:
        ref_base = _load_module_from_file(
            "tau_bench.envs.base",
            os.path.join(_REF_DIR, "base.py"),
        )
    except ImportError:
        ref_base = builtin_types.ModuleType("tau_bench.envs.base")
        sys.modules["tau_bench.envs.base"] = ref_base

    # --- Phase 3: Load domain sub-packages ---
    # For each domain, we need: data/, tools/, wiki.py, tasks_*.py
    # We load them with proper module names so their internal relative imports work.
    for domain in VALID_DOMAINS:
        domain_dir = os.path.join(_REF_DIR, domain)
        domain_pkg = f"rosetta.benchmark.tau.ref.{domain}"
        tau_domain_pkg = f"tau_bench.envs.{domain}"

        # Register the domain package with a dummy __init__ (skip the real one
        # which has problematic imports)
        dmod = builtin_types.ModuleType(domain_pkg)
        dmod.__path__ = [domain_dir]
        dmod.__file__ = os.path.join(domain_dir, "__init__.py")
        sys.modules[domain_pkg] = dmod
        # Alias
        sys.modules[tau_domain_pkg] = dmod

        # data/__init__.py
        data_dir = os.path.join(domain_dir, "data")
        data_init = os.path.join(data_dir, "__init__.py")
        if os.path.exists(data_init):
            data_mod = _load_module_from_file(
                f"{domain_pkg}.data", data_init, is_package=True
            )
            sys.modules[f"{tau_domain_pkg}.data"] = data_mod

        # wiki.py
        wiki_file = os.path.join(domain_dir, "wiki.py")
        if os.path.exists(wiki_file):
            wiki_mod = _load_module_from_file(f"{domain_pkg}.wiki", wiki_file)
            sys.modules[f"{tau_domain_pkg}.wiki"] = wiki_mod

        # tools/ package — need to load __init__.py which imports all tool classes
        tools_dir = os.path.join(domain_dir, "tools")
        tools_init = os.path.join(tools_dir, "__init__.py")
        if os.path.exists(tools_init):
            # First register the tools package
            tools_pkg = builtin_types.ModuleType(f"{domain_pkg}.tools")
            tools_pkg.__path__ = [tools_dir]
            tools_pkg.__file__ = tools_init
            sys.modules[f"{domain_pkg}.tools"] = tools_pkg
            sys.modules[f"{tau_domain_pkg}.tools"] = tools_pkg

            # Load individual tool files (they do `from tau_bench.envs.tool import Tool`)
            for fname in os.listdir(tools_dir):
                if fname.endswith(".py") and fname != "__init__.py":
                    mod_name = fname[:-3]
                    _load_module_from_file(
                        f"{domain_pkg}.tools.{mod_name}",
                        os.path.join(tools_dir, fname),
                    )

            # Now execute the tools __init__.py which imports from the tool submodules
            spec = importlib.util.spec_from_file_location(
                f"{domain_pkg}.tools", tools_init,
                submodule_search_locations=[tools_dir],
            )
            spec.loader.exec_module(tools_pkg)

        # Task files
        for split_name in ("tasks_test", "tasks_train", "tasks_dev"):
            task_file = os.path.join(domain_dir, f"{split_name}.py")
            if os.path.exists(task_file):
                task_mod = _load_module_from_file(
                    f"{domain_pkg}.{split_name}", task_file
                )
                sys.modules[f"{tau_domain_pkg}.{split_name}"] = task_mod

    # Also register the ref package itself (dummy, without running its __init__)
    ref_pkg = builtin_types.ModuleType("rosetta.benchmark.tau.ref")
    ref_pkg.__path__ = [_REF_DIR]
    sys.modules["rosetta.benchmark.tau.ref"] = ref_pkg


# Install shims on first import of this module
_install_tau_bench_shims()


def _get_domain_modules(domain: str):
    """Return (tools_module, data_module, wiki_module) for a domain."""
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Valid: {VALID_DOMAINS}")

    base = f"rosetta.benchmark.tau.ref.{domain}"
    tools_mod = sys.modules[f"{base}.tools"]
    data_mod = sys.modules[f"{base}.data"]
    wiki_mod = sys.modules[f"{base}.wiki"]
    return tools_mod, data_mod, wiki_mod


def load_tau_tools(
    domain: str,
    tool_names: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> List[FunctionTool]:
    """Load tau-bench tools as FunctionTool objects for the eval pipeline.

    Args:
        domain: "airline" or "retail".
        tool_names: Optional filter -- only return tools with these names.
            If None, returns all tools for the domain.
        data: Optional pre-loaded data dict. If None, loads fresh data.
            Pass a shared data dict to maintain state across tool calls
            within a single evaluation episode.

    Returns:
        List of FunctionTool objects compatible with singletool.py.
    """
    tools_mod, data_mod, _ = _get_domain_modules(domain)
    all_tool_classes = tools_mod.ALL_TOOLS

    if data is None:
        data = data_mod.load_data()

    result = []
    for tool_cls in all_tool_classes:
        schema = tool_cls.get_info()
        func_name = schema["function"]["name"]

        if tool_names is not None and func_name not in tool_names:
            continue

        # Create closure that binds tool_cls and data
        def _make_fn(cls, shared_data):
            def fn(**kwargs):
                return cls.invoke(data=shared_data, **kwargs)
            fn.__name__ = cls.get_info()["function"]["name"]
            fn.__doc__ = cls.get_info()["function"]["description"]
            return fn

        func = _make_fn(tool_cls, data)
        tool = FunctionTool(func=func, openai_tool_schema=schema)
        result.append(tool)

    return result


def load_tau_tools_with_data(
    domain: str,
    tool_names: Optional[List[str]] = None,
) -> tuple:
    """Load tau-bench tools and return (tools, data) so caller can reset data.

    Returns:
        (tools, data) where tools is List[FunctionTool] and data is the
        shared mutable dict. Caller can deep-copy data to reset state.
    """
    _, data_mod, _ = _get_domain_modules(domain)
    data = data_mod.load_data()
    tools = load_tau_tools(domain, tool_names=tool_names, data=data)
    return tools, data


def reset_data(domain: str) -> Dict[str, Any]:
    """Load a fresh copy of domain data (for resetting between episodes)."""
    _, data_mod, _ = _get_domain_modules(domain)
    return data_mod.load_data()


def load_tau_tasks(domain: str, split: str = "test") -> list:
    """Load task definitions for a tau-bench domain.

    Args:
        domain: "airline" or "retail".
        split: "test", "train", or "dev". Not all domains have all splits.

    Returns:
        List of Task dataclass instances.
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Valid: {VALID_DOMAINS}")

    base = f"rosetta.benchmark.tau.ref.{domain}"
    split_map = {
        "test": "tasks_test",
        "train": "tasks_train",
        "dev": "tasks_dev",
    }
    if split not in split_map:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(split_map.keys())}")

    mod_name = f"{base}.{split_map[split]}"
    if mod_name not in sys.modules:
        raise RuntimeError(
            f"Tasks not available: {domain}/{split}. "
            f"File may not exist at ref/{domain}/{split_map[split]}.py"
        )
    mod = sys.modules[mod_name]

    # Airline uses TASKS, retail uses TASKS_TEST/TASKS_TRAIN/TASKS_DEV
    for attr in ("TASKS", f"TASKS_{split.upper()}"):
        if hasattr(mod, attr):
            return getattr(mod, attr)

    raise RuntimeError(f"No tasks found in {mod_name}")


def get_system_prompt(domain: str) -> str:
    """Get the wiki-based system prompt for a tau-bench domain.

    Args:
        domain: "airline" or "retail".

    Returns:
        The full wiki text used as the agent's system prompt.
    """
    _, _, wiki_mod = _get_domain_modules(domain)
    return wiki_mod.WIKI


def get_all_tool_names(domain: str) -> List[str]:
    """Return the names of all tools available in a tau-bench domain.

    Args:
        domain: "airline" or "retail".

    Returns:
        List of tool name strings.
    """
    tools_mod, _, _ = _get_domain_modules(domain)
    return [
        cls.get_info()["function"]["name"] for cls in tools_mod.ALL_TOOLS
    ]


def get_tools_map(domain: str) -> Dict[str, Any]:
    """Return {tool_name: tool_class} for reward computation.

    Args:
        domain: "airline" or "retail".

    Returns:
        Dict mapping tool function names to their Tool classes.
    """
    tools_mod, _, _ = _get_domain_modules(domain)
    return {
        cls.get_info()["function"]["name"]: cls for cls in tools_mod.ALL_TOOLS
    }


def get_data_load_func(domain: str):
    """Return callable that loads fresh data dict.

    Args:
        domain: "airline" or "retail".

    Returns:
        Callable that returns a fresh data dict.
    """
    _, data_mod, _ = _get_domain_modules(domain)
    return data_mod.load_data


def get_tools_info(domain: str) -> List[Dict]:
    """Return list of OpenAI tool schemas for the agent.

    Args:
        domain: "airline" or "retail".

    Returns:
        List of OpenAI-format tool schema dicts.
    """
    tools_mod, _, _ = _get_domain_modules(domain)
    return [cls.get_info() for cls in tools_mod.ALL_TOOLS]
