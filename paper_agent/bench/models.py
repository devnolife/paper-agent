"""Models under test: ``engine:model[@effort]`` specs and the study roster (``bench/models.yaml``).

The roster assigns each model a *tier* (``local``, ``economy``, ``standard``, ``premium``) and a *family* (vendor of the
base model). Families keep LLM judges independent of the model they grade; tiers drive the cost gate: premium models
and high reasoning effort only run with an explicit approval flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

TIERS = ("local", "economy", "standard", "premium")
HIGH_EFFORT = ("high", "xhigh", "max")
FIXED_CTX = 16384
_FAMILY_PREFIX = [
    (r"^claude", "anthropic"), (r"^gpt-oss", "openai"), (r"^(gpt|o\d|chatgpt)", "openai"),
    (r"^(gemini|gemma)", "google"), (r"sahabat", "google"), (r"^kimi", "moonshot"), (r"^(mai-|phi)", "microsoft"),
    (r"^qwen", "alibaba"), (r"^llama", "meta"), (r"^grok", "xai"), (r"^(mistral|mixtral|ministral)", "mistral"),
    (r"^deepseek", "deepseek"), (r"^glm", "zhipu"),
]


def family_of(model: str) -> str:
    name = model.lower().split("/")[-1]
    for rx, fam in _FAMILY_PREFIX:
        if re.search(rx, name):
            return fam
    return "other"


@dataclass(frozen=True)
class ModelSpec:
    engine: str
    model: str
    effort: Optional[str] = None

    @classmethod
    def parse(cls, s: str) -> "ModelSpec":
        """``copilot:gpt-5-mini@low`` or ``ollama:qwen2.5:7b-instruct``; the engine is the part before the first colon."""
        engine, _, rest = s.strip().partition(":")
        if engine not in ("ollama", "copilot") or not rest:
            raise ValueError(f"model spec must be <ollama|copilot>:<model>[@effort], got {s!r}")
        model, _, effort = rest.rpartition("@") if "@" in rest else (rest, "", "")
        return cls(engine, model, effort or None)

    @property
    def label(self) -> str:
        return f"{self.engine}:{self.model}" + (f"@{self.effort}" if self.effort else "")

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{self.engine}__{self.model}__{self.effort or 'default'}")


@dataclass
class ModelInfo:
    spec: ModelSpec
    tier: str
    family: str
    name: str = ""
    think: object = None                  # ollama: True/False/"low"... sent as the request's ``think``
    ctx_max: Optional[int] = None         # ollama: context-window cap (model's trained context)
    options: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def high_cost(self) -> bool:
        return self.tier == "premium" or (self.spec.effort or "") in HIGH_EFFORT

    @property
    def reasons(self) -> bool:
        """Spends output tokens on hidden reasoning (local thinking models)."""
        return bool(self.think) or "reasoning" in self.spec.model

    def engine_options(self, num_predict: int) -> dict:
        """Ollama options: answer budget (+ room for hidden reasoning), ``think``, and one fixed context window per
        model (``FIXED_CTX`` capped by the model's context) so the model is loaded once, not per prompt length."""
        if self.spec.engine != "ollama":
            return {}
        ctx = min(self.ctx_max or FIXED_CTX, FIXED_CTX)
        budget = num_predict + (6144 if self.reasons else 0)
        opts: dict = {"num_ctx": ctx, "num_predict": min(budget, ctx // 2), **self.options}
        if self.think is not None:
            opts["think"] = self.think
        return opts


def load_roster(path: Path, effort: Optional[str] = None) -> List[ModelInfo]:
    """Roster YAML -> models. ``effort`` overrides the roster's default effort for models that support effort."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    default_effort = effort or cfg.get("effort")
    out: List[ModelInfo] = []
    cop = cfg.get("copilot") or {}
    no_effort = set(cop.get("no_effort") or [])
    for tier in ("economy", "standard", "premium"):
        for m in cop.get(tier) or []:
            eff = None if m in no_effort else default_effort
            out.append(ModelInfo(ModelSpec("copilot", m, eff), tier, family_of(m), m))
    for e in cfg.get("ollama") or []:
        e = {"model": e} if isinstance(e, str) else dict(e)
        m = e.pop("model")
        out.append(ModelInfo(ModelSpec("ollama", m, None), "local", e.pop("family", None) or family_of(m),
                             e.pop("name", m), e.pop("think", None), e.pop("ctx_max", None), e.pop("options", {}) or {}))
    return out


def select(roster: List[ModelInfo], specs: Optional[List[str]] = None, tiers: Optional[List[str]] = None,
           effort: Optional[str] = None) -> List[ModelInfo]:
    """Models from the roster by tier and/or by spec; specs not in the roster are added with a guessed tier.

    ``effort`` replaces the effort of the selected API models that support it (models without effort keep none)."""
    picked: List[ModelInfo] = []
    by_model = {(m.spec.engine, m.spec.model): m for m in roster}
    if tiers:
        picked += [m for m in roster if m.tier in tiers]
    for s in specs or []:
        sp = ModelSpec.parse(s)
        base = by_model.get((sp.engine, sp.model))
        if base is None:
            base = ModelInfo(sp, "local" if sp.engine == "ollama" else "standard", family_of(sp.model), sp.model)
        info = ModelInfo(ModelSpec(sp.engine, sp.model, sp.effort if sp.effort else base.spec.effort), base.tier,
                         base.family, base.name, base.think, base.ctx_max, dict(base.options))
        picked.append(info)
    if effort:
        picked = [ModelInfo(ModelSpec(m.spec.engine, m.spec.model, effort if m.spec.effort else None), m.tier, m.family,
                            m.name, m.think, m.ctx_max, dict(m.options)) for m in picked]
    seen, out = set(), []
    for m in picked:
        if m.label not in seen:
            seen.add(m.label)
            out.append(m)
    return out
