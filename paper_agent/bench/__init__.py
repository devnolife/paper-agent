"""SkripsiBench: Indonesian academic-writing and research-coding benchmark for local and API models.

Eight tasks (``structure``, ``methodology``, ``extraction``, ``grounded``, ``references``, ``language``, ``coding``,
``latex``), scored deterministically where possible; cross-vendor LLM judges and human raters only for prose.
Every call records the model that actually answered, its reasoning effort, tokens, Copilot AI units and, for local
models, measured GPU energy. See ``bench/README.md``.
"""

TASKS = ("structure", "methodology", "extraction", "grounded", "references", "language", "coding", "latex")
ACADEMIC_TASKS = ("structure", "methodology", "extraction", "grounded", "references", "language")
CODING_TASKS = ("coding", "latex")
