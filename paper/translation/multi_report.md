# Translation report — pipeline `multi`

- units: 131 (4203 words); untranslated (kept source language): 0
- roles: draft=copilot:gpt-5-mini, review=copilot:claude-haiku-4.5, qa=ollama:gemma3:27b, fallback=ollama:gemma3:27b, critic=copilot:claude-haiku-4.5, critic2=ollama:gemma3:27b, planner=copilot:gpt-5-mini
- served models (copilot may substitute): {'copilot:gpt-5-mini': 30, 'copilot:claude-haiku-4.5': 30, 'ollama:gemma3:27b': 180}
- wall time this run: 102 s
- glossary compliance: 336/352

## stage `draft`
- s_per_unit: 7.473
- cosine_direct: 0.881
- cosine_back: 0.959

## stage `review`
- s_per_unit: 7.701
- cosine_direct: 0.881
- cosine_back: 0.959
- edit_ratio: 0.009
- changed_units: 24

- selected: {'draft': 3, 'review': 128}
