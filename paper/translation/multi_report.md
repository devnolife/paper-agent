# Translation report — pipeline `multi`

- units: 122 (4140 words); untranslated (kept source language): 0
- roles: draft=copilot:gpt-5-mini, review=copilot:claude-haiku-4.5, qa=ollama:gemma3:27b, fallback=ollama:gemma3:27b, critic=copilot:claude-haiku-4.5, critic2=ollama:gemma3:27b, planner=copilot:gpt-5-mini
- served models (copilot may substitute): {'copilot:gpt-5-mini': 21, 'copilot:claude-haiku-4.5': 21, 'ollama:gemma3:27b': 114}
- wall time this run: 95 s
- glossary compliance: 338/350

## stage `draft`
- s_per_unit: 7.52
- cosine_direct: 0.865
- cosine_back: 0.959

## stage `review`
- s_per_unit: 6.441
- cosine_direct: 0.867
- cosine_back: 0.961
- edit_ratio: 0.015
- changed_units: 33

- selected: {'draft': 3, 'review': 119}
