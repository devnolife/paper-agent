# Translation report — pipeline `multi`

- units: 111 (4713 words); untranslated (kept source language): 0
- roles: draft=copilot:gpt-5-mini, review=copilot:claude-haiku-4.5, qa=ollama:gemma3:27b, fallback=ollama:gemma3:27b, critic=copilot:claude-haiku-4.5, critic2=ollama:gemma3:27b, planner=copilot:gpt-5-mini
- served models (copilot may substitute): {'copilot:gpt-5-mini': 47, 'copilot:claude-haiku-4.5': 217, 'ollama:gemma3:27b': 296}
- wall time this run: 145 s
- glossary compliance: 296/312

## stage `draft`
- s_per_unit: 5.105
- cosine_direct: 0.861
- cosine_back: 0.94

## stage `review`
- s_per_unit: 9.224
- cosine_direct: 0.858
- cosine_back: 0.942
- edit_ratio: 0.014
- changed_units: 31

- selected: {'draft': 7, 'review': 104}
