# Translation report — pipeline `multi`

- units: 146 (4696 words); untranslated (kept English): 0
- roles: draft=copilot:gpt-5-mini, review=copilot:claude-haiku-4.5, qa=ollama:gemma3:27b, fallback=ollama:gemma3:27b
- served models (copilot may substitute): {'copilot:gpt-5-mini': 17, 'copilot:claude-haiku-4.5': 17}
- wall time this run: 145 s
- glossary compliance: 284/287 = 99.0%

## stage `draft`
- mean s/unit: 7.357
- cosine EN↔ID (direct): 0.823
- cosine EN↔back-translation: 0.955

## stage `review`
- mean s/unit: 5.398
- cosine EN↔ID (direct): 0.825
- cosine EN↔back-translation: 0.953
- reviewer edit ratio: mean 0.017, unchanged units: 108/146

- selected: draft=7, review=139
