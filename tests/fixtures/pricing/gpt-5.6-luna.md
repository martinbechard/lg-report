# GPT-5.6 Luna

> For the complete documentation index, see [llms.txt](/llms.txt). Markdown versions of documentation pages are available by appending `.md` to the page URL.

> GPT-5.6 model optimized for cost-sensitive workloads

Model ID: `gpt-5.6-luna`

GPT-5.6 Luna is designed for cost-sensitive, high-volume workloads. It
roughly corresponds to the nano model tier used in earlier GPT-5 families.
Reasoning.effort supports: none, low, medium (default), high, xhigh, and max.

## Model details

- Default snapshot: `gpt-5.6-luna`
- Input modalities: text, image
- Output modalities: text
- 1,050,000 context window
- Maximum input tokens: 922,000
- 128,000 max output tokens
- Feb 16, 2026 knowledge cutoff
- Reasoning token support

## Pricing

Pricing is based on the number of tokens used, or other metrics based on the model type. For tool-specific models, like search and computer use, there’s a fee per tool call. See details in the [pricing page](/api/docs/pricing).

### Text tokens

| Metric | Price | Unit |
| --- | ---: | --- |
| Input | $0.2 | 1M tokens |
| Cached input | $0.02 | 1M tokens |
| Output | $1.2 | 1M tokens |

- Prompts with >272K input tokens are priced at 2x input and 1.5x output for the full request.
- Cache writes are billed at 1.25x the uncached input token rate.
