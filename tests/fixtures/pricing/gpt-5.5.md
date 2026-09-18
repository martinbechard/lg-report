# GPT-5.5

> For the complete documentation index, see [llms.txt](/llms.txt). Markdown versions of documentation pages are available by appending `.md` to the page URL.

> A new class of intelligence for coding and professional work.

Model ID: `gpt-5.5`

GPT-5.5 is a flagship model for the most complex professional work.
Learn more in our [GPT-5.5 model guidance](/api/docs/guides/latest-model?model=gpt-5.5). Reasoning.effort supports: none, low, medium (default), high and xhigh.

## Model details

- Default snapshot: `gpt-5.5-2026-04-23`
- Input modalities: text, image
- Output modalities: text
- 1,050,000 context window
- 128,000 max output tokens
- Dec 01, 2025 knowledge cutoff
- Reasoning token support

## Pricing

Pricing is based on the number of tokens used, or other metrics based on the model type. For tool-specific models, like search and computer use, there’s a fee per tool call. See details in the [pricing page](/api/docs/pricing).

### Text tokens

| Metric | Price | Unit |
| --- | ---: | --- |
| Input | $5 | 1M tokens |
| Cached input | $0.5 | 1M tokens |
| Output | $30 | 1M tokens |

- For GPT-5.5, prompts with >272K input tokens are priced at 2x input and 1.5x output for the full session for standard, batch, and flex.
- Regional processing (data residency) endpoints are charged a 10% uplift for GPT-5.5.
