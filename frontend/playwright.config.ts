/** Run browser acceptance checks against a built UI and real local gateway.
 * Each test gets a fresh browser context; this server uses an isolated port.
 * AI attribution: Generated with AI assistance by Northstar.
 * Copyright (c) 2026 Martin.Bechard@DevConsult.ca
 */
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 90_000,
  // The first sample factory lazily imports Deep Agents. Wait for readiness,
  // not an arbitrary sleep, while allowing cold Python imports to finish.
  expect: { timeout: 30_000 },
  use: { baseURL: 'http://127.0.0.1:18765', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  // Select scripted responses explicitly so configured API keys cannot change
  // these deterministic acceptance checks. Exercise the same launcher as users. An explicit price snapshot keeps
  // startup offline and capacity unknown for context-meter tests. The report
  // path isolates all selected samples.
  webServer: {
    command: '../.venv/bin/python -m agent_runtime --client angular --demo --port 18765 --out ../reports/chat-e2e/simple_chat --prices ../tests/fixtures/accounting_prices.json --metadata-only',
    url: 'http://127.0.0.1:18765/api/samples',
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
