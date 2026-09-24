/** Exercise the compact chat through HTTP/SSE and deterministic sample graphs.
 * A gated HTTP fixture separately proves HttpAgent renders chunks before EOF
 * and aborts its fetch when Stop is clicked, without a paid model request.
 * AI attribution: Generated with AI assistance by Northstar.
 * Copyright (c) 2026 Martin.Bechard@DevConsult.ca
 */
import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';

test('real SSE sample retains attachments and a second turn in a single chat', async ({ page }) => {
  const requests: string[] = [];
  const sockets: string[] = [];
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('websocket', socket => sockets.push(socket.url()));
  page.on('request', request => {
    if (request.url().includes('/api/chat/')) requests.push(request.postData() ?? '');
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  await expect(page.locator('aside')).toHaveCount(0);
  await expect(page.getByRole('link')).toHaveCount(0);
  await page.locator('input[type=file]').setInputFiles({ name: 'notes.md', mimeType: 'text/markdown', buffer: Buffer.from('Attached facts.\r\nSecond line.') });
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('.prose')).toContainText('An agent observes its input');
  expect(JSON.parse(requests[0]).forwardedProps.attachments[0].content).toBe('Attached facts.\nSecond line.');
  await expect(page.getByRole('button', { name: 'Remove notes.md' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('.prose').last()).toContainText('A tool observation supplies evidence');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
  await expect(page.locator('.message')).toHaveCount(4);
  expect(sockets).toEqual([]);
  expect(errors).toEqual([]);
  await page.screenshot({ path: 'test-results/desktop-chat.png', fullPage: true });
  await page.getByRole('button', { name: 'New chat' }).click();
  await expect(page.locator('.message')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
});

test('rejects non-UTF-8, removes files, selects samples and fits a phone', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  await page.locator('input[type=file]').setInputFiles({ name: 'binary.dat', mimeType: 'application/octet-stream', buffer: Buffer.from([0xff, 0xfe]) });
  await expect(page.getByRole('alert')).toContainText('readable UTF-8');
  await page.locator('input[type=file]').setInputFiles({ name: 'valid.txt', mimeType: 'text/plain', buffer: Buffer.from('valid') });
  await page.getByRole('button', { name: 'Remove valid.txt' }).click();
  await expect(page.locator('.file-chip')).toHaveCount(0);
  await page.getByLabel('Sample', { exact: true }).selectOption('tool_chat');
  await expect(page.getByRole('heading', { name: 'Tool chat', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await page.screenshot({ path: 'test-results/mobile-chat.png', fullPage: true });
});

// Public sample IDs also select the audit link. Exercise each renamed lesson
// through the real catalog and one scripted turn so a stale ID-prefix check
// cannot silently hide the context evidence from the user.
for (const sample of ['edit-with-patched-state', 'edit-with-reloaded-state']) {
  test(`${sample} exposes its context audit after a turn`, async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
    await page.getByLabel('Sample', { exact: true }).selectOption(sample);
    await expect(page.getByRole('heading', { name: sample, exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
    await page.getByRole('button', { name: 'Send message' }).click();
    const audit = page.getByRole('link', { name: 'View context audit' });
    await expect(audit).toBeVisible();
    const response = await page.request.get((await audit.getAttribute('href'))!);
    expect(response.ok()).toBeTruthy();
    expect((await response.json()).mode).toBe(sample);
  });
}

test('attachment-only live input streams before EOF and Stop closes HTTP', async ({ page }) => {
  let submitted: any;
  let disconnected = false;
  // This fixture holds an actual HTTP response open. It does not replace the
  // browser SDK or implement application transport logic inside a mock socket.
  const server = createServer((request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Headers', '*');
    if (request.method === 'OPTIONS') { response.end(); return; }
    let body = '';
    request.on('data', chunk => body += chunk);
    request.on('end', () => {
      submitted = JSON.parse(body);
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      for (const event of [
        { type: 'RUN_STARTED', threadId: submitted.threadId, runId: submitted.runId },
        { type: 'TOOL_CALL_START', toolCallId: 'pending', toolCallName: 'lookup', parentMessageId: 'request' },
        { type: 'TOOL_CALL_ARGS', toolCallId: 'pending', delta: '{"topic":"demo"}' },
        { type: 'TOOL_CALL_END', toolCallId: 'pending' },
        { type: 'SUBAGENT_STARTED', subagentRunId: 'child', name: 'Researcher', parentToolCallId: 'pending' },
        { type: 'TOOL_CALL_START', subagentRunId: 'child', toolCallId: 'nested', toolCallName: 'task', parentMessageId: 'child-request' },
        { type: 'TOOL_CALL_ARGS', subagentRunId: 'child', toolCallId: 'nested', delta: '{}' },
        { type: 'TOOL_CALL_END', subagentRunId: 'child', toolCallId: 'nested' },
        { type: 'SUBAGENT_STARTED', subagentRunId: 'grandchild', parentSubagentRunId: 'child', parentToolCallId: 'nested', name: 'Verifier' },
        { type: 'TEXT_MESSAGE_START', subagentRunId: 'grandchild', messageId: 'nested-answer', role: 'assistant' },
        { type: 'TEXT_MESSAGE_CONTENT', subagentRunId: 'grandchild', messageId: 'nested-answer', delta: 'Evidence checked' },
        { type: 'TEXT_MESSAGE_END', subagentRunId: 'grandchild', messageId: 'nested-answer' },
        { type: 'SUBAGENT_ERROR', subagentRunId: 'grandchild', message: 'Fixture failure' },
        { type: 'TEXT_MESSAGE_START', messageId: 'answer', role: 'assistant' },
        { type: 'TEXT_MESSAGE_CONTENT', messageId: 'answer', delta: 'Partial ' },
        { type: 'TEXT_MESSAGE_CONTENT', messageId: 'answer', delta: 'response' },
      ]) response.write(`data: ${JSON.stringify(event)}\n\n`);
      response.on('close', () => disconnected = true);
    });
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    await page.route('**/api/samples', async route => {
      const response = await route.fetch();
      await route.fulfill({ json: { ...await response.json(), live: true } });
    });
    // This streaming fixture never invokes a paid provider. Session creation
    // uses a scripted graph while the live UI submits to the held HTTP stream.
    await page.route('**/api/sessions', route => route.continue({
      postData: JSON.stringify({ ...route.request().postDataJSON(), live: false }),
    }));
    await page.route('**/api/chat/*', route => route.continue({ url: `http://127.0.0.1:${(server.address() as AddressInfo).port}/` }));
    await page.goto('/');
    await expect(page.getByRole('button', { name: 'New chat' })).toBeEnabled();
    await page.locator('input[type=file]').setInputFiles({ name: 'data.json', mimeType: 'application/json', buffer: Buffer.from('{"ok":true}') });
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.locator('.transcript > .message .prose').last()).toHaveText('Partial response');
    await expect(page.locator('.tool-execution').first()).toContainText('Running…');
    await page.locator('.subagent-conversation > summary').first().click();
    await page.locator('.subagent-conversation .subagent-conversation > summary').click();
    await expect(page.locator('.subagent-conversation .subagent-conversation')).toContainText('Evidence checked');
    await expect(page.locator('.subagent-conversation .subagent-conversation')).toContainText('Failed');
    expect(submitted.forwardedProps.prompt).toBe('');
    expect(submitted.forwardedProps.attachments[0].name).toBe('data.json');
    await page.getByRole('button', { name: 'Stop response' }).click();
    await expect(page.locator('footer [role=alert]')).toContainText('Stopped');
    await expect.poll(() => disconnected).toBe(true);
    await expect(page.locator('.tool-execution').first()).toContainText('No result received');
    await expect(page.locator('.subagent-conversation').first()).toContainText('Ended without completion');
    await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
});

// The real adapter emits TOOL_CALL_* events even for the scripted tool sample.
// Their SDK projection must survive the final MESSAGES_SNAPSHOT replacement.
test('tool calls and their results appear inline with expandable details', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Sample', { exact: true }).selectOption('tool_chat');
  await page.getByRole('button', { name: 'Send message' }).click();
  const tool = page.locator('.tool-execution');
  await expect(tool).toContainText('Tool echo_tool');
  await expect(tool).toContainText('Result received');
  await expect(page.locator('.prose').last()).toContainText('The tool returned "Your input was: ReAct"');
  await tool.locator('summary').click();
  await expect(tool.locator('pre').first()).toContainText('ReAct');
  await expect(tool.locator('pre').last()).toContainText('Your input was: ReAct');
  await expect(page.locator('aside')).toHaveCount(0);
});

// Exercise real Deep Agents attribution, including final snapshot replacement
// and a later user turn. Child text belongs in the expandable transcript.
test('subagent conversation includes its task, tools and answer', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Sample', { exact: true }).selectOption('subagent_chat');
  await page.getByRole('button', { name: 'Send message' }).click();
  const child = page.locator('.subagent-conversation').first();
  await expect(child).toContainText('Completed');
  await expect(child.locator('summary').first()).toContainText('Subagent ');
  await expect(child.locator('.view-conversation').first()).toBeVisible();
  await child.locator('summary').first().click();
  await expect(child.locator('.view-conversation').first()).toBeHidden();
  await expect(page.locator('.tool-details > .subagent-conversation')).toHaveCount(1);
  // The task's direct children establish the visible sequence, independent of
  // content inside the nested subagent's own tools and messages.
  await expect(page.locator('.tool-details').first().locator(':scope > p')).toHaveText(['Arguments', 'Result']);
  await expect(page.locator('.tool-details').first().locator(':scope > pre + .subagent-conversation + p')).toHaveText('Result');
  await expect(page.locator('.agent-name').first()).toContainText('Agent delegating_parent');
  await expect(child.locator('.message')).not.toHaveCount(0);
  await expect(child).toContainText('The echo tool returned Your input was: ReAct');
  await expect(child).toContainText('echo_tool');
  await expect(child).toContainText('Call echo_tool with the text ReAct');
  await child.locator('.tool-execution > summary').click();
  await expect(child.locator('.tool-details pre').first()).toContainText('ReAct');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
});

test('earlier subagent conversations survive later turns', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Sample', { exact: true }).selectOption('expert_dispatch');
  await page.getByRole('button', { name: 'Send message' }).click();
  const child = page.locator('.subagent-conversation').first();
  await expect(child).toContainText('Completed');
  await expect(child.locator('summary').first()).toContainText('Subagent ');
  await expect(child.locator('.view-conversation').first()).toBeVisible();
  await child.locator('summary').first().click();
  const transcript = await child.locator('.subagent-body').innerText();
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('.subagent-conversation')).toHaveCount(2);
  await expect(page.locator('.subagent-conversation').last()).toContainText('Completed');
  await expect(child.locator('.subagent-body')).toHaveText(transcript, { useInnerText: true });
});

// MCP is explicit catalog provenance; ordinary tools with the same name should
// remain ordinary tools. This fixture tests presentation without a RAG download.
test('MCP transport provenance prefixes the tool label', async ({ page }) => {
  await page.route('**/api/samples', async route => {
    const response = await route.fetch();
    const catalog = await response.json();
    catalog.samples.find((sample: { id: string }) => sample.id === 'tool_chat').mcpTools = ['echo_tool'];
    await route.fulfill({ json: catalog });
  });
  await page.goto('/');
  await page.getByLabel('Sample', { exact: true }).selectOption('tool_chat');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('.tool-execution > summary')).toContainText('MCP echo_tool');
});

test('file approval pauses, rejects and resumes through the shared AG-UI driver', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Sample', { exact: true }).selectOption('file_approval');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByRole('heading', { name: 'Approve file change' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
  await page.getByRole('button', { name: 'Reject', exact: true }).click();
  await expect(page.locator('.interaction pre')).toContainText('Next step');
  await expect(page.locator('.interaction pre')).not.toContainText('Reviewed by');
  await page.getByRole('button', { name: 'Approve', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Approve file change' })).toHaveCount(0);
  await expect(page.locator('.prose').last()).toContainText('Finished');
  const output = await page.request.get(await page.getByRole('link', { name: /Download approved/ }).getAttribute('href') ?? '');
  expect(output.status()).toBe(200);
  expect(await output.text()).toContain('Next step');
  expect(await output.text()).not.toContain('Reviewed by');
});

test('quote clarification accepts an answer then workflow cancellation', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Sample', { exact: true }).selectOption('quote_request');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByRole('heading', { name: 'Clarification needed' })).toBeVisible();
  const first = await page.locator('.interaction').innerText();
  await page.getByRole('textbox', { name: 'Clarification answer' }).fill('Use twenty posters.');
  await page.getByRole('button', { name: 'Submit answer' }).click();
  await expect(page.locator('.interaction')).not.toHaveText(first);
  await page.getByRole('button', { name: 'Cancel workflow' }).click();
  await expect(page.locator('.prose').last()).toContainText('cancelled');
  await expect(page.getByRole('heading', { name: 'Clarification needed' })).toHaveCount(0);
});

// Real trace metadata supplies bytes even without saved message content. A
// supplied capacity fixture covers scale rendering independently of model access.
test('context scale shows bytes, percentages, unknown usage and resets', async ({ page }) => {
  await page.goto('/');
  const context = page.getByRole('region', { name: 'Context usage' });
  await expect(context).toContainText('Not available');
  await expect(context).toContainText('0%');
  await expect(context).toContainText('100%');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(context).toContainText(/\d[\d,]* bytes/);
  await context.getByText('View retained context', { exact: true }).click();
  await expect(context.locator('pre').first()).toContainText('An agent observes');
  await expect(context.locator('pre').last()).toContainText('read_file');
  await expect(context).toContainText('Not available');
  await page.getByRole('button', { name: 'New chat' }).click();
  await expect(context).toContainText('Bytes not available');
  await page.route('**/api/chat/*', async route => {
    const response = await route.fetch();
    const body = (await response.text()).replace(/"utilization":null/g,
      '"utilization":{"percent":50,"tokens":500,"capacity":1000,"illustrative":true}');
    await route.fulfill({ response, body });
  });
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByRole('meter', { name: 'Context used' })).toHaveAttribute('value', '50');
  await expect(context).toContainText('500 / 1,000 tokens (estimated) · Illustrative');
});

// Verify the user's choice reaches session creation and changes input behaviour.
// Stub only real-mode construction here; backend tests check factory selection.
test('response mode dropdown starts a new session and persists across samples', async ({ page }) => {
  const selections: { sample: string; live: boolean }[] = [];
  await page.route('**/api/sessions', async route => {
    const selected = route.request().postDataJSON();
    selections.push(selected);
    await route.continue({ postData: JSON.stringify({ ...selected, live: false }) });
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  const mode = page.getByLabel('Response mode');
  const message = page.getByRole('textbox', { name: 'Message', exact: true });
  await expect(message).toHaveJSProperty('readOnly', true);
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('.prose')).toContainText('An agent observes');
  await mode.selectOption({ label: 'Use real LLM' });
  await expect(message).toBeEnabled();
  await expect(message).toHaveJSProperty('readOnly', false);
  await expect(page.locator('.message')).toHaveCount(0);
  expect(selections.at(-1)).toEqual({ sample: 'simple_chat', live: true });
  await message.fill('An arbitrary question');
  await page.getByLabel('Sample', { exact: true }).selectOption('tool_chat');
  await expect(message).toBeEnabled();
  expect(selections.at(-1)).toEqual({ sample: 'tool_chat', live: true });
  await mode.selectOption({ label: 'Use scripted responses' });
  await expect(message).toBeEnabled();
  await expect(message).toHaveJSProperty('readOnly', true);
  expect(selections.at(-1)).toEqual({ sample: 'tool_chat', live: false });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});
