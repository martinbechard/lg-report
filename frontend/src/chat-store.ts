/** Bind Angular presentation to AG-UI's standard HTTP/SSE client.
 * HttpAgent owns event parsing, message accumulation, snapshots and aborts.
 * This store selects a sample and adds the shared text-attachment request.
 * AI attribution: Generated with AI assistance by Northstar.
 * Copyright (c) 2026 Martin.Bechard@DevConsult.ca
 */
import { Injectable, signal } from '@angular/core';
import { WebScriptPrompter } from './web-script-prompter';
import { HttpAgent, type Message as AgentMessage } from '@ag-ui/client';

export interface Sample { id: string; title: string; description: string; prompts: string[]; mcpTools?: string[] }
export interface Attachment { name: string; content: string }
/** A tool result confirms a response arrived; it does not prove tool success. */
export interface ToolExecution { id: string; name: string; arguments: string; result?: string; mcp?: boolean }
export interface Message { id: string; role: 'user' | 'assistant'; text: string; files: string[]; name?: string; tools?: ToolExecution[] }
/** One invocation, rather than one reusable agent name, owns each transcript. */
export interface SubagentConversation {
  id: string; name: string; description?: string; parentId?: string; toolId?: string;
  status: string; messages: Message[]; error?: string;
}
/** Retained context preview; tokens are estimates rather than provider receipts. */
export interface ContextUsage {
  bytes: number | null; model: string | null;
  messages: unknown[]; tools: unknown[] | null; note: string;
  utilization: { percent: number; tokens: number; capacity: number; illustrative: boolean } | null;
}
export interface PendingInteraction { id: string; payload: Record<string, unknown> }
type Status = 'waiting' | 'disconnected' | 'connecting' | 'ready' | 'running' | 'stopping' | 'failed';

/** Only the small catalog/session responses need application-side validation. */
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

@Injectable({ providedIn: 'root' })
export class ChatStore {
  readonly samples = signal<Sample[]>([]);
  readonly messages = signal<Message[]>([]);
  readonly subagents = signal<SubagentConversation[]>([]);
  readonly status = signal<Status>('disconnected');
  readonly error = signal('');
  readonly pending = signal<PendingInteraction[]>([]);
  readonly outputUrl = signal('');
  readonly contextUrl = signal('');
  readonly contextUsage = signal<ContextUsage | null>(null);
  readonly outputAvailable = signal(false);
  private answers = new Map<string, string>();
  // Report availability counts all turns, even after scripted prompts end.
  readonly completedTurns = signal(0);
  readonly scriptPrompter = new WebScriptPrompter();
  readonly live = signal(false);
  defaultSample = 'simple_chat';
  maxRequestBytes = 8 * 1024 * 1024;
  private agent?: HttpAgent;
  private opening?: AbortController;
  private threadId?: string;
  private generation = 0;
  private mcpTools = new Set<string>();
  private submitted = new Map<string, Message>();

  /** Load menu data without creating a graph or making a model request. */
  async loadSamples(): Promise<void> {
    try {
      const response = await fetch('/api/samples');
      if (!response.ok) throw new Error('Cannot load samples. Check the local server.');
      const data: unknown = await response.json();
      if (!record(data) || !Array.isArray(data['samples'])) throw new Error('Invalid sample catalog.');
      const samples = data['samples'];
      if (!samples.every((item: unknown) => record(item) && typeof item['id'] === 'string'
        && typeof item['title'] === 'string' && typeof item['description'] === 'string'
        && (item['mcpTools'] === undefined || Array.isArray(item['mcpTools']) && item['mcpTools'].every((name: unknown) => typeof name === 'string'))
        && Array.isArray(item['prompts']) && item['prompts'].every((p: unknown) => typeof p === 'string'))) {
        throw new Error('Invalid sample catalog.');
      }
      this.samples.set(samples as Sample[]);
      if (typeof data['defaultSample'] === 'string') this.defaultSample = data['defaultSample'];
      this.live.set(data['live'] === true);
      if (typeof data['maxRequestBytes'] === 'number') this.maxRequestBytes = data['maxRequestBytes'];
      this.error.set('');
    } catch (error) { this.error.set(error instanceof Error ? error.message : 'Cannot load samples.'); }
  }

  /** A new sample means a new model cursor and native LangGraph checkpoint. */
  async connect(sample: string): Promise<void> {
    this.disconnect();
    // Use explicit sample transport provenance, never guesses from tool names.
    this.mcpTools = new Set(this.samples().find(item => item.id === sample)?.mcpTools ?? []);
    const generation = this.generation;
    const opening = new AbortController();
    this.opening = opening;
    this.messages.set([]);
    this.pending.set([]);
    this.answers.clear();
    this.outputUrl.set('');
    this.contextUrl.set('');
    this.contextUsage.set(null);
    this.outputAvailable.set(false);
    this.subagents.set([]);
    this.submitted.clear();
    this.completedTurns.set(0);
    this.scriptPrompter.reset(this.samples().find(item => item.id === sample)?.prompts ?? []);
    this.error.set('');
    this.status.set('connecting');
    try {
      const response = await fetch('/api/sessions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample, live: this.live() }), signal: opening.signal,
      });
      if (!response.ok) throw new Error('Cannot open this sample. Check its server configuration, then start a new chat.');
      const data: unknown = await response.json();
      if (!record(data) || typeof data['threadId'] !== 'string') throw new Error('Invalid session response.');
      if (generation !== this.generation) { this.discard(data['threadId']); return; }
      this.threadId = data['threadId'];
      if (sample.startsWith('claims_context_')) this.contextUrl.set(`/api/sessions/${this.threadId}/context`);
      if (sample === 'file_approval') this.outputUrl.set(`/api/sessions/${this.threadId}/output`);
      this.agent = new HttpAgent({ url: `/api/chat/${this.threadId}`, threadId: this.threadId });
      this.agent.subscribe({
        onCustomEvent: ({ event }) => {
          // The server projects its authoritative checkpoint after the turn.
          // A generation check prevents an old session updating a new chat.
          if (generation === this.generation && event.name === 'retained_context') {
            this.contextUsage.set(event.value as ContextUsage);
          }
        },
        onMessagesChanged: ({ messages }) => {
          if (generation === this.generation) this.showMessages(messages);
        },
        onSubagentStartedEvent: ({ event }) => {
          if (generation !== this.generation) return;
          this.updateSubagent(event.subagentRunId, {
            name: event.name, description: event.description,
            parentId: event.parentSubagentRunId, toolId: event.parentToolCallId,
            status: 'Running',
          });
        },
        onSubagentFinishedEvent: ({ event }) => {
          if (generation !== this.generation) return;
          this.updateSubagent(event.subagentRunId, {
            status: event.outcome?.type === 'suspended' ? 'Paused' : 'Completed',
          });
        },
        onSubagentErrorEvent: ({ event }) => {
          if (generation !== this.generation) return;
          this.updateSubagent(event.subagentRunId, { status: 'Failed', error: event.message });
        },
        onRunErrorEvent: ({ event }) => {
          if (generation !== this.generation || this.status() === 'stopping') return;
          this.error.set(event.message);
          this.status.set('failed');
        },
        onRunFinishedEvent: (result) => {
          if (generation !== this.generation) return;
          if (result.outcome === 'success') {
            if (this.outputUrl()) void fetch(this.outputUrl(), { method: 'HEAD' }).then(response => {
              if (generation === this.generation) this.outputAvailable.set(response.ok);
            }).catch(() => undefined);
            this.pending.set([]);
            this.answers.clear();
            this.completedTurns.update(count => count + 1);
            this.scriptPrompter.completeTurn();
          } else if (result.outcome === 'interrupt') {
            // AG-UI carries the original workflow interaction in metadata.
            // A pause retains the session and does not consume a user turn.
            this.pending.set(result.interrupts.map(item => {
              const metadata = item.metadata;
              const lg = record(metadata) ? metadata['langgraph'] : undefined;
              const raw = record(lg) ? lg['raw'] : undefined;
              return { id: item.id, payload: record(raw) ? raw : {} };
            }));
            this.answers.clear();
            this.status.set('waiting');
          } else {
            this.error.set('Run cancelled. Start a new chat to continue.');
            this.status.set('failed');
          }
        },
      });
      this.status.set('ready');
    } catch (error) {
      if (generation !== this.generation) return;
      this.error.set(error instanceof Error ? error.message : 'Cannot open this sample.');
      this.status.set('failed');
    }
  }

  /** Start a standard run; leave draft clearing to the component on acceptance. */
  send(prompt: string, attachments: Attachment[]): boolean {
    const agent = this.agent;
    if (!agent || this.status() !== 'ready' || (!prompt.trim() && !attachments.length)) return false;
    // Include the SDK's outgoing history in the byte budget, not just the new
    // attachment. There is no custom SSE framing or decoder in the application.
    const id = crypto.randomUUID();
    const user: AgentMessage = { id, role: 'user', content: prompt || 'Attached context' };
    const forwardedProps = { prompt, attachments };
    const estimated = JSON.stringify({ messages: [...agent.messages, user], state: agent.state, forwardedProps });
    if (new TextEncoder().encode(estimated).length + 1024 > this.maxRequestBytes) {
      this.error.set('This conversation exceeds 8 MiB. Remove a file or start a new chat.');
      return false;
    }
    this.submitted.set(id, { id, role: 'user', text: prompt, files: attachments.map(file => file.name) });
    agent.addMessage(user);
    this.error.set('');
    this.status.set('running');
    void this.run(agent, forwardedProps, this.generation);
    return true;
  }

  /** Answer workflow interactions through the same driver as normal requests. */
  answer(id: string, value: string): void {
    if (!this.agent || this.status() !== 'waiting') return;
    this.answers.set(id, value);
    if (!this.pending().every(item => this.answers.has(item.id))) return;
    const resume = this.pending().map(item => ({ interruptId: item.id,
      status: 'resolved' as const, payload: this.answers.get(item.id)! }));
    this.status.set('running'); // Disable duplicate clicks before starting HTTP.
    this.error.set('');
    void this.run(this.agent, {}, this.generation, resume);
  }

  format(value: unknown): string { return JSON.stringify(value, null, 2); }

  /** The SDK aborts its HTTP fetch; disconnect cancels the server's native run. */
  stop(): void {
    if (this.status() !== 'running') return;
    this.status.set('stopping');
    this.agent?.abortRun();
  }

  disconnect(): void {
    this.generation += 1;
    this.opening?.abort();
    this.agent?.abortRun();
    if (this.threadId) this.discard(this.threadId);
    this.threadId = undefined;
    this.agent = undefined;
  }

  private async run(agent: HttpAgent, forwardedProps: object, generation: number, resume?: { interruptId: string; status: 'resolved'; payload: string }[]): Promise<void> {
    try { await agent.runAgent({ forwardedProps, resume }); }
    catch {
      if (generation === this.generation && this.status() !== 'stopping') {
        if (!this.error()) this.error.set('The response failed. Start a new chat to try again.');
        this.status.set('failed');
      }
    } finally {
      if (generation === this.generation) {
        if (this.status() === 'stopping') {
          this.error.set('Stopped. Start a new chat to continue.');
          this.status.set('failed');
        } else if (this.status() === 'running') this.status.set('ready');
        // A cancelled stream may end before a child lifecycle event arrives.
        // Do not leave its row claiming that it is still executing.
        this.subagents.update(items => items.map(item => item.status === 'Running'
          ? { ...item, status: 'Ended without completion' } : item));
      }
    }
  }

  /** Project the SDK's accumulated messages into compact conversation entries.
   * HttpAgent applies TOOL_CALL_* events and final snapshots. Joining results
   * by call ID works for streamed calls and snapshots without a second reducer.
   * TOOL_CALL_END only ends argument generation, so completion requires a result.
   */
  private showMessages(messages: ReadonlyArray<Readonly<AgentMessage>>): void {
    this.messages.set(this.projectMessages(messages));
    // Snapshots can contain only this run's child messages. Keep earlier child
    // conversations available across later turns instead of dropping them.
    for (const owner of new Set(messages.map(message => message.subagentRunId))) {
      if (owner) this.updateSubagent(owner, { messages: this.projectMessages(messages, owner) });
    }
  }

  private projectMessages(messages: ReadonlyArray<Readonly<AgentMessage>>, owner?: string): Message[] {
    const visible: Message[] = [];
    const results = new Map<string, string>();
    for (const message of messages) {
      if (message.role === 'tool' && message.subagentRunId === owner) results.set(message.toolCallId,
        typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2));
    }
    for (const message of messages) {
      if (message.role !== 'user' && message.role !== 'assistant') continue;
      if (message.subagentRunId !== owner) continue;
      const submitted = this.submitted.get(message.id);
      if (submitted) { visible.push(submitted); continue; }
      const text = typeof message.content === 'string' ? message.content : '';
      const tools = message.role === 'assistant' ? message.toolCalls?.map(call => ({
        id: call.id, name: call.function.name, mcp: this.mcpTools.has(call.function.name), arguments: call.function.arguments,
        result: results.get(call.id),
      })) : undefined;
      if (text || tools?.length) visible.push({ id: message.id, role: message.role, text, files: [], name: message.name, tools });
    }
    return visible;
  }

  /** Lifecycle metadata and SDK message projections share stable invocation IDs. */
  private updateSubagent(id: string, patch: Partial<SubagentConversation>): void {
    this.subagents.update(items => {
      const existing = items.find(item => item.id === id);
      const updated = { ...(existing ?? { id, name: 'Subagent', status: 'Running', messages: [] }), ...patch };
      return existing ? items.map(item => item.id === id ? updated : item) : [...items, updated];
    });
  }

  /** Attach children to their spawning tool; unlinked children remain visible. */
  children(owner?: string, toolId?: string): SubagentConversation[] {
    const messages = owner ? this.subagents().find(item => item.id === owner)?.messages ?? [] : this.messages();
    const calls = new Set(messages.flatMap(message => message.tools?.map(tool => tool.id) ?? []));
    return this.subagents().filter(item => item.parentId === owner &&
      (toolId ? item.toolId === toolId : !item.toolId || !calls.has(item.toolId)));
  }

  /** Cleanup is best effort on unload; server expiry also releases idle state. */
  private discard(threadId: string): void {
    void fetch(`/api/sessions/${threadId}`, { method: 'DELETE', keepalive: true }).catch(() => undefined);
  }
}
