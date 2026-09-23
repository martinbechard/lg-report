/** Present sample selection, streamed messages and explicitly queued text files.
 * Attachments use the same UTF-8/newline semantics as ConsoleClient's /attach.
 * Angular sanitizes Markdown HTML; model output never becomes trusted HTML.
 * AI attribution: Generated with AI assistance by Northstar.
 * Copyright (c) 2026 Martin.Bechard@DevConsult.ca
 */
import { afterRenderEffect, Component, computed, ElementRef, inject, OnDestroy, OnInit, Pipe, PipeTransform, signal, viewChild } from '@angular/core';
import { DecimalPipe, NgTemplateOutlet } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { marked } from 'marked';
import { Attachment, ChatStore } from './chat-store';

@Pipe({ name: 'markdown' })
class MarkdownPipe implements PipeTransform {
  /** Return ordinary HTML text, leaving Angular's innerHTML sanitizer enabled. */
  transform(text: string): string { return marked.parse(text, { async: false }); }
}

@Component({
  selector: 'chat-app',
  imports: [FormsModule, MarkdownPipe, NgTemplateOutlet, DecimalPipe],
  templateUrl: './chat-app.html',
})
export class ChatApp implements OnInit, OnDestroy {
  readonly chat = inject(ChatStore);
  readonly sampleId = signal('simple_chat');
  readonly live = this.chat.live;
  readonly draft = signal('');
  readonly clarification = signal('');
  readonly files = signal<Attachment[]>([]);
  readonly readingFiles = signal(false);
  readonly dragging = signal(false);
  readonly sample = computed(() => this.chat.samples().find(item => item.id === this.sampleId()));
  readonly busy = computed(() => ['running', 'stopping'].includes(this.chat.status()));
  readonly nextPrompt = this.chat.scriptPrompter.nextPrompt;
  readonly transcript = viewChild<ElementRef<HTMLElement>>('transcript');
  private followStream = true;

  constructor() {
    // Follow only while the reader is near the bottom. Streaming must not
    // repeatedly drag someone away from an earlier answer they are inspecting.
    afterRenderEffect(() => {
      this.chat.messages();
      this.chat.subagents();
      const element = this.transcript()?.nativeElement;
      if (element && this.followStream) element.scrollTop = element.scrollHeight;
    });
  }

  async ngOnInit(): Promise<void> {
    await this.chat.loadSamples();
    this.sampleId.set(this.chat.defaultSample);
    if (this.sample()) this.newChat();
  }

  ngOnDestroy(): void { this.chat.disconnect(); }

  /** Start a fresh sample session; the standard adapter owns its history. */
  newChat(): void {
    this.draft.set('');
    this.files.set([]);
    this.followStream = true;
    void this.chat.connect(this.sampleId());
  }

  chooseSample(id: string): void { this.sampleId.set(id); this.newChat(); }

  /** Mode changes need a fresh model and checkpoint, just like sample changes. */
  chooseMode(live: boolean): void { this.live.set(live); this.newChat(); }

  /** Scripted mode submits authored prompts; live mode accepts arbitrary input. */
  send(): void {
    const prompt = this.live() ? this.draft() : this.nextPrompt();
    if (!this.live() && !prompt) return;
    if (this.readingFiles()) return;
    if (this.chat.send(prompt, this.files())) {
      this.draft.set('');
      this.files.set([]);
      this.followStream = true;
    }
  }

  /** Example input is user data in either mode, never an agent instruction. */
  sendExample(): void {
    if (this.live()) this.draft.set(this.nextPrompt());
    this.send();
  }

  /** Preserve Shift+Enter and IME composition while making Enter send a turn. */
  onKey(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      this.send();
    }
  }

  onScroll(): void {
    const element = this.transcript()?.nativeElement;
    if (element) this.followStream = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
  }

  async fileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    await this.attach(Array.from(input.files ?? []));
    input.value = ''; // Selecting the same filename again must fire change.
  }

  async drop(event: DragEvent): Promise<void> {
    event.preventDefault();
    this.dragging.set(false);
    await this.attach(Array.from(event.dataTransfer?.files ?? []));
  }

  /** Read only user-selected files; names are labels and never server paths. */
  async attach(selected: File[]): Promise<void> {
    if (this.busy() || this.readingFiles()) return;
    this.readingFiles.set(true);
    this.chat.error.set('');
    try {
      for (const file of selected) {
        if (file.size > this.chat.maxRequestBytes) {
          this.chat.error.set(`${file.name} exceeds the 8 MiB request limit.`);
          continue;
        }
        try {
          // Fatal decoding rejects binary/non-UTF-8 content rather than sending
          // replacement characters. Python read_text normalizes newlines too.
          const content = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true })
            .decode(await file.arrayBuffer()).replace(/\r\n?/g, '\n');
          this.files.update(files => [...files, { name: file.name, content }]);
        } catch {
          this.chat.error.set(`Cannot attach ${file.name}: choose a readable UTF-8 text file.`);
        }
      }
    } finally { this.readingFiles.set(false); }
  }

  removeFile(index: number): void { this.files.update(files => files.filter((_, i) => i !== index)); }
}
