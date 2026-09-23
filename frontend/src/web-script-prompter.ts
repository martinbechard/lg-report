/** Own one browser conversation's position in its authored prompt sequence.
 * The catalog supplies data; this class advances only after a completed turn.
 * Interruptions and failed runs do not consume another prompt. Exhaustion never
 * switches to free-form chat or restarts a script implicitly.
 * AI attribution: Generated with AI assistance by Northstar.
 * Copyright (c) 2026 Martin.Bechard@DevConsult.ca
 */
import { computed, signal } from '@angular/core';

export class WebScriptPrompter {
  private readonly prompts = signal<readonly string[]>([]);
  private readonly position = signal(0);
  readonly nextPrompt = computed(() => this.prompts()[this.position()] ?? '');

  /** A selected sample or new session starts a fresh independent sequence. */
  reset(prompts: readonly string[]): void {
    this.prompts.set([...prompts]);
    this.position.set(0);
  }

  /** Called only after the workflow completes, including any resumed interrupts. */
  completeTurn(): void {
    this.position.update(index => Math.min(index + 1, this.prompts().length));
  }
}
