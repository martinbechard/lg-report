/** Start the standalone, zoneless Angular UI; Python owns models and history.
 * AI attribution: Generated with AI assistance by Northstar.
 * Copyright (c) 2026 Martin.Bechard@DevConsult.ca
 */
import { bootstrapApplication } from '@angular/platform-browser';
import { ChatApp } from './chat-app';

bootstrapApplication(ChatApp).catch((error: unknown) => console.error(error));
