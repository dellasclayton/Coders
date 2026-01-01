# CLAUDE.md

## Project Overview

Low-latency voice chat application with a FastAPI (Python) backend and vanilla JavaScript frontend. This is a **single-user application**—do not over-engineer for enterprise scale or multi-tenancy.

## Core Philosophy

- **Simplicity over cleverness.** Write code a human can read and maintain.
- **Do what is asked.** No future-proofing, no speculative edge cases, no unsolicited abstractions.
- **Minimal viable solution first.** Add complexity only when explicitly requested.

---

## Python / FastAPI Backend

### Style

- **Type hints:** Use everywhere (function signatures, variables where useful).
- **Docstrings:** Minimal or none. Code should be self-documenting via clear naming.
- **Naming:** `snake_case` for everything. Function names must be clear and descriptive—if a name requires mental gymnastics to understand, it's wrong.
- **Comments:** Include "what" and "why" where the code isn't immediately obvious. Skip the obvious.

### Async Patterns

Preferred patterns:
- `asyncio.create_task()` for fire-and-forget or managed background work
- Producer/consumer with `asyncio.Queue`
- `asyncio.as_completed()` when processing results as they arrive

Avoid:
- Overly nested async context managers
- Complex task orchestration frameworks when simple queues suffice

### Error Handling

Keep it simple:
- Log the error with **specific, actionable context** (what failed, relevant IDs/values, why it might have failed)
- Move on—don't wrap everything in elaborate try/except hierarchies
- No custom exception class hierarchies unless explicitly requested

```python
# Good
logger.error(f"TTS generation failed for chunk '{text[:50]}...': {e}")

# Bad
logger.error("An error occurred")
```

### Key Libraries

- **RealtimeSTT** (faster-whisper) — speech-to-text
- **RealtimeTTS** — text-to-speech streaming
- **stream2sentence** — sentence boundary detection
- **Higgs Audio** — TTS service
- **Supabase** — database
- **FastAPI** — API framework

---

## JavaScript Frontend

### Style

- **Vanilla JS only.** No frameworks.
- **Functional programming preferred.** Avoid classes and OOP patterns; use pure functions and closures.
- **ES Modules:** Keep imports/exports simple and flat. Avoid circular dependencies or complex re-exports that break loading.

### Structure

- Flat: `index.html`, `.js`, `.css` files in `/frontend`

---

## Code Organization

### Directory Structure

```
/backend
  fastserver.py
  database_director.py
/frontend
  index.html
  *.js
  *.css
```

### File Size

- Target: **500–600 lines max** per file
- If a file approaches this limit, split by responsibility
- Group related functions together, but don't let files bloat

---

## What NOT To Do

1. **Don't over-engineer.** No abstract base classes, factory patterns, or dependency injection unless explicitly asked.
2. **Don't add speculative features.** Solve the stated problem, nothing more.
3. **Don't create deep abstraction layers.** One level of indirection is usually enough.
4. **Don't use vague function names.** `process_data()` and `handle_stuff()` are not acceptable.
5. **Don't wrap simple operations in unnecessary classes.** A function is usually fine.
6. **Don't add extensive error recovery for edge cases not mentioned.** Log and continue.

---

## When In Doubt

- Ask for clarification rather than assuming complexity is needed.
- Prefer the boring, obvious solution.
- If the simple approach might not scale, **mention it briefly** but implement simple anyway unless told otherwise.