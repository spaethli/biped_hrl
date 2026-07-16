---
name: delegate
description: Turn a task into a ready-to-paste hand-off prompt for a fresh chat or a background subagent, with a model recommendation. Use when the user says "delegate", "hand this off", "write a prompt for another chat", or when the current (planning) chat catches itself doing implementation work.
---

# Delegate

Generate hand-off prompts so implementation runs in cheap, fresh-context chats/agents
while the planning chat keeps sequencing and decisions. Worklines registry:
`doc/hrl/worklines.md`.

## Self-check for the delegating chat (apply BEFORE writing the prompt)

If the current chat is a planning chat and has just spent >3 tool calls on
implementation (editing source, debugging, running trainings, long grep chains): stop.
That work is usually too easy for the expensive model and burns planning context.
Emit a hand-off prompt instead, or spawn a background subagent. Say so explicitly:
"delegating this rather than continuing inline."

## Choosing the executor

- **Background subagent (sonnet), spawned from this chat**: isolated one-off with an
  objective pass criterion, no files another workline owns, result returns here.
  Examples: a standalone script with a numeric pass gate, a benchmark run + table.
- **Fresh chat, sonnet class**: well-scoped implementation inside one subsystem with
  fixed protocol and verifiable outputs (reward-term arms, launch + bench batches,
  YAML/config work, analyzer extensions).
- **Fresh chat, opus/fable class**: mechanism diagnosis, architecture decisions,
  anything cross-system (training <-> deploy), and anything where a silently wrong
  answer poisons downstream work (root-cause verdicts, sim2real gates).
- **Haiku**: mechanical transformations with an exact spec and a checker (renames,
  table reformatting). Rare here.

## Hand-off prompt template

Every prompt must contain, in this order:

1. **Task** in one sentence, plus the spec-first reminder ("per CLAUDE.md: analysis /
   spec before code; present for approval before X" for anything non-trivial).
2. **Read first**: 2-4 canonical doc pointers (paths, not pasted content - CLAUDE.md
   auto-loads and indexes the rest). Include the findings-ledger row if one exists.
3. **Tasks in order**, each with its own verifiable output.
4. **Protocol constants**: launch conventions (conda activate + python, never
   `conda run`; N+1 iterations; naming convention; bench commands) only where the
   task runs them.
5. **Pass criteria / verdict format**: objective, checkable without trusting prose.
6. **Reporting target**: which doc/table the results land in (single canonical home,
   house style, honest reads), plus a sync line in `doc/hrl/worklines.md`.
7. **Do-not-touch list**: files/subsystems owned by other worklines.

## Rules

- One workline = one owner chat. Never hand two chats the same files.
- The prompt names a MODEL recommendation explicitly, with one clause of why.
- Trainings and benchmarks belong in the executor chat, not the planning chat.
- If the task's pass criterion cannot be stated objectively, it is not ready to
  delegate - grill it first (/grill-with-docs).
