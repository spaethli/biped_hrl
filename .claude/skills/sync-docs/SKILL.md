---
name: sync-docs
description: Sync this session's results, decisions, fixes, and conventions into the project documentation (the doc/ HRL plan, .claude/docs, CLAUDE.md) and auto-memory, each fact to its single canonical home, with a personal-identifier redaction gate before finishing. Use at the end of a work session or milestone, or whenever the user asks to "update the docs/memory", "write down what we did/learned", or "sync documentation".
---

# Sync session knowledge into docs & memory

You route what this session produced into the right documentation home. The cardinal
rules: **one canonical home per fact** (duplication desyncs), **CLAUDE.md stays short**
(links, not content), and **nothing personal enters the repo** (it is a public fork).

## 1. Gather

List what the session produced before writing anything:

- experiment results / run outcomes (with run names, log dirs, key numbers)
- design or experiment decisions (and who decided / why)
- bug fixes & root causes
- new conventions or workflow rules
- user preferences / feedback about how to work

If the session produced nothing in these categories, say so and stop.

## 2. Route — one canonical home per fact

| Fact type | Home | Notes |
|---|---|---|
| A<N> **design / as-built spec**, current status, open ablations | `doc/hrl/A<N>_HIRO.md` (A1: `doc/hrl/A1_HIRO.md`) | The "how it works now + where we are" doc. Update its status header. Also update the status-dashboard row in `doc/hrl/HRL_plan.md`. New architectures get their own `doc/hrl/A<N>_*.md`. |
| A<N> **run results, failed fixes, root-cause analyses, lessons** (chronological) | `doc/hrl/A<N>_findings.md` (A1: the A1 findings ledger (research KB)) | The findings ledger: what was tried → outcome → why → lesson; keep result tables verbatim. Carry run name/log dir/W&B id/key metrics. Don't duplicate the design spec — link to it. |
| **Cross-architecture HRL machinery** (co-train loop, goal space, warm-start, reward decomp, benchmark/probe tools, checkpoint/ONNX, shared gotchas) | `.claude/docs/hrl-infra.md` | Reused by all A1–A4. Put a mechanism here (not in a per-arch doc) the moment a 2nd architecture would touch it. |
| Stable project knowledge: file/class maps, obs layouts, cluster ops, deploy pipeline, experiment-design rules (A0–A4 matrix, RQ2 rule) | `.claude/docs/*.md` | Extend the matching existing file (`codebase-map`, `cluster`, `deployment`, `experiment-design`, `hrl-infra`); create a new file only for a genuinely new topic, then link it from CLAUDE.md / `doc/hrl/HRL_plan.md`. |
| Conventions needed at every launch or code edit (e.g. launch flags, max-iterations N+1, coding style) | `CLAUDE.md` | One line + link at most. If it needs a paragraph, it belongs in `.claude/docs/` with a CLAUDE.md link. |
| User preferences, feedback on how to work, session-scoped context, anything personal | auto-memory ONLY | Never into the repo. Follow the memory system's own conventions for file format and the MEMORY.md index. |

Never write the same content to two homes. If a repo doc and a memory overlap, the
repo doc is canonical for project facts; the memory may *point* to it.

## 3. Update

- Match each file's existing structure and terseness; update stale statements you
  touch (status headers, ✅/⬜ markers, "next steps").
- Convert relative dates ("yesterday") to absolute dates.
- Findings-ledger entries should carry enough identity to find the run again:
  run name, log dir, W&B run id, key metrics.

### Routine sync vs. a requested restructuring/cleanup pass

The "Conciseness budget" below governs the **routine** mode of this skill: routing one
new fact at a time into an already-healthy doc. When the user instead explicitly asks to
**restructure, clean up, or shrink** the `doc/hrl/` planning docs (`A1a_plan.md`,
`A1a_deploy_plan.md`, `worklines.md`, and their siblings for future architectures) — a
different, stricter rule applies, because **these docs are the user's primary thesis
report source material**, not disposable working notes: he queries them later for "what
happened and why" when writing the report, and a condensed "final state only" rewrite
destroys exactly the methodology narrative (what was tried, why it failed, how the root
cause was found) a thesis needs. Git history is not a substitute for this — the docs
themselves are the working knowledge base.

For a restructuring pass on these docs:
- **Default action is reorganize-in-place**: promote current-status dashboards to the
  top, fix identified stale status fields (✅/⬜/dates that no longer match later
  sections), improve headings/section order/navigability. This is not "cutting."
- **Only trim two things**: (a) content that is genuinely unimportant to the report or
  its results (not just verbose), or (b) a literal duplicate of a fuller copy that
  already lives elsewhere in the doc set — and even then, point to the canonical copy,
  never delete the only copy of something.
- **Do not condense, summarize, or cut**: narrative detail, numbers, superseded/dead-end
  findings, or the blow-by-blow of "bug found → fixed → re-measured" sagas, even when
  they read like a messy chronological journal. That messiness ledger *is* the report
  material.
- When in doubt whether a specific cut loses retrievable detail, keep it, or ask before
  making it — don't guess toward brevity.

### Conciseness budget (prevent doc bloat) — routine-sync mode only

Each sync should leave the doc **roughly flat in length** — a milestone *retires* stale
text, it doesn't only append. Concretely:

- **Edit-in-place over append.** First look for the existing line/row/status this fact
  updates and *replace* it. Only add a new line when nothing it supersedes exists.
- **Budget per fact: ~1 table row + ≤2 lines.** A new run result = one row in the
  results table (+ at most one sentence of interpretation). If you're writing a
  paragraph, you're duplicating a table or the linked doc — cut it.
- **Retire what the new fact obsoletes:** flip "running/TODO" → done, delete the old
  verdict the new result overturns, drop superseded hypotheses (don't keep both).
  Net new lines per sync should be small.
- **Numbers live in exactly one place.** Put the result table in its canonical doc;
  every other mention *links* to it rather than restating the figures.
- **Prefer tables/terse clauses over prose.** No restating context the reader can see
  one line up. Cut hedging and adjectives.
- If a section has grown sprawling, compact it *now* (merge rows, summarize a resolved
  thread to one line) rather than leaving it for a future cleanup pass.

## 4. Redaction gate (MANDATORY before finishing)

The repo is public. Grep everything tracked plus anything you created/edited for
personal identifiers — the user's name, student/cluster account IDs, local machine
usernames, personal emails:

```bash
{ git ls-files; git status --porcelain | awk '{print $2}'; } | sort -u \
  | xargs grep -nilE "<name>|<student-id>|<lab-username>|<email>" 2>/dev/null
```

(Ask the user for the identifier list if unknown; do not write the real identifiers
into this skill or any repo file — keep machine-specific absolute paths out of repo
files too, use relative or `$WORK`/`$USER` forms.) Neutralize hits: "the user
decided" instead of names, `$USER` instead of account IDs. `.mailmap` / git history
are out of scope (the user manages those).

## 5. Report

End with a short table: what was written where, plus anything that had two plausible
homes (and which one you chose). Do not commit — leave changes in the working tree
for review.
