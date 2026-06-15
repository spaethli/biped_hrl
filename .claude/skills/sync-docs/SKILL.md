---
name: sync-docs
description: Sync this session's results, decisions, fixes, and conventions into the project documentation (plan doc §0, .claude/docs, CLAUDE.md) and auto-memory, each fact to its single canonical home, with a personal-identifier redaction gate before finishing. Use at the end of a work session or milestone, or whenever the user asks to "update the docs/memory", "write down what we did/learned", or "sync documentation".
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
| A1 status, run results, decisions, root-cause analyses | `doc/A1_HIRO_implementation_plan.md` **§0** | §0 is authoritative over older sections; update the status header too. Other architectures (A2+): their own design doc under `doc/`. |
| Stable project knowledge: file/class maps, obs layouts, cluster ops, deploy pipeline, experiment-design rules | `.claude/docs/*.md` | Extend the matching existing file (`codebase-map`, `cluster`, `deployment`, `experiment-design`); create a new file only for a genuinely new topic, then link it from CLAUDE.md. |
| Conventions needed at every launch or code edit (e.g. launch flags, max-iterations N+1, coding style) | `CLAUDE.md` | One line + link at most. If it needs a paragraph, it belongs in `.claude/docs/` with a CLAUDE.md link. |
| User preferences, feedback on how to work, session-scoped context, anything personal | auto-memory ONLY | Never into the repo. Follow the memory system's own conventions for file format and the MEMORY.md index. |

Never write the same content to two homes. If a repo doc and a memory overlap, the
repo doc is canonical for project facts; the memory may *point* to it.

## 3. Update

- Match each file's existing structure and terseness; update stale statements you
  touch (status headers, ✅/⬜ markers, "next steps").
- Convert relative dates ("yesterday") to absolute dates.
- Plan-doc results entries should carry enough identity to find the run again:
  run name, log dir, W&B run id, key metrics.

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
