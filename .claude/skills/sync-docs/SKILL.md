---
name: sync-docs
description: Sync this session's results, decisions, fixes, and conventions to their canonical homes — the publishable repo docs (doc/, .claude/docs, docs/adr, CLAUDE.md), the research knowledge base at ~/biped_hrl_wiki for exploratory narrative, and auto-memory — with a personal-identifier redaction gate before finishing. Use at the end of a work session or milestone, or whenever the user asks to "update the docs/memory", "write down what we did/learned", or "sync documentation".
---

# Sync session knowledge into docs, the research KB, and memory

You route what this session produced into the right home. The cardinal rules:
**one canonical home per fact** (duplication desyncs), **CLAUDE.md stays short**
(links, not content), and **nothing personal enters the repo** (it is a public fork).

## 0. The split (read this first — it changed on 2026-07-28)

There are two documentation systems, with opposite rules. Putting a fact in the
wrong one is the main failure mode of this skill.

| | **This repo** (`doc/`, `.claude/docs/`, `docs/adr/`, `CLAUDE.md`) | **Research KB** (`~/biped_hrl_wiki`) |
|---|---|---|
| Holds | Current design and status: the options in use, how the working solution is implemented, decision records, gate ladders, safety procedures | Exploratory narrative: what was tried, what failed, why, root causes, dated session logs, full result tables, ideas backlogs |
| Rule | **Stay lean.** It gets published. Retire superseded text rather than accumulating it | **Never condense.** Preserve run names, log dirs, W&B ids and numbers verbatim |
| Reader | Someone using or evaluating the code | The thesis report |

Before 2026-07-28 the repo's `doc/hrl/` journals held both, and this skill told you
never to condense them. That is no longer true **for the repo**: the narrative was
migrated out so the repo could be published, and `doc/hrl/A1_findings.md`,
`A1_goal_achievability_probe.md` and `hierarchy_benefit_roadmap.md` no longer exist
here. The never-condense rule did not disappear — it moved to the KB. Do not
re-create findings ledgers or journals under `doc/hrl/`.

If the KB is unavailable (different machine, not checked out), do **not** fall back
to writing narrative into `doc/hrl/`. Say so, and hold the narrative in your report
for the user to file later.

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
| A\<N\> **design / as-built spec**, current status, open ablations | `doc/hrl/A<N>_*.md` (A1: `A1_HIRO.md`; A2: `A2_ARMA.md`) | The "how it works now + where we are" doc. Update its status header and the status-dashboard row in `doc/hrl/HRL_plan.md`. New architectures get their own file. |
| A\<N\> **run results, failed fixes, root-cause analyses, dead ends, lessons** (chronological) | **KB** `wiki/architectures/a<N>-findings-ledger.md`, `a1a-experiment-journal.md`, `a1a-deploy-journal.md` | The findings ledger: what was tried → outcome → why → lesson. Keep result tables verbatim; carry run name / log dir / W&B id / key metrics. Link to the design spec, don't restate it. |
| **Ideas backlogs, proposals, parked work** | **KB** `wiki/thesis/hierarchy-benefit-roadmap.md` (or `open-questions.md` for looser items) | Never in the repo. |
| **Closed workline verdicts, delivered hand-off prompts** | **KB** `wiki/architectures/worklines-archive.md` | `doc/hrl/worklines.md` keeps only the live registry: status table, locked decisions, standing rules. |
| **Literature** (a paper read, a concept, a comparison) | **KB** `wiki/papers/`, `wiki/concepts/`, `wiki/comparisons/` | One page per source. Follow the KB's own `CLAUDE.md`. |
| **Cross-architecture HRL machinery** (co-train loop, goal space, warm-start, reward decomp, benchmark/probe tools, checkpoint/ONNX, shared gotchas) | `.claude/docs/hrl-infra.md` | Reused by all A1–A4. Put a mechanism here the moment a 2nd architecture would touch it. |
| **Architecture decisions** (a choice made, its alternatives, its consequences) | `docs/adr/` | ADRs stay in the repo — they are reference, not narrative. The *investigation* that led to one belongs in the KB. |
| Stable project knowledge: file/class maps, obs layouts, cluster ops, deploy pipeline, experiment-design rules | `.claude/docs/*.md` | Extend the matching file (`codebase-map`, `cluster`, `deployment`, `experiment-design`, `hrl-infra`); create a new file only for a genuinely new topic, then link it from `CLAUDE.md` / `HRL_plan.md`. |
| Safety procedures, gate ladders, rollback rules | `doc/hrl/A1a_deploy_plan.md` | Operational reference someone may need at the robot. Keep in the repo even though the sessions that produced it are in the KB. |
| Conventions needed at every launch or code edit (launch flags, max-iterations N+1, coding style) | `CLAUDE.md` | One line + link at most. If it needs a paragraph, it belongs in `.claude/docs/` with a CLAUDE.md link. |
| User preferences, feedback on how to work, session-scoped context, anything personal | auto-memory ONLY | Never into the repo. Follow the memory system's conventions and update the `MEMORY.md` index. Keep entries compact; if the full record is in the KB, point at it. |

Never write the same content to two homes. When a repo doc and a KB page cover the
same work, the repo holds the current state and the KB holds how it was reached;
each links to the other rather than repeating it.

## 3. Writing to the repo — stay lean

Each sync should leave a repo doc **roughly flat in length**: a milestone *retires*
stale text, it doesn't only append.

- **Edit-in-place over append.** Find the existing line/row/status this fact updates
  and *replace* it. Add a new line only when nothing it supersedes exists.
- **Budget per fact: ~1 table row + ≤2 lines.** If you're writing a paragraph, you're
  duplicating a table, a linked doc, or narrative that belongs in the KB.
- **Retire what the new fact obsoletes:** flip "running/TODO" → done, delete the
  verdict the new result overturns, drop superseded hypotheses.
- **Numbers live in exactly one place.** Everything else links.
- Match each file's existing structure and terseness. Convert relative dates
  ("yesterday") to absolute ones.
- **Don't leave dangling pointers.** If you move or retire a section, fix every
  reference to it ("see X below") in the same pass — grep for it.

## 4. Writing to the research KB — never condense

The KB at `~/biped_hrl_wiki` is an **OKF v0.2 bundle**, and its own `CLAUDE.md` is
the authority. Read it before writing there. The rules you will get wrong otherwise:

- **Every page needs YAML frontmatter with a non-empty `type`** (`Paper`, `Concept`,
  `Architecture`, `Comparison`, `Reference`, `Synthesis`, `Open Questions`), plus
  `title` and a one-line standalone `description`.
- **`status` is OKF's lifecycle field** (`draft`/`stable`/`deprecated`) — project
  status goes in the body as a `**Project status:**` line, never in `status:`.
- **Never add `verified:`.** It means a human confirmed the page; only the user may
  authorize it (`scripts/okf-verify.py`).
- **Links are bundle-absolute markdown**, not Obsidian wikilinks:
  `[A1](/wiki/architectures/a1.md)`. Filenames are lowercase kebab-case slugs.
- **`index.md` and `log.md` are reserved** and carry no frontmatter (except the root
  `index.md`'s `okf_version`).
- **Append a `log.md` entry** under today's `## YYYY-MM-DD` heading, newest first.
- **Run `python3 scripts/okf-lint.py`** before finishing; it must report 0 errors.

Content rules there are the opposite of §3: preserve the blow-by-blow. Numbers,
run names, log dirs, W&B ids, superseded findings, dead ends and the "bug found →
fixed → re-measured" sagas are the point — that messiness is the report material,
and git history is not a substitute for it. When in doubt, keep it.

## 5. Redaction gate (MANDATORY before finishing — repo only)

The repo is public. Grep everything tracked plus anything you created/edited for
personal identifiers — the user's name, student/cluster account IDs, local machine
usernames, personal emails:

```bash
{ git ls-files; git status --porcelain | awk '{print $2}'; } | sort -u \
  | xargs grep -nilE "<name>|<student-id>|<lab-username>|<email>" 2>/dev/null
```

(Ask the user for the identifier list if unknown; do not write the real identifiers
into this skill or any repo file — keep machine-specific absolute paths out of repo
files too, use `~`, relative, or `$WORK`/`$USER` forms. Note that the KB's own path
contains a username: always write it as `~/biped_hrl_wiki` in repo files.)
Neutralize hits: "the user decided" instead of names, `$USER` instead of account
IDs. `.mailmap` / git history are out of scope (the user manages those).

**The KB has no redaction gate** — it is personal and not published. Don't apply
this repo's redaction habits there.

## 6. Report

End with a short table: what was written where, split by destination (repo / KB /
memory), plus anything that had two plausible homes and which one you chose. Note
the `okf-lint` result if you wrote to the KB. Do not commit — leave changes in the
working tree for review.
