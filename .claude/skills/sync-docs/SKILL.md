---
name: sync-docs
description: Sync this session's results, decisions, fixes, and conventions to their canonical homes — the publishable repo docs (doc/, .claude/docs, docs/adr, CLAUDE.md), an append-only engineering journal in the research KB at ~/biped_hrl_wiki/raw/engineering-journal/, and auto-memory — with a personal-identifier redaction gate before finishing. Use at the end of a work session or milestone, or whenever the user asks to "update the docs/memory", "write down what we did/learned", or "sync documentation".
---

# Sync session knowledge into docs, the journal, and memory

You route what this session produced into the right home. The cardinal rules:
**one canonical home per fact** (duplication desyncs), **CLAUDE.md stays short**
(links, not content), and **nothing personal enters the repo** (it is a public fork).

## 0. Three destinations (read this first)

| Destination | Holds | Rule |
|---|---|---|
| **This repo** (`doc/`, `.claude/docs/`, `docs/adr/`, `CLAUDE.md`) | Current design and status: the options in use, how the working solution is implemented, decision records, gate ladders, safety procedures | **Stay lean.** It gets published. Retire superseded text rather than accumulating it |
| **Engineering journal** (`~/biped_hrl_wiki/raw/engineering-journal/`) | The chronological narrative: what was tried, what failed, why, root causes, dated session logs, full result tables | **Append-only and verbatim.** Never condense, never rewrite a past entry |
| **auto-memory** | Collaboration feedback, user preferences, and compact pointers to the above | Keep entries short; point at the fuller record |

**You do not write wiki pages.** The journal is a *raw source*. A separate **Ingest**
operation, run inside the wiki vault, compiles it into `wiki/` pages. Writing directly
to `wiki/` from here would bypass that and re-create the problem this split exists to
fix — 20k-word pages nobody can query. Your job is to produce a good source and say
that an Ingest is pending.

Before 2026-07-28 this skill routed narrative into `doc/hrl/A<N>_findings.md` and told
you never to condense the `doc/hrl` planning docs. Both are obsolete: the narrative
moved out so the repo could be published, and those files no longer exist here. The
never-condense rule did not disappear — it now applies to the journal. **Do not
re-create findings ledgers or journals under `doc/hrl/`.**

If the KB is unavailable (different machine, not checked out), do **not** fall back to
writing narrative into `doc/hrl/`. Say so, and put the narrative in your report for the
user to file later.

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
| A\<N\> **design / as-built spec**, current status, open ablations | `doc/hrl/A<N>_*.md` (A1: `A1_HIRO.md`; A2: `A2_ARMA.md`) | The "how it works now + where we are" doc. Update its status header and the dashboard row in `doc/hrl/HRL_plan.md`. |
| **Architecture decisions** (a choice, its alternatives, its consequences) | `docs/adr/` | ADRs are reference and stay here. The *investigation* behind one goes in the journal. |
| **Cross-architecture HRL machinery** (co-train loop, goal space, warm-start, benchmark/probe tools, checkpoint/ONNX, shared gotchas) | `.claude/docs/hrl-infra.md` | Put a mechanism here the moment a 2nd architecture would touch it. |
| Stable project knowledge: file/class maps, obs layouts, cluster ops, deploy pipeline, experiment-design rules | `.claude/docs/*.md` | Extend the matching file; create a new one only for a genuinely new topic, then link it from `CLAUDE.md` / `HRL_plan.md`. |
| Safety procedures, gate ladders, rollback rules | `doc/hrl/A1a_deploy_plan.md` | Operational reference someone may need standing at the robot. Stays here even though the sessions that produced it are journalled. |
| Conventions needed at every launch or code edit | `CLAUDE.md` | One line + link at most. A paragraph belongs in `.claude/docs/`. |
| **Run results, failed fixes, root-cause analyses, dead ends, lessons, dated session logs** | **journal** — append to the matching file (below) | The whole chronological record. Verbatim. |
| **Ideas, proposals, parked work** | **journal** `hierarchy-benefit-roadmap.md` | Never in the repo. |
| **Closed workline verdicts, delivered hand-off prompts** | **journal** `worklines-archive.md` | `doc/hrl/worklines.md` keeps only the live registry. |
| A paper read, a concept learned | Drop the source in `~/biped_hrl_wiki/raw/papers/` and note that an Ingest is due | Literature is the wiki's Ingest job, not this skill's. |
| User preferences, feedback on how to work, anything personal | auto-memory ONLY | Never into the repo. Update the `MEMORY.md` index. |

Never write the same content to two homes. The repo holds the current state, the
journal holds how it was reached, and each points at the other.

## 3. Writing to the repo — stay lean

Each sync should leave a repo doc **roughly flat in length**: a milestone *retires*
stale text, it doesn't only append.

- **Edit-in-place over append.** Find the line/row/status this fact updates and
  *replace* it. Add a new line only when nothing it supersedes exists.
- **Budget per fact: ~1 table row + ≤2 lines.** Writing a paragraph means you are
  duplicating a table, a linked doc, or narrative that belongs in the journal.
- **Retire what the new fact obsoletes:** flip "running/TODO" → done, delete the
  verdict the new result overturns, drop superseded hypotheses.
- **Numbers live in exactly one place.** Everything else links.
- **Don't leave dangling pointers.** If you retire a section, grep for references to
  it ("see X below") and fix them in the same pass.

## 4. Appending to the engineering journal

Files (create a new one only for a genuinely new track, and say so in your report):

| File | Scope |
|---|---|
| `a1-findings-ledger.md` | A1 levers tried, outcomes, lessons |
| `a1a-experiment-journal.md` | A1a training/sim: stages, sweeps, result tables |
| `a1a-deploy-journal.md` | Bridge and hardware sessions, defects, root causes |
| `a0-model-delta-vs-upstream.md` | Fork-vs-upstream model differences |
| `worklines-archive.md` | Closed verdicts, delivered hand-off prompts |
| `hierarchy-benefit-roadmap.md` | Ideas backlog, parked work |

**Append a dated entry** at the end of the matching file:

```markdown
## 2026-07-28 — arm4d replicates on seed 123

<what was run, what happened, the numbers, the read>
```

Rules, all of which exist because this is a **source**:

- **Append only.** Never edit or condense an existing entry. If an earlier entry turns
  out to be wrong, write a new dated entry that says so and why — that correction is
  itself evidence, and the sequence of being wrong then right is what a thesis needs.
- **Verbatim numbers and identity.** Run name, log dir, W&B id, the metrics as
  measured. A result without its run identity cannot be re-benchmarked.
- **Keep the dead ends.** Failed fixes, falsified hypotheses and the "bug found →
  fixed → re-measured" sequence are the point, not noise.
- **No frontmatter needed.** These files sit outside the OKF bundle; don't add
  `type:`/`status:` to a journal entry and don't run the bundle linter over them.
- **Don't tidy.** Length is not a defect here. Compression happens downstream in
  Ingest, which can always re-read the source; the source cannot be recovered once
  summarized.

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
files too, use `~`, relative, or `$WORK`/`$USER` forms. The KB path contains a
username: always write it as `~/biped_hrl_wiki` in repo files.) Neutralize hits:
"the user decided" instead of names, `$USER` instead of account IDs.

**The journal and the KB have no redaction gate** — they are personal and not
published. Don't apply this repo's redaction habits there.

## 6. Hand off to Ingest

The journal entry is a source, so the wiki is now stale until it is compiled. Finish by
either running the wiki's **Ingest** operation (read `~/biped_hrl_wiki/CLAUDE.md`
first — it owns the page contract, OKF conventions and the `log.md` entry), or stating
clearly that **an Ingest is pending** and which journal files changed.

Ingest finds new material by date: it processes journal entries dated after the last
`Ingest` entry for the engineering journal in the wiki's `log.md`. So the dated heading
in §4 is load-bearing — without it your entry is invisible to the compile step.

## 7. Keep the report inbox current

`wiki/thesis/report-inbox.md` is not a fourth destination — it's a derived index over
(1) the repo and (2) the journal that mirrors `raw/latex/main.tex`'s own section
headers, so Liam can find what's new without re-reading either. Whenever this session
routed a fact that changes what's ready to go into the thesis — a solved/closed
milestone, a new architecture result, a formula that no longer matches the current
code, a hardware finding, a paper that grounds one of the above — check whether the
inbox needs a line:

- **Match, don't duplicate.** One line: the claim, its ✅/⏳ marker (✅ = already
  compiled into a wiki page, ⏳ = journal-only, pending Ingest) and a link to the
  canonical home from §2/§4 — never restate the finding itself.
- **File it under the `main.tex` section it belongs to**, not a new structure of your
  own — a paste-ready location is the entire point of the page.
- **Retire, don't accumulate** (same discipline as §3): if this session's fact turns an
  existing ⏳ line ✅ (an Ingest happened) or overturns one, fix that line in place
  rather than adding a new one under it.
- Same KB-unavailable rule as the journal (§0): if the vault isn't checked out, skip
  this step and say so in your report rather than recreating it in the repo.

## 8. Report

End with a short table: what was written where, split by destination (repo / journal /
memory / report inbox), plus anything that had two plausible homes and which one you
chose. State whether an Ingest was run or is pending. Do not commit — leave changes in
the working tree for review.
