# Security & privacy policy

## Reporting a vulnerability

Please report security or privacy issues **privately** — do not open a public issue. Use
GitHub's "Report a vulnerability" (Security Advisories) on this repository, or email the
maintainers listed in the repository metadata. We aim to acknowledge within a few days.

Helpful details: affected version/commit, repro steps (a seed + encounter is ideal), impact,
and any suggested fix.

## Scope & data-handling stance

TriageFlow Lite is a prototype, **not a medical device**. Still, it is designed around
privacy-preserving defaults:

- **No real patient data** is shipped or required. The project runs on synthetic data only.
- **Local inference.** The open-weights model runs locally via Ollama; patient data entered
  into a deployment does not leave the machine. The configured data region is surfaced in the
  UI.
- **Audit log contents.** `~/.triageflow/audit.{jsonl,db}` records tool inputs/outputs,
  timestamps, and approver names. In a real deployment this is sensitive operational data:
  it is git-ignored, and an operator must apply appropriate file permissions, retention, and
  access controls. Do not commit audit logs or databases.
- **Borrowed filesystem MCP** is started with a single allowed directory (`protocols/`) and is
  used read-only. Do not widen its allowed roots to directories containing sensitive data.
- **No telemetry.** The application makes no outbound network calls except, on first run, the
  model download (Ollama registry) and loading front-end assets from public CDNs.

## Things to check before any pilot

- Replace CDN-loaded front-end assets with vendored copies if operating offline or under a
  strict content policy.
- Put the audit store on encrypted, access-controlled storage with a defined retention policy.
- Confirm the clinical thresholds against the locally-adopted triage protocol, reviewed by a
  qualified clinician.
- Ensure any real data used in a pilot is covered by the facility's consent and ethics process.
