# Client migration focus

Window: 2026-07-28–2026-09-14 UTC. Source: `evalstate/hf-mcp-logs` @ `12c284c96e556c16aa7cb09ff1370dbbd6342941`.

Display label Hugging Face Chat UI means exactly chat-ui-mcp; chat-ui-intern remains separate. Codex CLI is the display label for codex-mcp-client and remains distinct from openai-mcp (Codex). CSV identities are unchanged.

Private review output. Fixed five excluded across every client/version; missing hashes retained. Self-reported identities, not users or proven product migrations.

Cyan daily shares; lime strict trailing-seven count-weighted shares; axes 0–100%. Six panels include chat-ui-mcp and codex-mcp-client; mcp (via mcp-remote) remains omitted. Mini bars retain actual daily all-protocol counts with independent per-client scales, but hide all numeric volume ticks and sample-size labels. Compare within-client activity patterns, not popularity. Counts remain available in CSVs.
Rates require ≥100 valid calls. Unknown/unreviewed protocols excluded from rate denominators, retained in volumes. Fewer than seven source days gives no last7 KPI. Blank rates are not zero; counts are never suppressed.
Missing source dates fail validation. No rows for a client on a covered date means zero observed calls. Coverage does not establish complete real-world traffic.

Contribution chart: top five of ALL retained identities by full-period modern calls + Other; percentages use full-period modern calls, not within-client shares.

## Computed insight table

| Identity | Last7 modern / valid calls | Last7 share | Period modern calls | Period contribution |
|---|---:|---:|---:|---:|
| claude-code | 68,541 / 71,971 | 95.2% | 226,374 | 31.9% |
| Anthropic/ClaudeAI | 39,697 / 39,967 | 99.3% | 181,224 | 25.5% |
| openai-mcp | 21,468 / 75,594 | 28.4% | 181,133 | 25.5% |
| chat-ui-mcp | 40,991 / 40,991 | 100.0% | 78,426 | 11.0% |
| chat-ui-intern | 17,158 / 17,158 | 100.0% | 17,402 | 2.5% |
| openai-mcp (Codex) | 2,615 / 17,313 | 15.1% | 11,910 | 1.7% |
| Other | 1,975 / 11,556 | 17.1% | 11,127 | 1.6% |
| antigravity-client | 420 / 420 | 100.0% | 1,251 | 0.2% |
| zcode | 66 / 66 | — | 476 | 0.1% |
| mcp | 111 / 10,273 | 1.1% | 306 | 0.0% |
| grok-shell-huggingface | 105 / 195 | 53.8% | 105 | 0.0% |
| opencode | 0 / 2,244 | 0.0% | 64 | 0.0% |
| unknown | 0 / 38 | — | 20 | 0.0% |
| claude-ai | 7 / 7 | — | 17 | 0.0% |
| github-copilot-developer | 0 / 42 | — | 7 | 0.0% |
| Anthropic/ClaudeDesign | 0 / 53 | — | 0 | 0.0% |
| Cursor | 0 / 6,800 | 0.0% | 0 | 0.0% |
| Gemini | 0 / 19 | — | 0 | 0.0% |
| Manus | 0 / 786 | 0.0% | 0 | 0.0% |
| Visual Studio Code | 0 / 432 | 0.0% | 0 | 0.0% |
| codex-mcp-client | 0 / 1,200 | 0.0% | 0 | 0.0% |
| connectors-manager | 0 / 88 | — | 0 | 0.0% |
| cursor-vscode | 0 / 6,818 | 0.0% | 0 | 0.0% |
| llama-ui-mcp | 0 / 2,699 | 0.0% | 0 | 0.0% |
| lmstudio-mcp-server-session | 0 / 851 | 0.0% | 0 | 0.0% |
| mcp (via mcp-remote) | 0 / 83,936 | 0.0% | 0 | 0.0% |

Source coverage limits: One padded partition each side; late ingestion beyond this is not covered. Calendar-completed days only. Source snapshot does not prove complete traffic.

Files: daily.csv (counts + rates), summary.csv (all identities), stats.json (including contribution membership/counts), provenance.json, two chart sets in PNG/SVG/PDF.

Reproduce (fresh output path required):
```bash
.venv/bin/python scripts/plot_client_migration_focus.py --source summaries/client-protocol-20260915-v1 --output NEW_DIRECTORY --allowlist-manifest /home/evalstate/source/data-analysis/summaries/manifest.json
```
