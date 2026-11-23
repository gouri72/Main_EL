# HumanFirst-Route — final research handoff

**This creates an evidence-backed review bundle, not a publisher-ready submission.**
Results derive from simulations and rule-based model evaluation. The final
submission requires the team's authorship approval and venue-specific formatting.

## Build the review package (Windows PowerShell)

```powershell
python -m pytest -q tests/
python -m paper.check_reported_results
python data/scripts/freeze_manifest.py --verify
python tools/build_research_handoff.py --out artifacts/research_handoff.zip --p1-archive artifacts/p1_v2_evidence.zip
Get-FileHash artifacts/research_handoff.zip -Algorithm SHA256
```

The bundled ZIP contains the manuscript, evidence inventory, policy and
cross-family result summaries, P3 audit, P4 recorded results and existing
figures. The optional nested original P1 ZIP is validated against its own
manifest; missing original data are never fabricated. Existing ZIP files
are never overwritten. The local `artifacts/` directory is gitignored.

## Remaining non-automatable or external work

- Team confirmation of author details/order, approval, venue requirements.
- Check of P4 raw paired outputs and trained LoRA adapter on team Drive.
- Genuine P3 independent human annotations and P4 blinded hint ratings, or
  disclosure that those studies were not conducted.
- Independent training seeds for P2 cross-family claims, or careful limitation
  of the current single-model results.
- Final figure/source licenses, reference review and publisher formatting.

Do not interpret the completed code and paper draft as evidence of real
student learning improvements.
