# InlayFake

Research code for **InlayFake: A Benchmark for Mixed-Origin Object Understanding**: can detectors and
image-forensic methods tell, *per object*, whether an object in a real photograph was photographed or
inserted by an AI image editor?

- `PRD.md`: goals, hypotheses, scope, timeline
- `ARCHITECTURE.md`: pipeline and data contracts
- `paper/PAPER.md`: living paper draft
- `docs/RESEARCH_LOG.md`: experiment log
- `AGENTS.md` / `CLAUDE.md`: instructions for AI coding agents

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate                              # Linux / WSL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
pytest -q
python -m inlayfake --help
```
