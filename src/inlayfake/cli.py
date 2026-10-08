"""Command-line entry point: ``python -m inlayfake <stage> --config configs/x.yaml``."""

from __future__ import annotations

import argparse

# Stage name -> one-line description (ARCHITECTURE.md §3). Implementations are added stage by stage.
STAGES = {
    "sample": "select COCO val2017 sources and assign splits",
    "audit": "flag candidate missing labels on C0 for manual review",
    "plan": "choose target category and insertion boxes (shared by C2-C5)",
    "generate": "produce C1-C5 images with the encoding policy",
    "verify": "verify insertions (OWLv2), SAM masks, actual-change masks, write ledger",
    "detect": "run the four detector families",
    "forensics": "run hand-crafted and learned forensic methods",
    "compose": "pair detections with origin scores; abstention",
    "postproc": "derive robustness variants of test images",
    "evaluate": "compute metrics with grouped bootstrap CIs",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inlayfake", description=__doc__)
    sub = parser.add_subparsers(dest="stage", metavar="stage")
    for name, desc in STAGES.items():
        p = sub.add_parser(name, help=desc)
        p.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    raise NotImplementedError(f"stage {args.stage!r} is not implemented yet (see ARCHITECTURE.md)")
