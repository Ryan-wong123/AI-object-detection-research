"""Command-line entry point: ``python -m inlayfake <stage> --config configs/x.yaml``."""

from __future__ import annotations

import argparse
import importlib
import json

from inlayfake.config import load_config

# Stage name -> one-line description (ARCHITECTURE.md §3). Implemented stages live in
# inlayfake.stages.<name> and expose run(cfg, **kwargs) -> dict.
STAGES = {
    "sample": "select COCO val2017 sources and assign splits",
    "plan": "choose target category and insertion boxes (shared by C2-C5)",
    "generate": "produce C0-C5 images with the encoding policy",
    "preview": "write an HTML contact sheet of generated conditions",
    "audit": "flag candidate missing labels on C0 for manual review",
    "verify": "verify insertions (OWLv2), SAM masks, actual-change masks, write ledger",
    "insert": "generate + verify in rounds until every edit passed or ran out of attempts",
    "detect": "run the four detector families",
    "forensics": "run hand-crafted and learned forensic methods",
    "compose": "pair detections with origin scores; abstention",
    "postproc": "derive robustness variants of test images",
    "evaluate": "compute metrics with grouped bootstrap CIs",
    "report": "per-image table of the photos used and what each detector found",
    "viewer": "web page comparing COCO labels with every detector's boxes",
}
IMPLEMENTED = {
    "sample",
    "plan",
    "generate",
    "preview",
    "detect",
    "forensics",
    "audit",
    "evaluate",
    "report",
    "viewer",
    "verify",
    "insert",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inlayfake", description=__doc__)
    sub = parser.add_subparsers(dest="stage", metavar="stage")
    for name, desc in STAGES.items():
        p = sub.add_parser(name, help=desc + ("" if name in IMPLEMENTED else " [not yet]"))
        p.add_argument("--config", default="configs/default.yaml")
        if name in ("generate", "preview", "insert"):
            p.add_argument("--limit", type=int, default=None, help="only the first N sources")
        if name == "preview":
            p.add_argument(
                "--all-attempts",
                action="store_true",
                help="also show rejected and unchecked attempts (default: benchmark images only)",
            )
        if name == "viewer":
            p.add_argument(
                "--share",
                action="store_true",
                help="self-contained copy in docs/results/<week folder>/ to commit or share",
            )
        if name == "audit":
            p.add_argument(
                "--review",
                action="store_true",
                help="only build <work>/audit_review.html from the current decisions",
            )
        if name == "preview":
            p.add_argument(
                "--share",
                action="store_true",
                help="self-contained copy of all photos in docs/results/<week folder>/ to share",
            )
    args = parser.parse_args(argv)
    if args.stage is None:
        parser.print_help()
        return 0
    if args.stage not in IMPLEMENTED:
        raise SystemExit(f"stage {args.stage!r} is not implemented yet (see ARCHITECTURE.md)")
    cfg = load_config(args.config)
    module = importlib.import_module(f"inlayfake.stages.{args.stage}")
    kwargs = {"limit": args.limit} if getattr(args, "limit", None) is not None else {}
    if getattr(args, "share", False):
        kwargs["share"] = True
    if getattr(args, "review", False):
        kwargs["review"] = True
    if getattr(args, "all_attempts", False):
        kwargs["all_attempts"] = True
    result = module.run(cfg, **kwargs)
    print(json.dumps(result, indent=2))
    return 0
