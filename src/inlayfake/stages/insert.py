"""Stage `insert`: generate → verify, repeated until every edit passed or ran out of attempts.

Convenience for the overnight run: failed edits (verifier did not find the object, or a sham edit
produced one) are regenerated with a new seed, up to edit.max_attempts per job. Each round frees
the GPU between the generator and the verifier, and every step is resumable.
"""

from __future__ import annotations

from inlayfake.stages import generate, verify


def run(cfg: dict, limit: int | None = None) -> dict:
    rounds = []
    # every box gets max_attempts seeds; with re-plans (edit.max_replans) a job may use new boxes
    n_rounds = cfg["edit"]["max_attempts"] * (1 + cfg["edit"].get("max_replans", 0))
    for i in range(n_rounds):
        g = generate.run(cfg, limit=limit)
        v = verify.run(cfg)
        rounds.append({"round": i + 1, "generate": g, "verify": v})
        if v["needs_retry"] == 0:
            break
    return {"rounds": len(rounds), "final": rounds[-1]["verify"], "history": rounds}
