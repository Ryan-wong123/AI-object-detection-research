"""Stage `generate`: produce C0–C5 for every planned source (ARCHITECTURE.md §3.4).

Outputs (under <work>):
  images/<image_id>.jpg          every image a model will see (encoding policy)
  masks/<image_id>_intended.png  intended edit mask (C2, C4, C5) / pasted mask (C3)
  records/<image_id>.json        one provisional ImageRecord per image
  ledger_generated.jsonl         all records, consolidated at the end

Records are *provisional*: inserted boxes are the planned boxes until `verify` replaces them with
verifier/SAM boxes and actual changed-region masks. The stage is resumable: an image whose record
already exists is skipped. Generators run one at a time (generator-major loop) so that only one
model occupies the 8 GB GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from inlayfake.edit.compose import local_edit
from inlayfake.edit.encode import load_rgb, save_policy, sha256_file
from inlayfake.edit.generators import build_inpainter
from inlayfake.edit.paste import choose_donor, paste_instance
from inlayfake.edit.placement import box_area, intersection
from inlayfake.ledger import EditInfo, ImageRecord, Instance, Origin
from inlayfake.runlog import write_run_info
from inlayfake.stages.common import load_coco, read_jsonl, work_dir
from inlayfake.stages.replan import (
    FREE,
    OCC,
    effective_plan,
    ensure_versions,
    job_seed,
    next_job,
    track_version,
)


def _original_instances(index, source_id: int) -> list[Instance]:
    return [
        Instance(
            instance_id=f"coco:{a['id']}",
            category=index.cat_name[a["category_id"]],
            bbox=[float(v) for v in a["bbox"]],
            origin=Origin.PHOTOGRAPHED,
            iscrowd=bool(a.get("iscrowd", 0)),
        )
        for a in index.image_anns(source_id)
    ]


def _save_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)
    return path.as_posix()


def _write_record(rec: ImageRecord, wd: Path) -> None:
    p = wd / "records" / f"{rec.image_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec.to_dict(), indent=1), encoding="utf-8")


def _exists(wd: Path, image_id: str) -> bool:
    return (wd / "records" / f"{image_id}.json").exists()


def _gen_enabled(cfg: dict) -> list[str]:
    """Generators to run. An explicit empty list means none (real-only runs)."""
    enabled = cfg.get("generators_enabled")
    return list(cfg["generators"].keys() if enabled is None else enabled)


def _non_generative(cfg, index, plan, src, wd, enc) -> int:
    """C0, C1 and C3 for one source. Returns number of new images."""
    sid = plan["source_id"]
    split = plan["split"]
    W, H = src["width"], src["height"]
    orig_path = index.image_path(sid)
    base = _original_instances(index, sid)
    n = 0

    rid = ImageRecord.make_id(sid, "C0", None, None)
    if not _exists(wd, rid):
        _write_record(
            ImageRecord(
                rid,
                sid,
                split,
                "C0",
                orig_path.as_posix(),
                W,
                H,
                sha256=sha256_file(orig_path),
                instances=base,
            ),
            wd,
        )
        n += 1

    img = load_rgb(orig_path)
    rid = ImageRecord.make_id(sid, "C1", None, None)
    if not _exists(wd, rid):
        path = wd / "images" / f"{rid}.jpg"
        sha = save_policy(img, path, enc)
        _write_record(
            ImageRecord(rid, sid, split, "C1", path.as_posix(), W, H, sha256=sha, instances=base),
            wd,
        )
        n += 1

    if "C3" in cfg["conditions"] and plan["box_free"] is not None:
        version = plan.get("free_version", 0)  # C3 follows the C4 box (re-plan on failure)
        seed = job_seed(cfg, sid, "C3", None, 0, version)
        rid = ImageRecord.make_id(sid, "C3", None, seed)
        if not _exists(wd, rid):
            rng = np.random.default_rng(seed)
            donor = choose_donor(index, plan["category"], sid, plan["box_free"], rng)
            if donor is not None:
                d_img = load_rgb(index.image_path(donor["image_id"]))
                d_mask = index.ann_mask(donor)
                out, pmask, pbox = paste_instance(
                    img, d_img, d_mask, donor["bbox"], plan["box_free"], cfg["edit"]["feather_px"]
                )
                path = wd / "images" / f"{rid}.jpg"
                sha = save_policy(out, path, enc)
                mpath = _save_mask(pmask, wd / "masks" / f"{rid}_intended.png")
                inst = base + [Instance(f"paste:{rid}", plan["category"], pbox, Origin.REAL_PASTED)]
                edit = EditInfo(
                    target_category=plan["category"],
                    seed=seed,
                    plan_box=plan["box_free"],
                    intended_mask_path=mpath,
                    donor={"image_id": donor["image_id"], "ann_id": donor["id"]},
                    plan_version=version,
                )
                _write_record(
                    ImageRecord(
                        rid,
                        sid,
                        split,
                        "C3",
                        path.as_posix(),
                        W,
                        H,
                        sha256=sha,
                        edit=edit,
                        instances=inst,
                    ),
                    wd,
                )
                n += 1
    return n


def _generative_jobs(cfg: dict, plan: dict, gen: str) -> list[tuple[str, list[float]]]:
    """(condition, box) pairs that generator ``gen`` must produce for this plan."""
    gcfg = cfg["generators"][gen]
    if plan["split"] not in gcfg.get("splits", ["train", "val", "test"]):
        return []
    jobs = []
    roles = cfg["generation"]
    free = plan["box_free"] is not None  # None: no box could be re-planned
    if "C2" in cfg["conditions"] and gen in roles["sham_generators"] and free:
        jobs.append(("C2", plan["box_free"]))
    if "C4" in cfg["conditions"] and gen in roles["insert_generators"] and free:
        jobs.append(("C4", plan["box_free"]))
    if "C5" in cfg["conditions"] and gen in roles["occlusion_generators"] and plan["box_occ"]:
        jobs.append(("C5", plan["box_occ"]))
    return jobs


def _generative_one(
    cfg, index, plan, src, gen, cond, box, inpainter, wd, enc, state, versions
) -> bool:
    sid, split = plan["source_id"], plan["split"]
    nxt = next_job(cfg, state, sid, cond, gen, versions)
    if nxt is None:  # already verified as passed, or out of attempts and re-plans
        return False
    version, attempt = nxt
    seed = job_seed(cfg, sid, cond, gen, attempt, version)
    rid = ImageRecord.make_id(sid, cond, gen, seed)
    if _exists(wd, rid):
        return False
    ecfg = cfg["edit"]
    cat = plan.get("category_occ", plan["category"]) if cond == "C5" else plan["category"]
    if cond == "C2":
        prompt, negative = ecfg["sham_prompt"], f"{cat}, object, " + ecfg["negative_prompt"]
    else:
        prompt, negative = ecfg["prompt"].format(category=cat), ecfg["negative_prompt"]
    img = load_rgb(index.image_path(sid))
    out, intended, win = local_edit(
        img,
        box,
        inpainter,
        inpainter.resolution,
        prompt,
        negative,
        seed,
        ecfg["window_scale"],
        ecfg["mask_dilate_px"],
        ecfg["feather_px"],
        ecfg.get("composite", "mask"),
        ecfg.get("window_min_px", 256),
    )
    path = wd / "images" / f"{rid}.jpg"
    sha = save_policy(out, path, enc)
    mpath = _save_mask(intended, wd / "masks" / f"{rid}_intended.png")

    inst = _original_instances(index, sid)
    if cond in ("C4", "C5"):
        inst.append(Instance(f"ins:{rid}", cat, [float(v) for v in box], Origin.AI_INSERTED))
    if cond == "C5":  # provisional visibility from boxes; `verify` recomputes it from masks
        occ_id = f"coco:{plan['occluded_ann_id']}"
        for i in inst:
            if i.instance_id == occ_id:
                i.visible_fraction = max(0.0, 1 - intersection(box, i.bbox) / box_area(i.bbox))
    gcfg = cfg["generators"][gen]
    edit = EditInfo(
        target_category=cat,
        seed=seed,
        plan_box=[float(v) for v in box],
        prompt=prompt,
        negative_prompt=negative,
        model_id=gcfg["model_id"],
        model_revision=gcfg.get("revision"),
        steps=gcfg.get("steps", ecfg["steps"]),
        guidance=gcfg.get("guidance", ecfg["guidance"]),
        crop_box=[float(v) for v in win],
        intended_mask_path=mpath,
        attempts=attempt + 1,
        plan_version=version,
    )
    _write_record(
        ImageRecord(
            rid,
            sid,
            split,
            cond,
            path.as_posix(),
            src["width"],
            src["height"],
            generator=gen,
            sha256=sha,
            edit=edit,
            instances=inst,
        ),
        wd,
    )
    return True


def _pending(cfg, plan, cond, gen, state, versions, wd: Path) -> bool:
    """True if this job still needs an image (first attempt, a retry after a failed verify, or
    the first attempt on a re-planned box)."""
    sid = plan["source_id"]
    nxt = next_job(cfg, state, sid, cond, gen, versions)
    if nxt is None:
        return False
    version, attempt = nxt
    seed = job_seed(cfg, sid, cond, gen, attempt, version)
    return not _exists(wd, ImageRecord.make_id(sid, cond, gen, seed))


def consolidate(wd: Path) -> int:
    rows = sorted((wd / "records").glob("*.json"))
    with (wd / "ledger_generated.jsonl").open("w", encoding="utf-8") as f:
        for p in rows:
            f.write(json.dumps(json.loads(p.read_text(encoding="utf-8"))) + "\n")
    return len(rows)


def run(cfg: dict, limit: int | None = None) -> dict:
    index = load_coco(cfg)
    wd = work_dir(cfg)
    enc = cfg["encoding"]
    sources = {s["source_id"]: s for s in read_jsonl(wd / "sources.jsonl")}
    from inlayfake.stages.verify import verify_state

    state = verify_state(wd)
    raw = read_jsonl(wd / "plans.jsonl")[:limit]
    versions = ensure_versions(cfg, index, sources, raw, state, wd)
    plans = [
        effective_plan(
            p,
            versions,
            track_version(cfg, state, p["source_id"], FREE, versions),
            track_version(cfg, state, p["source_id"], OCC, versions),
        )
        for p in raw
    ]
    counts: dict[str, int] = {"non_generative": 0}

    for plan in tqdm(plans, desc="C0/C1/C3"):
        counts["non_generative"] += _non_generative(
            cfg, index, plan, sources[plan["source_id"]], wd, enc
        )

    for gen in _gen_enabled(cfg):
        jobs = [
            (p, c, b)
            for p in plans
            for c, b in _generative_jobs(cfg, p, gen)
            if _pending(cfg, p, c, gen, state, versions, wd)
        ]
        if not jobs:  # nothing to (re)generate: do not load the model
            counts[gen] = 0
            continue
        inpainter = build_inpainter(cfg["generators"][gen], cfg["edit"], cfg["hardware"])
        done = 0
        for plan, cond, box in tqdm(jobs, desc=f"{gen}"):
            done += _generative_one(
                cfg,
                index,
                plan,
                sources[plan["source_id"]],
                gen,
                cond,
                box,
                inpainter,
                wd,
                enc,
                state,
                versions,
            )
        counts[gen] = done
        del inpainter
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    counts["ledger_records"] = consolidate(wd)
    write_run_info(wd / "runs", "generate", cfg, counts)
    return counts
