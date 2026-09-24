"""Stage-major execution with fingerprint-based caching.

Stage-major means: run stage 1 on every source, then stage 2 on every source,
and so on. Each model is loaded once per run instead of once per source, and
only one stage's models sit in GPU memory at a time.

To scale out, run one process per GPU or machine over disjoint shards
(`--shard i/N`) sharing a work_dir, then run `soundakira build` once.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

from soundakira.components.base import ComponentContext
from soundakira.config import PipelineConfig
from soundakira.pipeline.stages import STAGE_NAMES, STAGES, Stage
from soundakira.pipeline.workspace import SourceWorkspace, now_iso
from soundakira.sources.resolve import Source
from soundakira.utils.device import resolve_device
from soundakira.utils.hashing import stable_hash

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    counts: dict[str, Counter] = field(default_factory=dict)

    def add(self, stage: str, outcome: str) -> None:
        self.counts.setdefault(stage, Counter())[outcome] += 1

    @property
    def failed(self) -> int:
        return sum(c["failed"] for c in self.counts.values())


def downstream_of(stages: Iterable[str]) -> set[str]:
    """The given stages plus everything that (transitively) depends on them."""
    out = set(stages)
    changed = True
    while changed:
        changed = False
        for s in STAGES:
            if s.name not in out and any(r in out for r in s.requires):
                out.add(s.name)
                changed = True
    return out


class Runner:
    def __init__(self, cfg: PipelineConfig, ctx: ComponentContext | None = None):
        self.cfg = cfg
        self.ctx = ctx or ComponentContext(
            device=resolve_device(cfg.device), hf_token=cfg.resolved_hf_token()
        )

    def prepare(self, sources: Iterable[Source]) -> list[SourceWorkspace]:
        return [SourceWorkspace(self.cfg.work_dir, s) for s in sources]

    @staticmethod
    def fingerprint(ws: SourceWorkspace, stage: Stage, identity: dict) -> str | None:
        upstream = {}
        for req in stage.requires:
            rec = ws.stage_record(req)
            if not rec or rec.get("status") != "done":
                return None
            upstream[req] = rec["fingerprint"]
        return stable_hash({"stage": stage.name, "version": stage.version,
                            "identity": identity, "upstream": upstream})

    def run(
        self,
        workspaces: list[SourceWorkspace],
        until: str | None = None,
        force: Iterable[str] = (),
    ) -> RunSummary:
        if until is not None and until not in STAGE_NAMES:
            raise ValueError(f"unknown stage {until!r}; stages: {STAGE_NAMES}")
        forced = downstream_of(force)
        if forced:
            for ws in workspaces:
                ws.invalidate(sorted(forced))

        summary = RunSummary()
        stage_classes = STAGES[: STAGE_NAMES.index(until) + 1] if until else STAGES
        for stage_cls in stage_classes:
            stage = stage_cls(self.cfg, self.ctx)
            identity = stage.identity()
            pending: list[tuple[SourceWorkspace, str]] = []
            for ws in workspaces:
                fp = self.fingerprint(ws, stage, identity)
                if fp is None:
                    summary.add(stage.name, "blocked")
                    continue
                rec = ws.stage_record(stage.name)
                if (rec and rec.get("status") == "done" and rec.get("fingerprint") == fp
                        and all(p.exists() for p in stage.outputs(ws))):
                    summary.add(stage.name, "cached")
                    continue
                pending.append((ws, fp))
            if not pending:
                continue

            log.info("stage %-10s %d source(s) to process", stage.name, len(pending))
            stage.setup()
            try:
                bar = tqdm(total=len(pending), desc=stage.name, unit="src", leave=False)
                if stage.workers > 1:
                    with ThreadPoolExecutor(stage.workers) as pool:
                        futures = [pool.submit(self._run_one, stage, ws, fp) for ws, fp in pending]
                        for fut in as_completed(futures):
                            summary.add(stage.name, fut.result())
                            bar.update()
                else:
                    for ws, fp in pending:
                        summary.add(stage.name, self._run_one(stage, ws, fp))
                        bar.update()
                bar.close()
            finally:
                stage.teardown()
        return summary

    @staticmethod
    def _run_one(stage: Stage, ws: SourceWorkspace, fp: str) -> str:
        t0 = time.monotonic()
        try:
            stats = stage.run(ws) or {}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error("[%s] stage %s failed: %s", ws.source_id, stage.name, e)
            log.debug("traceback", exc_info=True)
            ws.update_stage(stage.name, {
                "status": "failed", "fingerprint": fp, "finished_at": now_iso(),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=8)[-4000:],
            })
            return "failed"
        ws.update_stage(stage.name, {
            "status": "done", "fingerprint": fp, "finished_at": now_iso(),
            "elapsed_s": round(time.monotonic() - t0, 2), "stats": stats,
        })
        return "done"
