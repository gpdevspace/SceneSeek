# SceneSeek

Natural-language scene retrieval over movie stills, built to test one claim:

> A plain image/text embedding treats "two people at a table in warm light" as
> interchangeable, so **who is on screen must be a hard filter and appearance only a
> soft ranker.**

The deliverable is not the demo — it is the measured comparison of `filter_rank`
against an `appearance_only` baseline on a hand-labelled query set (Phase 5).

Corpus: three films (The Godfather 1972, Inglourious Basterds 2009, Rocky IV 1985),
~60+ web stills each. No video files, no shot segmentation, no dialogue channel.

## Setup

```bash
uv venv --python 3.12
uv sync
uv run python scripts/spike_env.py    # Phase 0 gate -- must pass before Phase 1
```

## Models

| Role | Model | Size | Notes |
|---|---|---|---|
| Face detect | RetinaFace `det_10g` | in `buffalo_l` pack | via `insightface` |
| Face recognize | ArcFace `w600k_r50` | ~330 MB pack total | 512-d, L2-normalized |
| Visual encoder | `google/siglip2-base-patch16-384` | ~1.5 GB | fp32 on MPS |

Face weights auto-download to `~/.insightface/models/`; SigLIP to the HF cache.
Fallbacks for both are listed in [config.py](src/sceneseek/config.py).

## Layout

```
src/sceneseek/
  config.py     paths, model ids, thresholds -- single source of truth
  films.py      film registry: slug, tmdb id, cast, character aliases
  acquire/      tmdb.py  supplement.py  dedupe.py
  enrich/       faces.py  cluster.py  review.py  gallery.py  visual.py  build.py
  retrieve/     parse.py  search.py
  evaluate/     queries.yaml  run.py
  web/          app.py  static/index.html
scripts/spike_env.py    Phase 0 gate
data/                   gitignored; sources.jsonl is the reproducible artifact
```

## Status

- [x] **Phase 0** — skeleton + both models proven on MPS
- [ ] Phase 1 — corpus (>=60 stills/film with provenance)
- [ ] Phase 2 — face clusters -> labelled gallery + calibrated threshold
- [ ] Phase 3 — SigLIP vectors + SQLite index
- [ ] Phase 4 — query parser, three search modes, CLI
- [ ] Phase 5 — 30-query labelled eval + report
- [ ] Phase 6 — FastAPI side-by-side web demo

See `plans/sceneseek-prototype-plan.md` (gitignored) for the full plan.
