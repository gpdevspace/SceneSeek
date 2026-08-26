"""Phase 0 gate: prove both models load and behave before any pipeline code exists.

Downloads four CC-licensed portraits (two actors x two photos) plus one landscape
from Wikimedia Commons into `data/spike/`, then checks four things:

  1. the face backend returns >=1 detection on a known face image
  2. same-actor cosine > different-actor cosine, with a clear margin
  3. the visual encoder actually runs on MPS (not silently CPU) and ranks an
     obvious caption above an obvious wrong one
  4. peak RSS stays inside the memory budget

Exits non-zero if any gate fails. Usage:
    uv run python scripts/spike_env.py [--visual-model ID] [--skip-visual]
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sceneseek import config  # noqa: E402

# Wikimedia Commons fixtures: two photos each of two actors (a genuine pair and an
# impostor pair for the recognition gate), plus a landscape for the text gate.
FIXTURES = {
    "pacino_a": "Al Pacino at the 2011 National Medal of Arts Ceremony.jpg",
    "pacino_b": "Al Pacino at the Academy in 2016.png",
    "waltz_a": "Christoph Waltz 2009.jpg",
    "waltz_b": "Christoph Waltz 2013 Django avp.jpg",
    "snow": "Peyto Lake in winter.jpg",
}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MEMORY_BUDGET_GB = 12.0


def peak_rss_gb() -> float:
    """macOS reports ru_maxrss in bytes; Linux in kilobytes."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024**3 if sys.platform == "darwin" else raw / 1024**2


@contextmanager
def stage(name: str):
    print(f"\n=== {name}")
    t0 = time.perf_counter()
    yield
    print(f"--- {name}: {time.perf_counter() - t0:.1f}s  peak RSS {peak_rss_gb():.2f} GB")


def _get(url: str, attempts: int = 5) -> bytes:
    """Wikimedia rate-limits hard; back off rather than hammering it."""
    parts = urllib.parse.urlsplit(url)
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query)
            if not k.startswith("utm_")]
    url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(kept)))
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT,
                                                   "Accept": "*/*"})
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == attempts - 1:
                raise
            wait = 5 * 2**attempt
            print(f"    HTTP {exc.code}, retrying in {wait}s "
                  f"(attempt {attempt + 2}/{attempts})")
            time.sleep(wait)
    raise SystemExit("unreachable")


def fetch_fixtures() -> dict[str, Path]:
    """Download once, cache forever. Fixtures are gitignored with the rest of data/."""
    config.SPIKE.mkdir(parents=True, exist_ok=True)
    paths = {key: config.SPIKE / f"{key}.jpg" for key in FIXTURES}
    missing = {k: FIXTURES[k] for k, p in paths.items() if not p.exists()}
    if not missing:
        print(f"all {len(paths)} fixtures cached in {config.SPIKE}")
        return paths

    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url", "iiurlwidth": "960",
        "titles": "|".join(f"File:{t}" for t in missing.values()),
    })
    import json
    pages = json.loads(_get(f"{COMMONS_API}?{query}"))["query"]["pages"]
    by_title = {p["title"]: p["imageinfo"][0].get("thumburl") or p["imageinfo"][0]["url"]
                for p in pages.values() if p.get("imageinfo")}

    for key, title in missing.items():
        url = by_title.get(f"File:{title}")
        if not url:
            raise SystemExit(f"Commons has no image info for {title!r}")
        data = _get(url)
        tmp = paths[key].with_suffix(".download")
        tmp.write_bytes(data)
        Image.open(tmp).convert("RGB").save(paths[key], "JPEG", quality=95)
        tmp.unlink()
        print(f"fetched {key:9s} {len(data) / 1024:6.0f} KB  {title}")
        time.sleep(config.SUPPLEMENT_RATE_LIMIT_S)
    return paths


def _area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


def run_faces(paths: dict[str, Path], results: list) -> None:
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available = ort.get_available_providers()
    providers = [p for p in config.ONNX_PROVIDERS if p in available] or ["CPUExecutionProvider"]
    print(f"onnxruntime {ort.__version__}  available={available}\n  using={providers}")

    with stage("face model load"):
        app = FaceAnalysis(name=config.FACE_MODEL, root=str(config.FACE_MODEL_ROOT),
                           allowed_modules=list(config.FACE_MODULES), providers=providers)
        app.prepare(ctx_id=-1, det_thresh=config.FACE_DET_THRESH,
                    det_size=config.FACE_DET_SIZE)

    embeddings: dict[str, np.ndarray] = {}
    with stage("face detect + embed"):
        for key in ("pacino_a", "pacino_b", "waltz_a", "waltz_b"):
            rgb = np.asarray(Image.open(paths[key]).convert("RGB"))
            t0 = time.perf_counter()
            faces = app.get(rgb[:, :, ::-1])  # insightface wants BGR
            dt = time.perf_counter() - t0
            if not faces:
                print(f"  {key:9s} NO FACES  ({dt:.2f}s)")
                continue
            # Pick the LARGEST face, not the highest-scoring one. det_score measures
            # detector confidence, not subject salience: these fixtures contain small
            # background bystanders that outscore the actual subject. The same trap
            # applies to movie stills full of extras -- see Phase 2.
            best = max(faces, key=_area)
            embeddings[key] = best.normed_embedding.astype(np.float32)
            top_det = max(faces, key=lambda f: f.det_score)
            note = "" if top_det is best else f"  (top det_score={top_det.det_score:.3f} is a smaller face)"
            print(f"  {key:9s} {len(faces)} face(s)  chose area={_area(best):.0f} px^2 "
                  f"det_score={best.det_score:.3f}  ({dt:.2f}s){note}")

    results.append(("face backend returns >=1 detection on known faces",
                    len(embeddings) == 4,
                    f"{len(embeddings)}/4 fixtures produced an embedding"))

    if len(embeddings) < 4:
        results.append(("same-actor cosine > different-actor cosine", False,
                        "skipped -- not all fixtures embedded"))
        return

    def cos(a: str, b: str) -> float:
        return float(embeddings[a] @ embeddings[b])

    genuine = {"pacino_a/pacino_b": cos("pacino_a", "pacino_b"),
               "waltz_a/waltz_b": cos("waltz_a", "waltz_b")}
    impostor = {f"{a}/{b}": cos(a, b) for a, b in
                [("pacino_a", "waltz_a"), ("pacino_a", "waltz_b"),
                 ("pacino_b", "waltz_a"), ("pacino_b", "waltz_b")]}

    print("\n  genuine (same actor):")
    for k, v in genuine.items():
        print(f"    {k:24s} {v:+.4f}")
    print("  impostor (different actors):")
    for k, v in impostor.items():
        print(f"    {k:24s} {v:+.4f}")
    margin = min(genuine.values()) - max(impostor.values())
    print(f"  separation margin: {margin:+.4f}")

    results.append(("same-actor cosine > different-actor cosine", margin > 0.10,
                    f"worst genuine {min(genuine.values()):.4f} vs "
                    f"best impostor {max(impostor.values()):.4f}, margin {margin:+.4f}"))


def _pooled(out):
    """transformers 5.x returns the full vision/text model output from
    get_image_features / get_text_features, not a pooled tensor (the docstring
    still shows the old contract). Accept either shape."""
    return out if hasattr(out, "device") else out.pooler_output


def run_visual(paths: dict[str, Path], model_id: str, results: list) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.backends.mps.is_available():
        results.append(("visual encoder runs on MPS", False, "MPS unavailable"))
        return

    dtype = getattr(torch, config.VISUAL_DTYPE)
    with stage(f"visual model load ({model_id})"):
        model = AutoModel.from_pretrained(model_id, dtype=dtype).to(config.DEVICE).eval()
        processor = AutoProcessor.from_pretrained(model_id)

    param_device = next(model.parameters()).device.type
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params {n_params / 1e6:.0f}M  dtype {next(model.parameters()).dtype}  device {param_device}")

    # Two images with unambiguous captions, plus a decoy caption for each.
    image_keys = ["snow", "pacino_a"]
    captions = ["a snowy mountain lake in winter",
                "a portrait of a man in a suit at a formal ceremony"]

    with stage("visual encode (2 images + 2 captions)"):
        images = [Image.open(paths[k]).convert("RGB") for k in image_keys]
        img_in = processor(images=images, return_tensors="pt").to(config.DEVICE)
        txt_in = processor(text=captions, padding="max_length",
                           max_length=config.TEXT_MAX_LENGTH, truncation=True,
                           return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            img_emb = _pooled(model.get_image_features(**img_in))
            txt_emb = _pooled(model.get_text_features(**txt_in))
        out_device = img_emb.device.type
        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)
        sim = (img_emb @ txt_emb.T).float().cpu().numpy()

    print(f"  output device: {out_device}   embedding dim: {img_emb.shape[-1]}")
    print(f"\n  {'image':10s} " + "  ".join(f"{c[:26]:>28s}" for c in captions))
    for i, key in enumerate(image_keys):
        print(f"  {key:10s} " + "  ".join(f"{sim[i, j]:+28.4f}" for j in range(len(captions))))

    on_mps = param_device == "mps" and out_device == "mps"
    results.append(("visual encoder runs on MPS (not silently CPU)", on_mps,
                    f"params on {param_device}, output on {out_device}"))

    correct = bool(sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0])
    results.append(("text->image cosine ranks the right caption first", correct,
                    f"snow: {sim[0, 0]:+.4f} vs {sim[0, 1]:+.4f} | "
                    f"portrait: {sim[1, 1]:+.4f} vs {sim[1, 0]:+.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-model", default=config.VISUAL_MODEL)
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--skip-faces", action="store_true")
    args = ap.parse_args()

    results: list[tuple[str, bool, str]] = []
    with stage("fetch fixtures"):
        paths = fetch_fixtures()

    if not args.skip_faces:
        run_faces(paths, results)
    if not args.skip_visual:
        run_visual(paths, args.visual_model, results)

    peak = peak_rss_gb()
    results.append((f"peak RSS within {MEMORY_BUDGET_GB:.0f} GB budget",
                    peak < MEMORY_BUDGET_GB, f"{peak:.2f} GB"))

    print("\n" + "=" * 78)
    print("PHASE 0 GATE")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}\n          {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print("=" * 78)
    print(f"{len(results) - len(failed)}/{len(results)} gates passed"
          + (f" -- FAILED: {', '.join(failed)}" if failed else " -- clear to proceed to Phase 1"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
