#!/usr/bin/env python3
"""Verify that a RetinaScope deployment is serving the real trained model.

Point it at any running instance -- local, container, or the public URL:

    python deploy/smoke_test.py                              # localhost:7860
    python deploy/smoke_test.py https://you-retinascope.hf.space

It uploads the 12 labelled verification images to the live service and scores
what comes back against their known grades, so "the deployment works" is a
measurement rather than an impression. Standard library only: it has to run on
a reviewer's machine with nothing installed.

Exit code 0 = every check passed.
"""
from __future__ import annotations

import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# Ground truth for outputs/verification_set/images/, from the filenames these
# were generated under (grade0_case1 -> image1, ... grade4_case2 -> image12).
TRUTH = {"image1": 0, "image2": 0, "image3": 0, "image4": 1, "image5": 1, "image6": 1,
         "image7": 2, "image8": 2, "image9": 3, "image10": 3, "image11": 4, "image12": 4}
REFERABLE_AT = 2

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "outputs" / "verification_set" / "images"

OK, BAD, DIM = "\033[32m", "\033[31m", "\033[2m"
RESET, BOLD = "\033[0m", "\033[1m"


def get(base: str, path: str, timeout: int = 180):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def post_image(base: str, img: Path, timeout: int = 300) -> dict:
    """Multipart upload without the requests library."""
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(img.name)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{img.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        img.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        base + "/screen?explain=true", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860").rstrip("/")
    print(f"{BOLD}RetinaScope deployment check{RESET}  ->  {base}\n")
    failures: list[str] = []

    # -- 1. is the real model answering? ------------------------------------
    try:
        h = get(base, "/health", timeout=120)
    except urllib.error.URLError as e:
        print(f"{BAD}FAIL{RESET} cannot reach {base}/health: {e}")
        print(f"{DIM}      If this is a free-tier host the container may be "
              f"asleep; open the URL in a browser first.{RESET}")
        return 1

    print(f"  warm={h.get('warm')}  device={h.get('device')}  "
          f"version={h.get('model_version')}")
    if h.get("trained_weights") and h.get("grader_loaded") and h.get("segmentation_loaded"):
        print(f"  {OK}PASS{RESET} trained weights loaded (not the rule-based fallback)")
    else:
        # This is the failure that matters most: the portal looks identical
        # either way, but the grades stop being the validated model's.
        print(f"  {BAD}FAIL{RESET} trained weights NOT loaded -- "
              f"grades would come from the rule-based fallback. Check that "
              f"outputs/artifacts/ made it into the image.")
        failures.append("weights")

    # -- 2. sample pack reachable -------------------------------------------
    try:
        s = get(base, "/samples", timeout=60)
        print(f"  {OK}PASS{RESET} /samples serves {s.get('count')} test images")
    except Exception as e:
        print(f"  {BAD}FAIL{RESET} /samples unreachable: {e}")
        failures.append("samples")

    if not IMAGES.is_dir():
        print(f"\n{BAD}FAIL{RESET} {IMAGES} not found -- run this from a repo clone.")
        return 1

    # -- 3. score the labelled set through the live service -----------------
    print(f"\n{BOLD}Screening 12 labelled images through the live service{RESET}")
    print(f"  {'image':<10} {'true':>4} {'pred':>4}  {'match':<9} {'P(refer)':>8} "
          f"{'conf':>5}  decision")
    exact = within1 = refer_ok = 0
    results = {}
    for name, truth in TRUTH.items():
        img = IMAGES / f"{name}.jpg"
        if not img.exists():
            print(f"  {BAD}missing{RESET} {img}")
            failures.append(f"missing {name}")
            continue
        try:
            d = post_image(base, img)
        except Exception as e:
            print(f"  {BAD}FAIL{RESET} {name}: {e}")
            failures.append(name)
            continue
        g = d.get("grade", -1)
        results[name] = g
        exact += g == truth
        within1 += abs(g - truth) <= 1
        # The clinically load-bearing number: did it refer the cases that
        # need an ophthalmologist, and not refer the ones that don't?
        correct_referral = bool(d.get("referable")) == (truth >= REFERABLE_AT)
        refer_ok += correct_referral
        mark = (f"{OK}exact{RESET}" if g == truth else
                (f"{DIM}within 1{RESET}" if abs(g - truth) <= 1 else f"{BAD}off by {abs(g-truth)}{RESET}"))
        flag = "" if correct_referral else f"  {BAD}<- wrong referral{RESET}"
        print(f"  {name:<10} {truth:>4} {g:>4}  {mark:<20} "
              f"{d.get('referable_probability',0):>8.3f} {d.get('confidence',0):>5.2f}  "
              f"{d.get('decision','?')}/{d.get('urgency','?')}{flag}")

    n = len(results)
    if n:
        print(f"\n  exact grade       {exact}/{n}")
        print(f"  within one grade  {within1}/{n}")
        print(f"  referral decision {refer_ok}/{n}   {DIM}(refer iff true grade >= 2){RESET}")
        if refer_ok < n:
            failures.append("referral")

    # -- 4. same image twice must give the same grade -----------------------
    print(f"\n{BOLD}Reproducibility{RESET}")
    probe = IMAGES / "image11.jpg"
    if probe.exists():
        try:
            a, b = post_image(base, probe), post_image(base, probe)
            if a.get("grade") == b.get("grade") and \
               abs(a.get("referable_probability", 0) - b.get("referable_probability", 1)) < 1e-9:
                print(f"  {OK}PASS{RESET} image11 screened twice -> identical grade "
                      f"{a.get('grade')} and identical P(refer)")
            else:
                print(f"  {BAD}FAIL{RESET} same image gave grade {a.get('grade')} then "
                      f"{b.get('grade')} -- MC-dropout seeding is not active")
                failures.append("determinism")
        except Exception as e:
            print(f"  {BAD}FAIL{RESET} {e}")
            failures.append("determinism")

    print()
    if failures:
        print(f"{BAD}{BOLD}FAILED{RESET}: {', '.join(failures)}")
        return 1
    print(f"{OK}{BOLD}All checks passed.{RESET} This deployment is serving the trained model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
