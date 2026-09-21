#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Push RetinaScope to a Hugging Face Space (free CPU tier, no credit card).
#
#   export HF_TOKEN=hf_xxxxxxxxxxxx          # Settings -> Access Tokens (write)
#   ./deploy/deploy_hf_space.sh yourname/retinascope
#
# Builds a clean staging copy rather than pushing this repo: the project's git
# history is ~130 MB of superseded checkpoints and report images, none of which
# the running service needs, and Spaces would store every byte of it.
# ---------------------------------------------------------------------------
set -euo pipefail

SPACE="${1:-}"
if [[ -z "$SPACE" || "$SPACE" != */* ]]; then
  echo "usage: $0 <hf-username>/<space-name>" >&2
  echo "   e.g. $0 yashchoudhary/retinascope" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN is not set." >&2
  echo "  Create one at https://huggingface.co/settings/tokens (role: write), then:" >&2
  echo "    export HF_TOKEN=hf_xxxxxxxxxxxx" >&2
  exit 2
fi
# The weights are 18 MB each; Spaces rejects non-LFS files over 10 MB.
if ! git lfs version >/dev/null 2>&1; then
  echo "error: git-lfs is not installed (required: the model weights are 18 MB each)." >&2
  echo "  macOS: brew install git-lfs     Ubuntu: sudo apt install git-lfs" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> staging deployment files"
cd "$ROOT"
for p in Dockerfile deploy/requirements-serve.txt src web \
         outputs/artifacts outputs/validation \
         outputs/verification_set/images outputs/verification_set/EXPECTED_GRADES.md; do
  [[ -e "$p" ]] || { echo "error: missing $p" >&2; exit 1; }
  mkdir -p "$STAGE/$(dirname "$p")"
  cp -R "$p" "$STAGE/$(dirname "$p")/"
done
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true

[[ -f "$STAGE/outputs/artifacts/grader.pt" ]] || { echo "error: grader.pt missing from stage" >&2; exit 1; }

# The Space's own README doubles as its config; without this front matter HF
# does not know to build the Dockerfile.
cat > "$STAGE/README.md" <<'MD'
---
title: RetinaScope DR Screening
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Explainable diabetic retinopathy screening — upload a fundus image
---

# RetinaScope — diabetic retinopathy screening

Upload a retinal (fundus) photograph and the service returns an ICDR grade
(0–4), a referral decision, and a Grad-CAM++ heatmap showing the regions the
model actually used.

**No fundus images to hand?** The portal has a *Download the 12 test images*
button. The zip includes `EXPECTED_GRADES.md` listing the grade each image
should receive, so the model can be checked rather than taken on trust.

Every result is computed on this container by running the uploaded image
through the trained pipeline. Nothing is pre-recorded.

Not a medical device. Research and education use only.
MD

echo "==> initialising Space repo"
cd "$STAGE"
git init -q -b main
git lfs install --local >/dev/null
git lfs track "*.pt" >/dev/null
git add .gitattributes
git add -A
git -c user.email=deploy@local -c user.name=deploy commit -qm "Deploy RetinaScope screening portal"

echo "==> pushing to https://huggingface.co/spaces/$SPACE"
git remote add origin "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}"
if ! git push -q --force origin main; then
  echo >&2
  echo "push failed. The most common cause is that the Space does not exist yet." >&2
  echo "Create it at https://huggingface.co/new-space" >&2
  echo "  Space name: ${SPACE#*/}    SDK: Docker -> Blank    Hardware: CPU basic (free)" >&2
  echo "Then re-run this script." >&2
  exit 1
fi

echo
echo "Done. The Space is building (first build takes ~5-10 minutes)."
echo "  Portal:  https://huggingface.co/spaces/${SPACE}"
echo "  Direct:  https://${SPACE/\//-}.hf.space"
echo
echo "When the build finishes, verify it is serving the real model:"
echo "  python deploy/smoke_test.py https://${SPACE/\//-}.hf.space"
