# DR-screening verification set (12 phantoms)

Raw synthetic fundus images run through the **full serve pipeline** (`DRScreeningPipeline.run`) — the same path `scripts/run_demo.py` uses.
Each case has a raw image (`images/`), plus a heatmap panel + HTML report + JSON (`reports/`).
All images are seed-reproducible: `from drscreen.data.synthetic import generate`, then `generate(grade=G, seed=SEED, severity=SEV, camera=CAM, size=512)`.

| case | true grade | predicted | match | conf | P(refer) | decision/urgency | seed | camera |
|------|-----------|-----------|-------|------|----------|------------------|------|--------|
| grade0_case1 | 0 (No apparent DR) | 0 (No apparent DR) | EXACT | 0.99 | 0.00 | auto_report/routine | `1063` | canon_cr2 |
| grade0_case2 | 0 (No apparent DR) | 0 (No apparent DR) | EXACT | 0.99 | 0.00 | auto_report/routine | `1007` | canon_cr2 |
| grade0_case3 | 0 (No apparent DR) | 0 (No apparent DR) | EXACT | 0.99 | 0.00 | auto_report/routine | `1045` | canon_cr2 |
| grade1_case1 | 1 (Mild NPDR) | 1 (Mild NPDR) | EXACT | 0.62 | 0.40 | defer_to_human/soon | `2025` | canon_cr2 |
| grade1_case2 | 1 (Mild NPDR) | 1 (Mild NPDR) | EXACT | 0.50 | 0.10 | auto_report/routine | `2056` | topcon_nw400 |
| grade1_case3 | 1 (Mild NPDR) | 1 (Mild NPDR) | EXACT | 0.58 | 0.17 | auto_report/routine | `2057` | canon_cr2 |
| grade2_case1 | 2 (Moderate NPDR) | 2 (Moderate NPDR) | EXACT | 0.64 | 0.77 | refer/urgent | `3002` | topcon_nw400 |
| grade2_case2 | 2 (Moderate NPDR) | 2 (Moderate NPDR) | EXACT | 0.65 | 0.77 | refer/urgent | `3032` | topcon_nw400 |
| grade3_case1 | 3 (Severe NPDR) | 4 (Proliferative DR) | NO (pred 4) | 0.38 | 0.97 | refer/urgent | `4010` | topcon_nw400 |
| grade3_case2 | 3 (Severe NPDR) | 2 (Moderate NPDR) | NO (pred 2) | 0.47 | 0.82 | refer/urgent | `4030` | topcon_nw400 |
| grade4_case1 | 4 (Proliferative DR) | 4 (Proliferative DR) | EXACT | 0.62 | 0.82 | refer/urgent | `5037` | canon_cr2 |
| grade4_case2 | 4 (Proliferative DR) | 4 (Proliferative DR) | EXACT | 0.63 | 0.97 | refer/urgent | `5025` | canon_cr2 |

**Exact-grade matches: 10/12.** Sight-threatening (true grade >= 2) correctly referred: 6/6.

Grade 3 (severe NPDR) is never predicted exactly here — it collapses to the adjacent grade 2 or 4 — but is still correctly flagged refer/urgent. This is the known open issue on branch `fix/sight-threatening-grades`.

The predicted **grade is deterministic** (same on every machine, every run). Only the
confidence / P(refer) numbers move slightly between runs (MC-dropout), so on the two
borderline grade-1 cases the *decision wording* may flip between `auto_report` and
`defer_to_human` — the grade stays 1.

---

## Run it on a fresh clone (verified end-to-end)

```bash
# 1. clone
git clone -b ppt_related https://github.com/yashchoudhary14/SIH-26038.git
cd SIH-26038

# 2. environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. install PyTorch — pick ONE:
pip install torch torchvision                                             # Mac / CPU-only laptop
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU

# 4. install the rest
pip install -r requirements.txt && pip install -e .
```

The trained model weights (`outputs/artifacts/*.pt`) are committed, so no download or
training is needed.

**Demo option A — browser UI (best for presenting):**

```bash
python -m uvicorn drscreen.api:app --port 8000
```

Open http://localhost:8000, upload any image from `outputs/verification_set/images/`,
and read the grade + referral decision + Grad-CAM++ heatmap live.

**Demo option B — grade all 12 at once and save reports:**

```bash
python scripts/run_demo.py --dir outputs/verification_set/images --out demo_out
```

Prints the grade per image and writes a heatmap panel + HTML report into `demo_out/`.
Compare each printed grade against the filename and the table above.

**Grade your own image:**

```bash
python scripts/run_demo.py --image path/to/fundus.jpg --out demo_out
```
