"""Build web/district-model.js from the Simulink district model.

The dossier's simulator draws the SimEvents block diagram and compares its
own engine with MATLAB. Both come from here, so the page cannot drift from
the model file:

* the diagram — every block's position and type, and every connection routed
  exactly as Simulink routes it — read from the .slx package;
* the reference results — MATLAB replications written by
  `Simulink Model/run_district_model.m` into `Simulink Model/validation/`.

    python scripts/build_district_web.py
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Simulink Model"
OUT = ROOT / "web" / "district-model.js"


def _model_path(*names: str) -> Path | None:
    for name in names:
        if (MODEL_DIR / name).is_file():
            return MODEL_DIR / name
    return None


def _nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")]


def _points(text: str | None) -> list[tuple[float, float]]:
    v = _nums(text or "")
    return [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]


def read_diagram(slx: Path) -> dict:
    with zipfile.ZipFile(slx) as z:
        root = ET.fromstring(z.read("simulink/systems/system_root.xml"))

    blocks, by_sid = [], {}
    for b in root.findall("Block"):
        p = {e.get("Name"): (e.text or "") for e in b.findall("P")}
        l, t, r, bt = _nums(p.get("Position", "0,0,0,0"))
        pc = b.find("PortCounts")
        blk = {"sid": b.get("SID"), "name": b.get("Name"), "type": b.get("BlockType"),
               "x": l, "y": t, "w": r - l, "h": bt - t,
               "in": int(pc.get("in", 0)) if pc is not None else 0,
               "out": int(pc.get("out", 0)) if pc is not None else 0}
        if b.get("BlockType") == "EntityResourcePool":
            blk["resource"] = p.get("ResourceName", "")
            blk["amount"] = p.get("ResourceAmount", "1")
        blocks.append(blk)
        by_sid[blk["sid"]] = blk

    def port(ref: str) -> tuple[float, float]:
        # "SID#out:2" -> the port's position, spaced as Simulink spaces them
        sid, rest = ref.split("#")
        side, k = rest.split(":")
        b, k = by_sid[sid], int(k)
        n = b["out"] if side == "out" else b["in"]
        y = b["y"] + b["h"] * (2 * k - 1) / (2 * max(n, 1))
        return (b["x"] + b["w"], y) if side == "out" else (b["x"], y)

    sinks = {"Display", "Scope"}
    lines = []

    def route(start, pts_text, dst_ref, kind, src_ref=None):
        pts, (x, y) = [start], start
        for dx, dy in _points(pts_text):
            x, y = x + dx, y + dy
            pts.append((x, y))
        if dst_ref:
            dx, dy = port(dst_ref)
            if abs(dy - y) > 0.5 and abs(dx - x) > 0.5:      # square off the last leg
                pts.append((x, dy))
            pts.append((dx, dy))
            dst = by_sid[dst_ref.split("#")[0]]
            kind = "signal" if dst["type"] in sinks else kind
        line = {"pts": [[round(a, 1), round(b, 1)] for a, b in pts], "kind": kind}
        if src_ref and dst_ref:       # which block feeds which, for the page's own layout
            line["src"] = by_sid[src_ref.split("#")[0]]["name"]
            line["srcPort"] = int(src_ref.split(":")[1])
            line["dst"] = by_sid[dst_ref.split("#")[0]]["name"]
        lines.append(line)
        return (x, y)

    for ln in root.findall("Line"):
        p = {e.get("Name"): (e.text or "") for e in ln.findall("P")}
        src = p.get("Src")
        if not src:
            continue
        start = port(src)
        kind = "entity"
        branches = ln.findall("Branch")
        if not branches:
            route(start, p.get("Points"), p.get("Dst"), kind, src)
            continue
        trunk_end = route(start, p.get("Points"), None, kind)
        for br in branches:
            q = {e.get("Name"): (e.text or "") for e in br.findall("P")}
            route(trunk_end, q.get("Points"), q.get("Dst"), kind, src)

    xs = [b["x"] for b in blocks] + [b["x"] + b["w"] for b in blocks]
    ys = [b["y"] for b in blocks] + [b["y"] + b["h"] for b in blocks]
    for ln in lines:
        xs += [p[0] for p in ln["pts"]]; ys += [p[1] for p in ln["pts"]]
    return {"source": slx.name, "blocks": blocks, "lines": lines,
            "bounds": [min(xs), min(ys), max(xs), max(ys)]}


def read_references() -> dict:
    refs = {}
    vdir = MODEL_DIR / "validation"
    for f in sorted(vdir.glob("*.json")) if vdir.is_dir() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        keep = {k: {"mean": v["mean"], "sd": v["sd"]} for k, v in d.items()
                if isinstance(v, dict) and "mean" in v}
        refs[f.stem] = {"model": d.get("model"), "reps": d.get("reps"), "stop_min": d.get("stop_min"),
                        "seed_all": d.get("seed_all"), "p": d.get("p"), "stats": keep}
    return refs


def main() -> None:
    # the model as shipped and the improved one: the page shows whichever is selected
    v1 = _model_path("district_model.slx", "district_model.slx.zip")
    v2 = _model_path("district_model_v2.slx")
    if not v1:
        sys.exit("no district model found in 'Simulink Model/'")
    data = {"diagrams": {"asbuilt": read_diagram(v1), "improved": read_diagram(v2 or v1)}}
    data["references"] = read_references()
    OUT.write_text("/* generated by scripts/build_district_web.py from the Simulink model; do not edit */\n"
                   "window.DISTRICT_MODEL = " + json.dumps(data, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    d = data["diagrams"]
    print(f"wrote {OUT.relative_to(ROOT)}: diagrams from {d['asbuilt']['source']} and {d['improved']['source']} "
          f"({len(d['improved']['blocks'])} blocks), {len(data['references'])} reference runs")


if __name__ == "__main__":
    main()
