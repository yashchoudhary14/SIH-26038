"""Build web/district-model.js from the Simulink district model.

The dossier's simulator draws the SimEvents block diagram and compares its
own engine with MATLAB. Both come from here, so the page cannot drift from
the model file:

* the diagram — every block's position and type, every connection routed
  exactly as Simulink routes it, and the stage panels and branch labels —
  read from `simulink/district_model.slx` (built by
  build_district_model.m), and from its recapture variant
  `simulink/figures/district_model_recapture.slx` when present;
* the reference results — MATLAB replications written by
  `simulink/run_district_model.m` into `simulink/validation/`.

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
MODEL_DIR = ROOT / "simulink"
OUT = ROOT / "web" / "district-model.js"

# blocks with no entity output: every output port they have is a statistic
NO_ENTITY_OUT = {"EntityTerminator", "EntityResourcePool", "Scope", "Display"}
PORT = 5                  # how far a port stands off its block


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
        n_out = int(pc.get("out", 0)) if pc is not None else 0
        # which outputs carry entities ("m") and which are statistics ("n")
        modes = [m.strip() for m in p.get("OutputPortMessageModes", "").split(",") if m.strip()]
        if not modes:
            modes = ["n" if b.get("BlockType") in NO_ENTITY_OUT else "m"] * n_out
        blk = {"sid": b.get("SID"), "name": b.get("Name"), "type": b.get("BlockType"),
               "x": l, "y": t, "w": r - l, "h": bt - t,
               "in": int(pc.get("in", 0)) if pc is not None else 0, "out": n_out, "modes": modes}
        if b.get("BlockType") == "EntityResourcePool":
            blk["resource"] = p.get("ResourceName", "")
            blk["amount"] = p.get("ResourceAmount", "1")
        blocks.append(blk)
        by_sid[blk["sid"]] = blk

    def port(ref: str) -> tuple[float, float]:
        # "SID#out:2" -> the port's position, where Simulink puts it: entity
        # ports spaced down the block's side, statistic ports on its top, each
        # standing PORT pixels off the block (line points are relative to it)
        sid, rest = ref.split("#")
        side, k = rest.split(":")
        b, k = by_sid[sid], int(k)
        if side == "in":
            return (b["x"] - PORT, b["y"] + b["h"] * (2 * k - 1) / (2 * max(b["in"], 1)))
        modes = b["modes"]
        if k - 1 < len(modes) and modes[k - 1] == "n":
            return (b["x"] + b["w"] / 2, b["y"] - PORT)
        n = sum(1 for m in modes if m == "m")
        j = sum(1 for m in modes[:k] if m == "m")
        return (b["x"] + b["w"] + PORT, b["y"] + b["h"] * (2 * j - 1) / (2 * max(n, 1)))

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
        if src_ref and dst_ref:       # which block feeds which
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
        branches = ln.findall("Branch")
        if not branches:
            route(start, p.get("Points"), p.get("Dst"), "entity", src)
            continue
        trunk_end = route(start, p.get("Points"), None, "entity")
        for br in branches:
            q = {e.get("Name"): (e.text or "") for e in br.findall("P")}
            route(trunk_end, q.get("Points"), q.get("Dst"), "entity", src)

    # the stage panels and the branch labels
    areas, notes = [], []
    for a in root.findall("Annotation"):
        p = {e.get("Name"): (e.text or "") for e in a.findall("P")}
        pos = _nums(p.get("Position", ""))
        if len(pos) != 4:
            continue
        item = {"text": p.get("Name", ""), "x": pos[0], "y": pos[1], "w": pos[2] - pos[0], "h": pos[3] - pos[1]}
        if p.get("AnnotationType") == "area_annotation":
            areas.append(item)
        elif p.get("Interpreter") != "tex":            # (the page sets the arrival formula itself)
            notes.append(item)

    xs = [b["x"] for b in blocks] + [b["x"] + b["w"] for b in blocks] + [a["x"] + a["w"] for a in areas]
    ys = [b["y"] for b in blocks] + [b["y"] + b["h"] for b in blocks] + [a["y"] + a["h"] for a in areas]
    xs += [a["x"] for a in areas]; ys += [a["y"] for a in areas]
    for ln in lines:
        xs += [p[0] for p in ln["pts"]]; ys += [p[1] for p in ln["pts"]]
    return {"source": slx.name, "blocks": blocks, "lines": lines, "areas": areas, "notes": notes,
            "bounds": [min(xs), min(ys), max(xs), max(ys)]}


def read_references() -> dict:
    refs = {}
    vdir = MODEL_DIR / "validation"
    for f in sorted(vdir.glob("*.json")) if vdir.is_dir() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        keep = {k: {"mean": v["mean"], "sd": v["sd"]} for k, v in d.items()
                if isinstance(v, dict) and "mean" in v}
        refs[f.stem] = {"model": d.get("model"), "reps": d.get("reps"), "stop_min": d.get("stop_min"),
                        "p": d.get("p"), "stats": keep}
    return refs


def main() -> None:
    slx = MODEL_DIR / "district_model.slx"
    if not slx.is_file():
        sys.exit("no district_model.slx in 'simulink/' (run build_district_model in MATLAB)")
    data = {"diagram": read_diagram(slx), "references": read_references()}
    # the recapture variant (include_gate = 1), built by export_district_model_figure.m
    gate = MODEL_DIR / "figures" / "district_model_recapture.slx"
    if gate.is_file():
        data["diagramGate"] = read_diagram(gate)
    OUT.write_text("/* generated by scripts/build_district_web.py from the Simulink model; do not edit */\n"
                   "window.DISTRICT_MODEL = " + json.dumps(data, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    d = data["diagram"]
    print(f"wrote {OUT.relative_to(ROOT)}: {d['source']} ({len(d['blocks'])} blocks, {len(d['lines'])} lines, "
          f"{len(d['areas'])} panels){' and its recapture variant' if 'diagramGate' in data else ''}, "
          f"{len(data['references'])} reference runs")


if __name__ == "__main__":
    main()
