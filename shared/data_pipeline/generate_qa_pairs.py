import json
from pathlib import Path
from typing import Iterator


def parse_graphml(path: str) -> Iterator[dict]:
    import networkx as nx

    G = nx.read_graphml(path)

    component_types: dict[str, list[str]] = {}
    for node, data in G.nodes(data=True):
        ctype = data.get("label", data.get("type", "unknown"))
        component_types.setdefault(ctype, []).append(node)

    for ctype, nodes in component_types.items():
        if len(nodes) >= 1:
            yield {
                "instruction": f"List all {ctype} components in this P&ID.",
                "input": "",
                "output": json.dumps({"components": nodes, "count": len(nodes)}),
            }

    pump_nodes = [
        n for n, d in G.nodes(data=True) if "pump" in d.get("label", d.get("type", "")).lower()
    ]
    for pump in pump_nodes[:5]:
        reachable = list(nx.bfs_tree(G, pump, depth_limit=3).nodes())
        valves = [
            n
            for n in reachable
            if "valve" in G.nodes[n].get("label", G.nodes[n].get("type", "")).lower()
        ]
        if valves:
            yield {
                "instruction": f"What valves must be closed to isolate {pump}?",
                "input": "",
                "output": json.dumps({"pump": pump, "isolation_valves": valves}),
            }

    vessel_nodes = [
        n for n, d in G.nodes(data=True) if "vessel" in d.get("label", d.get("type", "")).lower()
    ]
    for vessel in vessel_nodes[:3]:
        if nx.is_directed(G):
            predecessors = list(nx.ancestors(G, vessel))
        else:
            predecessors = list(G.neighbors(vessel))
        spf = [n for n in predecessors if G.degree(n) == 1]
        yield {
            "instruction": f"Identify single-point failures that could affect {vessel}.",
            "input": "",
            "output": json.dumps({"vessel": vessel, "single_point_failures": spf}),
        }

    # Count components by label
    label_counts: dict[str, int] = {}
    for node, data in G.nodes(data=True):
        lbl = data.get("label", data.get("type", "unknown"))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    yield {
        "instruction": "How many components of each type are in this P&ID?",
        "input": "",
        "output": json.dumps(label_counts),
    }

    # List all instrumentation nodes
    instr_nodes = [
        n for n, d in G.nodes(data=True)
        if "instrument" in d.get("label", d.get("type", "")).lower()
    ]
    if instr_nodes:
        yield {
            "instruction": "List all instrumentation components in this P&ID.",
            "input": "",
            "output": json.dumps({"instrumentation": instr_nodes, "count": len(instr_nodes)}),
        }

    # List connectors
    connector_nodes = [
        n for n, d in G.nodes(data=True)
        if "connector" in d.get("label", d.get("type", "")).lower()
    ]
    if connector_nodes:
        yield {
            "instruction": "List all connectors in this P&ID.",
            "input": "",
            "output": json.dumps({"connectors": connector_nodes[:20], "total_count": len(connector_nodes)}),
        }


def parse_cghd(annotation_path: str) -> Iterator[dict]:
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    comp_names = [a["category"] for a in annotations if a.get("category")]

    if not comp_names:
        return

    yield {
        "instruction": "Describe what this circuit does based on its components.",
        "input": "",
        "output": (
            f"This circuit contains: {', '.join(set(comp_names))}. "
            + _circuit_description(comp_names)
        ),
    }

    resistors = [c for c in comp_names if "resistor" in c.lower()]
    leds = [c for c in comp_names if "led" in c.lower() or "diode" in c.lower()]
    if resistors and leds:
        yield {
            "instruction": f"What component limits current through {leds[0]}?",
            "input": "",
            "output": (
                f"{resistors[0]} acts as the current-limiting resistor for {leds[0]}."
            ),
        }

    yield {
        "instruction": "Generate a SPICE-compatible netlist for this circuit.",
        "input": "",
        "output": _netlist_from_components(comp_names),
    }


def _circuit_description(components: list[str]) -> str:
    has_resistor = any("resistor" in c.lower() for c in components)
    has_capacitor = any("capacitor" in c.lower() for c in components)
    has_led = any("led" in c.lower() for c in components)
    has_opamp = any("op" in c.lower() for c in components)
    if has_opamp:
        return "This appears to be an amplifier circuit."
    if has_resistor and has_capacitor:
        return "This appears to be an RC filter circuit."
    if has_resistor and has_led:
        return "This appears to be a simple LED driver circuit."
    return "This is an electronic circuit."


def _netlist_from_components(components: list[str]) -> str:
    lines = ["* Auto-generated netlist"]
    r_idx = c_idx = l_idx = d_idx = 1
    for comp in components:
        cl = comp.lower()
        if "resistor" in cl:
            lines.append(f"R{r_idx} N{r_idx} N{r_idx+1} 1k")
            r_idx += 1
        elif "capacitor" in cl:
            lines.append(f"C{c_idx} N{c_idx} 0 100n")
            c_idx += 1
        elif "inductor" in cl:
            lines.append(f"L{l_idx} N{l_idx} N{l_idx+1} 10m")
            l_idx += 1
        elif "led" in cl or "diode" in cl:
            lines.append(f"D{d_idx} N{d_idx} 0 1N4148")
            d_idx += 1
    lines.append(".end")
    return "\n".join(lines)


CIRCUIT1K_CLASSES = {
    0: "battery",
    1: "resistor",
    2: "capacitor",
    3: "inductor",
    4: "diode",
}


def parse_circuit1k_yolo(annotation_path: str) -> Iterator[dict]:
    """Parse circuit1k YOLO annotation .txt file, generate circuit QA pairs."""
    path = Path(annotation_path)
    if not path.exists() or path.stat().st_size == 0:
        return

    components: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                components.append(CIRCUIT1K_CLASSES.get(class_id, f"component_{class_id}"))

    if not components:
        return

    comp_counts: dict[str, int] = {}
    for c in components:
        comp_counts[c] = comp_counts.get(c, 0) + 1

    # Q1: describe the circuit
    yield {
        "instruction": "Describe what this circuit does based on its components.",
        "input": "",
        "output": (
            f"This circuit contains: {', '.join(f'{v}x {k}' for k,v in comp_counts.items())}. "
            + _circuit_description(list(comp_counts.keys()))
        ),
    }

    # Q2: component count
    yield {
        "instruction": "How many components are in this circuit?",
        "input": "",
        "output": json.dumps({"total": len(components), "by_type": comp_counts}),
    }

    # Q3: current limiter if resistor + diode
    resistors = [c for c in components if c == "resistor"]
    diodes = [c for c in components if c == "diode"]
    if resistors and diodes:
        yield {
            "instruction": "What limits the current through the diode in this circuit?",
            "input": "",
            "output": "A resistor in series with the diode limits the current flow.",
        }

    # Q4: SPICE netlist
    yield {
        "instruction": "Generate a SPICE-compatible netlist for this circuit.",
        "input": "",
        "output": _netlist_from_components(components),
    }

    # Q5: what-if resistor doubles
    if "resistor" in comp_counts:
        yield {
            "instruction": "If all resistors were doubled in value, how would that affect the circuit?",
            "input": "",
            "output": "Doubling the resistance halves the current (by Ohm's law: I = V/R), reducing power dissipation and potentially dimming any LEDs or diodes.",
        }


def parse_as1100(annotation_path: str) -> Iterator[dict]:
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    dimensions: list[dict] = data.get("dimensions", [])

    if dimensions:
        yield {
            "instruction": "List all critical dimensions in this engineering drawing.",
            "input": "",
            "output": json.dumps(
                {"dimensions": [d["label"] for d in dimensions if "label" in d]}
            ),
        }

    tols = []
    for d in dimensions:
        if "tolerance" not in d or "label" not in d:
            continue
        try:
            tols.append((d["label"], float(d["tolerance"])))
        except (ValueError, TypeError):
            continue
    if tols:
        tightest = min(tols, key=lambda x: x[1])
        yield {
            "instruction": (
                "Which feature has the tightest tolerance and may be most costly to machine?"
            ),
            "input": "",
            "output": (
                f"Feature '{tightest[0]}' has the tightest tolerance of "
                f"±{tightest[1]}, making it the most costly to machine."
            ),
        }

    yield {
        "instruction": "Create a manufacturing checklist for this part.",
        "input": "",
        "output": _machining_checklist(dimensions),
    }


def _machining_checklist(dimensions: list[dict]) -> str:
    items = ["[ ] Review drawing for completeness and revision level"]
    for d in dimensions[:5]:
        label = d.get("label", "feature")
        items.append(f"[ ] Machine {label} to specified dimension")
        if "tolerance" in d:
            items.append(f"[ ] Verify {label} within tolerance ±{d['tolerance']}")
    items.append("[ ] Final dimensional inspection and sign-off")
    return "\n".join(items)


def parse_as1100_parquet(parquet_path: str) -> Iterator[dict]:
    """Extract QA pairs directly from AS1100 parquet files (pre-formatted dataset)."""
    try:
        import pandas as pd
    except ImportError:
        return

    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        user_q = str(row.get("user_query", "")).strip()
        assistant_r = str(row.get("assistant_response", "")).strip()
        if user_q and assistant_r:
            yield {
                "instruction": user_q,
                "input": str(row.get("system_prompt", "")).strip(),
                "output": assistant_r,
            }


def parse_dimensioning_yolo(annotation_path: str) -> Iterator[dict]:
    """Parse YOLO dimension annotation .txt, generate drawing QA pairs."""
    path = Path(annotation_path)
    if not path.exists() or path.stat().st_size == 0:
        return

    dt_count = 0
    text_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                if parts[0] == "0":
                    dt_count += 1
                elif parts[0] == "1":
                    text_count += 1

    if dt_count + text_count == 0:
        return

    yield {
        "instruction": "How many dimension annotations are visible in this engineering drawing?",
        "input": "",
        "output": json.dumps({
            "dimension_tags": dt_count,
            "text_labels": text_count,
            "total_annotations": dt_count + text_count,
        }),
    }

    yield {
        "instruction": "Create a manufacturing checklist for this engineering drawing.",
        "input": "",
        "output": _drawing_checklist(dt_count, text_count),
    }

    if dt_count > 0:
        yield {
            "instruction": "List the types of annotations present in this drawing.",
            "input": "",
            "output": (
                f"This drawing contains {dt_count} dimension tag(s) "
                f"and {text_count} text label(s)."
            ),
        }


def _drawing_checklist(dt_count: int, text_count: int) -> str:
    items = [
        "[ ] Verify title block completeness (part number, revision, scale, material)",
        "[ ] Check all dimension tags are legible and unambiguous",
    ]
    if dt_count > 5:
        items.append(f"[ ] Inspect all {dt_count} dimension tags for tolerance compliance")
    if text_count > 0:
        items.append(
            f"[ ] Cross-reference {text_count} text annotation(s) with part specification"
        )
    items += [
        "[ ] Verify surface finish callouts",
        "[ ] Confirm all geometric tolerances are achievable on target machine",
        "[ ] Final inspection sign-off",
    ]
    return "\n".join(items)


_PARSERS = {
    "graphml": parse_graphml,
    "cghd": parse_cghd,
    "circuit1k_yolo": parse_circuit1k_yolo,
    "as1100": parse_as1100,
    "as1100_parquet": parse_as1100_parquet,
    "dimensioning_yolo": parse_dimensioning_yolo,
}


def parse(format: str, path: str) -> list[dict]:
    if format not in _PARSERS:
        raise ValueError(f"Unknown format '{format}'. Choose from: {list(_PARSERS)}")
    return list(_PARSERS[format](path))


def save_jsonl(pairs: list[dict], dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
