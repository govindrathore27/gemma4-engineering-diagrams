import json
from pathlib import Path
from typing import Iterator


def parse_graphml(path: str) -> Iterator[dict]:
    import networkx as nx

    G = nx.read_graphml(path)

    component_types: dict[str, list[str]] = {}
    for node, data in G.nodes(data=True):
        ctype = data.get("type", "unknown")
        component_types.setdefault(ctype, []).append(node)

    for ctype, nodes in component_types.items():
        if len(nodes) >= 1:
            yield {
                "instruction": f"List all {ctype} components in this P&ID.",
                "input": "",
                "output": json.dumps({"components": nodes, "count": len(nodes)}),
            }

    pump_nodes = [
        n for n, d in G.nodes(data=True) if "pump" in d.get("type", "").lower()
    ]
    for pump in pump_nodes[:5]:
        reachable = list(nx.bfs_tree(G, pump, depth_limit=3).nodes())
        valves = [
            n
            for n in reachable
            if "valve" in G.nodes[n].get("type", "").lower()
        ]
        if valves:
            yield {
                "instruction": f"What valves must be closed to isolate {pump}?",
                "input": "",
                "output": json.dumps({"pump": pump, "isolation_valves": valves}),
            }

    vessel_nodes = [
        n for n, d in G.nodes(data=True) if "vessel" in d.get("type", "").lower()
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


def parse_cghd(annotation_path: str) -> Iterator[dict]:
    with open(annotation_path) as f:
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
    r_idx = c_idx = d_idx = 1
    for comp in components:
        cl = comp.lower()
        if "resistor" in cl:
            lines.append(f"R{r_idx} N{r_idx} N{r_idx+1} 1k")
            r_idx += 1
        elif "capacitor" in cl:
            lines.append(f"C{c_idx} N{c_idx} 0 100n")
            c_idx += 1
        elif "led" in cl or "diode" in cl:
            lines.append(f"D{d_idx} N{d_idx} 0 1N4148")
            d_idx += 1
    lines.append(".end")
    return "\n".join(lines)


def parse_as1100(annotation_path: str) -> Iterator[dict]:
    with open(annotation_path) as f:
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

    tols = [
        (d["label"], float(d["tolerance"]))
        for d in dimensions
        if "tolerance" in d and "label" in d
    ]
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


_PARSERS = {
    "graphml": parse_graphml,
    "cghd": parse_cghd,
    "as1100": parse_as1100,
}


def parse(format: str, path: str) -> list[dict]:
    if format not in _PARSERS:
        raise ValueError(f"Unknown format '{format}'. Choose from: {list(_PARSERS)}")
    return list(_PARSERS[format](path))


def save_jsonl(pairs: list[dict], dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
