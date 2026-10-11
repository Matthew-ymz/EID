"""Declared Boolean response networks, separately from their inferred SPTs.

One gate graph drives both the notebook PNG and the editable draw.io source.
Its Boolean outputs are checked exhaustively against the actual full TPMs.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from plotting import COLORS, STYLE, panel

from eid.discrete import enumerate_binary_states

INK = "#25364B"
SOURCE_FILL = "#ECF2F7"
GATE_FILL = "#E3ECF5"
HIGHLIGHT_FILL = "#FCECE4"
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True)
class Node:
    key: str
    xy: tuple[float, float]
    kind: str
    label: str
    parents: tuple[str, ...] = ()
    operation: str = "copy"
    highlighted: bool = False

    @property
    def size(self):
        if self.kind == "gate":
            return (86, 68) if self.operation in ("mux", "majority") else (76, 42)
        return (36, 36) if self.kind == "target" else (32, 32)


@dataclass(frozen=True)
class Mechanism:
    nodes: tuple[Node, ...]
    formula: str
    note: str = ""


def _symbol(prefix, index):
    return f"{prefix}{index}".translate(SUBSCRIPTS)


def _inputs(levels, *, highlight=None):
    return tuple(
        Node(f"x{i}", (55, y), "input", _symbol("x", i), highlighted=i == highlight)
        for i, y in enumerate(levels)
    )


def _gate(key, xy, parents, operation="xor", label="XOR"):
    return Node(key, xy, "gate", label, tuple(parents), operation)


def _target(index, xy, parent):
    return Node(f"y{index}", xy, "target", _symbol("y", index), (parent,))


def mechanism_graphs():
    """Fixed diagrams of the six mechanisms defined in spt_examples.py."""
    return {
        "Pair + triad": Mechanism(
            _inputs((280, 230, 130, 80, 30))
            + (
                _gate("pair", (300, 255), ("x0", "x1")),
                _gate("triad", (300, 80), ("x2", "x3", "x4")),
                _target(0, (565, 255), "pair"),
                _target(2, (565, 80), "triad"),
            ),
            "y₀ = x₀ ⊕ x₁;   y₂ = x₂ ⊕ x₃ ⊕ x₄",
            "Two separate source modules",
        ),
        "Global parity": Mechanism(
            _inputs((280, 232, 184, 136, 88, 40))
            + (
                _gate("parity", (300, 160), tuple(f"x{i}" for i in range(6))),
                _target(0, (565, 160), "parity"),
            ),
            "y₀ = x₀ ⊕ x₁ ⊕ x₂ ⊕ x₃ ⊕ x₄ ⊕ x₅",
            "One six-input parity interaction",
        ),
        "Gated parity": Mechanism(
            _inputs((280, 232, 184, 136, 88, 40))
            + (
                _gate("pair01", (235, 256), ("x0", "x1")),
                _gate("pair23", (235, 160), ("x2", "x3")),
                _gate("pair45", (235, 64), ("x4", "x5")),
                _gate("and", (435, 160), ("pair01", "pair23", "pair45"), "and", "AND"),
                _target(0, (565, 160), "and"),
            ),
            "y₀ = (x₀ ⊕ x₁) ∧ (x₂ ⊕ x₃) ∧ (x₄ ⊕ x₅)",
            "Three local parity gates, then joint gating",
        ),
        "Overlapping parity": Mechanism(
            _inputs((280, 230, 155, 80, 30), highlight=2)
            + (
                _gate("left", (300, 235), ("x0", "x1", "x2")),
                _gate("right", (300, 75), ("x2", "x3", "x4")),
                _target(0, (565, 235), "left"),
                _target(1, (565, 75), "right"),
            ),
            "y₀ = x₀ ⊕ x₁ ⊕ x₂;   y₁ = x₂ ⊕ x₃ ⊕ x₄",
            "Source x₂ participates in both interactions",
        ),
        "Majority vote": Mechanism(
            _inputs((280, 220, 160, 100, 40))
            + (
                _gate("vote", (300, 160), tuple(f"x{i}" for i in range(5)), "majority", "≥ 3 of 5"),
                _target(0, (565, 160), "vote"),
            ),
            "y₀ = 1 if x₀ + x₁ + x₂ + x₃ + x₄ ≥ 3; else 0",
            "Five equally weighted votes; threshold = 3",
        ),
        "Switched logic": Mechanism(
            _inputs((45, 280, 230, 125), highlight=0)
            + (
                _gate("pair12", (235, 255), ("x1", "x2")),
                _gate("mux", (435, 185), ("x0", "pair12", "x3"), "mux", "MUX\nselect"),
                _target(0, (565, 185), "mux"),
            ),
            "y₀ = (x₁ ⊕ x₂) if x₀ = 1; else x₃",
            "Source x₀ selects the branch",
        ),
    }


def verify_mechanisms(cases, graphs):
    """Check all states, selected targets, and the stated auxiliary copies."""
    if tuple(cases) != tuple(graphs):
        raise ValueError("Diagram cases must match the ordered structure gallery.")
    checked = 0
    for name, case in cases.items():
        states = enumerate_binary_states(case.n_nodes)
        values, targets = {}, []
        for node in graphs[name].nodes:
            if node.kind == "input":
                values[node.key] = states[:, int(node.key[1:])]
                continue
            parents = [values[key] for key in node.parents]
            if node.operation == "xor":
                value = np.bitwise_xor.reduce(parents, axis=0)
            elif node.operation == "and":
                value = np.bitwise_and.reduce(parents, axis=0)
            elif node.operation == "majority":
                value = np.sum(parents, axis=0) >= 3
            elif node.operation == "mux":
                value = np.where(parents[0], parents[1], parents[2])
            else:
                value = parents[0]
            values[node.key] = value
            if node.kind == "target":
                targets.append(int(node.key[1:]))
        if tuple(targets) != case.targets or case.horizon != 1:
            raise ValueError(f"Diagram target/horizon mismatch for {name}.")
        drawn_output = states.copy()
        for target in targets:
            drawn_output[:, target] = values[f"y{target}"]
        actual_output = states[np.argmax(case.tpm, axis=1)]
        if not np.all(np.max(case.tpm, axis=1) == 1) or not np.array_equal(actual_output, drawn_output):
            raise ValueError(f"Diagram rules or auxiliary copies disagree with the TPM for {name}.")
        checked += len(states)
    return checked


def _colors(node):
    if node.highlighted:
        return HIGHLIGHT_FILL, COLORS["accent"], INK
    if node.kind == "target":
        return COLORS["xi"], COLORS["xi"], "white"
    return (GATE_FILL if node.kind == "gate" else SOURCE_FILL), COLORS["single"], INK


def _boundary(node, toward):
    direction = np.asarray(toward) - node.xy
    half_size = np.asarray(node.size) / 2
    if node.kind == "gate":
        distance = min(half_size[i] / abs(direction[i]) for i in range(2) if direction[i] != 0)
    else:
        distance = half_size[0] / np.linalg.norm(direction)
    return np.asarray(node.xy) + direction * distance


def _route(source, target):
    control = target.operation == "mux" and source.key == "x0"
    waypoints = [(target.xy[0], source.xy[1])] if control else []
    first, last = waypoints[0] if waypoints else target.xy, waypoints[-1] if waypoints else source.xy
    return [_boundary(source, first), *waypoints, _boundary(target, last)], control


def _copy_label(case):
    remaining = [i for i in range(case.n_nodes) if i not in case.targets]
    return "Other updates: " + "; ".join(f"{_symbol('y', i)} = {_symbol('x', i)}" for i in remaining)


def _plot_graph(ax, name, case, graph, letter):
    by_key = {node.key: node for node in graph.nodes}
    for target in graph.nodes:
        for source_key in target.parents:
            points, control = _route(by_key[source_key], target)
            color = COLORS["accent"] if control else COLORS["single"]
            if len(points) > 2:
                segment = np.asarray(points[:-1])
                ax.plot(*segment.T, color=color, lw=1.5, ls="--")
            ax.add_patch(
                FancyArrowPatch(
                    points[-2],
                    points[-1],
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color=color,
                    lw=1.3,
                    linestyle="--" if control else "-",
                    shrinkA=1.5,
                    shrinkB=2,
                )
            )
    for node in graph.nodes:
        face, edge, text = _colors(node)
        width, height = node.size
        if node.kind == "gate":
            patch = FancyBboxPatch(
                (node.xy[0] - width / 2, node.xy[1] - height / 2),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=7",
                facecolor=face,
                edgecolor=edge,
                lw=1.3,
            )
        else:
            patch = Circle(node.xy, width / 2, facecolor=face, edgecolor=edge, lw=1.3)
        ax.add_patch(patch)
        ax.text(*node.xy, node.label, ha="center", va="center", color=text, fontsize=10)
    if name == "Overlapping parity":
        ax.text(99, 155, "shared", color=COLORS["accent"], ha="center", va="center", fontsize=8.5)
    if name == "Switched logic":
        ax.text(270, 29, "selector x₀", color=COLORS["accent"], ha="center", fontsize=8.5)
        ax.text(372, 225, "1", color=INK, fontsize=9)
        ax.text(372, 148, "0", color=INK, fontsize=9)
    for x, label in ((55, "Inputs · t"), (320, "Interaction rules"), (550, "Targets · t+1")):
        ax.text(x, 323, label, ha="center", fontsize=9, color=COLORS["single"])
    ax.text(310, -23, graph.formula, ha="center", fontsize=9.5, color=INK)
    ax.text(310, -48, graph.note, ha="center", fontsize=9, color=COLORS["single"])
    ax.text(310, -72, _copy_label(case), ha="center", fontsize=8.5, color=COLORS["single"])
    ax.set(xlim=(0, 620), ylim=(-85, 340), aspect="equal")
    ax.axis("off")
    panel(ax, letter, name)


def _write_drawio(cases, graphs, path):
    """Editable source with the same gate graph and layout; no desktop dependency."""
    document = ET.Element("mxfile", host="drawio")
    diagram = ET.SubElement(document, "diagram", name="Boolean response networks", id="boolean-mechanisms")
    model = ET.SubElement(diagram, "mxGraphModel", page="1", pageWidth="1340", pageHeight="1400")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    next_id = 2

    def vertex(value, style, xy, size, parent="1"):
        nonlocal next_id
        key = str(next_id)
        next_id += 1
        cell = ET.SubElement(root, "mxCell", id=key, value=value, style=style, vertex="1", parent=parent)
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(xy[0]),
            y=str(xy[1]),
            width=str(size[0]),
            height=str(size[1]),
            **{"as": "geometry"},
        )
        return key

    for index, (name, graph) in enumerate(graphs.items()):
        parent = vertex(
            f"{'abcdef'[index]}  {name}",
            "swimlane;startSize=30;html=1;fillColor=#FFFFFF;strokeColor=#CCD5DF;"
            "fontColor=#25364B;fontSize=15;container=1;pointerEvents=0;",
            (10 + index % 2 * 670, 10 + index // 2 * 450),
            (640, 430),
        )
        ids = {}
        for node in graph.nodes:
            width, height = node.size
            face, edge, text = _colors(node)
            shape = "rounded=1;" if node.kind == "gate" else "ellipse;"
            ids[node.key] = vertex(
                node.label,
                f"{shape}whiteSpace=wrap;html=1;fillColor={face};strokeColor={edge};"
                f"fontColor={text};fontSize=14;strokeWidth=1.3;",
                (node.xy[0] - width / 2, 342 - node.xy[1] - height / 2),
                (width, height),
                parent,
            )
        for target in graph.nodes:
            for source_key in target.parents:
                points, control = _route(next(n for n in graph.nodes if n.key == source_key), target)
                color = COLORS["accent"] if control else COLORS["single"]
                branch = (
                    "" if target.operation != "mux" or control else ("1" if source_key == "pair12" else "0")
                )
                cell = ET.SubElement(
                    root,
                    "mxCell",
                    id=str(next_id),
                    value=branch,
                    edge="1",
                    parent=parent,
                    source=ids[source_key],
                    target=ids[target.key],
                    style=f"edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;"
                    f"strokeColor={color};strokeWidth=1.3;dashed={int(control)};fontSize=12;",
                )
                next_id += 1
                geometry = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
                if len(points) > 2:
                    waypoints = ET.SubElement(geometry, "Array", **{"as": "points"})
                    for x, y in points[1:-1]:
                        ET.SubElement(waypoints, "mxPoint", x=str(x), y=str(342 - y))
        labels = [
            (55, 323, "Inputs · t"),
            (320, 323, "Interaction rules"),
            (550, 323, "Targets · t+1"),
            (310, -23, graph.formula),
            (310, -48, graph.note),
            (310, -72, _copy_label(cases[name])),
        ]
        if name == "Overlapping parity":
            labels.append((99, 155, "shared"))
        if name == "Switched logic":
            labels.append((270, 29, "selector x₀"))
        for x, y, value in labels:
            width = 620 if y < 0 else 150
            vertex(
                value,
                "text;html=1;align=center;verticalAlign=middle;fontColor=#53657B;fontSize=12;",
                (x - width / 2, 342 - y - 10),
                (width, 20),
                parent,
            )
    ET.indent(document)
    ET.ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)


def show_boolean_mechanisms(cases, path):
    """Render six original response networks and save an editable companion."""
    graphs = mechanism_graphs()
    checked = verify_mechanisms(cases, graphs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.0), constrained_layout=True)
        for index, ((name, graph), ax) in enumerate(zip(graphs.items(), axes.flat)):
            _plot_graph(ax, name, cases[name], graph, "abcdef"[index])
        fig.legend(
            [
                Line2D([], [], marker="o", mfc=SOURCE_FILL, mec=COLORS["single"], ls="none", ms=8),
                Line2D([], [], marker="s", mfc=GATE_FILL, mec=COLORS["single"], ls="none", ms=8),
                Line2D([], [], marker="o", mfc=COLORS["xi"], mec=COLORS["xi"], ls="none", ms=8),
                Line2D([], [], color=COLORS["accent"], ls="--"),
            ],
            ["Binary input", "Logical operation", "Analyzed output", "Selector control"],
            loc="outside upper center",
            ncol=4,
        )
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        display(fig)
        plt.close(fig)
    _write_drawio(cases, graphs, path.with_suffix(".drawio"))
    print(f"Diagram rules and auxiliary copies match all {checked} full TPM states across six systems.")
