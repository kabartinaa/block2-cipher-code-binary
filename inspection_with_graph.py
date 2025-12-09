import os
import math
from collections import Counter

import angr
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
from networkx.drawing.nx_pydot import write_dot  # for DOT export


def entropy(data):
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    ent = 0.0
    for f in freq:
        if f > 0:
            p = f / len(data)
            ent -= p * math.log2(p)
    return ent


def main():
    BIN_PATH = "samples/sample_0000_x86.bin"

    # Load raw bytes
    with open(BIN_PATH, "rb") as f:
        raw = f.read()

    proj = angr.Project(
        BIN_PATH,
        main_opts={
            "backend": "blob",   # force blob loader
            "arch": "x86",       # adjust to "x86_64", "ARM", etc. if needed
            "base_addr": 0x1000  # arbitrary load address
        },
        auto_load_libs=False
    )

    print("\n=== BINARY STRUCTURE ===")
    print("Architecture:", proj.arch)
    print("Endianness:", proj.arch.memory_endness)
    print("Bits:", proj.arch.bits)
    print("Entry point:", hex(proj.entry))
    print("File size:", len(raw))
    print("Binary entropy:", entropy(raw))

    print("\n=== BUILDING CFG ===")
    cfg = proj.analyses.CFGFast()
    g = cfg.graph  # this is a networkx DiGraph

    num_nodes = len(g.nodes())
    num_edges = len(g.edges())
    print("CFG nodes:", num_nodes)
    print("CFG edges:", num_edges)

    # ----------------------------------------------------
    # EXPORT CFG AS GRAPHVIZ DOT
    # ----------------------------------------------------
    out_path = "cfg.dot"
    write_dot(g, out_path)
    print(f"\n[+] CFG exported in Graphviz DOT format to: {out_path}")
    print("    You can render it with, for example:")
    print("      dot -Tpng cfg.dot -o cfg.png")
    print("    or open cfg.dot in a Graphviz/Gephi viewer.\n")


if __name__ == "__main__":
    main()
