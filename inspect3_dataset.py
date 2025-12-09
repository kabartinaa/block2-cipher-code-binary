import angr
from capstone import *
from collections import Counter
import math

def entropy(data):
    freq = [0]*256
    for b in data:
        freq[b] += 1
    ent = 0
    for f in freq:
        if f > 0:
            p = f / len(data)
            ent -= p * math.log2(p)
    return ent

# Load raw binary blob explicitly
raw = open("samples/sample_0061_x86.bin", "rb").read()
proj = angr.Project(
    "samples/sample_0000_x86.bin",
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

print("\n=== CFG FEATURES ===")
cfg = proj.analyses.CFGFast()
num_nodes = len(cfg.graph.nodes())
num_edges = len(cfg.graph.edges())
print("CFG nodes:", num_nodes)
print("CFG edges:", num_edges)
print("CFG connectivity:", num_edges / num_nodes if num_nodes else 0)

avg_degree = sum(dict(cfg.graph.degree()).values()) / num_nodes if num_nodes else 0
print("Average CFG degree:", avg_degree)

density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
print("CFG density:", density)

print("\n=== FUNCTION FEATURES ===")
functions = proj.kb.functions
print("Total functions:", len(functions))

sizes = [len(list(f.blocks)) for f in functions.values()]
print("Average function size:", sum(sizes) / len(sizes) if sizes else 0)
print("Max function size:", max(sizes) if sizes else 0)

print("\n=== BASIC BLOCK FEATURES ===")
total_blocks = sum(len(list(f.blocks)) for f in functions.values())
print("Total basic blocks:", total_blocks)

block_sizes = [b.size for f in functions.values() for b in f.blocks]
print("Average block size:", sum(block_sizes) / len(block_sizes) if block_sizes else 0)
print("Max block size:", max(block_sizes) if block_sizes else 0)

print("\n=== OPCODE FEATURES ===")
md = Cs(CS_ARCH_X86, CS_MODE_32)  # use CS_MODE_64 if binary is 64-bit
opcodes = [ins.mnemonic for ins in md.disasm(raw, 0x1000)]
print("Opcode count:", len(opcodes))
print("Unique opcodes:", len(set(opcodes)))
print("First 20 opcodes:", opcodes[:20])
print("Opcode histogram:", Counter(opcodes))

print("\n=== CALL GRAPH FEATURES ===")
cg = proj.kb.callgraph
print("Call graph nodes:", len(cg.nodes()))
print("Call graph edges:", len(cg.edges()))

#print("\n=== SECTIONS ===")
for sec in proj.loader.main_object.sections:
    print("Section:", sec.name, "Size:", sec.size, "Addr:", hex(sec.vaddr))
