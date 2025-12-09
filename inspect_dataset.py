import angr

proj = angr.Project(
    "samples/sample_0000_x86.bin",
    main_opts={
        "backend": "blob",      # force blob loader
        "arch": "x86",          # architecture of your binary
        "entry": 0x0,           # starting address (adjust if known)
        "base_addr": 0x400000   # load address in memory
    },
    auto_load_libs=False
)

print("Architecture:", proj.arch)
print("Entry point:", hex(proj.entry))
cfg = proj.analyses.CFGFast()
print(cfg)
num_nodes = len(cfg.graph.nodes())
num_edges = len(cfg.graph.edges())
print(f"CFG has {num_nodes} nodes and {num_edges} edges.")


opcodes = []

for func in proj.kb.functions.values():
    for block in func.blocks:
        for ins in block.capstone.insns:
            opcodes.append(ins.mnemonic)
print("First 10 opcodes:", opcodes[:10])


import math

def entropy(data):
    if len(data) == 0:
        return 0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    e = 0
    for f in freq:
        if f > 0:
            p = f / len(data)
            e -= p * math.log2(p)
    return e

raw = open("samples/sample_0000_x86.bin", "rb").read()
ent = entropy(raw)
print("Entropy of binary:", ent)


print("Architecture:", proj.arch)
print("Entry point:", hex(proj.entry))
print("File size:", len(raw))
print("Binary entropy:", entropy(raw))
print("Endianness:", proj.arch.memory_endness)
print("Bits:", proj.arch.bits)



cfg = proj.analyses.CFGFast()
print("CFG object:", cfg)

num_nodes = len(cfg.graph.nodes())
num_edges = len(cfg.graph.edges())

print("CFG nodes:", num_nodes)
print("CFG edges:", num_edges)
print("CFG connectivity:", num_edges / num_nodes if num_nodes else 0)


functions = proj.kb.functions
print("Total functions:", len(functions))

sizes = [len(list(f.blocks)) for f in functions.values()]
print("Average function size:", sum(sizes)/len(sizes) if sizes else 0)
print("Max function size:", max(sizes) if sizes else 0)


total_blocks = sum(len(list(f.blocks)) for f in functions.values())
print("Total basic blocks:", total_blocks)


block_sizes = [b.size for f in functions.values() for b in f.blocks]
print("Average block size:", sum(block_sizes)/len(block_sizes) if block_sizes else 0)
print("Max block size:", max(block_sizes) if block_sizes else 0)
