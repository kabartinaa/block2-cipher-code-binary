import os
from capstone import *

BIN_PATH = r"D:\P_2_S_3.bin"

# ---- Build a big multi-arch candidate list safely ----
CANDIDATES = {}

def add_candidate(name, arch_const, mode):
    """Create a Cs() handle if possible and register it."""
    try:
        md = Cs(arch_const, mode)
    except Exception:
        return
    CANDIDATES[name] = md

def add_if_available(name, arch_sym, mode_syms=()):
    """
    Look up CS_ARCH_* and CS_MODE_* symbols by name.
    If they exist in this Capstone build, add a candidate.
    """
    g = globals()
    arch = g.get(arch_sym)
    if arch is None:
        return

    mode = 0
    for m in mode_syms:
        v = g.get(m)
        if v is not None:
            mode |= v

    add_candidate(name, arch, mode)

# --- ARM / AArch64 ---
add_if_available("ARM32",       "CS_ARCH_ARM",    ["CS_MODE_ARM"])
add_if_available("ARM THUMB",   "CS_ARCH_ARM",    ["CS_MODE_THUMB"])
add_if_available("ARMv7-M",     "CS_ARCH_ARM",    ["CS_MODE_THUMB", "CS_MODE_MCLASS"])
add_if_available("AARCH64 LE",  "CS_ARCH_ARM64",  ["CS_MODE_LITTLE_ENDIAN"])
add_if_available("AARCH64 BE",  "CS_ARCH_ARM64",  ["CS_MODE_BIG_ENDIAN"])

# --- X86 family ---
add_if_available("x86 16",      "CS_ARCH_X86",    ["CS_MODE_16"])
add_if_available("x86 32",      "CS_ARCH_X86",    ["CS_MODE_32"])
add_if_available("x86 64",      "CS_ARCH_X86",    ["CS_MODE_64"])

# --- MIPS ---
add_if_available("MIPS32 LE",   "CS_ARCH_MIPS",   ["CS_MODE_MIPS32", "CS_MODE_LITTLE_ENDIAN"])
add_if_available("MIPS32 BE",   "CS_ARCH_MIPS",   ["CS_MODE_MIPS32", "CS_MODE_BIG_ENDIAN"])
add_if_available("MIPS64 LE",   "CS_ARCH_MIPS",   ["CS_MODE_MIPS64", "CS_MODE_LITTLE_ENDIAN"])
add_if_available("MIPS64 BE",   "CS_ARCH_MIPS",   ["CS_MODE_MIPS64", "CS_MODE_BIG_ENDIAN"])

# --- RISC-V ---
add_if_available("RISCV32",     "CS_ARCH_RISCV",  ["CS_MODE_RISCV32"])
add_if_available("RISCV64",     "CS_ARCH_RISCV",  ["CS_MODE_RISCV64"])

# --- PowerPC ---
add_if_available("PPC32 BE",    "CS_ARCH_PPC",    ["CS_MODE_32", "CS_MODE_BIG_ENDIAN"])
add_if_available("PPC32 LE",    "CS_ARCH_PPC",    ["CS_MODE_32", "CS_MODE_LITTLE_ENDIAN"])
add_if_available("PPC64 BE",    "CS_ARCH_PPC",    ["CS_MODE_64", "CS_MODE_BIG_ENDIAN"])
add_if_available("PPC64 LE",    "CS_ARCH_PPC",    ["CS_MODE_64", "CS_MODE_LITTLE_ENDIAN"])

# --- SPARC ---
add_if_available("SPARC32",     "CS_ARCH_SPARC",  ["CS_MODE_BIG_ENDIAN"])
add_if_available("SPARC V9",    "CS_ARCH_SPARC",  ["CS_MODE_V9", "CS_MODE_BIG_ENDIAN"])

# --- SystemZ / s390x ---
add_if_available("SystemZ",     "CS_ARCH_SYSZ",   [])

# --- XCore ---
add_if_available("XCORE",       "CS_ARCH_XCORE",  [])

# --- DSP / embedded / legacy cores ---
add_if_available("TMS320C64X",  "CS_ARCH_TMS320C64X", [])
add_if_available("M68K",        "CS_ARCH_M68K",   ["CS_MODE_BIG_ENDIAN"])
add_if_available("M680X",       "CS_ARCH_M680X",  [])
add_if_available("MOS65XX",     "CS_ARCH_MOS65XX",[])
add_if_available("SH",          "CS_ARCH_SH",     [])
add_if_available("TriCore",     "CS_ARCH_TRICORE",[])
add_if_available("HPPA",        "CS_ARCH_HPPA",   [])
add_if_available("Alpha",       "CS_ARCH_ALPHA",  [])
add_if_available("ARC",         "CS_ARCH_ARC",    [])
add_if_available("LoongArch",   "CS_ARCH_LOONGARCH", [])
add_if_available("Xtensa",      "CS_ARCH_XTENSA", [])

# --- “VM” style archs ---
add_if_available("BPF",         "CS_ARCH_BPF",    [])
add_if_available("eBPF",        "CS_ARCH_BPF",    ["CS_MODE_BPF_EXTENDED"])
add_if_available("EVM",         "CS_ARCH_EVM",    [])
add_if_available("WASM",        "CS_ARCH_WASM",   [])

print(f"Total candidate architectures loaded: {len(CANDIDATES)}")

# ---- The rest of your original logic stays the same ----
with open(BIN_PATH, "rb") as f:
    data = f.read()

best_arch = None
highest_ratio = 0.0

CHUNK_SIZE = 0x1000  # 4 KB
THRESHOLD = 0.30     # minimum ratio required to consider the architecture valid

for arch_name, md in CANDIDATES.items():
    total_insn = 0
    valid_insn = 0

    for offset in range(0, len(data), CHUNK_SIZE):
        chunk = data[offset:offset + CHUNK_SIZE]
        try:
            count = 0
            for _ in md.disasm(chunk, 0x0):
                count += 1
            valid_insn += count
            total_insn += max(len(chunk) // 4, 1)
        except Exception:
            continue

    if total_insn == 0:
        continue

    ratio = valid_insn / total_insn
    print(f"[{arch_name}] Valid instructions ratio: {ratio:.2f}")

    if ratio > highest_ratio:
        highest_ratio = ratio
        best_arch = arch_name

print("\n--- Results ---")
if best_arch is None or highest_ratio < THRESHOLD:
    print(f"Detected Architecture: Unknown (highest ratio {highest_ratio:.2f} < threshold {THRESHOLD})")
else:
    print(f"Detected Architecture: {best_arch} ({highest_ratio:.2f} valid instruction ratio)")
