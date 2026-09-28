#!/usr/bin/env python3
import re
import struct
import subprocess
import sys
import tempfile
import os


def carve_code_objects(binary, tmp):
    sec = subprocess.run(["readelf", "-x", ".hip_fatbin", binary],
                         capture_output=True, text=True).stdout
    raw = bytearray()
    for line in sec.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0].startswith("0x"):
            for hx in parts[1:5]:
                try:
                    raw += bytes.fromhex(hx)
                except ValueError:
                    pass
    if not raw:
        return []
    idx = [m.start() for m in re.finditer(b"\x7fELF", bytes(raw))]
    out = []
    for i, off in enumerate(idx):
        end = idx[i + 1] if i + 1 < len(idx) else len(raw)
        path = os.path.join(tmp, f"co{i}.o")
        with open(path, "wb") as f:
            f.write(bytes(raw[off:end]))
        out.append(path)
    return out


def read_section_bytes(path, section):
    sec = subprocess.run(["readelf", "-x", section, path],
                         capture_output=True, text=True).stdout
    data = {}
    for line in sec.splitlines():
        parts = line.split()
        if len(parts) >= 17 and parts[0].startswith("0x"):
            off = int(parts[0], 16)
            for col in range(16):
                hx = parts[1 + col]
                if re.fullmatch(r"[0-9a-f]{2}", hx):
                    data[off + col] = int(hx, 16)
    if not data:
        return None
    end = max(data) + 1
    return bytes(data.get(i, 0) for i in range(end))


def read_symbols(path):
    out = subprocess.run(["readelf", "-sW", path],
                         capture_output=True, text=True).stdout
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[3] != "FUNC" or not parts[7]:
            continue
        try:
            val = int(parts[1], 16)
            size = int(parts[2])
        except ValueError:
            continue
        syms.setdefault(parts[7], (val, size))
    return syms


def note_metadata(co_bytes):
    note = read_section_bytes_note(co_bytes)
    return note


def read_section_bytes_note(co_bytes):
    pos = co_bytes.find(b"\x7fELF")
    if pos < 0:
        return None
    e_shoff = struct.unpack_from("<Q", co_bytes, pos + 0x28)[0]
    e_shentsize = struct.unpack_from("<H", co_bytes, pos + 0x3A)[0]
    e_shnum = struct.unpack_from("<H", co_bytes, pos + 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", co_bytes, pos + 0x3E)[0]
    sections = []
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        name_off = struct.unpack_from("<I", co_bytes, base)[0]
        stype = struct.unpack_from("<I", co_bytes, base + 4)[0]
        shoff = struct.unpack_from("<Q", co_bytes, base + 0x18)[0]
        shsize = struct.unpack_from("<Q", co_bytes, base + 0x20)[0]
        sections.append((name_off, stype, shoff, shsize))
    strndx = sections[e_shstrndx]
    strtab = co_bytes[strndx[2]:strndx[2] + strndx[3]]

    def cstr(off):
        end = strtab.find(b"\x00", off)
        return strtab[off:end].decode()

    for name_off, stype, shoff, shsize in sections:
        if cstr(name_off) == ".note":
            blob = co_bytes[shoff:shoff + shsize]
            out = b""
            p = 0
            while p + 12 <= len(blob):
                namesz, descsz, ntype = struct.unpack_from("<III", blob, p)
                p += 12
                desc = blob[p + namesz:(p + namesz + descsz)]
                p += (namesz + 3) & ~3
                p += (descsz + 3) & ~3
                out += desc
            return out
    return None


def to_symtab_name(name):
    i = name.find("ZN")
    if i > 0:
        return "_" + name[i:]
    return name


KEYS = {
    b".sgpr_count": "sgpr_count",
    b".sgpr_spill_count": "sgpr_spill_count",
    b".vgpr_count": "vgpr_count",
    b".vgpr_spill_count": "vgpr_spill_count",
    b".wavefront_size": "wavefront_size",
    b".private_segment_fixed_size": "private_bytes",
    b".group_segment_fixed_size": "group_fixed_bytes",
    b".kernarg_segment_size": "kernarg_bytes",
    b".kernarg_segment_align": "kernarg_align",
}


def parse_kernel_records(meta):
    if meta is None:
        return []
    names = [m.start() for m in re.finditer(re.escape(b".name"), meta)]
    recs = []
    for i in names:
        content = i + len(b".name")
        if meta[content] >= 0xA0:
            start = content + 1
        else:
            start = content + 2
        end = meta.find(b".private_segment_fixed_size", start)
        if end < 0:
            end = len(meta)
        else:
            end -= 1
        name = meta[start:end].decode(errors="replace")
        recs.append({"name": name})
        for key, field in KEYS.items():
            occ = [m.start() for m in re.finditer(re.escape(key), meta)]
            for k in range(min(len(occ), len(recs))):
                j = occ[k]
                recs[k][field] = msgpack_uint(meta, j + len(key))
    return recs


def msgpack_uint(meta, pos):
    if pos >= len(meta):
        return -1
    b = meta[pos]
    if b < 0x80:
        return b
    if b == 0xCC and pos + 1 < len(meta):
        return meta[pos + 1]
    if b == 0xCD and pos + 2 < len(meta):
        return (meta[pos + 1] << 8) | meta[pos + 2]
    if b == 0xCE and pos + 4 < len(meta):
        return int.from_bytes(meta[pos + 1:pos + 5], "big")
    if b == 0xD0 and pos + 1 < len(meta):
        v = meta[pos + 1]
        return v - 256 if v >= 128 else v
    return -2


def main():
    if len(sys.argv) < 2:
        print("usage: extract_kernel_resources.py <binary> [filter]")
        sys.exit(2)
    binary = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else "gemm"
    tmp = tempfile.mkdtemp()
    rows = []
    for co in carve_code_objects(binary, tmp):
        co_bytes = open(co, "rb").read()
        meta = note_metadata(co_bytes)
        syms = read_symbols(co)
        for rec in parse_kernel_records(meta):
            if filt not in rec["name"]:
                continue
            size = syms.get(to_symtab_name(rec["name"]), (0, 0))[1]
            rows.append((rec["name"], rec, size))
    def demangle(sym):
        p = subprocess.run(["c++filt", to_symtab_name(sym)], capture_output=True, text=True)
        out = p.stdout.strip()
        return out if out else sym

    print("symbol,sgpr,sgpr_spill,vgpr,vgpr_spill,scratch_bytes,group_fixed_bytes,kernarg_bytes,kernarg_align,wavefront,code_size")
    for name, rec, size in sorted(rows):
        print("{},{},{},{},{},{},{},{},{},{},{}".format(
            demangle(name),
            rec.get("sgpr_count", -1), rec.get("sgpr_spill_count", -1),
            rec.get("vgpr_count", -1), rec.get("vgpr_spill_count", -1),
            rec.get("private_bytes", -1), rec.get("group_fixed_bytes", -1),
            rec.get("kernarg_bytes", -1), rec.get("kernarg_align", -1),
            rec.get("wavefront_size", -1), size))
    subprocess.run(["rm", "-rf", tmp])


if __name__ == "__main__":
    main()
