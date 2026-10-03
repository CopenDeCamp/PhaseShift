"""weight scale 表現スキーム別の再構成 RMS（PSQ4 E2M1 / PSQ8 E4M3）。

baseline: per-32 bf16 scale
schemes : per-32 scale を coarse な log2 grid / exponent+residual に置換
          fold: scale を operand へ畳む（operand = fp8(code * 2^(p/2^pbits)), 指数は整数 e）

usage:
    analyze_scale_schemes.py <model_dir> [rows] [alpha] [e2m1|e4m3]
"""

import json
import math
import os
import struct
import sys

import numpy as np

E2M1_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = np.array(E2M1_MAGS + [-m for m in E2M1_MAGS[1:]])
CB10_MAGS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]


def make_fmt(expb, manb, bias):
    vals = {0.0}
    emax = (1 << expb) - 1
    for e in range(1, emax + 1):
        for m in range(1 << manb):
            if e == emax and m == (1 << manb) - 1:
                continue  # inf / nan
            vals.add((1.0 + m / (1 << manb)) * 2.0 ** (e - bias))
    for m in range(1, 1 << manb):
        vals.add((m / (1 << manb)) * 2.0 ** (1 - bias))
    return np.array(sorted(vals))


E4M3 = make_fmt(4, 3, 7)
E5M2 = make_fmt(5, 2, 15)

BLOCK = 32
CODEBOOK = None
CMAX = None


def set_format(name):
    global CODEBOOK, CMAX
    if name == 'e4m3':
        CODEBOOK = E4M3[E4M3 >= 0.0]
        CMAX = float(CODEBOOK[-1])
    elif name == 'cb10':
        CODEBOOK = np.array(CB10_MAGS)
        CMAX = 10.0
    else:
        CODEBOOK = np.array(E2M1_MAGS)
        CMAX = 6.0


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) & 0xFFFF0000).view(np.float32)


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_names(model_dir):
    idx = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))
    return {n: os.path.join(model_dir, f) for n, f in idx['weight_map'].items()}


def read_rows(path, first, count, hdr, base):
    e = hdr[path]
    rows, k = e['shape']
    n = min(count, rows - first)
    off = base + e['data_offsets'][0] + first * k * 2
    with open(path, 'rb') as f:
        f.seek(off)
        buf = f.read(n * k * 2)
    u = np.frombuffer(buf, dtype='<u2').astype(np.uint32)
    return (u << 16).view(np.float32).reshape(n, k)


def blocks_of(w):
    k = w.shape[1] // BLOCK * BLOCK
    return w[:, :k].reshape(-1, BLOCK).astype(np.float64)


def s0_of(b):
    return np.maximum(np.abs(b).max(1) / CMAX, 1e-30)


def quant_err(b, s, codebook):
    mag = np.abs(b)
    idx = np.abs(mag[:, :, None] / s[:, None, None]
                 - codebook[None, None, :]).argmin(-1)
    rec = codebook[idx] * s[:, None]
    return ((rec - mag) ** 2).sum(1)


def eval_grid(b, step, span, codebook):
    s0 = s0_of(b)
    c = np.round(np.log2(s0) / step)[:, None] + np.arange(-span, span + 1)[None, :]
    cand = np.power(2.0, c * step)
    best = np.full(b.shape[0], np.inf)
    for j in range(cand.shape[1]):
        best = np.minimum(best, quant_err(b, cand[:, j], codebook))
    return best


def eval_residual(b, group, span, codebook, fine=8):
    s0 = s0_of(b)
    ng = b.shape[0] // group * group
    b = b[:ng]
    s0g = s0[:ng].reshape(-1, group)
    gm = np.exp(np.log(s0g).mean(1))
    rcand = gm[:, None] * np.power(2.0, np.arange(-span, span + 1)[None, :] / fine)
    best = np.full(ng // group, np.inf)
    for j in range(rcand.shape[1]):
        r = np.maximum(rcand[:, j], 1e-30)
        e = np.round(np.log2(s0g / r[:, None]))
        scale = (np.power(2.0, e) * r[:, None]).reshape(-1)
        best = np.minimum(best, quant_err(b, scale, codebook).reshape(-1, group).sum(1))
    return best


def nearest_arr(vals, x):
    return vals[np.abs(vals - x).argmin()]


def round_scale(x, fmt):
    """scale を保存形式へ丸める。fmt: bf16 / f16 / e4m3 / free。"""
    if fmt == 'bf16':
        return bf16(x)
    if fmt == 'f16':
        return np.asarray(x, dtype=np.float16).astype(np.float64)
    if fmt == 'e4m3':
        return round_fmt(x, E4M3)
    if fmt == 'e5m2':
        return round_fmt(x, E5M2)
    return x


def eval_scale_fmt(b, fmt, codebook):
    """scale を fmt 形式で保存。粗 exploration で octave を決め、そこで最良を探す。"""
    s0 = s0_of(b)
    lo = np.log2(s0)
    coarse = np.floor(lo * 8.0)[:, None] / 8.0 + np.arange(-32, 33)[None, :] / 8.0
    cand = round_scale(np.power(2.0, coarse), fmt)
    best = np.full(b.shape[0], np.inf)
    bj = np.zeros(b.shape[0], dtype=np.int64)
    for j in range(cand.shape[1]):
        e = quant_err(b, cand[:, j], codebook)
        m = e < best
        best = np.where(m, e, best)
        bj = np.where(m, j, bj)
    center = coarse[np.arange(b.shape[0]), bj]
    step, span = {'bf16': (1 / 256.0, 32), 'f16': (1 / 1024.0, 48),
                  'e4m3': (1 / 8.0, 4), 'e5m2': (1 / 4.0, 4),
                  'free': (1 / 2048.0, 300)}[fmt]
    fine = center[:, None] + np.arange(-span, span + 1)[None, :] * step
    cand = round_scale(np.power(2.0, fine), fmt)
    for j in range(cand.shape[1]):
        best = np.minimum(best, quant_err(b, cand[:, j], codebook))
    return best


def round_fmt(x, fmt):
    return fmt[np.abs(x[..., None] - fmt).argmin(-1)]


def eval_folded(b, pbits, fmt, codebook):
    """operand = fmt(codebook * 2^(p/2^pbits)), 指数 e は bit 加算後に再度 fmt へ丸め。"""
    s0 = s0_of(b)
    e0 = np.round(np.log2(s0))
    mag = np.abs(b)
    best = np.full(b.shape[0], np.inf)
    npc = 1 << pbits
    for de in range(-3, 4):
        twoe = np.power(2.0, e0 + de)
        for p in range(npc):
            ph = 2.0 ** (p / float(npc))
            lut = np.array([nearest_arr(fmt, c * ph) for c in codebook])
            cand = round_fmt(lut[None, :] * twoe[:, None], fmt)
            idx = np.abs(mag[:, :, None] - cand[:, None, :]).argmin(-1)
            sel = cand[np.arange(b.shape[0])[:, None], idx]
            best = np.minimum(best, ((sel - mag) ** 2).sum(1))
    return best


def schemes_for(fmtname, foldfmt):
    s = [
        ('baseline bf16/32', 'grid', (1 / 64.0, 48), None),
        ('sfmt bf16/32', 'sfmt', 'bf16', None),
        ('sfmt f16/32', 'sfmt', 'f16', None),
        ('sfmt e4m3/32', 'sfmt', 'e4m3', None),
        ('sfmt e5m2/32', 'sfmt', 'e5m2', None),
        ('E8M0/32', 'grid', (1.0, 42), None),
        ('log2 half/32', 'grid', (0.5, 42), None),
        ('log2 quarter/32', 'grid', (0.25, 42), None),
        ('log2 eighth/32', 'grid', (0.125, 42), None),
        ('exp/32+bf16 r/128', 'res', (4, 24), None),
        ('exp/32+bf16 r/256', 'res', (8, 24), None),
        ('FOLD pow2/32 ' + foldfmt, 'fold', (0, foldfmt), None),
        ('FOLD quarter/32 ' + foldfmt, 'fold', (2, foldfmt), None),
        ('FOLD eighth/32 ' + foldfmt, 'fold', (3, foldfmt), None),
        ('FOLD eighth/32 e5m2', 'fold', (3, 'e5m2'), None),
    ]
    return s


def main():
    model_dir = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    fmtname = sys.argv[4] if len(sys.argv) > 4 else 'e2m1'
    set_format(fmtname)
    fmts = {'e4m3': E4M3, 'e5m2': E5M2}

    wm = tensor_names(model_dir)
    pats = ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'o_proj',
            'in_proj_qkv', 'in_proj_gate')
    picks = []
    for pat in pats:
        got = [n for n in sorted(wm) if 'layers.' in n and pat in n and n.endswith('.weight')]
        if got:
            picks.append(got[0])
    picks = sorted(set(picks))

    schemes = schemes_for(fmtname, 'e4m3')
    if len(sys.argv) > 5:
        toks = sys.argv[5].split(',')
        schemes = [s for s in schemes
                   if s[0] == 'baseline bf16/32' or any(t in s[0] for t in toks)]
    agg = {label: [0.0, 0.0] for label, _, _, _ in schemes}
    print('format=%s rows=%d alpha=%.2f tensors=%d codebook=%d' %
          (fmtname, rows, alpha, len(picks), len(CODEBOOK)))
    for name in picks:
        path = wm[name]
        hdr, base = read_header(path)
        w = read_rows(path, 0, rows, {path: hdr[name]}, base).astype(np.float64)
        k = w.shape[1] // BLOCK * BLOCK
        w = w[:, :k]
        if alpha > 0:
            su = np.maximum(np.abs(w).max(0), 1e-8) ** alpha
            su = su / su.mean()
            w = w / su[None, :]
        b = blocks_of(w)
        keep = np.abs(b).max(1) > 0
        b = b[keep]
        den = float((b ** 2).sum())
        for label, kind, arg, _ in schemes:
            if kind == 'grid':
                err = eval_grid(b, arg[0], arg[1], CODEBOOK).sum()
            elif kind == 'sfmt':
                err = eval_scale_fmt(b, arg, CODEBOOK).sum()
            elif kind == 'fold':
                err = eval_folded(b, arg[0], fmts[arg[1]], CODEBOOK).sum()
            else:
                err = eval_residual(b, arg[0], arg[1], CODEBOOK).sum()
            agg[label][0] += err
            agg[label][1] += den

    print('\n%-32s %10s %12s' % ('scheme', 'rms_rel', 'vs baseline'))
    base = None
    for label, kind, arg, _ in schemes:
        num, den = agg[label]
        r = math.sqrt(num / den) if den else 0.0
        if base is None:
            base = r
        print('%-32s %10.5f    %+7.1f%%' % (label, r, 100.0 * (r / base - 1.0)))


if __name__ == '__main__':
    main()
