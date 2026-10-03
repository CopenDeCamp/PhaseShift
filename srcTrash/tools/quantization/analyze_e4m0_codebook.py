import json
import struct
import sys
import math

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else '/opt/zen/wk/PhaseShift/models/Qwen3.5-4B'
NROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 24

CB10 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])


def ue4m3_decode(e):
    expf = e >> 3
    man = e & 7
    if expf != 0:
        return (1.0 + man / 8.0) * (2.0 ** (expf - 8))
    if man == 0:
        return 0.0
    msb = man.bit_length() - 1
    return (man / float(1 << msb)) * (2.0 ** (msb - 10))


UE = np.array([ue4m3_decode(e) for e in range(256)])
UE_POS = np.unique(UE[UE > 0])


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_rows(shard, entry, nrows):
    hdr, base = read_header(shard)
    e = hdr[entry]
    _, k = e['shape']
    off = base + e['data_offsets'][0]
    out = []
    with open(shard, 'rb') as f:
        for r in range(min(nrows, e['shape'][0])):
            f.seek(off + r * k * 2)
            u16 = np.frombuffer(f.read(k * 2), dtype='<u2').astype(np.uint32)
            out.append((u16 << 16).view(np.float32))
    return np.stack(out)


def band_ue(s0, n=256):
    k0 = np.searchsorted(UE_POS, np.maximum(s0, 1e-30))
    idx = np.clip(k0[:, None] + (np.arange(n) - n // 2)[None, :], 0, UE_POS.size - 1)
    return UE_POS[idx]


def band_pow2(s0, n=256, unit=0.0625):
    k0 = np.log2(np.maximum(s0, 1e-30)) / unit
    ks = np.stack([np.round(k0) + (o - n // 2) for o in range(n)], axis=1)
    return np.power(2.0, ks * unit)


def quant_err(blocks, cb, scales):
    a = np.abs(blocks)
    sgn = np.sign(blocks)
    best = np.full(blocks.shape[0], np.inf)
    for k in range(scales.shape[1]):
        s = scales[:, k:k + 1]
        d = np.abs(a[:, :, None] / s[:, :, None] - cb[None, None, :])
        j = d.argmin(-1)
        rec = sgn * (cb[j] * s)
        err = ((rec - blocks) ** 2).sum(1)
        best = np.minimum(best, err)
    return best


def kmeans_cb(vals, levels=8, iters=60):
    v = np.sort(vals)
    qs = (np.arange(levels) + 0.5) / levels
    c = np.quantile(v, qs)
    c[0] = 0.0
    for _ in range(iters):
        j = np.abs(v[:, None] - c[None, :]).argmin(-1)
        nc = np.array([v[j == i].mean() if np.any(j == i) else c[i] for i in range(levels)])
        nc[0] = 0.0
        if np.allclose(nc, c):
            break
        c = nc
    c = np.sort(np.maximum(c, 0.0))
    return c


def main():
    idx = json.load(open(SRC + '/model.safetensors.index.json'))
    wm = idx['weight_map']
    names = [n for n in wm if 'layers.' in n and n.endswith('.weight')]
    picks = []
    for pat in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'self_attn.q_proj',
                'self_attn.o_proj', 'linear_attn.in_proj'):
        for n in sorted(names):
            if pat in n and n.endswith('.weight'):
                picks.append(n)
                break
    for l in (4, 16, 32):
        for n in sorted(names):
            if ('layers.%d.' % l) in n and 'mlp.gate_proj' in n:
                picks.append(n)
                break

    blocks = []
    for name in picks:
        shard = SRC + '/' + wm[name]
        rows = read_rows(shard, name, NROWS)
        b = rows[:, :rows.shape[1] // 16 * 16].reshape(-1, 16).astype(np.float64)
        blocks.append(b)
        print('read %-56s %s rows=%d blocks=%d' % (name, rows.shape, NROWS, b.shape[0]))
    blocks = np.concatenate(blocks, axis=0)
    nz = np.abs(blocks).max(1) > 0
    blocks = blocks[nz]
    print('total blocks %d elems %d' % (blocks.shape[0], blocks.size))

    s0 = np.abs(blocks).max(1) / 10.0

    norm = (np.abs(blocks) / s0[:, None]).reshape(-1)
    fit_blocks = blocks[:blocks.shape[0] // 2]
    fit_s0 = np.abs(fit_blocks).max(1) / 10.0
    fit_norm = (np.abs(fit_blocks) / fit_s0[:, None]).reshape(-1)
    LB = kmeans_cb(fit_norm, 8)
    LB = LB / LB.max() * 10.0
    NF4 = np.array([0.0, 0.4023, 0.7129, 1.1211, 1.8027, 3.0469, 5.5156, 10.0])
    print('cb10    ', np.round(CB10, 4))
    print('learned ', np.round(LB, 4))

    den = (blocks ** 2).sum()
    hist, edges = np.histogram(fit_norm, bins=10, range=(0.0, 10.0))
    print('normalized |w|/(max/10) histogram:',
          ' '.join('%d:%.0f%%' % (int(edges[i]), 100.0 * hist[i] / hist.sum()) for i in range(10)))
    variants = [
        ('cb10 + ue4m3 (current)', CB10, band_ue(s0, 256)),
        ('cb10 + e4m0 (16 exp)', CB10, band_pow2(s0, 16, 1.0)),
        ('learned + e4m0 (16 exp)', LB, band_pow2(s0, 16, 1.0)),
        ('cb10 + e4m0+sub4 (1/16)', CB10, band_pow2(s0, 256, 0.0625)),
        ('cb10 + free scale', CB10, band_pow2(s0, 256, 0.015625)),
        ('learned + ue4m3', LB, band_ue(s0, 256)),
        ('learned + e4m0+sub4', LB, band_pow2(s0, 256, 0.0625)),
        ('learned + free scale', LB, band_pow2(s0, 256, 0.015625)),
        ('nf4 + e4m0+sub4', NF4, band_pow2(s0, 256, 0.0625)),
    ]
    print()
    print('%-28s %14s %14s' % ('scheme', 'rms rel err', 'mean blk rel'))
    for label, cb, sc in variants:
        err = quant_err(blocks, cb, sc)
        rms = math.sqrt(err.sum() / den)
        blk = np.sqrt(err / (blocks ** 2).sum(1)).mean()
        print('%-28s %14.5f %14.5f' % (label, rms, blk))

    print()
    print('--- per-32 block, 16bit-class scale (Halogen Q4C-P hypothesis) ---')
    b32 = []
    for name in picks:
        rows = read_rows(SRC + '/' + wm[name], name, NROWS)
        k32 = rows.shape[1] // 32 * 32
        b32.append(rows[:, :k32].reshape(-1, 32).astype(np.float64))
    b32 = np.concatenate(b32, axis=0)
    b32 = b32[np.abs(b32).max(1) > 0]
    s32 = np.abs(b32).max(1) / 10.0
    den32 = (b32 ** 2).sum()
    den32b = (b32 ** 2).sum(1)
    v32 = [
        ('cb10 + ue4m3/16  (4.50bpw)', CB10, band_ue(s0, 256)),
        ('cb10 + ue4m3/32  (4.25bpw)', CB10, band_ue(s32, 256)),
        ('cb10 + f16like/32 (4.50bpw)', CB10, band_pow2(s32, 96, 1.0 / 1024.0)),
        ('cb10 + f16like/16 (5.00bpw)', CB10, band_pow2(s0, 96, 1.0 / 1024.0)),
        ('cb10 + bf16like/32(4.50bpw)', CB10, band_pow2(s32, 96, 1.0 / 256.0)),
        ('cb10 + e8m0/32   (4.25bpw)', CB10, band_pow2(s32, 16, 1.0)),
        ('learned + f16like/32 (4.5)', LB, band_pow2(s32, 96, 1.0 / 1024.0)),
    ]
    print('%-30s %14s %14s' % ('scheme', 'rms rel err', 'mean blk rel'))
    for label, cb, sc in v32:
        grid16 = label.endswith('(4.50bpw)') and ' /16' in label
        base = blocks if '/16 ' in label else b32
        deno = den if '/16 ' in label else den32
        denb = (blocks ** 2).sum(1) if '/16 ' in label else den32b
        err = quant_err(base, cb, sc)
        rms = math.sqrt(err.sum() / deno)
        blk = np.sqrt(err / denb).mean()
        print('%-30s %14.5f %14.5f' % (label, rms, blk))


main()
