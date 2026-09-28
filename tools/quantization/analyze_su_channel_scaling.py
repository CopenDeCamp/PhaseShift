import json
import math
import struct
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else '/opt/zen/wk/PhaseShift/models/Qwen3.5-4B'
NROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 24

CB10 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])


def ue4m3_decode(e):
    expf = e >> 3
    man = e & 7
    if expf:
        return (1.0 + man / 8.0) * 2.0 ** (expf - 8)
    if man == 0:
        return 0.0
    msb = man.bit_length() - 1
    return (man / float(1 << msb)) * 2.0 ** (msb - 10)


UE = np.array([ue4m3_decode(e) for e in range(256)])
UEP = UE[UE > 0]


def band_ue(s, n=256):
    k0 = np.searchsorted(UEP, np.maximum(s, 1e-30))
    idx = np.clip(k0[:, None] + (np.arange(n) - n // 2)[None, :], 0, UEP.size - 1)
    return UEP[idx]


def band_pow2(s, n=96, unit=1.0 / 256.0):
    k0 = np.log2(np.maximum(s, 1e-30)) / unit
    ks = np.stack([np.round(k0) + (o - n // 2) for o in range(n)], axis=1)
    return np.power(2.0, ks * unit)


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def rows_of(shard, entry, nr):
    h, base = read_header(shard)
    e = h[entry]
    k = e['shape'][1]
    off = base + e['data_offsets'][0]
    out = []
    with open(shard, 'rb') as f:
        for r in range(min(nr, e['shape'][0])):
            f.seek(off + r * k * 2)
            u = np.frombuffer(f.read(k * 2), dtype='<u2').astype(np.uint32)
            out.append((u << 16).view(np.float32))
    return np.stack(out)


def evaluate(blocks, cb, band, su=None):
    s0 = np.abs(blocks).max(1) / 10.0
    sc = band(s0)
    a = np.abs(blocks)
    sgn = np.sign(blocks)
    best = np.full(blocks.shape[0], np.inf)
    for k in range(sc.shape[1]):
        s = sc[:, k:k + 1]
        j = np.abs(a[:, :, None] / s[:, :, None] - cb[None, None, :]).argmin(-1)
        rec = sgn * (cb[j] * s)
        e = (rec - blocks) ** 2
        if su is not None:
            e = e * (su ** 2)
        best = np.minimum(best, e.sum(1))
    if su is None:
        den = (blocks ** 2).sum()
    else:
        den = ((blocks ** 2) * (su ** 2)).sum()
    return math.sqrt(float(best.sum()) / float(den))


def main():
    idx = json.load(open(SRC + '/model.safetensors.index.json'))
    wm = idx['weight_map']
    names = [n for n in wm if 'layers.' in n and n.endswith('.weight')]
    picks = []
    for pat in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'self_attn.q_proj',
                'self_attn.o_proj', 'linear_attn.in_proj'):
        got = [n for n in sorted(names) if pat in n]
        if got:
            picks.append(got[0])
    for l in (4, 16, 32):
        got = [n for n in sorted(names) if ('layers.%d.' % l) in n and 'mlp.gate_proj' in n]
        if got:
            picks.append(got[0])

    raw = [rows_of(SRC + '/' + wm[n], n, NROWS) for n in picks]
    print('tensors %d rows/tensor %d' % (len(raw), NROWS))

    def blocks_of(bs):
        out = []
        for r in raw:
            k = r.shape[1] // bs * bs
            out.append(r[:, :k].reshape(-1, bs).astype(np.float64))
        b = np.concatenate(out, axis=0)
        return b[np.abs(b).max(1) > 0]

    b16 = blocks_of(16)
    b32 = blocks_of(32)
    print('blocks16 %d  blocks32 %d' % (b16.shape[0], b32.shape[0]))

    schemes16 = [('cb10+ue4m3/16 (4.50)', CB10, band_ue)]
    schemes32 = [('cb10+ue4m3/32 (4.25)', CB10, band_ue),
                 ('cb10+bf16scale/32 (4.50)', CB10, lambda s: band_pow2(s, 96, 1.0 / 256.0)),
                 ('cb10+f16scale/32 (4.50)', CB10, lambda s: band_pow2(s, 200, 1.0 / 2048.0)),
                 ('cb10+e8m0/32 (4.25)', CB10, lambda s: band_pow2(s, 16, 1.0)),
                 ('cb10+free/32', CB10, lambda s: band_pow2(s, 200, 1.0 / 4096.0))]

    def make_dev(bs, alpha):
        B, S = [], []
        for r in raw:
            k = r.shape[1] // bs * bs
            w = r[:, :k].astype(np.float64)
            if alpha is None:
                su = np.ones(k)
            else:
                su = np.amax(np.abs(w), axis=0) ** alpha
                su = np.maximum(su, 1e-8)
                su = su / su.mean()
            bb = (w / su[None, :]).reshape(-1, bs)
            ss = np.tile(su, w.shape[0]).reshape(-1, bs)
            m = np.abs(bb).max(1) > 0
            B.append(bb[m])
            S.append(ss[m])
        return np.concatenate(B, 0), np.concatenate(S, 0)

    for alpha in (None, 0.5, 1.0):
        tag = 'none' if alpha is None else 'amax^%.1f' % alpha
        print()
        print('--- su = %s ---' % tag)
        for bs, schemes in ((16, schemes16), (32, schemes32)):
            blocks, su_el = make_dev(bs, alpha)
            for label, cb, band in schemes:
                rms = evaluate(blocks, cb, band, su_el)
                print('  %-28s bs=%d  rms=%.5f' % (label, bs, rms))


main()
