"""PSQ4 -- 4bit Codebook10 + per-32 bf16 scale (4.5 bpw).

block = 32 weight (最後の次元に沿って連続):

    bytes 0..15 : 4bit code 32 個。byte j の下位ニブル = w[2j]、上位ニブル = w[2j+1]
                  code 0..7  =  0, 1, 2, 3, 4, 6, 8, 10
                  code 8..15 = -0,-1,-2,-3,-4,-6,-8,-10
    bytes 16..17: bf16 scale s (little endian)

    w[i] ~= CB[code[i]] * s   (CB is CB10 or E2M1)

1 block = 18 byte / 32 weight = 4.5 bpw。

usage:
    psq4.py eval <model_dir> [rows_per_tensor] [alpha]
    psq4.py pack <model_dir> <out_prefix> [rows_per_tensor] [alpha]
"""

import json
import math
import os
import struct
import sys

import numpy as np

CB10 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
                 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -10.0])
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
CB = CB10
CMAX = 10.0


def set_codebook(name):
    global CB, CMAX
    if name == 'e2m1':
        CB, CMAX = E2M1, 6.0
    else:
        CB, CMAX = CB10, 10.0
    return CB.tolist()
BLOCK = 32
HEADER = 18


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
    wm = idx['weight_map']
    return {n: os.path.join(model_dir, f) for n, f in wm.items()}


def read_rows(path, entry, first, count, hdr=None, base=None):
    if hdr is None:
        hdr, base = read_header(path)
    e = hdr[entry]
    rows, k = e['shape']
    n = min(count, rows - first)
    off = base + e['data_offsets'][0] + first * k * 2
    with open(path, 'rb') as f:
        f.seek(off)
        buf = f.read(n * k * 2)
    u = np.frombuffer(buf, dtype='<u2').astype(np.uint32)
    return (u << 16).view(np.float32).reshape(n, k)


def scale_candidates(s0, half=32):
    fac = np.power(2.0, np.arange(-half, half + 1) / 64.0)
    return bf16(np.maximum(s0, 1e-30)[:, None] * fac[None, :])


def quant_block(blocks, half=32):
    """blocks (B,32) -> (codes (B,32) uint8, scale (B,) f32, err (B,))"""
    s0 = np.abs(blocks).max(1) / CMAX
    cand = scale_candidates(s0, half)
    sgn = np.sign(blocks)
    mag = np.abs(blocks)
    best_err = np.full(blocks.shape[0], np.inf)
    best_code = np.zeros(blocks.shape, dtype=np.uint8)
    best_scale = np.ones(blocks.shape[0], dtype=np.float32)
    for j in range(cand.shape[1]):
        s = cand[:, j:j + 1]
        idx = np.abs(mag[:, :, None] / s[:, :, None]
                     - np.abs(CB)[None, None, :]).argmin(-1)
        code = (idx | np.where(blocks < 0, 8, 0)).astype(np.uint8)
        rec = CB[code] * s
        err = ((rec - blocks) ** 2).sum(1)
        upd = err < best_err
        best_err = np.where(upd, err, best_err)
        best_scale = np.where(upd, s[:, 0], best_scale)
        best_code[upd] = code[upd]
    return best_code, best_scale, best_err


def pack_block(codes, scale):
    """codes (B,32) -> byte blob (B,18)"""
    b = np.zeros((codes.shape[0], HEADER), dtype=np.uint8)
    c = codes.astype(np.uint16)
    lo = c[:, 0::2]
    hi = c[:, 1::2]
    b[:, :16] = (lo | (hi << 4)).astype(np.uint8)
    b[:, 16:18] = (bf16(scale).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(-1, 2)
    return b


def unpack_block(b):
    """(B,18) -> (codes (B,32) uint8, scale (B,) f32)"""
    packed = b[:, :16]
    s = np.frombuffer(np.ascontiguousarray(b[:, 16:18]).tobytes(),
                      dtype='<u2').astype(np.uint32)
    scale = (s << 16).view(np.float32)
    codes = np.empty((b.shape[0], 32), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes, scale


def dequant_block(codes, scale):
    return CB[codes] * scale[:, None]


def rms(num, den):
    return math.sqrt(float(num) / float(den)) if den > 0 else 0.0


def pick_tensors(names):
    picks = []
    for pat in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'self_attn.q_proj',
                'self_attn.o_proj', 'linear_attn.in_proj'):
        got = [n for n in sorted(names) if 'layers.' in n and pat in n
               and n.endswith('.weight')]
        if got:
            picks.append(got[0])
    for l in (4, 16, 32):
        got = [n for n in sorted(names)
               if ('layers.%d.' % l) in n and 'mlp.gate_proj' in n and n.endswith('.weight')]
        if got:
            picks.append(got[0])
    return picks


def cmd_eval(model_dir, rows=24, alpha=0.5, codebook='cb10'):
    set_codebook(codebook)
    wm = tensor_names(model_dir)
    picks = pick_tensors(wm)
    num = den = 0.0
    for name in picks:
        w = read_rows(wm[name], name, 0, rows).astype(np.float64)
        k = w.shape[1] // BLOCK * BLOCK
        w = w[:, :k]
        if alpha > 0:
            su = np.maximum(np.abs(w).max(0), 1e-8) ** alpha
            su = su / su.mean()
        else:
            su = np.ones(w.shape[1])
        wn = w / su[None, :]
        b = wn.reshape(-1, BLOCK)
        keep = np.abs(b).max(1) > 0
        b = b[keep]
        su_el = np.tile(su, w.shape[0]).reshape(-1, BLOCK)[keep]
        codes, scale, err = quant_block(b)
        rec = dequant_block(codes, scale)
        e = ((rec - b) ** 2 * su_el ** 2).sum()
        d = ((b ** 2) * su_el ** 2).sum()
        num += e
        den += d
        print('  %-52s blocks=%6d rms=%.5f' % (name[:52], b.shape[0], rms(e, d)))
    print('PSQ4 total rms rel = %.5f (alpha=%.1f, rows=%d, cb=%s)' % (rms(num, den), alpha, rows, codebook))


def cmd_pack(model_dir, out_prefix, rows=0, alpha=0.5, codebook='cb10'):
    set_codebook(codebook)
    wm = tensor_names(model_dir)
    entries = []
    total_bytes = 0
    with open(out_prefix + '.psq4', 'wb') as f:
        for i, name in enumerate(sorted(wm)):
            shard = wm[name]
            hdr, base = read_header(shard)
            e = hdr.get(name)
            if e is None or len(e['shape']) < 2:
                continue
            R, K = e['shape'][0], e['shape'][1]
            if rows:
                R = min(R, rows)
            k = K // BLOCK * BLOCK
            if k == 0:
                continue
            off = 0
            num = den = 0.0
            su = None
            if alpha > 0:
                w0 = read_rows(shard, name, 0, min(R, 16), hdr, base).astype(np.float64)
                su = np.maximum(np.abs(w0[:, :k]).max(0), 1e-8) ** alpha
                su = su / su.mean()
            for r0 in range(0, R, 16):
                w = read_rows(shard, name, r0, min(16, R - r0), hdr, base).astype(np.float64)
                w = w[:, :k]
                if su is None:
                    wn = w
                    swe = np.ones(k)
                else:
                    wn = w / su[None, :]
                    swe = su
                b = wn.reshape(-1, BLOCK)
                keep = np.abs(b).max(1) > 0
                if not keep.any():
                    continue
                bb = b[keep]
                codes, scale, err = quant_block(bb)
                blob = pack_block(codes, scale)
                f.write(blob.tobytes())
                off += blob.shape[0] * HEADER
                rec = dequant_block(codes, scale)
                ee = bb - rec
                num += float((ee ** 2).sum())
                den += float((bb ** 2).sum())
            entries.append({'name': name, 'shape': [int(R), int(k)], 'bytes': int(off),
                            'rms': rms(num, den)})
            total_bytes += off
            if i % 50 == 0:
                print('  packed %-52s %8d B rms=%.5f' % (name[:52], off, entries[-1]['rms']))
    json.dump({'format': 'PSQ4', 'version': 1, 'codebook_name': codebook, 'block': BLOCK, 'header_bytes': HEADER,
               'bpw': 4.5, 'codebook': CB.tolist(), 'alpha': alpha,
               'tensors': entries, 'payload_bytes': total_bytes},
              open(out_prefix + '.json', 'w'), indent=1)
    print('PSQ4 packed %d tensors, %d bytes (%.4f bpw effective)'
          % (len(entries), total_bytes, 18.0 * 8 / 32))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'eval':
        cmd_eval(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24,
                 float(sys.argv[4]) if len(sys.argv) > 4 else 0.5,
                 sys.argv[5] if len(sys.argv) > 5 else 'cb10')
    elif mode == 'pack':
        cmd_pack(sys.argv[2], sys.argv[3],
                 int(sys.argv[4]) if len(sys.argv) > 4 else 0,
                 float(sys.argv[5]) if len(sys.argv) > 5 else 0.5,
                 sys.argv[6] if len(sys.argv) > 6 else 'cb10')
    else:
        print(__doc__)
