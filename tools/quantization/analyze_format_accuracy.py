import json
import struct
import sys
import math
import random
from collections import defaultdict

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    '/opt/zen/wk/PhaseShift/models/Qwen3.5-4B-PSQ/model.safetensors'
QUANT = MODEL.replace('model.safetensors', 'phaseshift_quantization.json')
PLAIN = len(sys.argv) > 2
PLAIN_TENSOR = sys.argv[2] if PLAIN else None

CB10 = [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10]
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def ue4m3_decode(e):
    expf = e >> 3
    man = e & 7
    if expf != 0:
        return (1.0 + man / 8.0) * (2.0 ** (expf - 8))
    if man == 0:
        return 0.0
    msb = man.bit_length() - 1
    return (man / float(1 << msb)) * (2.0 ** (msb - 10))


UE4M3 = [ue4m3_decode(e) for e in range(256)]


def nearest(vals, x):
    best = vals[0]
    be = abs(vals[0] - x)
    for v in vals[1:]:
        e = abs(v - x)
        if e < be:
            be = e
            best = v
    return best


def quant_cb10_ue4m3(block):
    mx = max(abs(v) for v in block)
    if mx == 0.0:
        return [0.0] * len(block)
    target = mx / 10.0
    s = nearest(UE4M3, target)
    if s <= 0.0:
        s = UE4M3[1]
    return [nearest(CB10, v / s) * s for v in block]


def quant_mxfp4(block):
    mx = max(abs(v) for v in block)
    if mx == 0.0:
        return [0.0] * len(block)
    e = math.floor(math.log2(mx / 6.0))
    s = 2.0 ** e
    return [nearest(E2M1, abs(v) / s) * s * (1.0 if v >= 0 else -1.0) for v in block]


E4M3 = []
for e in range(128):
    exp = e >> 3
    man = e & 7
    if exp == 15 and man == 7:
        continue
    if exp:
        E4M3.append((1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    else:
        E4M3.append((man / 8.0) * (2.0 ** -6) if man else 0.0)
E4M3 = sorted(set(E4M3))


def quant_cb10_then_e4m3(block):
    rec = quant_cb10_ue4m3(block)
    out = []
    for v in rec:
        sgn = 1.0 if v >= 0 else -1.0
        av = abs(v)
        if av == 0.0:
            out.append(0.0)
            continue
        out.append(sgn * nearest(E4M3, av))
    return out


def quant_fold_norm(block):
    import math as _m
    rec = quant_cb10_ue4m3(block)
    mx = max((abs(v) for v in rec), default=0.0)
    if mx <= 0.0:
        return list(rec)
    g = 2.0 ** round(_m.log2(mx / 64.0))
    if g <= 0.0:
        g = 1.0
    out = []
    for v in rec:
        av = abs(v) / g
        nv = nearest(E4M3, av) * g
        out.append(nv if v >= 0 else -nv)
    return out


def quant_cb10_then_bf16(block):
    rec = quant_cb10_ue4m3(block)
    import struct as _s
    out = []
    for v in rec:
        u = _s.unpack('<I', _s.pack('<f', v))[0]
        out.append(_s.unpack('<f', _s.pack('<I', (u + 0x8000) & 0xFFFF0000))[0])
    return out


def quant_e4m3_scaled(block):
    mx = max((abs(v) for v in block), default=0.0)
    if mx <= 0.0:
        return [0.0] * len(block)
    sc = mx / 448.0
    out = []
    for v in block:
        av = abs(v) / sc
        nv = nearest(E4M3, av) * sc
        out.append(nv if v >= 0 else -nv)
    return out


def quant_int8(block):
    mx = max(abs(v) for v in block)
    if mx == 0.0:
        return [0.0] * len(block)
    s = mx / 127.0
    return [max(-127.0, min(127.0, round(v / s))) * s for v in block]


SCHEMES = [
    ('CB10 + UE4M3/16', 16, quant_cb10_ue4m3),
    ('CB10 + UE4M3/32', 32, quant_cb10_ue4m3),
    ('CB10 + UE4M3/64', 64, quant_cb10_ue4m3),
    ('MXFP4 E2M1 + E8M0/32', 32, quant_mxfp4),
    ('MXFP4 E2M1 + E8M0/16', 16, quant_mxfp4),
    ('MXFP4 E2M1 + E8M0/64', 64, quant_mxfp4),
    ('plain E4M3 per-16', 16, lambda b: [ (1.0 if v>=0 else -1.0)*nearest(E4M3, abs(v)) if v!=0 else 0.0 for v in b ]),
    ('plain E4M3 per-32', 32, lambda b: [ (1.0 if v>=0 else -1.0)*nearest(E4M3, abs(v)) if v!=0 else 0.0 for v in b ]),
    ('CB10+UE4M3/16 -> e4m3', 16, quant_cb10_then_e4m3),
    ('fold -> e4m3 + pow2/16', 16, quant_fold_norm),
    ('CB10+UE4M3/32 -> e4m3', 32, quant_cb10_then_e4m3),
    ('fold -> e4m3 + pow2/32', 32, quant_fold_norm),
    ('CB10+UE4M3/64 -> e4m3', 64, quant_cb10_then_e4m3),
    ('fold -> e4m3 + pow2/64', 64, quant_fold_norm),
    ('CB10+UE4M3/16 -> bf16', 16, quant_cb10_then_bf16),
    ('E4M3/16 + block scale', 16, quant_e4m3_scaled),
    ('E4M3/32 + block scale', 32, quant_e4m3_scaled),
    ('E4M3/128 + block scale', 128, quant_e4m3_scaled),
    ('E4M3 per-row scale', 5120, quant_e4m3_scaled),
    ('E4M3 per-row scale (k=2560)', 2560, quant_e4m3_scaled),
    ('INT8 per-16', 16, quant_int8),
    ('INT8 per-32', 32, quant_int8),
]


def load_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, base, entry):
    o0, o1 = entry['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + o0)
        return f.read(o1 - o0)


def main():
    hdr, base = load_header(MODEL)
    if PLAIN:
        e = hdr[PLAIN_TENSOR]
        out_features, k = e['shape']
        rng = random.Random(1234)
        stats = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
        print('plain bf16 tensor %s [%d, %d]' % (PLAIN_TENSOR, out_features, k))
        step = max(1, out_features // 48)
        with open(MODEL, 'rb') as f:
            for r in range(0, out_features, step):
                f.seek(base + e['data_offsets'][0] + r * k * 2)
                u16 = struct.unpack('<%dH' % k, f.read(k * 2))
                row = [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in u16]
                x = [rng.gauss(0.0, 1.0) for _ in range(k)]
                nx = math.sqrt(sum(v * v for v in x))
                x = [v / nx for v in x]
                ref = sum(a * b for a, b in zip(row, x))
                nrm = math.sqrt(sum(v * v for v in row))
                for label, blk, fn in SCHEMES:
                    if k % blk:
                        continue
                    num = 0.0
                    den = 0.0
                    dot = 0.0
                    for i in range(0, k, blk):
                        rec = fn(row[i:i + blk])
                        for a, b, xx in zip(rec, row[i:i + blk], x[i:i + blk]):
                            d = a - b
                            num += d * d
                            den += b * b
                            dot += a * xx
                    wrel = math.sqrt(num / den) if den > 0 else 0.0
                    drel = abs(dot - ref) / max(abs(ref), 1e-12)
                    rec = stats[label]
                    rec[0] += wrel
                    rec[1] += drel
                    rec[2] += 1
                    if wrel > rec[3]:
                        rec[3] = wrel
                    if drel > rec[4]:
                        rec[4] = drel
        print()
        print('%-26s %10s %10s %10s %10s' % ('scheme', 'w_rel', 'dot_rel', 'w_max', 'dot_max'))
        for label, blk, fn in SCHEMES:
            rec = stats.get(label)
            if rec is None or rec[2] == 0:
                continue
            n = rec[2]
            print('%-26s %10.5f %10.5f %10.5f %10.5f' % (
                label, rec[0] / n, rec[1] / n, rec[3], rec[4]))
        return
    quant = json.load(open(QUANT))['tensors']
    candidates = []
    for name, meta in quant.items():
        if meta.get('encoding') != 'bf16':
            continue
        shp = meta.get('logical_shape')
        if shp is None or len(shp) != 2:
            continue
        out_features, k = shp
        if out_features < 256 or k < 512:
            continue
        candidates.append((name, out_features, k))
    candidates.sort(key=lambda t: -(t[1] * t[2]))
    picked = candidates[:6]

    print('bf16 ground-truth tensors: %d, using:' % len(candidates))
    for n, o, k in picked:
        print('   %s [%d, %d]' % (n, o, k))

    rng = random.Random(1234)
    stats = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for name, out_features, k in picked:
        meta = quant[name]
        raw = read_tensor(MODEL, base, hdr[meta['data']['tensor']])
        step = max(1, out_features // 48)
        for r in range(0, out_features, step):
            row = struct.unpack_from('<%df' % k, b'\0' * 0, 0) if False else None
            off = r * k * 2
            u16 = struct.unpack_from('<%dH' % k, raw, off)
            row = [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in u16]
            x = [rng.gauss(0.0, 1.0) for _ in range(k)]
            nx = math.sqrt(sum(v * v for v in x))
            x = [v / nx for v in x]
            ref = sum(a * b for a, b in zip(row, x))
            nrm = math.sqrt(sum(v * v for v in row))
            for label, blk, fn in SCHEMES:
                if k % blk:
                    continue
                num = 0.0
                den = 0.0
                dot = 0.0
                for i in range(0, k, blk):
                    rec = fn(row[i:i + blk])
                    for a, b, xx in zip(rec, row[i:i + blk], x[i:i + blk]):
                        d = a - b
                        num += d * d
                        den += b * b
                        dot += a * xx
                wrel = math.sqrt(num / den) if den > 0 else 0.0
                drel = abs(dot - ref) / max(abs(ref), 1e-12)
                rec = stats[label]
                rec[0] += wrel
                rec[1] += drel
                rec[2] += 1
                if wrel > rec[3]:
                    rec[3] = wrel
                if drel > rec[4]:
                    rec[4] = drel

    print()
    print('%-26s %10s %10s %10s %10s' % ('scheme', 'w_rel', 'dot_rel', 'w_max', 'dot_max'))
    for label, blk, fn in SCHEMES:
        rec = stats.get(label)
        if rec is None or rec[2] == 0:
            continue
        n = rec[2]
        print('%-26s %10.5f %10.5f %10.5f %10.5f' % (
            label, rec[0] / n, rec[1] / n, rec[3], rec[4]))


main()
