import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open


def _mapped_name(name: str):
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("lm_head."):
        return name
    return None


def _load_index(model_dir: Path):
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        weight_map = idx["weight_map"]
        per_file = {}
        for k, f in weight_map.items():
            per_file.setdefault(f, []).append(k)
        return {f: per_file[f] for f in sorted(per_file)}
    single = model_dir / "model.safetensors"
    with safe_open(str(single), framework="pt", device="cpu") as f:
        return {single.name: list(f.keys())}


def _module_for(module, param_name):
    mod_path, _, _ = param_name.rpartition(".")
    return module.get_submodule(mod_path) if mod_path else module


def stream_load(model_dir, model_class, config, dtype=torch.bfloat16, gpu_ids=(0, 1, 2),
                mem_gib=22, verbose=True):
    model_dir = Path(model_dir)
    with init_empty_weights(include_buffers=False):
        model = model_class.from_config(config)

    names = [n for n, p in model.named_parameters()]
    shapes = {n: p.shape for n, p in model.named_parameters()}
    req = {n: p.requires_grad for n, p in model.named_parameters()}

    from accelerate.utils import infer_auto_device_map

    max_memory = {g: f"{mem_gib}GiB" for g in gpu_ids}
    nsc = list(getattr(model, "_no_split_modules", None) or [])
    for extra in ("Qwen3_5MLP", "Qwen3_5DecoderLayer", "Qwen3_5Attention", "Qwen3_5GatedDeltaNet"):
        if extra not in nsc:
            nsc.append(extra)
    dmap = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=nsc,
                                 dtype=dtype)

    def dev_of(name):
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            if key in dmap:
                v = dmap[key]
                return v
        return dmap.get("", gpu_ids[0])

    def to_dev(v):
        return f"cuda:{v}" if isinstance(v, int) else v

    used = {}
    owner = {}
    for name in names:
        v = dev_of(name)
        owner[name] = v
        used[v] = used.get(v, 0) + shapes[name].numel() * dtype.itemsize
    if verbose:
        for v in sorted(used, key=str):
            print(f"[load] {to_dev(v)}: {used[v] / 1e9:.2f} GB params", flush=True)

    for mod_name, mod in model.named_modules():
        for pname, p in list(mod._parameters.items()):
            if p is None or not p.is_meta:
                continue
            full = (mod_name + "." + pname) if mod_name else pname
            dev = to_dev(owner[full])
            mod._parameters[pname] = torch.nn.Parameter(
                torch.empty(p.shape, dtype=dtype, device=dev), requires_grad=req[full])

    params = dict(model.named_parameters())
    for name, b in model.named_buffers():
        if b.is_meta:
            continue
        mod = _module_for(model, name)
        dev = None
        for p in mod.parameters():
            dev = p.device
            break
        if dev is None:
            dev = f"cuda:{gpu_ids[-1]}"
        b.data = b.data.to(dev)

    per_file = _load_index(model_dir)
    missing = set(params.keys())
    unexpected = []
    loaded = 0
    for fname, keys in per_file.items():
        path = model_dir / fname
        with safe_open(str(path), framework="pt", device="cpu") as f:
            present = set(f.keys())
            for ck in keys:
                if ck not in present:
                    continue
                mapped = _mapped_name(ck)
                if mapped is None or mapped not in params:
                    unexpected.append(ck)
                    continue
                t = f.get_tensor(ck)
                dst = params[mapped]
                if tuple(t.shape) != tuple(dst.shape):
                    raise RuntimeError(
                        f"shape mismatch {ck} {tuple(t.shape)} vs {tuple(dst.shape)}")
                dst.data.copy_(t.to(dst.device, non_blocking=False))
                missing.discard(mapped)
                loaded += 1
    model.tie_weights()
    from accelerate import dispatch_model
    model = dispatch_model(model, device_map=dmap, main_device=f"cuda:{gpu_ids[0]}")
    model.eval()
    if verbose:
        print(f"[load] loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
        if missing:
            print("[load] missing sample:", sorted(missing)[:8], flush=True)
        if unexpected:
            print("[load] unexpected sample:", unexpected[:8], flush=True)
    return model, {"loaded": loaded, "missing": sorted(missing), "unexpected": unexpected}
