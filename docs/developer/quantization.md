# Quantization

source of truth: `include/phaseshift/quantization/**`,
`src/phaseshift/quantization/**`, `include/phaseshift/weights/**`,
`src/phaseshift/weights/**`, `src/phaseshift/io/**`, `src/apps/quantizer/**`。

canonical payload の stream / metadata / size formula / native preshuffle は
[psq_canonical_soa_payload.md](psq_canonical_soa_payload.md) を正本とする。
本ドキュメントは format の概要と offline / runtime の経路を示す。

## フォーマット

- `QuantSpecId` / `ComputeSpecId` — 量子化仕様ID（`quantization/quantization_types.h`）。
- `QuantFormatId` — weight storage / compute formatのID（`quantization/quant_format.h`）。
  numeric valueはin-memory専用で、file / manifestには保存しない。
  on-diskのencodingは `phaseshift_quantization.json` の文字列名で管理する。
- 対応 encoding: BF16 / PSQ4 / PSQ8 / FP8 E4M3 block128 / MXFP4。
  `quantization/fpx/types.h` の `WeightEncoding` と `QuantizedEncoding` が同じ番号体系を持つ。
  各formatの canonical stream、metadata、preshuffle は
  [psq_canonical_soa_payload.md](psq_canonical_soa_payload.md) を参照。
- self-contained quantized safetensors — `phaseshift-quantizer quantize` の出力。
  `phaseshift_quantization.json` 付きで `phaseshift-compute` が直接ロード可能。
  各tensorに `QuantSpecId` / `ComputeSpecId` / CRC32 payloadを保持する。
- PSQ4 / PSQ8 の reference mathは `phaseshift_quant_reference`（pure CPU）。
- iMatrix — `quantize` はPSIM fileを `--imatrix` で入力に使う
  （`imatrix_format.h` / `imatrix_format.cpp`）。PSIMの生成は `phaseshift-quantizer imatrix`
  が行う（`imatrix_collect.h` / `gpu_collector.hip` / `model_sites.cpp`）。
  runtime側は `LinearNode` 系dispatchの入力activationを `ImatrixProbe` としてprogramへ登録し、
  collectorがattachされている時だけ `Σx²` を蓄積する。

## Weight loading（generic layer）

重みロードは model非依存の `phaseshift_weights`（`weights/weight_loader.h`）が担当する。

```text
Qwen35 model（tensor name -> Qwen35ModelWeights field の bindingのみ）
    ↓
ps::weights loader
    safetensors / quantized safetensors の解決
    BF16 / PSQ4 / PSQ8 / FP8_BLOCK128 / MXFP4 の format解釈
    canonical payload validation / preshuffle_native / GPU upload
    MatrixWeight（quant_spec / compute_spec 含む）の生成
    ↓
fpx format / quant reference / io / core / gpu
```

- model側はencodingの分岐（`if PSQ4` 等）を持たない。
- BF16もquantized model内のweight formatの一種として同一経路でロードする
  （量子化されていないことを特別扱いしない）。
- `WeightLoadOptions.preshuffle` が load-time preshuffle のスイッチである。
  format別の native layout と `preshuffle = false` の制約は
  [psq_canonical_soa_payload.md](psq_canonical_soa_payload.md) を参照。
- offline側の `quantization/offline/qwen35_adapter.{h,cpp}` は
  tensor name → `TensorRole` の分類（Qwen3.5 tensor schema）のみを担い、
  量子化math（`quantizer.cpp` / `CanonicalQuantStore`）とは分離されている。
- architecture の判定と分類は `quantization/offline/adapter_dispatch.{h,cpp}` が仲介する。
  Qwen3.5 dense に加えて、DFlash2 draft（`DFlash2DraftModel` →
  `Architecture::DFlash2Draft`）を扱う。
  DFlash2 draftのroleは `dflash_backbone_linear`（PSQ4にするlinear行列）/
  `dflash_selector_linear` / `dflash_small`（BF16固定）の3つで、
  `quantize` は DFlash2 draftでは `--preset psq` と iMatrix無しのみ受け付ける。
- PPL / KLD / imatrix collectなどQwen35 runtimeを実行する評価機能は
  applicationレベルであり、model非依存化しない。

## Offline pipeline

```text
quantize  → self-contained quantized safetensors
verify    → 量子化model directoryの自己検証
kld       → BF16 vs 量子化のlogit KLD（PSKLDTOK corpus / shadow model）
imatrix   → 校正corpusのactivation統計をPSIMへ（quantizeの--imatrix入力）
ppl       → corpusのteacher-forced perplexity
```

`kld` はGPU shadow modelでlogit cacheをbuildし、
window / stride単位でKLDを評価する（`kld.cpp` / `gpu_kld.hip` /
`kld_shadow_model.cpp`）。

## Runtime contract

runtime は weight を `MatrixEncoding`（`weights/matrix_weight.h`）として扱い、
`WeightSlot.encoding` で linear launcher を選ぶ。encoding と KernelId の対応は
[qwen35.md](qwen35.md) を参照。

- `QuantSpecId` と `ComputeSpecId` の互換性は `fpx/runtime_resolution.h` の
  `is_gemm_compute_compatible` が定義し、量子化 storage ごとに実行契約が1つに定まる。
- 量子化 linear は対応 weight が native layout に preshuffle 済みであることを
  dispatch 時に検証する（未 preshuffle なら失敗する）。
- canonical payload を production GPU kernel が直接参照することはない。

## KV quantization

推論時のKVキャッシュは `PagedKVPool` の storage dtype として選択する。CLI では
`--kv-cache-dtype` で指定し、quantization pipeline とは独立。

| dtype | code | scale | head_dim |
| --- | --- | --- | --- |
| `bf16` | bf16 | なし | 任意 |
| `fp8_e4m3` | E4M3 8bit | per (token, kv_head) f32 | 任意 |
| `psq4` | CB10 4bit | per 32 value bf16 | 256 |
| `psq8` | E4M3 8bit | per 32 value bf16 | 256 |

- PSQ4 / PSQ8 は prefill / decode とも専用カーネルを持つ。長 context decode は
  BF16 と同じ KV split/reduce（flash-decoding 型 2 kernel、`max_visible >= 2048`）を使う。
- PSQ8 の code は weight PSQ8 と同じ E4M3。block は 32 value（`kPsq8BlocksPerHead = 8`）。
- prefix cache は全 KV dtype を cache する（`PrefixCache::create` は active pool と同じ
  dtype の cache pool を作る）。詳細は [prefix_cache.md](prefix_cache.md) を参照。
