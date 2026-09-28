# GPU-resident Prefix Cache

この文書は PhaseShift runtime の GPU-resident Agent prefix cache を記述する。

## 目的

Agent の multi-turn 推論では、次の turn の入力が前の turn のある token prefix を含む。
prefix cache はこの共通 prefix の KV と GDN state を保持し、次 request で再 prefill せずに
D2D restore する。

```
request が stable prompt boundary まで prefill 完了
    ↓
active KV  -> cache KV pool   (D2D)
active GDN -> cache GDN pool  (D2D)

next request (hit)
    ↓
cache KV  -> active KV        (D2D)
cache GDN -> active GDN       (D2D)
    ↓
remaining prefill only
```

Host へ KV / GDN payload を退避しない。Host に残るのは token IDs と小さな metadata だけである。

## Checkpoint policy

snapshot には 2 種類ある。

- **prompt-boundary checkpoint**: request が明示的な prompt boundary を持つ場合にその位置で
  保存する。token 列は `input_tokens[0:boundary]` exact で、generated token を含まない。
  multi-turn で再利用されるのはこちらである。
- **terminal checkpoint**: 明示 boundary を持たない request が `Eos` / `MaxNewTokens` で
  終了したときに保存する。token 列は `input_tokens + committed generated` である。
  direct compute の exact-continuation 用途を維持する。

優先順位:

```
explicit boundary > 0:
  prompt-boundary checkpoint のみ（terminal は保存しない）
boundary == 0:
  terminal checkpoint
```

1 request で 2 つの GDN snapshot を持たないため、明示 boundary 時は terminal を抑制する。

## Storage model

cache は active request の pool と分離した専用 pool を持つ。

```
active PagedKVPool   --D2D-->  cache PagedKVPool
active GdnStatePool  --D2D-->  cache GdnStatePool
```

- `cache_kv_pool` は active pool と同じ `page_tokens`, attention layers, kv heads,
  head dim, KV dtype を使う。
- `cache_gdn_pool` は active と同じ `GdnStatePoolLayout` を使う。
- cache page と active page は ID 空間を共有しない。`PrefixCheckpoint` は
  `cache_pages` / `cache_state_slot` として保持する。
- reference count や copy-on-write は無い。restore は cache を read-only source として
  active へ D2D copy するため、同一 entry への concurrent / repeated hit が可能である。

### VRAM model

```
total arena usage
  = model
  + active KV pool
  + active GDN pool
  + prefix KV pool
  + prefix GDN pool
  + executor scratch
```

prefix cache は server では既定で有効であり、arena 予約量を含めて計算する。arena に入らない
設定は startup failure であり、silent shrink しない。`kv_cache_capacity_tokens`（active）とは
別物である。低 VRAM では `--prefix-cache-capacity-tokens 0` で完全に無効化できる。

### dtype

`bf16` / `fp8_e4m3` / `psq4` / `psq8` を cache する。FP8 は K/V bytes に加えて
K/V scales（f32）を、PSQ4 / PSQ8 は code bytes に加えて per-block scale（bf16）を
D2D copy する。dtype による silent disable は行わない。

PSQ4 / PSQ8 の `head_dim == 256` 制約は active pool 側のものであり、cache pool は
同じ dtype・同じ geometry を active pool から受け継ぐ。

## Data movement

save / restore は `hipMemcpyAsync(..., hipMemcpyDeviceToDevice, stream)` のみを使う。
production の prefix cache source に `hipMemcpyDeviceToHost` / `hipMemcpyHostToDevice` は
存在しない（test の verification copy は除く）。

追加の `hipStreamSynchronize` / `hipDeviceSynchronize` も置かない。save / restore と
後続 kernel は runtime の単一 stream へ enqueue し、stream ordering で保証する。

```
same-stream ownership contract:
  prefix cache copy と runtime execution は Qwen35ComputeRuntime::stream_ で行う。
  別 stream copy は行わない。
```

save は `finish_request` 内で `release_paged_sequence_state()` より前に enqueue する。
release 側の既存 sync により save D2D 完了後に active page が解放されるため、
prefix cache 側で追加 sync は不要である。

## Policy

- lookup は token 列の exact full prefix comparison。hash だけで hit にしない。
- 同じ token 列が既に存在する場合は entry を増やさず MRU へ promote する。
- insert / hit は MRU、eviction は true LRU。
- lookup は request より厳密に短い checkpoint だけを restore 候補にする。prompt が cache
  boundary とちょうど同じ長さでも、より短い有効な prefix へ restore する。
- cache 対象は prompt boundary、または boundary が無い場合の `Eos` / `MaxNewTokens`。
  `Cancelled` / `Error` は terminal を保存しない。
- prompt boundary まで正常 prefill 済みなら、その後 generation が cancel / error になっても
  その snapshot は保持する。snapshot は deterministic な input prefix state だけを持つ。
  boundary 到達前の未完成 state は保存しない。
- 短い prefix（既定 64 token 未満）は cache しない。
- cache capacity を超える prefix、空きが確保できない場合は benign skip であり request を
  失敗させない。HIP copy failure は GPU/runtime error として扱い、黙って無視しない。
- partial final page も page 全体を cache する。restore 後は `position` だけを commit し、
  page 内の position 以降は未使用として扱う。append は active 側へ copy 済みなので
  cache page を変更しない。

## Turn boundary

server は `add_generation_prompt=False` で rendering した会話履歴を stable prefix とし、それが
`add_generation_prompt=True` の full prompt の exact token prefix である場合だけ、その長さを
prompt boundary として送る。exact prefix でなければ boundary=0 として通常 generation を続ける。
`tools` と `tool_choice` は両方の render で同一に渡す。

Qwen3.5 chat template の generation prompt には thinking prelude が付く一方、multi-turn の
assistant message rendering には付かない。これは generation prompt の末尾にのみ現れるため、
add_generation_prompt=False の履歴は次 turn の full prompt の prefix として一致する。

```
turn 1 full   = history + generation prompt
prompt boundary = len(history)

turn 2 full   = history + rendered assistant + next user + generation prompt
                  ^^^^^^^ boundary が一致 → restore
```

したがって通常の Chat multi-turn、tool round-trip、Responses function-call output で hit する。
direct compute で boundary を指定しない場合は従来どおり terminal checkpoint を使う。

## Statistics / trace

`PrefixCacheStats` は lookups / hits / misses / inserts / duplicate_hits / evictions /
skipped_too_large / restored_tokens / saved_tokens / save_d2d_bytes / restore_d2d_bytes を
持つ。runtime owner は単一 thread なので atomic は使わない。

`PHASESHIFT_PREFIX_CACHE_TRACE=1` のとき stderr へ次を出す（stdout へは出さない）。

```
PREFIX_CACHE_HIT tokens=...
PREFIX_CACHE_MISS
PREFIX_CACHE_CHECKPOINT tokens=... pages=...   (prompt-boundary snapshot)
PREFIX_CACHE_INSERT tokens=... pages=...       (terminal snapshot)
PREFIX_CACHE_SKIP tokens=... reason=min_prefix|too_large|no_room
PREFIX_CACHE_EVICT tokens=...
PREFIX_CACHE_RESTORE tokens=...
```

`PREFIX_CACHE_SKIP` は benign skip（短すぎる prefix、capacity 超過、空きなし）を示す。
request は失敗させない。Agent の system prompt + tool schema は数 1000 token に達するため、
server の default capacity が小さすぎると snapshot はこの理由で skip される。

backend が stable boundary を確定できない場合は、trace 有効時のみ stderr へ
`PREFIX_CACHE_BOUNDARY_UNAVAILABLE <reason>` を出す。request は失敗させない。

## Configuration

```
--prefix-cache-capacity-tokens N   GPU prefix cache KV token capacity (0 = disabled)
--prefix-cache-max-entries N       max cached prefixes / GDN snapshots
```

server default は `16384 / 8` で有効である。`capacity=0` は完全 disabled であり、低 VRAM や
benchmark 用途ではこれを指定する。`capacity>0 && max_entries=0` は invalid である。
compute 単体の default は `0`（disabled）のままである。

## Non-goals

KV page reference counting、shared page ownership、copy-on-write、zero-copy restore、
distributed / cross-GPU / disk / host prefix cache、semantic prompt cache、hash-only
approximate matching はこの段階では扱わない。将来の multi-stream 化には event dependency
が必要になる。
