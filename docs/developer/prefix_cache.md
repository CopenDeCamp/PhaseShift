# Prefix Cache

この文書は prefix cache の **protocol contract** を記述する。GPU-resident cache の実装は
削除済みであり、現在 `capabilities.prefix_cache` は `false` である。

## 現状

```
chat stable boundary
      ↓
prefix_cache_checkpoint_position
      ↓
JSONL contract
      ↓
capabilities.prefix_cache = false
```

GPU 実装は存在しない。KV / GDN の D2D save・restore、LRU eviction、DFlash2 ring、
arena 予約はいずれも削除された。復元手段は Git history である。

## 保有する contract

### server 側

Hugging Face chat template の stable prefix boundary を計算する責務は server に残る。

- `prompt_ids_with_cache_boundary()` が返す `PromptRender.cache_boundary`
- capability が `false` のとき server は常に `prefix_cache_checkpoint_position = 0` を送る
  （fail-closed。boundary 自体は計算してよいが compute へ要求しない）

### JSONL request

`phaseshift-compute --serve-stdio` の `generate` は常に
`prefix_cache_checkpoint_position` を受理する。ただし `0` 以外は
`event=error` / `code=invalid_argument` で拒否する。silent ignore はしない。
この validation は `main.hip` の serve protocol 境界で行い、
Qwen runtime や GPU へこの値は渡さない。

### JSONL response

`done` event は以下を必ず返す。

```json
{
  "event": "done",
  "request_id": 1,
  "generated_ids": [...],
  "finish_reason": "stop",
  "prompt_tokens": 123,
  "prefill_tokens": 123,
  "restored_tokens": 0,
  "cache_checkpoint_tokens": 0
}
```

現在は `restored_tokens = 0` / `cache_checkpoint_tokens = 0` で、
`prefill_tokens == prompt_tokens` である。prefix cache を再実装したときは
`restored_tokens > 0` / `prefill_tokens < prompt_tokens` に変えられる。

### capability

`ping` の応答:

```json
{
  "event": "pong",
  "protocol_version": 1,
  "capabilities": {
    "prefix_cache": false
  }
}
```

`capabilities` は map である。将来の capability はここへ追加し、
top-level boolean を増やさない。未知の key は server が無視してよい。

再実装時は `capabilities.prefix_cache` を `true` へ変更する。

## 検証

`tests/server/test_compute_prefix_contract.py` が上記 contract を固定する。
