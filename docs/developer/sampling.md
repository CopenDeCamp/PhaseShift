# Target LM sampling

Qwen3.5 / Qwen3.8 target runtime の sampling 契約。DFlash2 の CandidateSelector や
MTP の sampling とは別物で、target LM の logits から次 token を決める経路だけを扱う。

## 範囲

- 対象: target logits → greedy / temperature / top-k / top-p → next token
- 対象外: top-k 以外の truncation（min-p / typical-p）、penalty 系、beam search、
  DFlash2 CandidateSelector
- DFlash2 verify は `SamplingConfig` と `sample_index` を `ScheduledRequest` に渡す。
  stochastic の verify は target の sample 結果と draft を比較する sample-and-compare
  で、round ごとに行消費 row 数だけ `sample_index` を進める
  （[dflash2.md](dflash2.md) を参照）。

## SamplingConfig

`include/phaseshift/models/qwen35/runtime/sampling_params.h`

| field | default | 意味 |
| --- | --- | --- |
| `temperature` | `0.0f` | `0` は greedy。`> 0` で stochastic |
| `top_p` | `1.0f` | nucleus の累積 mass 閾値。`(0, 1]` |
| `top_k` | `0` | 上位 k token だけを残す。`0` は無効 |
| `seed` | `0` | RNG seed |

validation:

- `temperature` は finite かつ `>= 0`
- `top_p` は finite かつ `> 0` かつ `<= 1`
- `top_k` は任意（`0` が無効。語彙数以上を指定しても `0` と同じ）

`temperature == 0` のとき `top_p` と `top_k` は結果に影響しない。

## mode 決定

`make_sampling_request(config, sample, sample_index)` が次のように決める。

| sample | temperature | mode |
| --- | --- | --- |
| false | 任意 | `None` |
| true | `0` | `Greedy` |
| true | `> 0` | `Stochastic` |

`SamplingMode` は `DeviceSamplingParams::mode` にそのまま入る
（`None = 0` / `Greedy = 1` / `Stochastic = 2`）。

## sample_index

`sample_index = request.generated_tokens`（その request が既に生成した token 数）。

- prefill 後の最初の sample は `0`
- 次は `1`

`sequence.position` は使わない。prefix cache の利用状況や batch 構成が変わっても
同じ prompt + 同じ seed なら同じ sampling sequence になるようにするため。

## RNG contract

`include/phaseshift/models/qwen35/kernels/optimized/sampling_common.h`

counter-based hash（SplitMix64 系）で、key は

```
(seed, sample_index, attempt, token_id)
```

だけ。**batch index、sequence slot、kernel partition id、block index は key に入れない。**
したがって同じ `(seed, sample_index)` と同じ logits なら、batch のどこに置かれても
同じ token を sample する。

uniform は `(uint32_bits + 0.5) * 2^-32` で `(0, 1)` に入る。integer hash は host と
device で bit-exact（`test_qwen35_sampling_rng`）。

## Gumbel-Max

stochastic では full vocab sort を使わない。各 token について

```
u_i = RNG(seed, sample_index, attempt, i)
g_i = -log(-log(u_i))
s_i = (l_i - max_logit) / temperature + g_i
candidate = argmax_i(s_i)
```

とする。`argmax` は score が同値なら token id が小さい方を選ぶ。これで
`softmax(logits / temperature)` からの categorical sample と厳密に同じ分布になる。

## top-k / top-p membership

candidate `c` について、full softmax 上の mass を

```
w_i = exp((l_i - max_logit) / temperature)
Z   = Σ_i w_i
```

とし、次の両方を満たすときだけ accept する。

```
better_count(c) < top_k        (top_k > 0 のとき)
better_mass(c)  < top_p * Z    (top_p < 1 のとき)
```

ここで `better` は

```
l_i > l_c  または  (l_i == l_c かつ i < c)
```

を満たす `i`。つまり `better_count` は「c より上位の token 数」、
`better_mass` はその mass 和。

- `top_k = 0` は top-k 条件を無効化
- `top_p = 1` は top-p 条件を無効化

`top_k` と `top_p` はそれぞれ full softmax 上で定義し、両方を満たす token だけを
accept する（積集合）。`top_k` だけなら通常の top-k sampling、`top_p` だけなら
通常の nucleus sampling と一致する。

## rejection sampling

accept されなければ `attempt` を 1 増やして Gumbel-Max をやり直す。full softmax から
sample して accept 集合に含まれる token だけを残すので、accept 後の分布は
「softmax を accept 集合に制限して再正規化したもの」になる。sort は不要。

accept 確率は accept 集合の mass なので、top-p では `>= top_p`。期待試行回数は
`<= 1 / top_p` 程度（`top_p = 0.9` で約 1.11）。`top_k` を小さくすると accept 確率は
top-k 集合の mass まで下がるため、試行回数が増える。

上限は `kMaxSamplingAttempts = 4096`。到達した場合は
`ProgramStatus::DISPATCH_FAILED` を `error_word` に書き、token は `-1` にする
（閾値を緩めて別の token を返すことはしない）。

## 数値

- logits は F32
- scaled logits と Gumbel score は F32
- `expf` を使う
- mass（`Z`、`better_mass`）の accumulation は F64

## invalid logits

row max が finite でない、`Z` が finite でない、`Z <= 0` のいずれかなら
`DISPATCH_FAILED` を書き、token は `-1`。NaN / Inf を黙って sample しない。

## greedy fast path

`temperature == 0` は既存の partitioned argmax をそのまま使う。stochastic 用の
kernel には入らない。tie break は従来どおり「logit が大きい方、同値なら token id が
小さい方」。

stochastic が 1 row でも含まれる batch では、sampling 対象の全 row を
generic な stochastic kernel（1 workgroup = 1 output row）で処理する。
stochastic が 0 なら従来の greedy path を使う。

## Device contract

- `DeviceSamplingParams`（32 byte）が `DeviceRequestDescriptor` に入る
- `DeviceBatchContext::output_sampling_params` に per-output の sampling params が並ぶ
- `prepare_batch_descriptor_kernel` が `output_index + j` へ書き、multi-output では
  `sample_index + j` とする
- logits は device に留まる。host は logits を読まないし sort もしない
- host に戻るのは sampled token（I32）だけ

`SamplingSelectorInput::stochastic_outputs` は host 側で数えた値を使う。selector の
判断のために sampling params を D2H しない。

## 実装の配置

| 役割 | ファイル |
| --- | --- |
| public contract | `include/phaseshift/models/qwen35/runtime/sampling_params.h` |
| device helper / row sampler | `include/phaseshift/models/qwen35/kernels/optimized/sampling_common.h` |
| optimized kernel | `include|src/.../kernels/optimized/stochastic_sampling.{h,hip}` |
| greedy kernel | `include|src/.../kernels/optimized/sampling.{h,hip}` |
| correctness backend | `src/.../kernels/correctness/detail/output.inc` |
| dispatch | `src/.../runtime/sampling_dispatch.hip` |
| selector | `src/.../runtime/sampling_selector.{h,cpp}` |

correctness backend も optimized kernel と同じ `sampling_sample_row` を呼ぶため、
RNG と数学は 1 か所にしかない。
