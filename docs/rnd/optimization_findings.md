> Status: historical R&D record（現在の仕様・性能値ではない。現在の仕様は `docs/developer/`、現在の性能は `docs/perf/` を参照）

# RDNA4 (gfx1201) optimization findings

RDNA4 (gfx1201) 上で実際に計測して確定した cross-cutting な最適化知見と、topic 別の研究記録への index。数値はすべて同リポジトリの実測。時系列の個別実験は topic 別ファイルへ分離した（[Topic research index](#topic-research-index)）。移動先でも見出し番号（§7.x）は元 dump の通し番号を保持する。

---

## Measurement discipline

測定原則の正本は [docs/perf/methodology.md](../perf/methodology.md)。原則の重複記載はしない。ここには、実際に測定を壊した具体例だけを残す。

- `hipSetDevice` が weight/workspace alloc より後 → `--device N` でも GPU 0 に確保（PCIe 経由計測）。
- CLI option の非対称（ある bench に `--seed` が無い）→ usage error で perf が欠測。
- code object metadata の msgpack int が 128 以上で decode 失敗（`-2`）→ private scratch を誤判定。
- bench の check が no-op（`--check` は受理されるが何も比較しない）。
- bench の入力がカーネルの前提を満たしていない（q/k の L2 正規化なし → chunked GDN の C×C solve が不適条件）。
- shape によって **per-call 固定オーバーヘッド**が支配することがある（実測: fp4 GEMM で ~33us/call。n=32 でも 33us）。

---

## Production reachability

- source tree に存在する / CMake に入る / public bench から叩ける ≠ production reachable。
- reachability は **runtime selector / matcher の rule table** から導出する。
  - candidate-level（rule が存在し、少なくとも 1 つの profile で選ばれる）
  - profile-level（rows/geometry ごとの `runtime_selected`）
- rule が無い経路は **NON_PRODUCTION**。release blocker に数えない（将来の promotion blocker として記録）。
- **「負けるカーネルを選ぶ rule」は promotion contract violation**（本物の FAIL）。
  実測例: fused FP4 GEMV は rows≤2 で選ばれていたが、同条件の unfused（WMMA GEMM + GEMV 混成）に 1.5〜2x 負けていた。
  → 勝つ側（fused WMMA GEMM）を選ぶよう rule を直すのが正解。kernel を微修正しても解決しない。

---

## Numeric correctness

- **hot loop に double を入れない**。`pow/cos/sin(double)` は f64 マクロ列になる。
  実測: attention_prep が v_fma_f64 200 + v_add_f64 366 で sgpr 107（上限 106 超過）→ spill 23〜25。
  RoPE の周波数計算を f32 化するだけで sgpr 41 / spill 0 / 約 2x 高速化。
  出力が bf16 なら tolerance は bf16 級（`2*2^-8`）で、f32 で十分。
- **縮約順序は contract**。bit-exact を要求する kernel では、加算順序を変える縮約最適化が
  ULP 差でテストを落とす（実例: `test_gdn_norm_gate` の tid0 線形和 → warp 縮約で mm=1。
  対象 kernel と条件は [quantization 履歴](quantization/kernel_optimization_history.md) §7.21 を参照）。
- bf16 staging を挟む最適化は数値損失を生む。損失量・受容判断は topic 別ファイルに記録する
  （例: GDN recurrence の bf16 staging は [gdn 履歴](gdn/optimization_history.md) §7.15）。

---

## Dispatch / launch floor

- shape によって **per-call 固定オーバーヘッド**が支配する。実測: fp4 GEMM で ~33us/call（n=32 でも 33us）。
- 小 kernel のコストは work ではなく launch に支配される。実測: bf16 GEMM `rows=1` で n=8〜512 が
  すべて ~7us/launch（work が 64 倍でも一定）。詳細は [runtime 履歴](runtime/execution_overhead.md) §7.59。
- したがって小 kernel の削減は「本体を速くする」ではなく「launch 数を減らす」で効かせる。
  ただし適用可能な fusion が無い場合や、逆に遅くなる場合は採用しない（同 §7.59）。
- launch 間 gap も無視できない。実測: prefill の host gap 9.9%（1669 dispatch / 98ms）、
  decode の gap 約 2ms（5.5%）。capture/keep-alive の採否は [runtime 履歴](runtime/execution_overhead.md) を参照。
- **standalone kernel の高速化が in-context wall に現れないことがある**。
  改善が効くかは、その kernel が launch floor 由来か occupancy 由来かで決まる
  （[kernel 履歴](kernel/optimization_history.md) §7.99）。

---

## WMMA / memory / occupancy general rules

### 引数渡し
- **by-value aggregate に対する device 側の書き込みは禁止**。
  書き込むと aggregate 全体が writable private frame に materialize される。
  実測: `args.projection_count = kX;` が device 側に 1 行あるだけで
  private scratch 136〜216B、sgpr 54→25、code size -13%、scratch_load/store が hot path に出現。
  host 側で既に設定している冗長な代入を消すだけで全て消えた。
- 大きな構造体は `const&` で受ける。もしくは scalar に分解して渡す。

### decode / unpack
- FP4 codebook の decode（分岐付き非線形テーブル、約 7〜8 命令/nibble × 16/nibble-block）が
  fp4 GEMM の支配コストになっていた。
- **16-entry の LDS LUT 化は bit-exact で、format 不変**。
  実測（rows=128, 4B shapes）: mlp_gate_up 394→350us、mlp_down 382→306、attn_q 150→119、attn_kv 53→44。
  vgpr 59→56、code size -27%。
- **LDS LUT より register permute（`v_perm_b32`）の方が速い**（bit-exact, format 不変）。
  Codebook10 の値は全て int8 に収まる整数なので、4 つの定数
  `[0,1,2,3] [4,6,8,10] [0,-1,-2,-3] [-4,-6,-8,-10]` と `__builtin_amdgcn_perm` だけで
  nibble→int8 を LDS 依存なしに展開できる（ROCmFPX の正攻法）。
  セレクタは「nibble の下位 3bit + 符号 bit」をそのまま使う。
  実測（rows=128, 同一条件 A/B, 4B shapes）: mlp_gate_up 203.5→184.9us / attn_q p50 -16%、
  end-to-end PP128 1608.5→1653.9、PP512 2842.2→2905.0、TG 41.0→42.5 tok/s。
  `__builtin_amdgcn_perm(a,b,sel)` の意味は **sel byte 0-3 → b、4-7 → a**（実測確認）。
  GEMV 側は 1byte 単位で選択子構築が増えるため LDS のまま（未 A/B）。
- scale の粒度が効く: **per-32 (UE4M3) は per-256 (MQ4) より decode が 8 倍**。
- 重みへの scale 折り込みは **scale が 2 の冪（E8M0）のときだけ無料**（指数部の整数加算）。
  UE4M3 のように仮数を持つ scale は per-element 乗算が残る。
- 4bit コードを「byte の上位 nibble に置いた two's complement int4」にすると
  unpack が **2 AND + 1 SHL / 8weight** になる（libr4d）。ただし codebook 形式が変わる。

### WMMA
- gfx1201 の実測スループット（libr4d 計測）:
  ```text
  v_wmma_f32_16x16x16_f16      207 TF/s (5.95 cyc)
  v_wmma_f32_16x16x16_fp8_fp8  412      (2.99)
  v_wmma_i32_16x16x16_iu8      407      (3.03)
  v_wmma_i32_16x16x32_iu4      830      (2.97)
  ```
  **iu8 は速い**。遅さは命令ではなく、unpack / `v_cvt_f32_i32` / scale の shuffle 側。
- wave32 fragment layout: `idx = lane % 16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`。
- **A/B operand を入れ替えると結果が転置される（無料）**。
- preshuffle: 重みを fragment order で保持 → **lane あたり b128 1 回 / 4 k-step**。
- activation preshuffle: k-step 内の 16byte を `[half0 8B][half1 8B]` に並べ替え → **b64 1 回/lane/k-step**。
  `[2][M][K/2]` の plane 分割は負ける（レーンが 2 領域に散り、cache line が倍）。
- per-row の activation scale は k-loop の外に出せる（store で 1 回）。per-32 scale は出せない。
- occupancy: 32 thread / 16×16 tile より、**最大 1024 thread + split-K + LDS reduction**。
- gfx1201 に **`v_cvt_pk_bf16_f32` / direct-to-LDS / `ds_read_b64_tr_b16` は無い**。
  bf16 pack は software（最安は 2 add + `v_perm_b32`）。
- **LLVM は LDS を single-buffer し、毎 WMMA の前に drain する**。
  `__builtin_amdgcn_sched_group_barrier` で制御する。**粒度が内容より重要（coarser is better）**。

---

## Rejected-pattern checklist

```text
数値を緩めてgateを通す
汚染された測定値を再利用する
同一GPUで複数benchを並走させる
correctness/referenceを性能baselineにする
source/CMake/benchにあるだけでproduction candidateとしてgateする
非production candidateをrelease blockerに数える
効果を計測せずに（ISAの理屈だけで）最適化を入れる
benchの前提条件（正規化・layout）をカーネル契約と揃えずに測る
```

---

## Cross-cutting lessons

### cross-lane（DS vs VALU）
- `__shfl*` は **DS 経路**（`ds_bpermute` + `s_wait_dscnt`）に落ちる。
- 代替: DPP（`v_add_*_dpp` の row_shl8/4/2/1）と `v_permlanex16` は **VALU 経路**。
  - GEMV の reduction（shfl_down ×8 + broadcast ×4）を DPP + permlanex16 に置換 → ds_bpermute 12→0、s_wait_dscnt 12→0。
  - GEMM の per-lane gather は `v_permlane16_var_b32` で置換可能（`D[lane] = A/B[(lane&16)|(sel&15)]`）。
- **しかし DS が律速とは限らない**。実測:
  - GEMV の DPP 化: p50 は中立（±0.5%）。
  - GEMM の permlanex16_var 化: **+2〜4% 悪化**（VALU pipe が FMA と競合）。
  - 手動の WMMA 再 schedule（WMMA0→WMMA1→consume）: **+1〜5.8% 悪化**。
  - 手動の double-buffer prefetch: **+4〜11% 悪化**（VGPR 59→72）。
  → 「ISA で説明できるから速くなる」は誤り。**必ず同一条件で A/B を取る**。compiler の schedule は既に良いことが多い。

### 最適化の順序
```text
layout (命令側) → 再利用 (算術強度) → 並列度 (SK/WV/occupancy)
各段で: required 78/78 / bench --check 0 mismatch / 分離GEMM p50 / PP/TG end-to-end
小shapeの固定費 (~30us/call) を悪化させないこと。悪化したら shape dispatch を足す。
```

### 参照実装から借用した事実（一次ソース）
- **hipfire**（外部 kernel optimization reference。当時 snapshot を参照したが repo には vendor しない）: gfx12 の **retained-PM4 tape replay**（dispatch DAG を record → PM4 → 1 AQL packet）で
  per-launch overhead を消す。per-op fusion（conv1d+silu+split+qknorm 等）。chunked GDN。tape は ~833 dispatch/token。
- **libr4d** (codeberg StillDeadcode/libr4d): gfx1201 専用カーネル群（conv_prep / KKT solve / chunk scan / skinny GEMM）。
  実測 WMMA throughput、fragment layout、gfx12 の ISA 欠落と `sched_group_barrier` はここが一次情報。
- **vllm-radiance**: MXFP4 weight + 動的 FP8 活性化で FP8 WMMA、preshuffle、decode(M≤64)/prefill の分離、non-temporal load。
  R9700 (TP2) で 27B prefill 4,000+ TPS。

---

## Topic research index

- [attention/optimization_history.md](attention/optimization_history.md) — paged attention decode / prefill、KV split/reduce。
  KV page pruning 系の PoC は同 directory の既存文書。
- [gdn/optimization_history.md](gdn/optimization_history.md) — GDN recurrence / conv / chunked scan / WMMA 数値挙動。
- [quantization/kernel_optimization_history.md](quantization/kernel_optimization_history.md) — PSQ4/PSQ8 GEMM、preshuffle、scale、codebook、iMatrix、W4A8/W8A8、FP8/MXFP4。
- [runtime/execution_overhead.md](runtime/execution_overhead.md) — prefill/decode のコスト内訳、kernel gap、HIP Graph capture / keep-alive。
- [kernel/optimization_history.md](kernel/optimization_history.md) — RoPE など、他 topic に属さない単独 kernel。
- [dflash2/optimization_history.md](dflash2/optimization_history.md) — DFlash2 drafter、speculative verify、INT2 coarse head。
  DFlash2 全体の Gate 記録は [dflash2/dflash2.md](dflash2/dflash2.md)。
- [mtp/optimization_history.md](mtp/optimization_history.md) — MTP の correctness / acceptance / Gate 検証。
  MTP の現状診断は [mtp/mtp.md](mtp/mtp.md)。
