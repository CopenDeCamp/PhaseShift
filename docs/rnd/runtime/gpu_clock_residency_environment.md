# GPU clock residency PoC — environment and controls (Gate 0)

R9700 (gfx1201) 上で GPU の idle / 低活動状態遷移と Prefill 性能の関係を調べる PoC の
Gate 0 記録である。branch `poc/gpu-clock-residency`。ここには環境、取得可能な
telemetry、制御可能な項目、root 要否のみを記録する。測定結果は
[gpu_clock_residency_poc.md](gpu_clock_residency_poc.md) を参照。

## 1. 環境

| 項目 | 値 |
| --- | --- |
| host | `zen-develop` / Linux 7.0.0-31-generic |
| CPU | AMD Ryzen 7 7800X3D |
| target GPU | AMD Radeon AI PRO R9700 (gfx1201, Navi 48, 32GB, PCI `0000:0a:00.0`) |
| GPU 数 | R9700 ×4 (PCI `07:00.0` / `0a:00.0` / `0f:00.0` / `12:00.0`) + Raphael iGPU (gfx1036) |
| ROCm | 10.0.0 (`amd-smi` 27.0.0) |
| HIP | 7.15 (hipcc, AMD clang 23.0.0git) |
| kernel driver | amdgpu 7.1.3.31500000 |
| vbios / IFWI | `00158738` / ASRock Navi48 XTW 32GB 300W |
| power limit | PPT0 min 210 W / max 300 W |

### GPU の同定

card 番号を決め打ちせず、PCI BDF と ASIC serial で対象を確定する。

- HIP device index は rocminfo の列挙順に一致し、`0000:07:00.0`=0, `0a:00.0`=1,
  `0f:00.0`=2, `12:00.0`=3, iGPU=4 である。
- 本 PoC の対象は **device 1 = `0000:0a:00.0` = `/sys/class/drm/card2`**（`docs/perf` と同じ）。
- `amd-smi static --asic --bus` の `ASIC_SERIAL` と rocminfo の GPU UUID が対応する。
  例: device 1 は `ASIC_SERIAL 0x5881986843433642` / `BDF 0000:0a:00.0`。

### 注意

sway (Wayland) が全 DRM card を開いており、`zed` / `firefox` / `code` などが
render node を保持している。amd-smi 上は compute process は 0 だが、desktop
compositor の activity が telemetry に乗り得る。測定は device 1 に固定し、
同時に同一 GPU で複数 workload を重ねない。

## 2. 取得可能な telemetry

すべて非 root で read 可能。値は card2 で実測した代表値。

| 種別 | interface | 単位 / 値 | 備考 |
| --- | --- | --- | --- |
| GFXCLK/SCLK | `.../hwmon/hwmon5/freq1_input` | Hz | label は `sclk`。idle 0〜100 MHz、load 最大 ~3.3 GHz |
| MCLK/UCLK | `.../hwmon/hwmon5/freq2_input` | Hz | label は `mclk`。idle 96 MHz、mem load 時 1258 MHz |
| board power | `.../hwmon/hwmon5/power1_average` | µW | label は `PPT` |
| temperature | `.../hwmon/hwmon5/temp1_input` | m°C | edge / hotspot / mem は temp1/2/3 |
| GFX busy | `.../gpu_busy_percent` | % | |
| MEM busy | `.../mem_busy_percent` | % | |
| SCLK DPM table | `.../pp_dpm_sclk` | 現在 level に `*` | `S:2Mhz / 1:500Mhz / 2:2350Mhz` |
| MCLK DPM table | `.../pp_dpm_mclk` | 現在 level に `*` | `0:96 / 1:456 / 2:772 / 3:875 / 4:1124 / 5:1258` MHz |
| SOC/FCLK | `.../pp_dpm_socclk` / `pp_dpm_fclk` | | |
| GPU metrics | `.../gpu_metrics` | binary 4096 B | `amd_gpu_metrics`。未使用 |
| amd-smi | `amd-smi metric --gpu 1` | 集約表示 | GFX/MEM clk, power, temp, activity, throttle, deep sleep 状態を含む |
| rocm-smi | `rocm-smi` | | 本環境の dGPU (0x7551) は telemetry `N/A` で使えない |

`amd-smi metric` で確認した idle 時:
`GFX_0 CLK 13 MHz / DEEP_SLEEP ENABLED`、`MEM_0 CLK 96 MHz / DEEP_SLEEP DISABLED`。

## 3. 制御可能な項目

| 項目 | interface | 権限 | 本環境での可否 |
| --- | --- | --- | --- |
| performance level | `power_dpm_force_performance_level` | root write (0644 root:root) | 可（非 root は Permission denied） |
| DPM state | `power_dpm_state` | root write | 現在 `performance` |
| power profile mode | `pp_power_profile_mode` | root write | card2 で read 可。index 0 `BOOTUP_DEFAULT*`。card1 は boot GPU / sway 使用のため `Device or resource busy` |
| OD clock/voltage | `pp_od_clk_voltage` | root | card2 には存在しない |
| pp_features | `pp_features` | read | DEEP_SLEEP 等の feature bit を表示 |

`power_dpm_force_performance_level` の設定値: `auto` / `low` / `high` / `manual` /
`profile_standard` / `profile_min_sclk` / `profile_min_mclk` / `profile_peak`。
名前の意味は driver/API 依存のため、必ず `freq1_input` / `freq2_input` の実測で
意図した状態かを検証する。

### root 運用

本環境の sudo は password を要求するため、agent からは sysfs を write できない。
PoC に限り `tools/poc/clock_residency/poc_clock_state.sh`（save/set/restore/status）を
用意し、利用者が `sudo` で実行する。production runtime へは sysfs 制御を入れない。
restore は開始前値を保存し、read-back で確認する。

## 4. 現状の performance profile（未変更時）

```
card2 power_dpm_force_performance_level = auto
card2 power_dpm_state                   = performance
card2 pp_power_profile_mode             = 0 BOOTUP_DEFAULT*
```

## 5. 再現コマンド

```sh
amd-smi static --asic --bus --gpu 1
amd-smi metric --gpu 1
cat /sys/class/drm/card2/device/hwmon/hwmon5/freq1_input   # sclk Hz
cat /sys/class/drm/card2/device/hwmon/hwmon5/freq2_input   # mclk Hz
cat /sys/class/drm/card2/device/hwmon/hwmon5/power1_average # uW
```
