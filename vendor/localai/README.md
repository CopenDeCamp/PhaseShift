# vendor/localai

LocalAI v4.10.0 の backend protocol 定義を固定するためのvendor directory。

- `VERSION` — 互換対象のLocalAI tag（`v4.10.0`）
- `backend.proto` — `LocalAI v4.10.0` の `backend/backend.proto` をそのまま配置
- `LICENSE` — LocalAIのLICENSE（MIT）

LocalAI本体のsource treeはvendorしない。plugin protocol driftを防ぐため、
backend protocolはこのrevisionに固定する。

## 出所

```text
repo:   https://github.com/mudler/LocalAI
tag:    v4.10.0
path:   backend/backend.proto
```

## Python protobuf stubの生成

生成物は `src/apps/server/localai_proto/` に保持し、必ずこの `backend.proto` から生成する。
生成には `grpcio-tools` が必要。

```bash
python3 -m grpc_tools.protoc \
    -Ivendor/localai \
    --python_out=src/apps/server/localai_proto \
    --grpc_python_out=src/apps/server/localai_proto \
    vendor/localai/backend.proto
touch src/apps/server/localai_proto/__init__.py
# grpc stubのimportをpackage-relativeへ修正する
sed -i 's/^import backend_pb2 as backend__pb2$/from . import backend_pb2 as backend__pb2/' \
    src/apps/server/localai_proto/backend_pb2_grpc.py
```

`backend.proto` を更新する場合は `VERSION` も同時に更新し、同じtagから
protobuf stubを再生成する。

## PhaseShift patch

PhaseShiftはLocalAI v4.10.0に対し`patches/`配下のpatchを適用したruntimeを
必要とする。詳細は`README.phaseshift.md`を参照。`VERSION`はupstream tagのまま
変更しない。
