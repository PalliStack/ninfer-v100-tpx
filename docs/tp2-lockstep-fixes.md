# TP2 Lockstep Reliability — Detailed Commit Record

**Repository**: https://github.com/PalliStack/ninfer-v100-tpx
**Base commit**: `9913bc22` — nvfp4 qpn prepacked: double-buffered 4-group prefetch in the main loop; Nsight Compute findings
**Head commit**: `1e9c6c1a`
**Commits recorded**: 7 (2 loader/model-layout commits + 5 engine-reliability commits)
**Diff size**: 12 files changed, +376 / −64
**Runtime**: 2x NVIDIA V100 32 GB, tensor-parallel degree 2, one process per rank, Qwen3.8-27B NVFP4, DFlash2/MTP speculative decoding

Every claim below is backed by a harness run on the machine that produced these commits; no estimate or projection is stated as a measurement.

---

## English

### 0. System model (why all five engine fixes exist)

Under tensor parallelism the two ranks must issue the **same GPU unit in the same order**. The unit exchange (`tp::Record` in a shared block, plus an agreed cancellation mask and an admission budget) is the only channel that may change state consistently. The invariant established by this work is:

> A rank-local input (wall clock, socket state, response-release timing) may only raise an **intent**. Every FIFO change, every lane termination, and every admission decision must travel through the agreed unit exchange.

Breaking that invariant wedges both ranks: one rank spins in a collective while the other waits at a host exchange, and the supervisor eventually restarts the pair.

### 1. `9d021cd3` — local tuning: dflash2 QKV 6144/MLP 34816 layout fix, proxy NVLink IPC probe bypass, long split keys 1920 (2026-10-08)

**Files**: `src/targets/qwen3_6_27b_tp2/impl/load/bindings.cpp` (+8/−8), `tools/tp2/tp2_proxy.py` (+2), `tools/tp2/shard_qwen38_27b.py` (−2)

**Cause**: the DFlash2 draft layer reads fused tensors, and the on-disk artifact stores the *full* fused widths. The loader was binding narrower shapes, so the draft model read a truncated layout.

**Change**:
- `query_key_value` rows 3072 → 6144; the `context_key` / `context_value` row views widen 512 → 1024
- `gate_up` rows 17408 → 34816; `down` 5120x8704 → 5120x17408
- proxy: `NINFER_TP_NVLINK=0` when NCCL point-to-point is disabled (development machine has no NVLink, plain PCIe), and `NINFER_SM70_LONG_SPLIT_KEYS=1920` by default
- shard script: blank-line cleanup only

### 2. `de8a896d` — refactor(dflash2): drive QKV/MLP layout from DFlashConfig constants (2026-10-13)

**Files**: `bindings.cpp` (+21/−9)

**Change**: the widths introduced by `9d021cd3` are no longer literals; they are derived from `DFlashConfig`. Numeric equivalence was verified against the previous literals: `hidden=5120`, `intermediate=17408`, `query_size=4096`, `kv_size=1024` give `query_key_value = query_size + kv_size = 6144`, `gate_up = 2 * intermediate = 34816`, `down = 5120x17408`, and the QKV offsets 4096 / 5120 with width 1024. This commit is behaviour-preserving and was **ruled out** as a cause of the outage investigated the same day.

### 3. `1ce94433` — tp2: keep client-disconnect cancellation inside the agreed unit mask

**Files**: `src/runtime/engine/engine_core.h`, `src/runtime/engine/tp_lockstep.h`, `src/serve/generation_service.cpp` (+56/−9)

**Cause**: rank 0 faces the client. When a client closed the socket during prepare, rank 0 dropped the request locally while rank 1 kept it, so rank 1 waited at the lockstep exchange forever. Observed signature: rank 0 `req#1 cancelled during prepare ... HTTP 499 | client disconnected`, rank 1 `req#1 started` with no completion, the unit counters frozen at `(4, 5)`, one GPU at 100% spinning in a collective and the other at 0%, and the proxy finally logging `RESTART both ranks: no lockstep progress for 180s`.

**Change**:
- `tp::lockstep_enabled()` — a single predicate for "this process is one rank of a lockstep pair"
- `preparation_cancellation_view()` — empty under TP, so prepare never aborts on a rank-local disconnect
- `cancelled_on_this_rank()` / `local_cancellation_view()` — all five rank-local decision sites now read the agreed mask instead of local state

**Verification**: a 932 KB payload (112 messages, 20 tools, 171,675 prompt tokens) sent with the socket closed immediately at 0.0 s / 0.05 s / 2.0 s. Before the fix the engine wedged; after the fix all three runs returned HTTP 200 and rank 1 logged `done | cancelled` after 3.4 s.

### 4. `9b3b98a2` — tp2: queue expiry is an intent, not a rank-local FIFO removal

**Files**: `engine_core.h`, `request_record.h`, `src/serve/generation_service.cpp`

**Cause**: the admission deadline is evaluated against wall clock. Removing an expired request from the local FIFO changes which request is admitted, and the two ranks can reach that decision at different moments — the divergence window was measured at 0 events in production (the proxy serialises with `max_inflight=1`, maximum observed FIFO wait 51 ms) but the defect was latent and reachable.

**Change**:
- `QueueTerminationCause { None, Cancelled, QueueExpired }` plus an `expired` flag on the request record
- the four expiry sites set the flag only; under TP they never remove the entry or reject the submission
- under TP `submit` no longer rejects on `pending_deadline` or `outstanding_ >= max_outstanding_`, and the serve-layer capacity rejection is bypassed (the symmetric bound is the proxy's, enforced before dispatch)

**Verification**: `expiry_repro.py --skew-ms 90000` makes rank 1 expire 90 s earlier than rank 0. Before the fix: wedge with units frozen at `(5, 4)` and a restart loop. After the fix: both ranks answered HTTP 200, units `(18, 18)`, engine alive.

### 5. `68c3eb7b` — tp2: defer instead of failing when a lane cannot be aborted, and end expired lanes through the agreed mask

**Files**: `engine_core.h`, `src/runtime/engine/resource_manager.h`

**Cause (a follow-on defect found by measurement)**: once expiry rode in the agreed termination mask, `Program::abort` reached `clear_lane_strict`, which refuses while the lane holds staged prompt state (`std::logic_error("Program did not consume aborted sequence")`). The worker's catch-all then called `fail_all_locked`, turning the whole engine into 503s — strictly worse than the stall it was meant to fix. Instrumentation showed the refusing unit had `state=2 (Prefill) computed=51 capture=0`, i.e. the unit that reaches the admitted prompt frontier.

**Change**:
- `ResourceManager::abort()` → `try_abort()` returning `std::optional<AbortResult>`; a refusal leaves the lane untouched and defers the termination
- the deferred request is ended by an override in `complete_success()` and reported as 503 `request_queue_timeout`, preserving the HTTP contract
- expiry travels in the upper bits of the agreed mask (`kExpiryMaskShift = 32`) with cause adoption, so both ranks terminate the same lane at the same unit and report the same reason

**Verification**: `expiry_repro.py --skew-ms 90000` → both ranks HTTP 503 `request_queue_timeout` in 1.24 s, units `(12, 12)`, engine alive. `expiry_contract.py` → at least one request received 503 `request_queue_timeout`, all requests answered, engine alive.

### 6. `e5860473` — tp2: propagate a rank failure instead of letting the pair hang

**Files**: `engine_core.h`, `tp_lockstep.h` (+45/−1)

**Cause**: a failed engine worker left its rank alive serving 503s while the peer waited. Measured with an injected failure: the peer spun **inside a prefill GPU unit** (host work 0.0%, 0 us) and only died from `cudaErrorLaunchFailure` in `ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:520` about 60 s later, because it never reached a host-level exchange where a failure could be observed. A peer-side flag alone cannot help a peer that is stuck in a collective.

**Change**:
- `SharedBlock.poison` added in former padding, so `published` keeps offset 64 (read by the TP2 proxy) and the block size stays 384 bytes; verified offsets: `published=64`, `poison=80`, `records=128`
- `poison_peer()` plus a poison check in the `publish_and_wait` spin loop, which fails with `peer engine failed (poison set) while waiting for unit N`
- `fail_all_locked()` poisons the peer, drains 200 ms so the HTTP error responses are flushed, writes the cause to stderr and **terminates the rank** — a failed worker can never serve again (`failed_` stays set), and the exit is what the peer's collectives and the proxy's 2 s liveness check actually notice
- test hook `NINFER_TP_FAIL_WORKER_AFTER_UNITS` (0 ships)

**Verification**: with the injected failure — poison alone: the peer died from a CUDA launch failure after about 62 s; with the process exit: the proxy logged `RESTART both ranks: a rank process exited` and the pair recovered in **19.4 s**.

### 7. `1e9c6c1a` — tp2: ask the Program whether a lane can be aborted instead of predicting it

**Files**: `engine_core.h`, `resource_manager.h`, `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h`, `.../impl/runtime/api_impl.h`, `.../impl/runtime/program.h`, `.../impl/runtime/program_impl.h` (+35/−8)

**Cause**: the engine predicted lane clearability from token counters (`computed_prompt_tokens` against the admitted prompt suffix). Measurement showed the refusal is decided by the Program's staged prompt state, which the engine cannot mirror with a counter.

**Change**:
- `Program::can_abort(sequence)` exposed through four layers, reusing `abort()`'s own guards (open transaction, sequence validity, lane lifecycle) and `can_clear_lane_strict`, which `clear_lane_strict()` itself gates on
- `try_abort()` asks first, so a termination that will be deferred never touches lane state
- the prefill abort guard uses the Program's verdict; the tolerant fallback stays as a safety net for a state change between the check and the call

**Verification**: full regression re-run — deadline-skew stress (both ranks 503 `request_queue_timeout` in 1.24 s, units `(12, 12)`, engine alive), expiry contract (503 `request_queue_timeout`), disconnect during prepare (cancelled after 3.4 s, follow-up 200), disconnect during decode (cancelled within one second, follow-up 200), normal traffic (200, 29 and 64 output tokens).

### Verification harnesses

| Harness | What it proves |
|---|---|
| `repro2.py <delay>` | client disconnect during prepare terminates both ranks through the agreed mask |
| `expiry_repro.py --skew-ms N` | rank-local wall-clock skew no longer diverges the FIFOs |
| `expiry_contract.py` | an expired request still receives 503 `request_queue_timeout` |
| `decode_abort.py` | a mid-decode disconnect terminates the lane early, engine survives |
| `poison_ab.py --fail-after N` | a rank failure propagates instead of hanging the pair |

---

## 简体中文

### 0. 系统模型（五个引擎修复的共同前提）

在张量并行下，两个 rank 必须**按相同顺序提交相同的 GPU 单元**。单元交换（共享内存中的 `tp::Record`，加上一致的取消掩码与准入预算）是唯一可以一致地改变状态的通道。本工作确立的不变式是：

> 任何 rank 本地输入（墙上时钟、套接字状态、响应释放时机）只能提出**意图**。所有 FIFO 变更、所有 lane 终止、所有准入决定都必须经过约定的单元交换。

违反该不变式会同时卡死两个 rank：一个 rank 在集合通信中空转，另一个在主机侧交换处等待，最终由监督进程重启两者。

### 1. `9d021cd3` — local tuning: dflash2 QKV 6144/MLP 34816 layout fix, proxy NVLink IPC probe bypass, long split keys 1920（2026-10-08）

**文件**：`bindings.cpp`（+8/−8）、`tools/tp2/tp2_proxy.py`（+2）、`tools/tp2/shard_qwen38_27b.py`（−2）

**原因**：DFlash2 草稿层读取融合张量，磁盘上的权重保存的是**完整**融合宽度，而加载器绑定的是较窄的形状，导致草稿模型读取了被截断的布局。

**变更**：
- `query_key_value` 行数 3072 → 6144；`context_key` / `context_value` 的行视图宽度 512 → 1024
- `gate_up` 行数 17408 → 34816；`down` 由 5120x8704 → 5120x17408
- 代理：当 NCCL 点对点关闭时设置 `NINFER_TP_NVLINK=0`（开发机没有 NVLink，只有普通 PCIe），并默认 `NINFER_SM70_LONG_SPLIT_KEYS=1920`
- 分片脚本：仅清理空行

### 2. `de8a896d` — refactor(dflash2): drive QKV/MLP layout from DFlashConfig constants（2026-10-13）

**文件**：`bindings.cpp`（+21/−9）

**变更**：`9d021cd3` 引入的宽度不再写成字面量，而是由 `DFlashConfig` 推导。已对旧字面量做等值核对：`hidden=5120`、`intermediate=17408`、`query_size=4096`、`kv_size=1024` 推出 `query_key_value = 4096 + 1024 = 6144`、`gate_up = 2 * 17408 = 34816`、`down = 5120x17408`，QKV 偏移 4096 / 5120、宽度 1024 全部一致。此提交保持行为不变，并已**排除**其作为当日故障原因的嫌疑。

### 3. `1ce94433` — tp2: keep client-disconnect cancellation inside the agreed unit mask

**文件**：`engine_core.h`、`tp_lockstep.h`、`src/serve/generation_service.cpp`（+56/−9）

**原因**：rank 0 面向客户端。当客户端在 prepare 阶段关闭套接字时，rank 0 在本地丢弃请求而 rank 1 仍然保留，于是 rank 1 永远等待在锁步交换处。观察到的特征：rank 0 `cancelled during prepare ... HTTP 499 | client disconnected`，rank 1 `req#1 started` 之后不再推进，单元计数冻结在 `(4, 5)`，一张 GPU 在集合通信中 100% 空转、另一张为 0%，代理最终记录 `RESTART both ranks: no lockstep progress for 180s`。

**变更**：
- `tp::lockstep_enabled()`：统一判断“本进程是否为锁步对中的一个 rank”
- `preparation_cancellation_view()`：在 TP 下为空，使 prepare 不再因 rank 本地断开而中止
- `cancelled_on_this_rank()` / `local_cancellation_view()`：五处 rank 本地判定改为读取一致的掩码

**验证**：发送 932 KB 负载（112 条消息、20 个工具、171,675 提示词 token），分别在 0.0 秒 / 0.05 秒 / 2.0 秒关闭套接字。修复前引擎卡死；修复后三次全部返回 HTTP 200，rank 1 在 3.4 秒后记录 `done | cancelled`。

### 4. `9b3b98a2` — tp2: queue expiry is an intent, not a rank-local FIFO removal

**文件**：`engine_core.h`、`request_record.h`、`src/serve/generation_service.cpp`

**原因**：准入截止时间按墙上时钟判定。从本地 FIFO 移除过期请求会改变被准入的请求，而两个 rank 可能在不同时刻做出该判断。生产环境实测该窗口为 0 次事件（代理以 `max_inflight=1` 串行化，观测到的最大 FIFO 等待为 51 毫秒），但缺陷是潜伏且可达的。

**变更**：
- 新增 `QueueTerminationCause { None, Cancelled, QueueExpired }` 与请求记录上的 `expired` 标志
- 四处过期判定只设置标志；在 TP 下不再移除条目或拒绝提交
- 在 TP 下 `submit` 不再因 `pending_deadline` 或 `outstanding_ >= max_outstanding_` 拒绝，并绕过服务层容量拒绝（对称上界由代理在派发前执行）

**验证**：`expiry_repro.py --skew-ms 90000` 让 rank 1 比 rank 0 提前 90 秒过期。修复前：卡死、单元冻结在 `(5, 4)` 并进入重启循环。修复后：两个 rank 均返回 HTTP 200，单元 `(18, 18)`，引擎存活。

### 5. `68c3eb7b` — tp2: defer instead of failing when a lane cannot be aborted, and end expired lanes through the agreed mask

**文件**：`engine_core.h`、`src/runtime/engine/resource_manager.h`

**原因（由实测发现的后续缺陷）**：一旦过期进入一致的终止掩码，`Program::abort` 会到达 `clear_lane_strict`，而该函数在 lane 持有已暂存的提示词状态时会拒绝（`std::logic_error("Program did not consume aborted sequence")`）。工作线程的兜底捕获随后调用 `fail_all_locked`，把整个引擎变成 503 —— 比它本要修复的停滞更糟。插桩显示被拒绝的单元为 `state=2 (Prefill) computed=51 capture=0`，即到达已准入提示词前沿的那个单元。

**变更**：
- `ResourceManager::abort()` → `try_abort()`，返回 `std::optional<AbortResult>`；被拒绝时不触碰 lane，改为推迟终止
- 被推迟的请求由 `complete_success()` 中的覆盖逻辑结束，并按 503 `request_queue_timeout` 上报，保持 HTTP 契约
- 过期信息通过一致掩码的高位（`kExpiryMaskShift = 32`）传递并在两 rank 间采纳原因，使两者在同一单元终止同一 lane 并上报相同原因

**验证**：`expiry_repro.py --skew-ms 90000` → 两个 rank 在 1.24 秒内返回 HTTP 503 `request_queue_timeout`，单元 `(12, 12)`，引擎存活。`expiry_contract.py` → 至少一个请求收到 503 `request_queue_timeout`，所有请求均有响应，引擎存活。

### 6. `e5860473` — tp2: propagate a rank failure instead of letting the pair hang

**文件**：`engine_core.h`、`tp_lockstep.h`（+45/−1）

**原因**：引擎工作线程失败后该 rank 仍然存活并持续返回 503，而同伴在等待。通过注入故障实测：同伴**卡在 prefill 的 GPU 单元内部**（主机占用 0.0%、0 微秒），约 60 秒后才因 `ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:520` 的 `cudaErrorLaunchFailure` 死亡，因为它从未到达可观测故障的主机侧交换点。仅靠同伴侧标志无法拯救卡在集合通信中的同伴。

**变更**：
- 在原填充位中加入 `SharedBlock.poison`，使 `published` 保持偏移 64（TP2 代理读取该偏移），块大小保持 384 字节；实测偏移 `published=64`、`poison=80`、`records=128`
- `poison_peer()` 与 `publish_and_wait` 自旋循环中的毒化检查，失败时报告 `peer engine failed (poison set) while waiting for unit N`
- `fail_all_locked()` 毒化同伴、等待 200 毫秒让 HTTP 错误响应写出、把原因写入标准错误并**终止本 rank** —— 失败的 worker 无法再提供服务（`failed_` 保持置位），而进程退出正是同伴的集合通信与代理 2 秒存活检查能立刻察觉的信号
- 测试钩子 `NINFER_TP_FAIL_WORKER_AFTER_UNITS`（发布版本为 0）

**验证**：注入故障下 —— 仅有毒化：同伴约 62 秒后因 CUDA 启动失败死亡；加入进程退出：代理记录 `RESTART both ranks: a rank process exited`，整对在 **19.4 秒**内恢复。

### 7. `1e9c6c1a` — tp2: ask the Program whether a lane can be aborted instead of predicting it

**文件**：`engine_core.h`、`resource_manager.h`、`runtime.h`、`api_impl.h`、`program.h`、`program_impl.h`（+35/−8）

**原因**：引擎用 token 计数（`computed_prompt_tokens` 与已准入提示词后缀比较）预测 lane 是否可清理。实测表明拒绝与否由 Program 的已暂存提示词状态决定，引擎无法用计数器模拟。

**变更**：
- `Program::can_abort(sequence)` 在四层中暴露，复用 `abort()` 自身的守卫（事务开启、序列有效性、lane 生命周期）以及 `clear_lane_strict()` 所依赖的 `can_clear_lane_strict`
- `try_abort()` 先询问，因此将被推迟的终止不再触碰 lane 状态
- prefill 中止守卫改用 Program 的判定；宽容回退保留，作为检查与调用之间状态变化的安全网

**验证**：完整回归重跑 —— 截止时间偏移压力（两个 rank 在 1.24 秒内 503 `request_queue_timeout`，单元 `(12, 12)`，引擎存活）、过期契约（503 `request_queue_timeout`）、prepare 阶段断开（3.4 秒后取消，后续请求 200）、解码中断开（一秒内取消，后续请求 200）、正常流量（200，输出 29 与 64 token）。

### 验证工具

| 工具 | 证明内容 |
|---|---|
| `repro2.py <delay>` | prepare 阶段客户端断开时，两个 rank 通过一致掩码终止 |
| `expiry_repro.py --skew-ms N` | rank 本地的时钟偏差不再导致 FIFO 分歧 |
| `expiry_contract.py` | 过期请求仍收到 503 `request_queue_timeout` |
| `decode_abort.py` | 解码中断开会提前终止 lane，引擎存活 |
| `poison_ab.py --fail-after N` | rank 故障会传播，而不是挂住整对 |

---

## 한국어

### 0. 구조 모델 (엔진 수정 다섯 건이 존재하는 이유)

텐서 병렬에서는 두 랭크가 **같은 GPU 유닛을 같은 순서로** 내보내야 합니다. 유닛 교환(공유 블록의 `tp::Record`, 합의된 취소 마스크와 진입 예산)이 상태를 일관되게 바꿀 수 있는 유일한 통로입니다. 이번 작업에서 세운 원칙은 다음과 같습니다.

> 랭크 로컬 입력(벽시계, 소켓 상태, 응답 반환 시점)은 오직 **의도**만 제기할 수 있습니다. 모든 대기열 변경, 모든 레인 종료, 모든 진입 결정은 합의된 유닛 교환을 거쳐야 합니다.

이 원칙이 깨지면 두 랭크가 함께 멈춥니다. 한 랭크는 집합 통신에서 헛돌고 다른 랭크는 호스트 교환 지점에서 기다리며, 결국 감시 프로세스가 둘을 재시작합니다.

### 1. `9d021cd3` — 로컬 튜닝: dflash2 QKV 6144/MLP 34816 배치 수정, 프록시 NVLink 탐색 우회, 긴 분할 키 1920 (2026-10-08)

**파일**: `bindings.cpp`(+8/−8), `tools/tp2/tp2_proxy.py`(+2), `tools/tp2/shard_qwen38_27b.py`(−2)

**원인**: DFlash2 초안 계층은 융합 텐서를 읽는데, 디스크의 가중치는 **전체** 융합 폭을 담고 있었고 로더는 더 좁은 모양을 묶고 있었습니다. 그래서 초안 모델이 잘린 배치를 읽었습니다.

**변경**:
- `query_key_value` 행 수 3072 → 6144, `context_key` / `context_value` 행 뷰 폭 512 → 1024
- `gate_up` 행 수 17408 → 34816, `down` 5120x8704 → 5120x17408
- 프록시: NCCL 점대점이 꺼져 있을 때 `NINFER_TP_NVLINK=0`(개발 장비에 NVLink가 없고 일반 PCIe만 사용), `NINFER_SM70_LONG_SPLIT_KEYS=1920` 기본값
- 분할 스크립트: 빈 줄 정리만

### 2. `de8a896d` — 리팩터(dflash2): QKV/MLP 배치를 DFlashConfig 상수로부터 유도 (2026-10-13)

**파일**: `bindings.cpp`(+21/−9)

**변경**: `9d021cd3`에서 넣은 폭을 더 이상 숫자로 적지 않고 `DFlashConfig`에서 유도합니다. 이전 숫자와 등가임을 확인했습니다. `hidden=5120`, `intermediate=17408`, `query_size=4096`, `kv_size=1024`이면 `query_key_value = 4096 + 1024 = 6144`, `gate_up = 2 * 17408 = 34816`, `down = 5120x17408`이고 QKV 오프셋 4096 / 5120, 폭 1024가 모두 일치합니다. 이 커밋은 동작을 바꾸지 않으며, 같은 날 조사한 장애의 원인에서 **제외**되었습니다.

### 3. `1ce94433` — tp2: 클라이언트 이탈 취소를 합의된 유닛 마스크 안으로

**파일**: `engine_core.h`, `tp_lockstep.h`, `src/serve/generation_service.cpp`(+56/−9)

**원인**: rank 0이 클라이언트를 마주합니다. 클라이언트가 prepare 도중 소켓을 닫으면 rank 0은 요청을 로컬에서 버렸지만 rank 1은 그대로 들고 있었고, rank 1은 락스텝 교환 지점에서 영원히 기다렸습니다. 관측된 증상: rank 0 `cancelled during prepare ... HTTP 499 | client disconnected`, rank 1 `req#1 started` 이후 진전 없음, 유닛 계수 `(4, 5)`에서 동결, GPU 한 장은 집합 통신에서 100% 헛돌고 다른 한 장은 0%, 프록시가 마지막에 `RESTART both ranks: no lockstep progress for 180s` 기록.

**변경**:
- `tp::lockstep_enabled()` — "이 프로세스가 락스텝 짝의 한 랭크인가"를 판단하는 단일 술어
- `preparation_cancellation_view()` — TP에서는 비어 있어 prepare 단계에서 랭크 로컬 이탈로 중단되지 않음
- `cancelled_on_this_rank()` / `local_cancellation_view()` — 다섯 곳의 랭크 로컬 판단이 로컬 상태 대신 합의 마스크를 읽음

**검증**: 932KB 페이로드(메시지 112건, 도구 20개, 프롬프트 171,675토큰)를 보내면서 소켓을 0.0초 / 0.05초 / 2.0초에 닫았습니다. 수정 전에는 엔진이 멈췄고, 수정 후에는 세 번 모두 HTTP 200을 반환했으며 rank 1이 3.4초 뒤 `done | cancelled`를 기록했습니다.

### 4. `9b3b98a2` — tp2: 대기열 만료는 의도이며, 랭크 로컬 제거가 아니다

**파일**: `engine_core.h`, `request_record.h`, `src/serve/generation_service.cpp`

**원인**: 진입 기한을 벽시계로 판정합니다. 만료된 요청을 로컬 대기열에서 빼면 어느 요청이 진입하는지가 달라지는데, 두 랭크가 서로 다른 순간에 그 판단에 이를 수 있습니다. 운영 환경에서 이 창은 실측 0건이었고(프록시가 `max_inflight=1`로 직렬화, 관측된 최대 대기열 대기는 51밀리초), 결함은 잠복 상태였지만 도달 가능했습니다.

**변경**:
- `QueueTerminationCause { None, Cancelled, QueueExpired }`와 요청 레코드의 `expired` 플래그 추가
- 네 곳의 만료 판정은 플래그만 세움. TP에서는 항목을 빼거나 제출을 거부하지 않음
- TP에서 `submit`은 `pending_deadline`과 `outstanding_ >= max_outstanding_`으로 거부하지 않고, 서비스 계층 용량 거부도 우회합니다(대칭 상한은 프록시가 파견 전에 강제)

**검증**: `expiry_repro.py --skew-ms 90000`으로 rank 1이 rank 0보다 90초 먼저 만료되게 했습니다. 수정 전: 유닛이 `(5, 4)`에서 동결되고 재시작 반복. 수정 후: 두 랭크 모두 HTTP 200, 유닛 `(18, 18)`, 엔진 생존.

### 5. `68c3eb7b` — tp2: 레인을 중단할 수 없으면 실패 대신 미루고, 만료 레인은 합의 마스크로 종료

**파일**: `engine_core.h`, `src/runtime/engine/resource_manager.h`

**원인(실측으로 찾은 후속 결함)**: 만료가 합의 종료 마스크에 실리자 `Program::abort`가 `clear_lane_strict`에 닿았고, 이 함수는 레인이 준비된 프롬프트 상태를 들고 있으면 거부했습니다(`std::logic_error("Program did not consume aborted sequence")`). 이어서 워커의 포괄 예외 처리가 `fail_all_locked`를 불러 엔진 전체를 503으로 만들었습니다. 이는 고치려던 멈춤보다 더 나쁩니다. 계측 결과 거부된 유닛은 `state=2 (Prefill) computed=51 capture=0`, 즉 진입한 프롬프트 경계에 도달한 유닛이었습니다.

**변경**:
- `ResourceManager::abort()` → `try_abort()`가 `std::optional<AbortResult>` 반환. 거부되면 레인을 건드리지 않고 종료를 미룸
- 미뤄진 요청은 `complete_success()`의 재정의가 끝내고, 503 `request_queue_timeout`으로 보고해 HTTP 계약을 지킴
- 만료는 합의 마스크 상위 비트(`kExpiryMaskShift = 32`)로 전달하고 사유를 서로 채택해, 두 랭크가 같은 유닛에서 같은 레인을 종료하고 같은 사유를 보고

**검증**: `expiry_repro.py --skew-ms 90000` → 두 랭크가 1.24초에 HTTP 503 `request_queue_timeout`, 유닛 `(12, 12)`, 엔진 생존. `expiry_contract.py` → 최소 한 요청이 503 `request_queue_timeout`을 받고, 모든 요청이 응답했으며, 엔진 생존.

### 6. `e5860473` — tp2: 짝을 멈추게 두지 말고 랭크 장애를 전파

**파일**: `engine_core.h`, `tp_lockstep.h`(+45/−1)

**원인**: 엔진 워커가 실패한 뒤에도 그 랭크는 살아서 503만 돌려주었고, 짝은 기다렸습니다. 주입 실험 실측: 짝은 **프리필 GPU 유닛 내부**에서 헛돌았고(호스트 점유 0.0%, 0마이크로초) 약 60초 뒤 `ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:520`의 `cudaErrorLaunchFailure`로 죽었습니다. 장애를 관측할 수 있는 호스트 교환 지점에 도달하지 못했기 때문입니다. 짝 쪽 플래그만으로는 집합 통신에 갇힌 짝을 구제할 수 없습니다.

**변경**:
- 원래 채움 공간에 `SharedBlock.poison` 추가. `published`는 프록시가 읽는 오프셋 64를 유지하고 블록 크기는 384바이트. 실측 오프셋 `published=64`, `poison=80`, `records=128`
- `poison_peer()`와 `publish_and_wait` 대기 루프의 오염 검사. 실패 시 `peer engine failed (poison set) while waiting for unit N` 보고
- `fail_all_locked()`가 짝을 오염시키고, HTTP 오류 응답이 나가도록 200밀리초를 비운 뒤, 사유를 표준 오류에 남기고 **랭크를 종료**합니다. 실패한 워커는 다시 서비스할 수 없고(`failed_` 유지), 프로세스 종료가 짝의 집합 통신과 프록시의 2초 생존 검사가 즉시 알아채는 신호입니다
- 시험용 훅 `NINFER_TP_FAIL_WORKER_AFTER_UNITS`(배포 시 0)

**검증**: 주입 실험에서 — 오염만 적용: 짝이 약 62초 뒤 CUDA 실행 실패로 죽음. 프로세스 종료까지 적용: 프록시가 `RESTART both ranks: a rank process exited`를 기록하고 짝 전체가 **19.4초**에 복구.

### 7. `1e9c6c1a` — tp2: 레인 중단 가능 여부를 추정하지 말고 Program에 묻기

**파일**: `engine_core.h`, `resource_manager.h`, `runtime.h`, `api_impl.h`, `program.h`, `program_impl.h`(+35/−8)

**원인**: 엔진이 토큰 계수(`computed_prompt_tokens`와 진입한 프롬프트 접미사 비교)로 레인 정리 가능 여부를 추정했습니다. 실측 결과 거부 여부는 Program이 들고 있는 준비 상태가 결정하며, 엔진이 계수로 흉내 낼 수 없습니다.

**변경**:
- `Program::can_abort(sequence)`를 네 계층에 노출. `abort()` 자체의 가드(트랜잭션 열림, 시퀀스 유효성, 레인 생애주기)와 `clear_lane_strict()`가 의존하는 `can_clear_lane_strict`를 재사용
- `try_abort()`가 먼저 물어보므로, 미뤄질 종료는 레인 상태를 건드리지 않음
- 프리필 중단 가드가 Program의 판정을 사용. 관용적 대체 경로는 검사와 호출 사이의 상태 변화에 대비한 안전망으로 유지

**검증**: 전체 회귀 재실행 — 기한 발산 스트레스(두 랭크 1.24초에 503 `request_queue_timeout`, 유닛 `(12, 12)`, 엔진 생존), 만료 계약(503 `request_queue_timeout`), prepare 중 이탈(3.4초 후 취소, 후속 200), 디코드 중 이탈(1초 안에 취소, 후속 200), 정상 트래픽(200, 출력 29·64토큰).

### 검증 도구

| 도구 | 입증 내용 |
|---|---|
| `repro2.py <delay>` | prepare 중 클라이언트 이탈 시 두 랭크가 합의 마스크로 종료 |
| `expiry_repro.py --skew-ms N` | 랭크 로컬 시계 발산이 더 이상 대기열을 갈라놓지 않음 |
| `expiry_contract.py` | 만료 요청도 503 `request_queue_timeout`을 받음 |
| `decode_abort.py` | 디코드 중 이탈이 레인을 조기에 종료하고 엔진 생존 |
| `poison_ab.py --fail-after N` | 랭크 장애가 짝을 멈추게 두지 않고 전파됨 |

---

## 日本語

### 0. システムモデル（エンジン修正 5 件が存在する理由）

テンソル並列では、2 つのランクが**同じ GPU ユニットを同じ順序で**発行しなければなりません。ユニット交換（共有ブロックの `tp::Record`、合意されたキャンセルマスクと受け入れ予算）が、状態を一貫して変更できる唯一の経路です。本作業で定めた原則は次のとおりです。

> ランクローカルの入力（壁時計、ソケット状態、応答返却のタイミング）は**意図**を提起できるだけです。すべての待ち行列の変更、すべてのレーン終了、すべての受け入れ判断は、合意されたユニット交換を経由しなければなりません。

この原則が破られると両ランクが同時に停止します。一方は集合通信で空回りし、他方はホスト側の交換点で待ち続け、最終的に監視プロセスが両方を再起動します。

### 1. `9d021cd3` — ローカル調整: dflash2 QKV 6144/MLP 34816 レイアウト修正、プロキシ NVLink 探索の迂回、長い分割キー 1920（2026-10-08）

**ファイル**: `bindings.cpp`（+8/−8）、`tools/tp2/tp2_proxy.py`（+2）、`tools/tp2/shard_qwen38_27b.py`（−2）

**原因**: DFlash2 のドラフト層は融合テンソルを読みますが、ディスク上の重みは**完全な**融合幅を保持しており、ローダーはより狭い形状を束ねていました。そのためドラフトモデルは切り詰められたレイアウトを読んでいました。

**変更**:
- `query_key_value` の行数 3072 → 6144、`context_key` / `context_value` の行ビュー幅 512 → 1024
- `gate_up` の行数 17408 → 34816、`down` 5120x8704 → 5120x17408
- プロキシ: NCCL の点対点が無効なとき `NINFER_TP_NVLINK=0`（開発機に NVLink はなく通常の PCIe のみ）、`NINFER_SM70_LONG_SPLIT_KEYS=1920` を既定に
- 分割スクリプト: 空行の整理のみ

### 2. `de8a896d` — リファクタ(dflash2): QKV/MLP レイアウトを DFlashConfig 定数から導出（2026-10-13）

**ファイル**: `bindings.cpp`（+21/−9）

**変更**: `9d021cd3` で導入した幅を数値リテラルではなく `DFlashConfig` から導出します。以前のリテラルとの等価性を確認しました。`hidden=5120`、`intermediate=17408`、`query_size=4096`、`kv_size=1024` から `query_key_value = 4096 + 1024 = 6144`、`gate_up = 2 * 17408 = 34816`、`down = 5120x17408`、QKV オフセット 4096 / 5120、幅 1024 がすべて一致します。本コミットは挙動を変えず、同日に調査した障害の原因からは**除外**されました。

### 3. `1ce94433` — tp2: クライアント切断によるキャンセルを合意済みユニットマスクの内側に保つ

**ファイル**: `engine_core.h`、`tp_lockstep.h`、`src/serve/generation_service.cpp`（+56/−9）

**原因**: rank 0 がクライアントに面しています。クライアントが prepare 中にソケットを閉じると、rank 0 は要求をローカルで破棄しましたが rank 1 は保持し続け、rank 1 はロックステップ交換点で永久に待ちました。観測された兆候: rank 0 は `cancelled during prepare ... HTTP 499 | client disconnected`、rank 1 は `req#1 started` の後まったく進まず、ユニットカウンタは `(4, 5)` で凍結、一方の GPU は集合通信で 100% 空回り、他方は 0%、最後にプロキシが `RESTART both ranks: no lockstep progress for 180s` を記録。

**変更**:
- `tp::lockstep_enabled()` — 「このプロセスがロックステップ対の一方のランクか」を判定する単一の述語
- `preparation_cancellation_view()` — TP では空であり、prepare がランクローカルの切断で中止されない
- `cancelled_on_this_rank()` / `local_cancellation_view()` — 5 か所のランクローカル判断がローカル状態ではなく合意済みマスクを読む

**検証**: 932 KB のペイロード（メッセージ 112 件、ツール 20 個、プロンプト 171,675 トークン）を送り、ソケットを 0.0 秒 / 0.05 秒 / 2.0 秒で閉じました。修正前はエンジンが停止し、修正後は 3 回とも HTTP 200 を返し、rank 1 が 3.4 秒後に `done | cancelled` を記録しました。

### 4. `9b3b98a2` — tp2: 待ち行列の期限切れは意図であり、ランクローカルの除去ではない

**ファイル**: `engine_core.h`、`request_record.h`、`src/serve/generation_service.cpp`

**原因**: 受け入れ期限は壁時計で判定されます。期限切れの要求をローカル待ち行列から取り除くと、どの要求が受け入れられるかが変わり、2 つのランクが異なる瞬間にその判断へ到達し得ます。本番でのこの窓は実測 0 件でした（プロキシが `max_inflight=1` で直列化、観測された最大待ちは 51 ミリ秒）が、欠陥は潜在かつ到達可能でした。

**変更**:
- `QueueTerminationCause { None, Cancelled, QueueExpired }` と要求レコードの `expired` フラグを追加
- 4 か所の期限切れ判定はフラグを立てるだけ。TP では項目を除去せず、投入も拒否しない
- TP では `submit` が `pending_deadline` や `outstanding_ >= max_outstanding_` で拒否せず、サービル層の容量拒否も迂回します（対称な上限はプロキシが派遣前に強制）

**検証**: `expiry_repro.py --skew-ms 90000` で rank 1 を rank 0 より 90 秒早く期限切れにしました。修正前: ユニットが `(5, 4)` で凍結し再起動を繰り返す。修正後: 両ランクとも HTTP 200、ユニット `(18, 18)`、エンジンは生存。

### 5. `68c3eb7b` — tp2: レーンを中止できないときは失敗ではなく延期し、期限切れレーンは合意マスクで終了する

**ファイル**: `engine_core.h`、`src/runtime/engine/resource_manager.h`

**原因（実測で見つかった後続の欠陥）**: 期限切れが合意された終了マスクに乗ると、`Program::abort` は `clear_lane_strict` に到達し、レーンがステージ済みのプロンプト状態を保持していると拒否しました（`std::logic_error("Program did not consume aborted sequence")`）。続いてワーカーの包括的な例外処理が `fail_all_locked` を呼び、エンジン全体を 503 にしました。これは直そうとしていた停止よりも悪い結果です。計測では拒否されたユニットは `state=2 (Prefill) computed=51 capture=0`、すなわち受け入れ済みプロンプトのフロンティアに到達したユニットでした。

**変更**:
- `ResourceManager::abort()` → `try_abort()` が `std::optional<AbortResult>` を返す。拒否時はレーンに触れず終了を延期する
- 延期された要求は `complete_success()` の上書きが終了させ、503 `request_queue_timeout` として報告し HTTP 契約を保つ
- 期限切れは合意マスクの上位ビット（`kExpiryMaskShift = 32`）で伝達し理由を相互に採用することで、両ランクが同じユニットで同じレーンを終了し同じ理由を報告する

**検証**: `expiry_repro.py --skew-ms 90000` → 両ランクが 1.24 秒で HTTP 503 `request_queue_timeout`、ユニット `(12, 12)`、エンジンは生存。`expiry_contract.py` → 少なくとも 1 件が 503 `request_queue_timeout` を受信、全要求が応答、エンジンは生存。

### 6. `e5860473` — tp2: 対を止めたままにせず、ランク障害を伝播させる

**ファイル**: `engine_core.h`、`tp_lockstep.h`（+45/−1）

**原因**: エンジンワーカーが失敗した後もそのランクは生き続けて 503 を返し、相手は待っていました。障害を注入した実測では、相手は**プリフィルの GPU ユニット内部**で空回りし（ホスト占有 0.0%、0 マイクロ秒）、約 60 秒後に `ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:520` の `cudaErrorLaunchFailure` で停止しました。障害を観測できるホスト側の交換点に到達しなかったためです。相手側のフラグだけでは、集合通信に閉じ込められた相手を救えません。

**変更**:
- 元の詰め物領域に `SharedBlock.poison` を追加。`published` はプロキシが読むオフセット 64 を維持し、ブロックサイズは 384 バイト。実測オフセットは `published=64`、`poison=80`、`records=128`
- `poison_peer()` と `publish_and_wait` の待機ループ内の汚染検査。失敗時は `peer engine failed (poison set) while waiting for unit N` を報告
- `fail_all_locked()` は相手を汚染し、HTTP エラー応答が送出されるよう 200 ミリ秒待ち、理由を標準エラーに書いてから**このランクを終了**します。失敗したワーカーは再びサービスできず（`failed_` は保持）、プロセス終了こそが相手の集合通信とプロキシの 2 秒生存確認が即座に気づく信号です
- 試験用フック `NINFER_TP_FAIL_WORKER_AFTER_UNITS`（出荷時は 0）

**検証**: 障害注入時 — 汚染のみ: 相手は約 62 秒後に CUDA 起動失敗で停止。プロセス終了を追加: プロキシが `RESTART both ranks: a rank process exited` を記録し、対は **19.4 秒**で復旧。

### 7. `1e9c6c1a` — tp2: レーンを中止できるかを推測せず Program に問い合わせる

**ファイル**: `engine_core.h`、`resource_manager.h`、`runtime.h`、`api_impl.h`、`program.h`、`program_impl.h`（+35/−8）

**原因**: エンジンはトークン数（`computed_prompt_tokens` と受け入れ済みプロンプト接尾辞の比較）でレーンの消去可否を推測していました。実測により、拒否の可否は Program が保持するステージ済みプロンプト状態が決めており、エンジンがカウンタで模倣できるものではありません。

**変更**:
- `Program::can_abort(sequence)` を 4 層で公開。`abort()` 自身のガード（トランザクションの開放、シーケンスの有効性、レーンのライフサイクル）と、`clear_lane_strict()` が依存する `can_clear_lane_strict` を再利用
- `try_abort()` が先に問い合わせるため、延期される終了はレーン状態に触れない
- プリフィル中止ガードは Program の判定を使用。寛容なフォールバックは検査と呼び出しの間の状態変化に対する安全網として維持

**検証**: 全回帰を再実行 — 期限発散ストレス（両ランクが 1.24 秒で 503 `request_queue_timeout`、ユニット `(12, 12)`、エンジン生存）、期限切れ契約（503 `request_queue_timeout`）、prepare 中の切断（3.4 秒後にキャンセル、後続 200）、デコード中の切断（1 秒以内にキャンセル、後続 200）、通常トラフィック（200、出力 29 および 64 トークン）。

### 検証ハーネス

| ハーネス | 証明する内容 |
|---|---|
| `repro2.py <delay>` | prepare 中のクライアント切断で両ランクが合意マスクにより終了する |
| `expiry_repro.py --skew-ms N` | ランクローカルの時計発散が待ち行列を分断しなくなる |
| `expiry_contract.py` | 期限切れ要求も 503 `request_queue_timeout` を受け取る |
| `decode_abort.py` | デコード中の切断がレーンを早期終了させ、エンジンは生存する |
| `poison_ab.py --fail-after N` | ランク障害が対を止めずに伝播する |
