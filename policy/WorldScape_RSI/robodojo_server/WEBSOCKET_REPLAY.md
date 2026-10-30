# WSP WebSocket 会话恢复协议

适用于 `serve_robodojo.sh flash|memory` 的 `official / isolated` 模式。
每张所选 GPU 一个会话，每会话容量 1–10，默认 3；每个动作需要一次观测反馈。
多环境必须提供稳定的 `env_idx`，同回合只缩批，增大需 reset。
结束或过期的 trial 不可复活，新回合使用新的 `trial_id`。
`legacy` / `explicit` 模式不提供本扩展。

## 内部鉴权与能力协商

按用户后续要求，两个 WSP isolated 入口默认支持恢复协议，无需额外设置口令。按现有内部网络
准入使用这些端口，不能直接公开。Router 在同一 WebSocket 内发送官方二进制 HELLO，payload 如下：

```json
{"replay":{"protocol":"r2si-replay-v1"}}
```

HELLO 外层应带本回合 evaluation_id/trial_id。初次连接可省略预期 boot；**重连必须在
replay 中携带原 server_instance_id**。认证失败、协议不支持、boot 不一致会明确报错。
尚未创建的会话不分 GPU，也没有伪造的 worker/session generation。

认证后的 HELLO_ACK 增加：

```json
{
  "server_instance_id":"<boot>",
  "replay_protocol":"r2si-replay-v1",
  "replay_retention_s":600,
  "disconnect_grace_s":180,
  "fingerprint":"tagged-sha256-v1",
  "durability":"process_memory",
  "token_required":false,
  "session_state":"not_created",
  "session_generation":null,
  "worker_generation":null
}
```

普通 HELLO 只有原协议基础字段和进程 ID，不公开恢复能力、模型/节点/容量/配置。
未设置 token 时，新私有恢复协议默认可协商，依赖现有内部网络准入；原有 SDK 调用仍可用。
若显式设置环境变量 `POLICY_REPLAY_TOKEN`，则启用附加口令校验，Router 的 replay 中必须带相同 token。
内部 ACK 的 token_required 表示是否需要该附加口令；未协商的普通客户端不会被强制改用新协议。
设置 token 后 `online_info`、`session_status`、`get_runtime_info` 也要求已认证的连接；
启动器 `--verify` 会从同一环境变量读取凭据并完成只读认证。凭据不写入身份文件或日志。
这是**单个可信 Router 信任域**，不是多租户授权系统；不能把共享 token 发给不互信租户，
不能将私有 ACK 或状态查询透传到公网客户端。

所有协商后的状态变更请求 payload 加上：

```json
{"_replay":{"server_instance_id":"<boot>","session_generation":"<session>","worker_generation":"<worker>"},
 "func_name":"get_action_batch","obs":[37,30,23,16,9,2]}
```

首次 RESET/PREPARE_CASE 尚无模型会话时只带 boot。创建响应 payload.replay 返回三种 ID。
创建响应丢失后，可重发原始创建帧，也可先查状态取回代际；后续新请求必须携带完整代际。
绑定字段在送入模型前剥离，不改变模型业务参数。新 socket 的 HELLO 接管连接所有权；旧
socket 可读相同原请求的结果，但新的状态变更会收到 CONNECTION_FENCED。
官方旧 SDK 的 HELLO 不带 trial_id，因此仅对未协商的 legacy 客户端，允许首个 scoped 请求
附着到尚在宽限期的 detached 会话；它不能绕过已协商会话的认证和代际校验。

## 请求账本与生命周期

去重键为 `(evaluation_id, trial_id, request_id)`。在任何模型派发之前登记 fingerprint 和
pending 任务，无 await 的登记区保证原子性。每会话按到达顺序获得会话锁与固定 GPU 锁，
多个独立 GPU 并行；不持全局锁等推理。取消一个 socket/等待者不会取消共享执行任务。

模型调用完成后，先用官方 codec 编码并保存不可变响应字节，再回复。重发不重新 decode 图像、
追加历史、计算模型、commit 或出队动作。再次返回的 envelope 保持原 request/trial/evaluation、
动作、latency 和 sent_at。响应过期仅删除完整字节，保留 digest 和 executed 标记，永不降为 missing。
未接受的准入失败不会派发；已登记的容量失败保存原错误，若要重新申请容量应使用新请求 ID。

账本是**有界进程内内存**，没有 SQLite/WAL/对象存储持久化，也不宣称跨进程 exactly-once。
正常结束与超时结束的 tombstone 保留到本进程退出；达到记录/字节上限拒绝新请求，不清旧记录
腾位置。pending 不受响应 TTL 清理。记录登记失败返回 NOT_ADMITTED；执行后编码/保存异常保留
已有的真实响应字节和 uncertain 标记，trial 标为 lost，不能再次执行原请求。
此时需要核对/新 trial 重测，不能将“保存异常”解释为“动作没生成”。

断线立即标 detached；默认 180 秒保留模型状态、动作缓存及在途请求，独立于通用 idle timeout。
超过宽限期先进入 closing，与重连使用同一事件循环内的原子检查；然后等待对应 session/GPU
锁完成收尾。正在执行的 GPU 不会提前给其他会话。清理失败则标 lost 并保留隔离槽，不循环
重试释放或把它冒充空闲。重复结束不再次释放；结束后晚到的 reset/update/action 被拒绝。

server_instance_id 每次构造服务生成；session_generation 每次合法分配新模型会话生成，
同回合 reset 不改变它。worker_generation 与实际 worker 对象、进程对象及 PID 绑定，变化或
死亡使原会话 lost。WSP 的项目内 pool 将致命错误/超时限制在
对应 worker，其他 GPU 的模型上下文保留。没有自动重建 worker 或会话迁移。

### 状态查询

认证连接使用 CALL `_replay_status`，外层仍为原 evaluation/trial，新查询 ID，payload：

```json
{"func_name":"_replay_status","obs":{"request_id":"<原请求ID>"},
 "_replay":{"server_instance_id":"<原boot>"}}
```

返回 CALL_RESULT `{ok:true,result:{state,digest,session_state,server_instance_id,
session_generation,worker_generation}}`。查询不创建模型/会话、不抢 GPU、不推进动作。
查询只要求 boot，便于找回创建响应中的代际；调用方必须核对返回值。

| state | 行为 |
| --- | --- |
| missing | 同 boot 内完整账本覆盖下未登记；允许同 ID/同内容提交，最终仍由原子登记去重 |
| pending | 排队/执行中；原请求重发附着同一个任务 |
| done | 原响应仍保存；重发原帧取回原响应，包括确定错误 |
| expired | 已登记并完成但响应过期；REPLAY_EXPIRED，绝不重新执行 |
| uncertain | 保存失败、会话 lost/closing/ended 等；不得自动恢复 |

session_state 为 not_created / active / detached / closing / ended / lost。
终态查询一律不授予继续执行权限；相同已完成请求可能仍能读到旧缓存，但不会复活会话。
`_replay_status` 的 missing 只对返回的 boot 有效，不能跨 boot 使用。

错误保持官方 ERROR envelope/code，机器判断使用 `payload.details.reason` 和 `admitted`：
REQUEST_ID_CONFLICT、SESSION_ENDED、SESSION_LOST、SERVER_INSTANCE_CHANGED、
WORKER_GENERATION_CHANGED、REPLAY_EXPIRED、REQUEST_UNCERTAIN、NOT_ADMITTED、
CAPACITY_EXHAUSTED、BATCH_CAPACITY_EXHAUSTED、CONNECTION_FENCED、TASK_IDENTITY_CONFLICT、
AUTH_REQUIRED、REPLAY_UNSUPPORTED、INVALID_REQUEST、MODEL_ERROR。
`admitted=false` 仅指该帧没有派发模型，不能推断相同 ID 之外的其他在途请求没有执行。
MODEL_ERROR 保留服务端异常证据，外部不返回文件路径或模型 traceback，并标记 batch_state=uncertain。批量调用异常不拼接部分成功，
按整个请求失败处理；原固定批次保护仍会在序列错误时要求显式 reset。

## 指纹规范 tagged-sha256-v1

输入是官方 codec 解码后的 Frame.to_wire_dict()；包含 message_type、message_id、evaluation_id、
trial_id、action_case_id、repeat_index、step、全部业务 payload。仅排除顶层 sent_at 与
payload._replay（后者每次独立核验）。图像是解码 envelope 后、图像预处理前的原始 bytes/数组；
重新压缩的 JPEG 即使像素近似相同也不是同一请求，应尽量保存并重发原编码帧。

所有长度为 8 字节无符号大端，`P(tag,data)=tag || uint64be(len(data)) || data`：

| 值 | 编码 |
| --- | --- |
| null / true / false | 单字节 n / t / f |
| Python int | P(i, 十进制 ASCII)，不带前导零 |
| Python float | d + IEEE754 binary64 大端，保留负零和位模式 |
| str | P(s, UTF-8)，不做 Unicode 归一化 |
| bytes/bytearray/memoryview | P(b, 原字节) |
| list/tuple | P(l, 依次拼接元素编码) |
| map | P(m, 按**编码后的 key 字节**排序，拼接每对 key/value 编码) |
| numpy array/scalar | P(a, encode(dtype.str) + encode(shape列表) + P(b,C顺序原始字节)) |

NumPy 标量按 shape=[] 数组处理，np.bool_ 按 bool；对象/结构 dtype 拒绝。
数组 dtype 字节序、shape 与原字节均参与摘要，不能把 float32 与 float64 混同。
最后 SHA-256，十六进制小写。固定测试向量位于 `tests/replay_vectors.json`；
例如 null 的 SHA-256 为 `1b16b1df538ba12dc3f97edbb85caa7050d46c148134290feba80f8236c83db9`。
另有 NumPy dtype/shape、图像变更、env 顺序与 codec 往返测试。

## Batch、容量与计时

`env_idx` 是稳定身份；update 可缩批/重排，不能把已退出环境加回当前历史。
get_action_batch 必须恰好包含最新活跃环境集合，响应顺序严格对应请求顺序。例如最新观测
env_idx=[2,9,16,23,30,37]，请求 `[37,30,23,16,9,2]`，响应六条 chunk 同序排列。
逐动作更新后缩为 `[9,30]`，后续动作只能取这两条。reset 清空原历史才可增加环境。
WSP chunk 保留各自原配置长度；每个动作仍是官方 joint 字典，未改单位。

task_name/task_config/task_id/action_case_id 在首次有值时固定为本 trial 的身份；之后不允许覆盖。
trialless PREPARE_CASE 通过 `(evaluation_id,action_case_id)` 保存身份并去重，后续同 case 的
trial 继承它，不分配提前 GPU。任务 `_random` 与普通任务不等同。**只检查显式任务字段**，
不从图片/自然语言推断任务配置；若调用方不携带这些字段，须由 Router 保证同配置并批。

| 设置 | 默认/边界 |
| --- | --- |
| max_sessions | 8，启动参数 `--max-sessions`，每卡1会话 |
| max_envs_per_session | 默认3，可 `--envs-per-session 10`；每台上限80环境 |
| WebSocket | 最多120连接，每帧最多64MiB；协议连接不等于模型槽位 |
| GPU 会话等待队列 | 0；槽满立即 CAPACITY_EXHAUSTED |
| 已登记 pending | 总128、每会话16；参数 `--replay-max-pending` / `--replay-per-session-pending` |
| 响应保留 | `--replay-retention-seconds 600`；启动要求不少于600且覆盖宽限期 |
| 断线宽限 | `--disconnect-grace-seconds 180` |
| 通用空闲 | 原 `--session-idle-seconds 1800` |
| 账本 | `--replay-max-records 200000`、`--replay-max-mib 512` |
| 响应预留 | 每 pending 256KiB，另预算3倍编码输入大小与记录元数据；超大异常输出标 uncertain |

字节预算是账本/请求的准入预算，不是 Python/codec/网络缓冲/模型内存总 RSS 硬限制。
达到记录上限不会自动轮转账本；需要停止新回合、受控排空后换 boot，不能清缓存继续旧 trial。
队列有数量上限，不因 socket 等待超时取消已登记请求；原 worker RPC 超时保留（来自部署 YAML 的 multi_gpu.worker_timeout_s），超时隔离相应 worker。Router 的连接/握手/查询超时与总恢复预算应独立，
总恢复预算应小于宽限期，不能每次重连重置推理截止时间。

内部日志记录请求/回合、digest、boot/session/worker、接收至保存时间、session/GPU 等待、
模型 RPC 墙钟时间、响应保存与发送、去重命中、断线/释放原因；不记录图像或 token。
模型 RPC 计时在实际执行线程内，包含模型进程通信，**不是 CUDA kernel 时间**。
每卡独占下 session 等待也就是该 GPU 的请求等待，不声称存在额外 GPU 并发队列。
状态分别提供连接数、会话数、环境容量、waiting、executing、pending、账本占用与重连计数；
执行结束立即清除 executing。生产入口将日志交给4096条有界后台队列，慢日志输出不阻塞推理事件循环；
队列满时记录 dropped_log_records，日志为尽力输出，不能替代去重账本。日志由部署进程现有输出
重定向收集，账本不依赖磁盘日志成功。

## 验证与恢复边界

当前随包提供 Harness CPU 测试与官方 Agent WebSocket 冒烟测试，运行方法与覆盖范围见
[服务器说明](README.md)。它们不验证本恢复扩展的请求去重、断线重连、连接接管、
鉴权、worker 隔离或批量缩减；这些能力仍需单独验证，也不代表机器人闭环成功或 GPU 容量验证。

服务端账本只保存在进程内存中。部署新实例、回滚或服务崩溃后必须使用新的 boot 和 trial，
不能将旧请求伪装成新请求继续执行。客户端需要持久保存请求 ID、原始帧、boot 标识及响应消费记录，
才能判断一次响应是否已执行；本服务不负责机器人端动作的重复消费防护。
