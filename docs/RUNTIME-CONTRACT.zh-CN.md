# 运行态契约

[English](RUNTIME-CONTRACT.md) · **简体中文**

同一个训练有三双眼睛在看：WebUI 看板、HTTP 接口、`moss-lora status`。三者能保持一致，不是因为互相通知，而是因为谁都不留自己的一份状态。磁盘上的工作区就是状态，其余都是读。

本文规定这份状态的文件格式、把文件读成状态的规则，以及证明三个读法仍然一致的那个测试。

## 为什么要有这份契约

一次训练跑几个小时，而启动它的连接只活几秒。Tailscale 上的 SSH 会掉线，浏览器标签会刷新，agent 的 shell 会退出。如果运行只活在那条连接里，连接一断看板就白屏，而显卡还在满功率跑。

所以训练进程边走边把状态写进磁盘。任何读者从同一批文件重建同一幅画面，服务重启、页面刷新、终端退出都不用付出代价。

## 目录布局

```
workspaces/<workspace>/
  .gpu.lock                     唯一的持有者：正在用显卡的进程
  runs/<run>/
    config.yaml                 这次训练实际用的配置
    training.json               心跳：谁在训、训到哪
    metrics.jsonl               每步一行 JSON
    checkpoints/  samples/
  logs/
    <job>.json                  作业记录：是谁启动了这条进程
    <run>.log                   worker 的标准输出与标准错误
```

`training.json` 是运行自己的口供，`logs/<job>.json` 是启动器留下的记录。只有作业记录没有心跳，说明进程在看到模型之前就挂了；只有心跳没有作业记录，说明是 `moss-lora train` 直接在终端起的。

## 一次运行用的是哪份配置

两个问题长得很像，但答案不能共用：

| 问题 | 源头 |
|---|---|
| 下次训练用什么 | `workspaces/<workspace>/configs/<配置名>.yaml`，你编辑的那份 |
| 这次训练用了什么 | `workspaces/<workspace>/runs/<run>/config.yaml` |

第二份由训练进程在构造函数里写，早于加载任何权重，别的地方都不写。里面是它实际跑起来的配置：预设已应用、命令行覆盖已应用、路径已解析、预览种子已定值。

读的地方只认这一份。没有这个文件的 run 就是没有配置记录，面板如实显示为空。早先的版本靠名字猜：先找 `logs/` 或 `configs/` 里同名的文件，再退到工作区默认值。结果是一个叫 `foo`、用 `bar.yaml` 训出来的 run 被标成 `foo.yaml`，而名字对不上任何配置的 run 被标成 `workspace.yaml`。

这样一来，配置文件是怎么写出来的就不再重要。网页保存也好、手写也好，它都只是输入；用了什么，记在 run 目录里。

## 心跳

`runs/<run>/training.json` 采用原子写：先写临时文件，再 `os.replace`。读者要么看到旧记录，要么看到新记录，不会读到一半新一半旧。

| 字段 | 含义 |
|---|---|
| `pid` | 持有这次运行的进程 |
| `step`、`total` | 进度计数 |
| `status` | `starting`、`running`、`stopping`、`done`、`failed`、`stopped` |
| `phase` | 这一步在做什么：`train`、`preview`、`export`、`done` |
| `run_id` | 运行名，与目录名同一字符串 |
| `started`、`time` | 启动时刻、最后一次心跳时刻 |
| `exit_code` | 仅终态：`0` 成功，`130` 被中断，其余为失败 |
| `finished` | 仅终态：结束时刻 |

训练进程每几十步重写一次，每次阶段切换也重写。写这个文件很便宜，而它是系统唯一需要的存活信号。

## 从文件到状态

只有一个函数做决定。`runtime.classify()` 接收一条记录，返回 `state`、`pid_alive`、`age`、`stale`；代码里没有第二处解释心跳的地方。

| 记录内容 | 进程 | 结果 |
|---|---|---|
| 有 `exit_code` | — | 0 为 `done`，状态说停则为 `stopped`，其余为 `failed` |
| 终态状态 | — | 该状态 |
| `stopping` | 活着 | `stopping` |
| `stopping` | 已死 | `stopped` |
| 其他 | 活着 | `running` |
| `starting` | — | 启动 30 秒内为 `starting`，否则 `interrupted` |
| 其他 | 已死 | `interrupted` |
| 空 | — | `unknown` |

存活只问内核。早先的版本还会把命令行和包名比对，结果是：用 `python -c` 包一层的 worker 明明活着却被判死；而 pid 复用的情况下它也证明不了谁死了。真正挂掉的训练不会再刷新心跳，`stale` 一分钟内就会把它标出来。

`stale` 标记心跳超过 60 秒而进程仍在的情况：可能卡在漫长的采样阶段，也可能已经僵死。看板显示这个标记，而不是替它编一个状态。

## 一次运行，一行

一次训练结束后会留下两份描述它的文件：作业记录和心跳。两者按 run 目录合并，不重复列出。

裁决权归训练进程。它知道退出码；作业记录只知道自己的子进程没了，这区分不了正常跑完和崩溃。两者冲突时以心跳为准。

进度来自 `metrics.jsonl`，从文件末尾往前读。eval 行没有 loss，读的时候跳过——否则一个跑完的运行会显示一个从未存在过的 0 loss。

## 显卡锁

两次训练不能共用一个显卡，而硬起的第二个进程通常死在第一个的显存分配错误里，看不出来是谁的问题。`workspaces/<workspace>/.gpu.lock` 负责挡住：

- 训练进程碰模型之前用 `O_CREAT | O_EXCL` 建这个文件。文件存在且持有者活着，就是 `GpuBusy`。
- 文件存在但持有者已死，说明上一次崩了：直接接管，不等人类去删文件。
- 之后还要扫一遍同级工作区，所以两个工作区也不会同时开训。
- 锁被占时 `POST /api/jobs` 返回 `409` 和 `Retry-After: 10`，并说明持有者是谁。
- 训练进程在 `finally` 里删锁，且只删自己持有的那把。

## 三个读法

| 读法 | 入口 |
|---|---|
| 命令行 | `moss-lora status [--workspace W] [--live] [--json]` |
| HTTP | `GET /api/jobs`、`GET /api/jobs/{id}` |
| 看板 | 每 1.2 秒轮询 `/api/jobs` |

三个入口都调用 `runtime.scan_jobs(root)`。命令行把行打印出来，接口补上看板仍在读的旧字段名，看板把它们画出来。以后再加第四个读法，要做的是调用同一个函数，而不是写第四种解释。

`moss-lora status` 不需要服务地址，所以 WebUI 停了也能用，发起运行的那台机器没了也能用。

## 验收测试

`tests/test_runtime_state.py` 覆盖投影层，不需要显卡：状态表、原子写、记录与心跳的合并、metrics 尾部、显卡锁。

`python .tmp/check_three_way.py` 覆盖它上面那一层。脚本起一个 worker，写出启动器会写的作业记录，然后用命令行、`/api/jobs` 和直接读盘三种方式读同一次运行——运行中读一次、几秒后再读一次、`SIGINT` 之后读一次。三次采样里三者的状态、pid、步数必须完全一致。谁先知道了别人不知道的事，它就会立刻报错。

一次真实运行与它的三个视角：

```
$ moss-lora status
STATE         AGE         STEP      PID   EXIT  JOB
running      2s         4210/8000  84120     -  <WORKSPACE>/<run>
                   loss 0.3148  avg 0.3302  <sec>s/step  vram <peak> GiB
gpu lock: <WORKSPACE>  pid 84120  kind train  run <run>
```

看板显示这一行，接口返回这一行，磁盘上的文件说的也是同一件事。没有任何一环需要去问一个活着的服务才知道发生了什么。
