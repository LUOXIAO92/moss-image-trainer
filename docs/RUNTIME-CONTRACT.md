# Runtime Contract

Three programs watch the same training: the WebUI dashboard, the HTTP API, and
`moss-lora status`. They agree because none of them keeps its own copy of the
truth. The workspace on disk is the state; everything else reads it.

This document defines that state, the rules that turn a file into a status, and
the one test that proves the three readers still agree.

## Why the question comes up

A training run lasts hours and the connection that started it lasts seconds. An
SSH session over a tailnet drops, a browser tab reloads, an agent's shell exits.
If the run lived only in the memory of the process that spawned it, a dropped
connection would blank the dashboard while the GPU kept working.

So the trainer writes its state to disk as it goes. Every reader rebuilds the
same picture from the same files, and a restart of anything — server, browser,
terminal — costs nothing.

## Directory layout

```
workspaces/<workspace>/
  .gpu.lock                     one holder: the process using the GPU
  .cache-progress.json          cache build progress: phase, done, total
  runs/<run>/
    config.yaml                 the config this run was trained with
    training.json               heartbeat: who is training and how far
    metrics.jsonl               one JSON object per step
    checkpoints/  samples/
  logs/
    <job>.json                  job record: what launched this process
    <run>.log                   stdout and stderr of the worker
```

`training.json` is the run's own voice. `logs/<job>.json` is the launcher's
record of starting it. A job can exist without a heartbeat — it failed before
the model loaded — and a heartbeat can exist without a job record, because
`moss-lora train` started it directly.

## Which config a run used

Two questions look alike and must not share an answer:

| question | source |
|---|---|
| what will the next run use | `workspaces/<workspace>/configs/<config>.yaml`, the file you edit |
| what did this run use | `workspaces/<workspace>/runs/<run>/config.yaml` |

The trainer writes the second one in its constructor, before it loads a weight,
and nothing else writes it. It holds the config as it actually ran: presets
applied, command-line overrides applied, paths resolved, preview seeds fixed.

Readers take it and only it. A run without the file has no config, and the
panel says so. Earlier versions guessed from the run's name — a same-named file
in `logs/` or `configs/`, then the workspace defaults — which labelled a
command-line run with whichever file happened to share its name. A run named
`foo` trained from `bar.yaml` came out as `foo.yaml`; a run named after nothing
inherited `workspace.yaml`.

The upshot is that how the config file was written stops mattering. Saved in
the WebUI or edited by hand, the file is the input; the run directory keeps the
record of what was used.

## Cache builds

A cache build writes `.cache-progress.json` in the workspace: the phase it is in
(`latent`, `text`), how many images or captions it has done, the rate, and a
`history` entry per phase so a reader that arrives late still sees the finished
latent bar while the text pass runs. Terminal records carry an exit code, the
same way a training heartbeat does.

`GET /api/cache` returns that record as `job`, so the cache panel shows a build
started in a terminal, and the page polls it faster than the page heartbeat
while the build is live.

## The heartbeat

`runs/<run>/training.json` is written atomically: a temporary file, then
`os.replace`. A reader sees the old record or the new one, never half of each.

| field | meaning |
|---|---|
| `pid` | process that owns the run |
| `step`, `total` | progress counters |
| `status` | `starting`, `running`, `stopping`, `done`, `failed`, `stopped` |
| `phase` | what the step means: `train`, `preview`, `export`, `done` |
| `run_id` | run name, the same string as the directory |
| `started`, `time` | wall clock: started, last heartbeat |
| `exit_code` | terminal only: `0` success, `130` interrupted, other = failure |
| `finished` | terminal only: when it ended |

The trainer rewrites the file every few dozen steps and at each phase change.
Writing it is cheap and it is the only liveness signal the system needs.

## From file to state

One function decides. `runtime.classify()` takes a record and returns
`state`, `pid_alive`, `age`, and `stale`; nothing else in the codebase
interprets a heartbeat on its own.

| record says | pid | result |
|---|---|---|
| `exit_code` set | — | `done` for 0, `stopped` if the status says stopping, `failed` otherwise |
| a terminal status | — | that status |
| `stopping` | alive | `stopping` |
| `stopping` | gone | `stopped` |
| anything else | alive | `running` |
| `starting` | — | `starting` within 30 s of spawn, else `interrupted` |
| anything else | gone | `interrupted` |
| nothing | — | `unknown` |

Liveness asks the kernel and only the kernel. An earlier version also matched
the command line against the package name, which reported a live worker as dead
whenever it was wrapped in `python -c` and could not prove a dead one was dead
anyway: pids get reused. A trainer that dies without writing a terminal status
stops refreshing its heartbeat, and `stale` catches it a minute later.

`stale` marks a heartbeat older than 60 seconds whose process is still alive: a
run inside a long sampling phase, or one that is wedged. The dashboard shows the
flag rather than inventing a state for it.

## One run, one row

A finished training leaves two files that describe it: the job record and the
heartbeat. They are linked by run directory, not listed twice.

The trainer owns the verdict. It knows the exit code; the job record only knows
that the pid it spawned has gone away, which cannot distinguish a clean finish
from a crash. When the two disagree, the heartbeat wins.

Progress comes from `metrics.jsonl`, read backwards from the end of the file.
Eval lines carry no loss and are skipped — otherwise a finished run would
display the zero loss of an eval record that never held one.

## The GPU lock

Two trainings cannot share one GPU, and a second one that starts anyway fails
somewhere inside the first one's allocation error. `workspaces/<workspace>/.gpu.lock`
prevents that:

- The trainer creates the file with `O_CREAT | O_EXCL` before it touches the
  model. Existing and alive means `GpuBusy`.
- Existing and dead means the previous holder crashed; the lock is reclaimed
  instead of waiting for a human to delete a file.
- A scan of sibling workspaces follows, so two workspaces cannot train at once.
- `POST /api/jobs` refuses with `409` and `Retry-After: 10` while a lock is held,
  and says who holds it.
- The trainer removes the lock in a `finally`, only when it still owns it.

## The three readers

| reader | entry point |
|---|---|
| CLI | `moss-lora status [--workspace W] [--live] [--json]` |
| HTTP | `GET /api/jobs`, `GET /api/jobs/{id}` |
| dashboard | polls `/api/jobs` every 1.2 seconds |

All three call `runtime.scan_jobs(root)`. The CLI prints the rows; the API adds
the legacy field names the dashboard still reads; the dashboard draws them.
Adding a fourth reader means calling the same function, not writing a fourth
interpretation.

`moss-lora status` takes no server address, so it works when the WebUI is down
and when the machine that started the run is gone.

## The acceptance test

`tests/test_runtime_state.py` covers the projection layer without a GPU: the
state table, atomic writes, the merge of record and heartbeat, the metrics tail,
the lock.

`python .tmp/check_three_way.py` covers the layer above it. The script starts a
worker, writes the job record a launcher would write, then reads the same run
through the CLI, through `/api/jobs`, and straight off disk — while it runs,
again a few seconds later, and after `SIGINT`. The three must report the same
state, pid, and step at every sample. It fails loudly the moment one of them
learns something the others do not.

A live run and its three views:

```
$ moss-lora status
STATE         AGE         STEP      PID   EXIT  JOB
running      2s         4210/8000  84120     -  <WORKSPACE>/<run>
                   loss 0.3148  avg 0.3302  <sec>s/step  vram <peak> GiB
gpu lock: <WORKSPACE>  pid 84120  kind train  run <run>
```

The dashboard shows that row, the API returns it, and the files on disk say the
same thing. Nothing polls a live server to find out.
