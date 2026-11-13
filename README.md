<p align="center">
  <img src="https://raw.githubusercontent.com/zweck/gpupark/main/assets/gpupark-logo.jpg" width="480" alt="gpupark">
</p>

# gpupark

**One GPU. Many tenants. Lease it, and park what's idle.**

`gpupark` is a single-node GPU lease broker. A resident model (or any service) is
registered as the *baseline*: it holds the card while nothing else needs it, and
gives it up the moment a job asks. When the queue drains, the baseline comes back.

It exists because "run ComfyUI next to a 21 GB local LLM on one 24 GB card" has no
good answer: llama-swap only swaps LLM backends, Slurm/Kueue are cluster schedulers,
and `flock` scripts cannot stop and restart a resident service.

```
$ gpupark run --need 22 --ttl 1800 -- comfy run workflow.json
queued 3f2a (need 22.0 GB, priority 0)
granted
... the job's own output ...
```

The broker parks the baseline before the job starts and restarts it when the queue
drains; every request is logged as one line (`gpupark: uid=1000 pid=4812 submit comfy …`),
which is the only audit trail.

## Install

**One line, Linux and macOS** — installs into `~/.local/share/gpupark`, links
`~/.local/bin/gpupark`, no sudo, nothing added to your system Python:

```sh
curl -fsSL https://raw.githubusercontent.com/zweck/gpupark/main/install.sh | sh
```

Then:

```sh
gpupark config init         # write ~/.config/gpupark/workloads.yaml
gpupark doctor              # check nvidia-smi, systemd, the daemon
gpupark install-service     # Linux: run the broker as a systemd --user unit
```

Uninstall with `curl -fsSL .../install.sh | sh -s -- --uninstall`; it refuses to delete
anything that is not a gpupark install and leaves a foreign `gpupark` binary alone.
The installer pins the latest release tag — set `GPUPARK_ALLOW_CUSTOM_SOURCE=1` to
override `GPUPARK_REF`/`GPUPARK_REPO`. Not on PyPI yet, so `pip install gpupark` will
not find it.

macOS: there is no `nvidia-smi`, so set `gpu_probe: none` in the config — the
queue still serialises jobs, it just cannot check VRAM. `gpupark daemon` runs in
the foreground (systemd is Linux-only).

## Commands

```bash
gpupark run --need 22 --ttl 1800 -- ./render.sh     # queue a command
gpupark lease comfyui --for 3600                    # hold a registered unit
gpupark pause [ID]                                  # give the card back, keep your place
gpupark resume [ID]                                 # take it again
gpupark status | cancel <id> | workloads | doctor
```

## Pausing

`gpupark pause` hands the card back without giving up the lease; `gpupark resume`
takes it again. Both act on the current lease by default and take a job id
otherwise (`gpupark pause 3f2a`).

- **A running unit lease** — the workload is stopped and the baseline restarted,
  so the resident model answers again while the lease stays yours. `resume` parks
  the baseline, waits for the VRAM to actually free, then starts the workload.
- **A queued job** — `gpupark pause <id>` holds it out of the queue: it is not
  granted until you resume it, and it keeps its priority and place in line. This
  is how you keep a submission for later without parking the card behind it.
- **A running command job** — the process group is SIGSTOPped. Suspending does
  **not** free VRAM (a stopped CUDA process keeps its allocations), so the
  baseline cannot come back: use it to give up the CPU, not the card. Pause the
  job while it is still queued if the GPU is what you want back.

Paused time does not count against `--ttl`: the deadline moves out by however
long the job was paused. `gpupark status` lists paused jobs under the queue with
the `resume` hint.

## Workloads

`~/.config/gpupark/workloads.yaml`:

```yaml
gpu: 0
cooldown_seconds: 5

workloads:
  tensorfold:
    kind: unit
    target: tensorfold-qwen27.service   # systemd --user unit
    vram_gb: 21
    baseline: true                      # runs whenever the queue is empty
    priority: -10
  comfyui:
    kind: unit
    target: comfyui.service
    vram_gb: 22
  game:
    kind: command
    target: ["./run-headless.sh"]
    vram_gb: 8
```

A workload is either a `unit` (systemd --user) or a `command` (argv). Exactly one
workload may set `baseline: true`; it is started at daemon boot and after every lease.

## CLI

| command | meaning |
|---|---|
| `gpupark run --need 22 -- CMD...` | queue an ad-hoc command, wait for the lease, run it |
| `gpupark lease comfyui --for 1800` | start a registered unit and hold the card for N seconds |
| `gpupark status` | lease, queue, VRAM, baseline state |
| `gpupark cancel ID` | drop a queued job or stop a running one |
| `gpupark pause [ID]` | hand the card back; keep the lease and the queue place |
| `gpupark resume [ID]` | re-grant a paused lease, or requeue a paused job |
| `gpupark doctor` | check the config, nvidia-smi, systemd and the daemon |
| `gpupark daemon` | run the broker in the foreground (what the unit runs) |

## Semantics

- **Exclusive by default.** One lease at a time; the broker verifies with `nvidia-smi`
  that the requested VRAM is actually free before granting (stopping a service is not
  the same as its memory being returned).
- **Priority then FIFO.** Higher `priority` first, ties in submission order.
- **The baseline is the only workload that may be preempted.** A running job is never
  evicted by a later arrival.
- **The TTL is running time, not wall time.** Time spent paused is subtracted before the
  deadline is checked. A command job stopped by its `--ttl` exits **124** with `lease
  expired`; a unit lease that reaches its `--for` simply ends.
- **A lease belongs to its client; a job belongs to the broker.** Closing the client that
  ran `gpupark lease` (Ctrl-C) releases the lease. A `gpupark run` job keeps running when its
  client goes away — stop it with `gpupark cancel <id>`.
- **Crash behaviour: the queue is in memory.** A `kill -9` of the broker loses queued jobs
  and can leave a running child or unit behind with no owner (the socket is stale, and the
  next start removes it). Restart the broker and check `gpupark status` for leftovers;
  journaling is not implemented yet.

## Security model

- **The socket is the whole boundary.** `$XDG_RUNTIME_DIR/gpupark.sock`, mode `0600`, inside a
  `0700` directory. There is no authentication: **whoever can connect can run commands as the
  user the broker runs as**, cancel or pause any job, and start/stop the configured units.
  gpupark is a single-user tool — do not share the socket with another user, and do not run
  the broker as root.
- **One broker per socket.** The daemon holds a `gpupark.lock` beside the socket and refuses
  to start when another broker has it, so a second daemon cannot take over the socket.
- **Jobs are not sandboxed.** They run as the broker's user, inherit its environment, and get
  no rlimits or VRAM enforcement after the grant. A job that needs isolation needs a
  container, not gpupark.
- **Every request is logged** as one line with the peer's uid and pid — that log is the only
  audit trail.

## Status

Milestone 1 (this release): daemon, queue, baseline preemption/restore, pause/resume,
CLI, systemd integration, tests with a fake GPU. Milestone 2: ComfyUI adapter (hold the
lease until the prompt finishes), multi-GPU pools, `gpupark top`. Milestone 3:
webhooks/metrics.

Apache-2.0.
