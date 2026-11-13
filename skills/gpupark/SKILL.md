---
name: gpupark
description: Use when a GPU job must queue for, or preempt, a resident model or render on one card.
---

# Using gpupark

`gpupark` is the lease broker for a single GPU. It owns one exclusive lease, a
priority queue, and a **baseline** workload (usually the resident LLM) that holds
the card while nothing else needs it and is parked the moment your job is granted.

Your job does not manage VRAM. It asks for a lease.

## Before you submit

```bash
gpupark status          # who holds the card, what is queued, how much is free
gpupark workloads       # names, kinds and the VRAM each declares
```

If `status` shows a lease you did not submit, **do not cancel it** — that is
someone's running work. Ask the human first.

## Submitting work

```bash
gpupark run --need 22 --ttl 1800 -- ./render-level.sh level3
gpupark run --need 8  --priority 5 --ttl 600 -- my-job --flag
gpupark lease comfyui --for 1800          # hold a registered unit (e.g. a UI session)
```

- `--need` is the VRAM your job actually uses, in GB. **Declare it honestly**: too
  high blocks the queue behind a card that will never be that empty; too low and
  your job OOMs after the baseline was already parked.
- `--ttl` is the lease ceiling in seconds. Always set it for renders and training
  runs. When it expires the broker stops your workload and restores the baseline.
- `--priority` is higher-first, FIFO on ties. Leave it at 0 unless the human asked
  you to jump the queue.
- The command's stdout/stderr stream to you and its exit code becomes `gpupark`'s.
  A non-zero exit means the job failed, not that the broker did.

## What happens to the machine

1. The baseline is stopped (the LLM goes away — a chat request during your job
   will fail or queue; that is expected, not a bug).
2. The broker polls `nvidia-smi` until the *memory* is actually free. A stopped
   service is not instantly a freed card, so this can take a few seconds.
3. Your workload starts and holds the exclusive lease.
4. On exit, TTL expiry, or `gpupark cancel`, the workload is stopped, a cooldown
   passes, and the baseline is restarted. Restarting a 27B model takes ~30-60 s;
   the card is idle during that time, not yours.

## When to ask the human

Ask before doing any of these:

- **preempting anything that is not the baseline** — a `lease` you did not take;
- **`gpupark cancel`** on a job whose id you do not recognise;
- **editing `~/.config/gpupark/workloads.yaml`** or restarting `gpupark.service`
  (a daemon restart drops every queued job and marks running leases cancelled);
- **a job that needs more VRAM than the card has** — it will never fit, and the
  broker will park the baseline only to fail after `grant_timeout`;
- **holding a lease while you do nothing** (a `sleep`, an interactive prompt, a
  long file download). Take the lease when the work starts, release it when it
  ends; a lease idle for minutes is a bug in your script;
- **`--priority` above 0** on shared machines.

Report a failed grant verbatim: `the GPU never freed N GB` means something outside
gpupark is holding the card — that is a human question, not something to retry in
a loop.

## Pitfalls

- Never run GPU work **outside** gpupark while another lease is held, and never
  start a second `gpupark run` expecting them to share: the lease is exclusive.
- Do not poll `nvidia-smi` to decide when to start. Submit to gpupark; it already
  waits for the memory and will tell you when it is granted.
- A client Ctrl-C cancels the job (the daemon terminates the process), so do not
  interrupt a `gpupark run` you want to keep.
- Jobs are children of the daemon: closing your terminal does **not** kill a
  granted job. Use `gpupark cancel <id>` when you mean it.
- `gpupark status --json` is the machine-readable form; prefer it over parsing text.

## Machine-readable facts

- Socket: `$XDG_RUNTIME_DIR/gpupark.sock` (override with `socket_path` in config).
- Protocol: one JSON object per line; `{"op":"submit","argv":[...],"need_gb":22,"ttl":1800}`.
- Every job gets `GPUPARK_JOB_ID` in its environment, for logs and cleanup.
