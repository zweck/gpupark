# Security

gpupark is a **single-user** tool. It is designed for one person's machine, where the
broker, the agents and the jobs all run as the same UNIX user.

## Trust model

- The daemon listens on a UNIX socket (`$XDG_RUNTIME_DIR/gpupark.sock`, mode `0600`, in a
  `0700` directory). It has no authentication. **Anyone who can connect to that socket can
  run arbitrary commands as the user the broker runs as**, cancel or pause any job, and
  start/stop the units named in `workloads.yaml`.
- Because of that, the socket is only as private as your user account. Do not point
  `socket_path` at a shared or world-writable directory, do not expose it to other users,
  and do not run the broker as root.
- Jobs are not sandboxed: they inherit the broker's environment and run with your
  privileges. The broker verifies free VRAM before a grant, not during the lease.
- The broker refuses to start if another broker holds the lock beside the socket, and it
  refuses a socket directory owned by a different user.

## Reporting a vulnerability

Email **philip.hauser@icloud.com** with the details and, if possible, a reproduction.
Please do not open a public issue for a security problem. You will get an answer, and
credit in the fix commit if you want it.

## Supported versions

The latest release tag on `main`. Fixes land there; older tags are not patched.
