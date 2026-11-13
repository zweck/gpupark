#!/bin/sh
# gpupark installer — one line, Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/zweck/gpupark/main/install.sh | sh
#
# Options (environment):
#   GPUPARK_HOME  where the venv lives      (default ~/.local/share/gpupark)
#   GPUPARK_BIN   where the command lands   (default ~/.local/bin)
#   GPUPARK_REF   git tag/branch to install (default main)
#   GPUPARK_REPO  repo URL                  (default github.com/zweck/gpupark)
#
# Uninstall:  curl -fsSL .../install.sh | sh -s -- --uninstall
set -eu

REPO="${GPUPARK_REPO:-https://github.com/zweck/gpupark.git}"
REF="${GPUPARK_REF:-main}"
HOME_DIR="${GPUPARK_HOME:-$HOME/.local/share/gpupark}"
BIN_DIR="${GPUPARK_BIN:-$HOME/.local/bin}"
VENV="$HOME_DIR/venv"
BIN="$BIN_DIR/gpupark"

say() { printf '%s\n' "$*"; }
die() { printf 'gpupark: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Linux | Darwin) ;;
  *) die "unsupported platform $(uname -s): gpupark installs on Linux and macOS" ;;
esac

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$BIN"
  rm -rf "$HOME_DIR"
  say "removed $BIN and $HOME_DIR"
  say "your config is untouched: ${XDG_CONFIG_HOME:-$HOME/.config}/gpupark/workloads.yaml"
  exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found — install Python 3.10 or newer first"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' ||
  die "Python 3.10 or newer is required, found $(python3 -V 2>&1)"

mkdir -p "$HOME_DIR" "$BIN_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" ||
    die "could not create a virtualenv (on Debian/Ubuntu: apt install python3-venv)"
fi

say "installing gpupark from ${REPO}@${REF}"
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
"$VENV/bin/python" -m pip install --quiet --upgrade "git+${REPO}@${REF}" ||
  die "pip install failed — see the output above"

ln -sf "$VENV/bin/gpupark" "$BIN"
[ -x "$BIN" ] || die "the gpupark command did not land at $BIN"

say ""
say "  gpupark $("$BIN" --version 2>/dev/null | awk '{print $2}') installed at $BIN"

case ":${PATH}:" in
  *":$BIN_DIR:"*) ;;
  *)
    say ""
    say "  $BIN_DIR is not on your PATH. Add it:"
    say "    echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile   # or ~/.zshrc"
    ;;
esac

say ""
say "next:"
say "  gpupark config init      # write ~/.config/gpupark/workloads.yaml"
say "  gpupark doctor           # check nvidia-smi, systemd, the daemon"
say "  gpupark install-service  # Linux: run the broker as a user service"
say "  gpupark daemon           # macOS: run it in the foreground, or wrap it yourself"
