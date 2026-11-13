#!/bin/sh
# gpupark installer — one line, Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/zweck/gpupark/main/install.sh | sh
#
# Installs the latest released tag by default. Options (environment):
#   GPUPARK_HOME  where the venv lives      (default ~/.local/share/gpupark)
#   GPUPARK_BIN   where the command lands   (default ~/.local/bin)
#   GPUPARK_REF   git tag/branch to install (default v0.2.1)
#   GPUPARK_REPO  repo URL                  (default github.com/zweck/gpupark)
#
# GPUPARK_REF and GPUPARK_REPO are ignored unless GPUPARK_ALLOW_CUSTOM_SOURCE=1:
# installing from an arbitrary repo or branch is opt-in.
#
# Uninstall:  curl -fsSL .../install.sh | sh -s -- --uninstall
set -eu

DEFAULT_REF=v0.2.1
DEFAULT_REPO=https://github.com/zweck/gpupark.git

say() { printf '%s\n' "$*"; }
die() { printf 'gpupark: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Linux | Darwin) ;;
  *) die "unsupported platform $(uname -s): gpupark installs on Linux and macOS" ;;
esac

[ -n "${HOME:-}" ] || die "HOME is not set — cannot choose an install location"

REPO="$DEFAULT_REPO"
REF="$DEFAULT_REF"
if [ "${GPUPARK_ALLOW_CUSTOM_SOURCE:-}" = "1" ]; then
  if [ -n "${GPUPARK_REPO:-}" ]; then REPO="$GPUPARK_REPO"; fi
  if [ -n "${GPUPARK_REF:-}" ]; then REF="$GPUPARK_REF"; fi
elif [ -n "${GPUPARK_REPO:-}" ] || [ -n "${GPUPARK_REF:-}" ]; then
  say "ignoring GPUPARK_REPO/REF; set GPUPARK_ALLOW_CUSTOM_SOURCE=1 to override"
fi

HOME_DIR="${GPUPARK_HOME-$HOME/.local/share/gpupark}"
BIN_DIR="${GPUPARK_BIN:-$HOME/.local/bin}"
[ -n "$HOME_DIR" ] || die "GPUPARK_HOME is empty — set it to a directory"
VENV="$HOME_DIR/venv"
BIN="$BIN_DIR/gpupark"

# physical path of an existing directory (readlink -f on Linux, cd + pwd -P on macOS)
physical() {
  [ -d "$1" ] || return 1
  if readlink -f / >/dev/null 2>&1; then
    readlink -f "$1" 2>/dev/null && return 0
  fi
  (cd "$1" 2>/dev/null && pwd -P)
}

# $BIN is ours only when it is a symlink whose target sits inside $HOME_DIR
bin_is_ours() {
  [ -L "$BIN" ] || return 1
  target=$(readlink "$BIN" 2>/dev/null) || return 1
  [ -n "$target" ] || return 1
  case "$target" in
    /*) ;;
    *) target="$BIN_DIR/$target" ;;
  esac
  dir_phys=$(physical "$HOME_DIR") || dir_phys="$HOME_DIR"
  case "$target" in
    "$HOME_DIR"/* | "$dir_phys"/*) return 0 ;;
  esac
  return 1
}

if [ "${1:-}" = "--uninstall" ]; then
  case "$HOME_DIR" in
    / | //) die "refusing to uninstall: GPUPARK_HOME is /" ;;
  esac
  home_phys=$(physical "$HOME") || home_phys="$HOME"
  dir_phys=$(physical "$HOME_DIR") || dir_phys="$HOME_DIR"
  [ "$dir_phys" != "$home_phys" ] ||
    die "refusing to uninstall: $HOME_DIR is your home directory"
  if [ -d "$HOME_DIR" ] && [ ! -e "$HOME_DIR/venv/bin/gpupark" ]; then
    die "refusing to uninstall: $HOME_DIR does not look like a gpupark install (no venv/bin/gpupark)"
  fi

  kept=0
  removed_bin=0
  removed_dir=0
  if [ -e "$BIN" ] || [ -L "$BIN" ]; then
    if bin_is_ours; then
      rm -f "$BIN"
      removed_bin=1
    else
      kept=1
      say "left $BIN in place: it is not a symlink into $HOME_DIR"
    fi
  fi
  if [ -d "$HOME_DIR" ]; then
    rm -rf "$HOME_DIR"
    removed_dir=1
  fi
  if [ "$removed_bin" = 1 ] && [ "$removed_dir" = 1 ]; then
    say "removed $BIN and $HOME_DIR"
  elif [ "$removed_dir" = 1 ]; then
    say "removed $HOME_DIR"
  elif [ "$removed_bin" = 1 ]; then
    say "removed $BIN"
  else
    say "nothing to remove: no install at $BIN or $HOME_DIR"
  fi
  say "your config is untouched: ${XDG_CONFIG_HOME:-$HOME/.config}/gpupark/workloads.yaml"
  [ "$kept" = 0 ] || exit 1
  exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found — install Python 3.10 or newer first"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' ||
  die "Python 3.10 or newer is required, found $(python3 -V 2>&1)"

if [ -e "$BIN" ] || [ -L "$BIN" ]; then
  bin_is_ours ||
    die "$BIN already exists and is not a gpupark symlink — move it aside, then re-run"
fi

mkdir -p "$HOME_DIR" "$BIN_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" ||
    die "could not create a virtualenv (on Debian/Ubuntu: apt install python3-venv)"
fi

say "installing gpupark from ${REPO}@${REF}"
if command -v git >/dev/null 2>&1; then
  resolved=$(git ls-remote "$REPO" "$REF" 2>/dev/null | awk '
    { if ($2 ~ /\^\{\}$/) { peeled = substr($1, 1, 12) } else if (first == "") { first = substr($1, 1, 12) } }
    END { print (peeled != "" ? peeled : first) }') || resolved=""
  if [ -n "$resolved" ]; then
    say "  ${REF} resolves to ${resolved}"
  fi
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
"$VENV/bin/python" -m pip install --quiet --upgrade "git+${REPO}@${REF}" ||
  die "pip install failed — see the output above"

ln -sf "$VENV/bin/gpupark" "$BIN"
[ -x "$BIN" ] || die "the gpupark command did not land at $BIN"

version=$("$VENV/bin/python" -c 'import gpupark; print(gpupark.__version__)' 2>/dev/null) || version=""

say ""
say "  gpupark ${version:-unknown} installed at $BIN"

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
