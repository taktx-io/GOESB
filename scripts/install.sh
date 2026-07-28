#!/bin/sh
# GOESB runner installer for macOS/Linux (including Raspberry Pi and other
# 64-bit ARM boards like rk3588) -- downloads the matching standalone
# PyInstaller binary from the latest GitHub release and installs it as
# `goesb` on PATH. No Python required.
#
# This is one of several ways to install the runner -- see
# https://goesb.com/docs/how-to for pipx/pip (needed for architectures this
# script doesn't cover: 32-bit ARM, Intel Mac) and building from source.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/taktx-io/GOESB/main/scripts/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/taktx-io/GOESB/main/scripts/install.sh | sh -s -- --engine vosk
#   GOESB_ENGINE=vosk GOESB_INSTALL_DIR=/usr/local/bin sh install.sh
set -eu

REPO="taktx-io/GOESB"
DOCS_URL="https://goesb.com/docs/how-to"
ENGINE="${GOESB_ENGINE:-faster-whisper}"
INSTALL_DIR="${GOESB_INSTALL_DIR:-$HOME/.local/bin}"

usage() {
  cat <<'EOF'
Usage: install.sh [--engine faster-whisper|vosk|whisper-cpp] [--dir <install-dir>]

Engines:
  faster-whisper  (default) -- best accuracy/speed on machines with a few
                    GB of RAM to spare; CUDA-accelerated automatically if
                    an NVIDIA GPU is present.
  vosk                      -- smallest footprint, no GPU support -- the
                    right choice on the most constrained boards.
  whisper-cpp               -- GGML, uses Metal (Apple Silicon) or CUDA
                    automatically when available, falls back to CPU.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$ENGINE" in
  faster-whisper|vosk|whisper-cpp) ;;
  *)
    echo "error: unknown engine '$ENGINE' (expected faster-whisper, vosk, or whisper-cpp)" >&2
    exit 1
    ;;
esac

os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Linux) platform_os="linux" ;;
  Darwin) platform_os="macos" ;;
  *)
    echo "error: unsupported OS '$os' -- on Windows, use install.ps1 instead: $DOCS_URL" >&2
    exit 1
    ;;
esac

case "$arch" in
  x86_64|amd64) platform_arch="x64" ;;
  aarch64|arm64) platform_arch="arm64" ;;
  *)
    echo "error: no prebuilt binary for CPU architecture '$arch'." >&2
    echo "This covers 32-bit ARM (e.g. older Raspberry Pi OS images, armv7)." >&2
    echo "Install via pipx/pip instead -- see $DOCS_URL" >&2
    exit 1
    ;;
esac

if [ "$platform_os" = "macos" ] && [ "$platform_arch" = "x64" ]; then
  echo "error: no prebuilt binary for Intel Mac (macos-x64) yet." >&2
  echo "Install via pipx/pip instead -- see $DOCS_URL" >&2
  exit 1
fi

asset="goesb-${ENGINE}-${platform_os}-${platform_arch}"
url="https://github.com/${REPO}/releases/latest/download/${asset}"

echo "Detected: ${platform_os}-${platform_arch}"
echo "Downloading ${asset} ..."

mkdir -p "$INSTALL_DIR"
tmp="$(mktemp)"
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT

if ! curl -fsSL "$url" -o "$tmp"; then
  echo "error: failed to download $url" >&2
  echo "check https://github.com/${REPO}/releases/latest for available assets." >&2
  exit 1
fi
chmod +x "$tmp"

# Keep each engine's binary around under its own name (so installing a
# second engine later doesn't destroy the first one you set up), and point
# the plain `goesb` name at whichever was installed most recently -- same
# "last one wins for the bare name" convention as e.g. python3 vs python.
mv "$tmp" "${INSTALL_DIR}/goesb-${ENGINE}"
ln -sf "${INSTALL_DIR}/goesb-${ENGINE}" "${INSTALL_DIR}/goesb"

echo "Installed goesb-${ENGINE} to ${INSTALL_DIR}/goesb-${ENGINE} (linked as ${INSTALL_DIR}/goesb)"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    echo ""
    echo "NOTE: ${INSTALL_DIR} is not on your PATH. Add this to your shell profile"
    echo "(~/.bashrc, ~/.zshrc, etc.) and restart your shell:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac

echo ""
echo "Run 'goesb --help' to get started."
echo "If you're on an NVIDIA GPU machine, run 'goesb doctor' first to check whether"
echo "--backend cuda will actually work here before your first real run."
