#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for cmd in qemu-system-i386 timeout; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "missing required command: $cmd" >&2
        echo "try: nix develop -c make screenshot" >&2
        exit 127
    fi
done

if [[ ! -f boot.img ]]; then
    echo "missing boot.img; run ./download.sh && python3 quantize.py first" >&2
    exit 1
fi

mkdir -p artifacts

if [[ -n "${QEMU_SCREENSHOT_DELAY:-}" ]]; then
    delay="$QEMU_SCREENSHOT_DELAY"
elif [[ "${SHORT:-0}" == 1 ]]; then
    delay=5
else
    delay=300
fi

if [[ -n "${QEMU_SCREENSHOT:-}" ]]; then
    ppm="$QEMU_SCREENSHOT"
elif [[ "${SHORT:-0}" == 1 ]]; then
    ppm="artifacts/qemu-screen-short.ppm"
else
    ppm="artifacts/qemu-screen.ppm"
fi

png="${ppm%.ppm}.png"
log="artifacts/qemu-screen.log"

rm -f "$ppm" "$png"

(
    sleep "$delay"
    printf 'screendump %s\nquit\n' "$ppm"
) | timeout "$((delay + 10))" qemu-system-i386 \
    -drive file=boot.img,format=raw \
    -display none \
    -monitor stdio \
    -serial none >"$log" 2>&1

if [[ ! -s "$ppm" ]]; then
    echo "FAIL: QEMU did not write $ppm; monitor log follows:" >&2
    cat "$log" >&2
    exit 1
fi

echo "OK: captured after ${delay}s"
echo "OK: wrote $ppm"

if command -v magick >/dev/null 2>&1; then
    magick "$ppm" "$png"
    echo "OK: wrote $png"
else
    echo "SKIP: ImageMagick not installed; leaving PPM only"
fi
