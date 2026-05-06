#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for cmd in nasm; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "missing required command: $cmd" >&2
        echo "try: nix develop -c make test" >&2
        exit 127
    fi
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

asm_log="$tmp/nasm.log"
nasm -f bin sectorllm.asm -o "$tmp/sectorllm.bin" 2>"$asm_log"
cat "$asm_log"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

image_size="$(wc -c < "$tmp/sectorllm.bin")"
if (( image_size != 1536 )); then
    fail "assembled binary is $image_size bytes, expected padded 1536 bytes"
fi

signature="$(od -An -tx1 -j 510 -N 2 "$tmp/sectorllm.bin")"
signature="${signature//[[:space:]]/}"
if [[ "$signature" != "55aa" ]]; then
    fail "boot signature is not 55 aa at offset 510"
fi

boot_size="$(sed -n 's/.*boot sector is \([0-9][0-9]*\) bytes.*/\1/p' "$asm_log")"
if [[ -z "$boot_size" ]]; then
    fail "NASM output did not report boot sector size"
fi
if (( boot_size > 510 )); then
    fail "boot sector code is $boot_size bytes, max is 510"
fi

code_size="$(sed -n 's/.*The total code is \([0-9][0-9]*\) bytes.*/\1/p' "$asm_log")"
if [[ -z "$code_size" ]]; then
    fail "NASM output did not report total code size"
fi
if (( code_size > 1536 )); then
    fail "total code is $code_size bytes, max padded image is 1536"
fi

echo "OK: boot=$boot_size bytes total=$code_size bytes image=$image_size bytes"

if [[ -f models/stories260K_int.bin ]]; then
    make boot.img
    boot_img_size="$(wc -c < boot.img)"
    if (( boot_img_size <= image_size )); then
        fail "boot.img does not include model payload"
    fi

    boot_prefix="$tmp/boot-prefix.bin"
    dd if=boot.img of="$boot_prefix" bs="$image_size" count=1 2>/dev/null
    if ! cmp -s "$tmp/sectorllm.bin" "$boot_prefix"; then
        fail "boot.img prefix does not match assembled code"
    fi

    echo "OK: boot.img=$boot_img_size bytes"
    QEMU_EXPECT_TEXT="${QEMU_TEST_EXPECT_TEXT:-Thank you, mommy!}" \
        QEMU_TEXT_ONLY="${QEMU_TEXT_ONLY:-1}" \
        ./tests/final-text.py
else
    echo "SKIP: models/stories260K_int.bin missing; create it to enable image tests"
fi

if [[ "${RUN_QEMU_SMOKE:-0}" == 1 ]]; then
    if ! command -v qemu-system-i386 >/dev/null 2>&1; then
        echo "missing required command: qemu-system-i386" >&2
        exit 127
    fi
    if [[ ! -f boot.img ]]; then
        echo "missing boot.img; run make boot.img first" >&2
        exit 1
    fi
    status=0
    timeout "${QEMU_TIMEOUT:-5}" qemu-system-i386 \
        -drive file=boot.img,format=raw \
        -display none \
        -no-reboot \
        -serial none \
        -monitor none || status=$?
    if [[ "$status" != 0 && "$status" != 124 ]]; then
        echo "FAIL: qemu exited with status $status" >&2
        exit "$status"
    fi
    echo "OK: qemu smoke ran for ${QEMU_TIMEOUT:-5}s"
fi
