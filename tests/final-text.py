#!/usr/bin/env python3
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CUR_POS_PHYS = 0x80000 + 0x09E8
TEXT_ONLY = os.environ.get("QEMU_TEXT_ONLY") == "1"
BOOT_MARKER = "Booting from Hard Disk..."
DEFAULT_GENERATED_TEXT = (
    ' Once upon a time, there was a little girl named Lily. She loved to play outside'
    ' in the park. One day, she saw a big, red ball. She wanted to play with it, but'
    ' it was too high. She asked her mom, "Mommy, can I have a ball?" Her mom said, "Yes,'
    ' it\'s time to go home."<0x0A>Lily was so happy and said, "I will help you." Her'
    ' mom smiled and said, "Yes, Lily. You can go to the park."<0x0A>Lily was so happy'
    ' and said, "Thank you, mommy!" Her mom smiled and said, "You\'re welcome, Lily.'
    ' You are a good friend."<0x0A>Lily was happy to have a new friend. She went back'
    ' to the park and said, "Thank you, mommy!"'
)


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_checks(raw):
    checks = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            fail(f"invalid QEMU_FINAL_CHECKS value: {part!r}")
        if value < 0:
            fail(f"invalid negative QEMU_FINAL_CHECKS value: {part!r}")
        checks.append(value)
    if not checks:
        fail("QEMU_FINAL_CHECKS did not contain any timestamps")
    return sorted(checks)


class Monitor:
    def __init__(self, sock_path, log):
        self.log = log
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 5
        while True:
            try:
                self.sock.connect(str(sock_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    fail("timed out waiting for QEMU monitor socket")
                time.sleep(0.05)
        self.sock.settimeout(10)
        self.read_until_prompt()

    def read_until_prompt(self, timeout=10):
        deadline = time.monotonic() + timeout
        chunks = []
        while time.monotonic() < deadline:
            self.sock.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                chunk = self.sock.recv(4096)
            except (TimeoutError, socket.timeout):
                break
            if not chunk:
                break
            chunk = chunk.decode("latin1")
            self.log.write(chunk)
            self.log.flush()
            chunks.append(chunk)
            text = "".join(chunks)
            if text.endswith("(qemu) "):
                return text
        fail("timed out waiting for QEMU monitor prompt")

    def command(self, command, timeout=10):
        self.sock.sendall((command + "\n").encode("ascii"))
        return self.read_until_prompt(timeout)

    def quit(self):
        self.sock.sendall(b"quit\n")


def read_cur_pos(mon):
    out = mon.command(f"xp /1uh 0x{CUR_POS_PHYS:x}")
    match = re.search(r":\s+([0-9]+)\s*(?:\r?\n|\(qemu\))", out)
    if not match:
        fail("could not parse CUR_POS from QEMU monitor output")
    return int(match.group(1))


def label_for(value):
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def vga_rows(path):
    data = path.read_bytes()
    if len(data) != 4000:
        fail(f"{path} is {len(data)} bytes, expected 4000 bytes")
    chars = data[0::2]
    rows = []
    for offset in range(0, 2000, 80):
        row = chars[offset : offset + 80]
        clean = bytes(byte if 32 <= byte < 127 else 32 for byte in row)
        rows.append(clean.decode("ascii"))
    return rows


def vga_text(path):
    return "\n".join(row.rstrip() for row in vga_rows(path)).rstrip()


def generated_text(rows):
    for index, row in enumerate(rows):
        marker_at = row.find(BOOT_MARKER)
        if marker_at == -1:
            continue
        tail = row[marker_at + len(BOOT_MARKER) :]
        story_rows = rows[index + 1 :]
        if tail.strip():
            story_rows = [tail] + story_rows
        return "".join(story_rows).rstrip()
    fail(f"could not find {BOOT_MARKER!r} in VGA text")


def normalized(text):
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def wait_for_done(mon):
    interval = float(os.environ.get("QEMU_DONE_POLL_INTERVAL") or "1")
    stable_seconds = float(os.environ.get("QEMU_DONE_STABLE_SECONDS") or "20")
    timeout = float(os.environ.get("QEMU_DONE_TIMEOUT") or "300")

    start = time.monotonic()
    last_change = start
    last_pos = None
    saw_progress = False
    while True:
        now = time.monotonic()
        if now - start > timeout:
            fail(f"timed out after {timeout:g}s waiting for CUR_POS to stop")

        pos = read_cur_pos(mon)
        if pos != last_pos:
            elapsed = now - start
            if not TEXT_ONLY:
                print(f"CUR_POS={pos} at {elapsed:.1f}s")
            if last_pos is not None:
                saw_progress = True
            last_pos = pos
            last_change = now
        elif saw_progress and now - last_change >= stable_seconds:
            elapsed = now - start
            if not TEXT_ONLY:
                print(f"OK: CUR_POS stable at {pos} for {stable_seconds:g}s after {elapsed:.1f}s")
            return elapsed, pos

        time.sleep(interval)


def capture(mon, label):
    vga = ARTIFACTS / f"qemu-vga-final-{label}.bin"
    txt = ARTIFACTS / f"qemu-vga-final-{label}.txt"
    for path in (vga, txt):
        path.unlink(missing_ok=True)

    mon.command(f'pmemsave 0xb8000 4000 "{vga}"', timeout=20)
    return vga, txt


def main():
    os.chdir(ROOT)
    qemu = shutil.which("qemu-system-i386")
    if not qemu:
        fail("missing required command: qemu-system-i386; try: nix develop -c make final-text")
    if not (ROOT / "boot.img").is_file():
        fail("missing boot.img; run make boot.img first")

    ARTIFACTS.mkdir(exist_ok=True)

    fixed_checks = os.environ.get("QEMU_FINAL_CHECKS")
    checks = parse_checks(fixed_checks) if fixed_checks else []
    expect = os.environ.get("QEMU_EXPECT_TEXT") or "Thank you, mommy!"
    expect_generated = os.environ.get("QEMU_EXPECT_GENERATED_TEXT") or DEFAULT_GENERATED_TEXT
    log_path = ARTIFACTS / "qemu-final-text.log"
    sock_path = ARTIFACTS / "qemu-final-text.sock"
    sock_path.unlink(missing_ok=True)

    timed_captures = []
    for check in checks:
        label = label_for(check)
        timed_captures.append((check, label))

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                qemu,
                "-drive",
                "file=boot.img,format=raw",
                "-display",
                "none",
                "-monitor",
                f"unix:{sock_path},server=on,wait=off",
                "-serial",
                "none",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        start = time.monotonic()
        captures = []
        try:
            mon = Monitor(sock_path, log)
            if timed_captures:
                for check, label in timed_captures:
                    remaining = start + check - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    captures.append((label, *capture(mon, label)))
            else:
                _elapsed, _pos = wait_for_done(mon)
                captures.append(("done", *capture(mon, "done")))

            mon.quit()
            proc.wait(timeout=10)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            raise
        finally:
            sock_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        fail(f"qemu exited with status {proc.returncode}; monitor log: {log_path}")

    for label, vga, txt in captures:
        if not vga.is_file() or vga.stat().st_size == 0:
            fail(f"QEMU did not write {vga}; monitor log: {log_path}")

        rows = vga_rows(vga)
        text = "\n".join(row.rstrip() for row in rows).rstrip()
        generated = generated_text(rows)
        txt.write_text(text + "\n")
        exact_match = generated == expect_generated

        when = f"{label}s" if label.replace("_", ".").replace(".", "", 1).isdigit() else label
        if TEXT_ONLY:
            print(generated)
        else:
            print(f"--- VGA text at {when} ---")
            print(text)
            print(f"--- Generated text at {when} ---")
            print(generated)
            print(f"OK: wrote {vga}")
            print(f"OK: wrote {txt}")

        if expect not in normalized(text):
            fail(f"{when} VGA text does not contain {expect!r}")
        if not exact_match:
            if TEXT_ONLY:
                print(generated)
            fail(f"{when} generated text did not exactly match expected text")
        if not TEXT_ONLY:
            print(f"OK: {when} contains {expect!r}")
            print(f"OK: {when} generated text exactly matches expected text")


if __name__ == "__main__":
    main()
