.PHONY: all run screenshot test clean download quantize size listing dump-boot smoke final-text check-qemu
QEMU_ACCEL ?= kvm:tcg
QEMU := qemu-system-i386 $(if $(QEMU_ACCEL),-machine accel=$(QEMU_ACCEL)) -drive file=boot.img,format=raw
QEMU_TIMEOUT ?= 5
QEMU_DONE_POLL_INTERVAL ?= 1
QEMU_DONE_STABLE_SECONDS ?= 20
QEMU_DONE_TIMEOUT ?= 300
QEMU_CUR_POS_PHYS ?= $(shell cur="$$(sed -n 's/^[[:space:]]*%define[[:space:]]\+CUR_POS[[:space:]]\+\(0x[0-9A-Fa-f]\+\).*/\1/p' sectorllm.asm)"; [ -n "$$cur" ] && printf '0x%x' "$$(( $(or $(QEMU_SCRATCH_PHYS),0x80000) + $$cur ))")
empty :=
space := $(empty) $(empty)
QEMU_EXPECT_GENERATED_TEXT ?= $(space)Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, butit was too high. She asked her mom, "Mommy, can I have a ball?" Her mom said, "Yes, it's time to go home."<0x0A>Lily was so happy and said, "I will help you." Her mom smiled and said, "Yes, Lily. You can go to the park."<0x0A>Lily was so happy and said, "Thank you, mommy!" Her mom smiled and said, "You're welcome, Lily. You are a good friend."<0x0A>Lily was happy to have a new friend. She went back to the park and said, "Thank you, mommy!"
export QEMU_EXPECT_GENERATED_TEXT
SCREENSHOT_PPM := $(if $(QEMU_SCREENSHOT),$(QEMU_SCREENSHOT),$(if $(filter 1,$(SHORT)),artifacts/qemu-screen-short.ppm,artifacts/qemu-screen.ppm))
SCREENSHOT_DELAY := $(if $(QEMU_SCREENSHOT_DELAY),$(QEMU_SCREENSHOT_DELAY),$(if $(filter 1,$(SHORT)),5,300))
NASM_SIZE_LOG ?= artifacts/nasm-size.log
LISTING ?= artifacts/sectorllm.lst
all: boot.img
download: models/stories260K.bin models/tok512.bin
models/stories260K.bin models/tok512.bin &: download.sh; ./download.sh; touch models/stories260K.bin models/tok512.bin
quantize: models/stories260K_int.bin
models/stories260K_int.bin: quantize.py models/stories260K.bin models/tok512.bin; python3 quantize.py
boot.bin: sectorllm.asm; nasm -f bin sectorllm.asm -o boot.bin
boot.img: boot.bin models/stories260K_int.bin quantize.py; cat boot.bin models/stories260K_int.bin | dd of=$@ bs=1024 conv=notrunc count=1440
size:; @mkdir -p artifacts; tmp="$$(mktemp)"; log="$(NASM_SIZE_LOG)"; nasm -f bin sectorllm.asm -o "$$tmp" 2>"$$log"; cat "$$log"; image_size="$$(wc -c <"$$tmp")"; signature="$$(od -An -tx1 -j 510 -N 2 "$$tmp" | tr -d '[:space:]')"; boot_size="$$(sed -n 's/.*boot sector is \([0-9][0-9]*\) bytes.*/\1/p' "$$log")"; code_size="$$(sed -n 's/.*The total code is \([0-9][0-9]*\) bytes.*/\1/p' "$$log")"; rm -f "$$tmp"; [ "$$image_size" = 1536 ] || { echo "FAIL: assembled binary is $$image_size bytes, expected 1536" >&2; exit 1; }; [ "$$signature" = 55aa ] || { echo "FAIL: boot signature is not 55 aa at offset 510" >&2; exit 1; }; [ -n "$$boot_size" ] && [ "$$boot_size" -le 510 ] || { echo "FAIL: boot sector code is $${boot_size:-unknown} bytes, max is 510" >&2; exit 1; }; [ -n "$$code_size" ] && [ "$$code_size" -le 1536 ] || { echo "FAIL: total code is $${code_size:-unknown} bytes, max padded image is 1536" >&2; exit 1; }; echo "OK: boot=$$boot_size bytes total=$$code_size bytes image=$$image_size bytes"
listing:; @mkdir -p "$$(dirname "$(LISTING)")"; nasm -f bin -l "$(LISTING)" sectorllm.asm -o boot.bin; echo "OK: wrote $(LISTING)"
dump-boot: boot.bin; xxd -g1 -l 512 boot.bin
smoke:; $(MAKE) RUN_QEMU_SMOKE=1 test
check-qemu:; @for c in qemu-system-i386 timeout; do command -v "$$c" >/dev/null || { echo "missing required command: $$c" >&2; exit 127; }; done
final-text: artifacts/qemu-vga-final.txt
	@story="$$(sed -n '/Booting from Hard Disk/,$$p' "$<" | sed '1s/.*Booting from Hard Disk...//' | tr -d '\n' | sed 's/[[:space:]]*$$//')"; printf '%s\n' "$$story"; if [ "$$story" != "$$QEMU_EXPECT_GENERATED_TEXT" ]; then echo "FAIL: generated text did not exactly match expected text" >&2; exit 1; fi; echo "OK: final generated text exactly matches expected text"
artifacts/qemu-vga-final.txt: boot.img Makefile | check-qemu
	@[ -n "$(QEMU_CUR_POS_PHYS)" ] || { echo "could not derive CUR_POS address from sectorllm.asm" >&2; exit 1; }
	@mkdir -p artifacts
	@set -e; bin="artifacts/qemu-vga-final.bin"; log="artifacts/qemu-final-text.log"; addr="$(QEMU_CUR_POS_PHYS)"; rm -f "$$bin" "$@"; : >"$$log"; start=$$(date +%s); last_change=$$start; last_pos=; saw_progress=0; { while :; do now=$$(date +%s); if [ $$(( now - start )) -gt "$(QEMU_DONE_TIMEOUT)" ]; then echo "FAIL: timed out after $(QEMU_DONE_TIMEOUT)s waiting for CUR_POS to stop; monitor log follows:" >&2; cat "$$log" >&2; exit 1; fi; printf 'xp /1uh %s\n' "$$addr"; sleep "$(QEMU_DONE_POLL_INTERVAL)"; pos="$$(sed -n "s/.*$${addr#0x}:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p" "$$log" | tail -n 1)"; [ -n "$$pos" ] || continue; if [ "$$pos" != "$$last_pos" ]; then [ -z "$$last_pos" ] && [ "$$pos" = 0 ] || saw_progress=1; last_pos=$$pos; last_change=$$now; echo "CUR_POS=$$pos at $$(( now - start ))s" >&2; elif [ "$$saw_progress" = 1 ] && [ $$(( now - last_change )) -ge "$(QEMU_DONE_STABLE_SECONDS)" ]; then echo "OK: CUR_POS stable at $$pos for $(QEMU_DONE_STABLE_SECONDS)s" >&2; break; fi; done; printf 'pmemsave 0xb8000 4000 %s\nquit\n' "$$bin"; } | timeout "$$(( $(QEMU_DONE_TIMEOUT) + 30 ))" $(QEMU) -display none -monitor stdio -serial none >"$$log" 2>&1 || { echo "FAIL: qemu final text capture failed; monitor log follows:" >&2; cat "$$log" >&2; exit 1; }; [ -s "$$bin" ] || { echo "FAIL: QEMU did not write $$bin; monitor log follows:" >&2; cat "$$log" >&2; exit 1; }
	@[ "$$(wc -c <artifacts/qemu-vga-final.bin)" = 4000 ] || { echo "FAIL: VGA dump is not 4000 bytes" >&2; exit 1; }
	@xxd -p -c2 artifacts/qemu-vga-final.bin | sed 's/..$$//' | xxd -r -p | LC_ALL=C tr -c ' -~' ' ' | fold -w 80 | sed 's/[[:space:]]*$$//' >"$@"
	@echo "OK: captured final VGA text after generation stopped"
run: boot.img; $(QEMU)
screenshot: $(SCREENSHOT_PPM)
$(SCREENSHOT_PPM): boot.img | check-qemu
	@mkdir -p "$(@D)" artifacts
	@png="$(patsubst %.ppm,%.png,$@)"; log="artifacts/qemu-screen.log"; rm -f "$@" "$$png"; ( sleep "$(SCREENSHOT_DELAY)"; printf 'screendump %s\nquit\n' "$@" ) | timeout "$$(( $(SCREENSHOT_DELAY) + 10 ))" $(QEMU) -display none -monitor stdio -serial none >"$$log" 2>&1; [ -s "$@" ] || { echo "FAIL: QEMU did not write $@; monitor log follows:" >&2; cat "$$log" >&2; exit 1; }; echo "OK: wrote $@"; if command -v magick >/dev/null 2>&1; then magick "$@" "$$png"; echo "OK: wrote $$png"; else echo "SKIP: ImageMagick not installed; leaving PPM only"; fi
test: size boot.img
	@echo "OK: built boot.img"
	@if [ "$(RUN_FINAL_TEXT)" = 1 ]; then $(MAKE) final-text; fi
	@if [ "$(RUN_QEMU_SMOKE)" = 1 ]; then $(MAKE) check-qemu || exit $$?; mkdir -p artifacts; status=0; timeout "$(QEMU_TIMEOUT)" $(QEMU) -display none -monitor none -serial none -no-reboot >artifacts/qemu-smoke.log 2>&1 || status=$$?; [ "$$status" = 0 ] || [ "$$status" = 124 ] || { echo "FAIL: qemu exited with status $$status; log: artifacts/qemu-smoke.log" >&2; exit "$$status"; }; echo "OK: qemu smoke ran for $(QEMU_TIMEOUT)s"; fi
clean:
	rm -rf boot.img boot.bin artifacts models/stories260K.bin models/stories260K_int.bin models/tok512.bin
