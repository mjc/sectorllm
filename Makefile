.PHONY: all run screenshot test clean download quantize size listing dump-boot smoke final-text check-qemu

QEMU_ACCEL ?= kvm:tcg
QEMU_ACCEL_ARG := $(if $(QEMU_ACCEL),-machine accel=$(QEMU_ACCEL))
QEMU := qemu-system-i386 $(QEMU_ACCEL_ARG) -drive file=boot.img,format=raw
QEMU_MONITOR := $(QEMU) -display none -monitor stdio -serial none
REQ := command -v
QEMU_TIMEOUT ?= 5
QEMU_DONE_POLL_INTERVAL ?= 1
QEMU_DONE_STABLE_SECONDS ?= 20
QEMU_DONE_TIMEOUT ?= 300
QEMU_SCRATCH_PHYS ?= 0x80000
QEMU_CUR_POS_OFFSET := $(shell sed -n 's/^[[:space:]]*%define[[:space:]]\+CUR_POS[[:space:]]\+\(0x[0-9A-Fa-f]\+\).*/\1/p' sectorllm.asm)
QEMU_CUR_POS_PHYS ?= $(shell cur="$(QEMU_CUR_POS_OFFSET)"; [ -n "$$cur" ] && printf '0x%x' "$$(( $(QEMU_SCRATCH_PHYS) + $$cur ))")
QEMU_CUR_POS_LOG_ADDR := $(patsubst 0x%,%,$(QEMU_CUR_POS_PHYS))
QEMU_EXPECT_TEXT ?= Thank you, mommy!
empty :=
space := $(empty) $(empty)
QEMU_EXPECT_GENERATED_TEXT ?= $(space)Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, butit was too high. She asked her mom, "Mommy, can I have a ball?" Her mom said, "Yes, it's time to go home."<0x0A>Lily was so happy and said, "I will help you." Her mom smiled and said, "Yes, Lily. You can go to the park."<0x0A>Lily was so happy and said, "Thank you, mommy!" Her mom smiled and said, "You're welcome, Lily. You are a good friend."<0x0A>Lily was happy to have a new friend. She went back to the park and said, "Thank you, mommy!"
export QEMU_EXPECT_TEXT
export QEMU_EXPECT_GENERATED_TEXT
SCREENSHOT_PPM := $(if $(QEMU_SCREENSHOT),$(QEMU_SCREENSHOT),$(if $(filter 1,$(SHORT)),artifacts/qemu-screen-short.ppm,artifacts/qemu-screen.ppm))
SCREENSHOT_DELAY := $(if $(QEMU_SCREENSHOT_DELAY),$(QEMU_SCREENSHOT_DELAY),$(if $(filter 1,$(SHORT)),5,300))
FINAL_TEXT_BIN := artifacts/qemu-vga-final.bin
FINAL_TEXT_TXT := artifacts/qemu-vga-final.txt
FINAL_TEXT_LOG := artifacts/qemu-final-text.log
FINAL_TEXT_FIFO := artifacts/qemu-final-text.in
SMOKE_LOG := artifacts/qemu-smoke.log
SCREENSHOT_LOG := artifacts/qemu-screen.log
NASM_SIZE_LOG ?= artifacts/nasm-size.log
LISTING ?= artifacts/sectorllm.lst

all: boot.img

download: models/stories260K.bin models/tok512.bin

models/stories260K.bin models/tok512.bin &: download.sh
	./download.sh
	touch models/stories260K.bin models/tok512.bin

quantize: models/stories260K_int.bin

models/stories260K_int.bin: quantize.py models/stories260K.bin models/tok512.bin
	python3 quantize.py

boot.bin: sectorllm.asm
	nasm -f bin sectorllm.asm -o boot.bin

boot.img: boot.bin models/stories260K_int.bin quantize.py
	cat boot.bin models/stories260K_int.bin | dd of=$@ bs=1024 conv=notrunc count=1440

size:
	@mkdir -p artifacts
	@tmp="$$(mktemp)"; log="$(NASM_SIZE_LOG)"; \
	nasm -f bin sectorllm.asm -o "$$tmp" 2>"$$log"; cat "$$log"; \
	image_size="$$(wc -c <"$$tmp")"; signature="$$(od -An -tx1 -j 510 -N 2 "$$tmp" | tr -d '[:space:]')"; \
	boot_size="$$(sed -n 's/.*boot sector is \([0-9][0-9]*\) bytes.*/\1/p' "$$log")"; code_size="$$(sed -n 's/.*The total code is \([0-9][0-9]*\) bytes.*/\1/p' "$$log")"; \
	rm -f "$$tmp"; \
	if [ "$$image_size" != 1536 ]; then echo "FAIL: assembled binary is $$image_size bytes, expected 1536" >&2; exit 1; fi; \
	if [ "$$signature" != 55aa ]; then echo "FAIL: boot signature is not 55 aa at offset 510" >&2; exit 1; fi; \
	if [ -z "$$boot_size" ] || [ "$$boot_size" -gt 510 ]; then echo "FAIL: boot sector code is $${boot_size:-unknown} bytes, max is 510" >&2; exit 1; fi; \
	if [ -z "$$code_size" ] || [ "$$code_size" -gt 1536 ]; then echo "FAIL: total code is $${code_size:-unknown} bytes, max padded image is 1536" >&2; exit 1; fi; \
	echo "OK: boot=$$boot_size bytes total=$$code_size bytes image=$$image_size bytes"

listing:
	@mkdir -p "$$(dirname "$(LISTING)")"
	nasm -f bin -l "$(LISTING)" sectorllm.asm -o boot.bin
	@echo "OK: wrote $(LISTING)"

dump-boot: boot.bin
	xxd -g1 -l 512 boot.bin

smoke:
	$(MAKE) RUN_QEMU_SMOKE=1 test

check-qemu:
	@$(REQ) qemu-system-i386 >/dev/null 2>&1 || { echo "missing required command: qemu-system-i386" >&2; exit 127; }
	@$(REQ) timeout >/dev/null 2>&1 || { echo "missing required command: timeout" >&2; exit 127; }

final-text: $(FINAL_TEXT_TXT)
	@story="$$(sed -n '/Booting from Hard Disk/,$$p' "$(FINAL_TEXT_TXT)" | sed '1s/.*Booting from Hard Disk...//' | tr -d '\n' | sed 's/[[:space:]]*$$//')"; \
	text="$$(tr '\n' ' ' < "$(FINAL_TEXT_TXT)")"; \
	printf '%s\n' "$$story"; \
	case "$$text" in *"$$QEMU_EXPECT_TEXT"*) ;; *) echo "FAIL: VGA text does not contain $$QEMU_EXPECT_TEXT" >&2; exit 1;; esac; \
	if [ "$$story" != "$$QEMU_EXPECT_GENERATED_TEXT" ]; then echo "FAIL: generated text did not exactly match expected text" >&2; exit 1; fi
	@echo "OK: final generated text exactly matches expected text"

$(FINAL_TEXT_TXT): boot.img Makefile | check-qemu
	@[ -n "$(QEMU_CUR_POS_PHYS)" ] || { echo "could not derive CUR_POS address from sectorllm.asm" >&2; exit 1; }
	@mkdir -p artifacts
	@set -e; \
	log="$(FINAL_TEXT_LOG)"; fifo="$(FINAL_TEXT_FIFO)"; \
	rm -f "$(FINAL_TEXT_BIN)" "$(FINAL_TEXT_TXT)" "$$fifo"; \
	mkfifo "$$fifo"; \
	qemu_pid=; \
	cleanup() { status=$$?; if [ -n "$$qemu_pid" ]; then printf 'quit\n' >&3 2>/dev/null || true; exec 3>&- 2>/dev/null || true; wait "$$qemu_pid" 2>/dev/null || true; else exec 3>&- 2>/dev/null || true; fi; rm -f "$$fifo"; exit "$$status"; }; \
	trap cleanup EXIT INT TERM; \
	exec 3<>"$$fifo"; \
	timeout "$$(( $(QEMU_DONE_TIMEOUT) + 30 ))" $(QEMU_MONITOR) <"$$fifo" >"$$log" 2>&1 & \
	qemu_pid=$$!; \
	start=$$(date +%s); \
	last_change=$$start; \
	last_pos=; \
	saw_progress=0; \
	while :; do \
		now=$$(date +%s); \
		if ! kill -0 "$$qemu_pid" 2>/dev/null; then \
			set +e; \
			wait "$$qemu_pid"; \
			status=$$?; \
			set -e; \
			qemu_pid=; \
			echo "FAIL: qemu exited with status $$status before final text capture; monitor log follows:" >&2; \
			cat "$$log" >&2; \
			exit "$$status"; \
		fi; \
		if [ $$(( now - start )) -gt "$(QEMU_DONE_TIMEOUT)" ]; then \
			echo "FAIL: timed out after $(QEMU_DONE_TIMEOUT)s waiting for CUR_POS to stop; monitor log follows:" >&2; \
			cat "$$log" >&2; \
			exit 1; \
		fi; \
		printf 'xp /1uh $(QEMU_CUR_POS_PHYS)\n' >&3; \
		sleep "$(QEMU_DONE_POLL_INTERVAL)"; \
		pos="$$(sed -n 's/.*$(QEMU_CUR_POS_LOG_ADDR):[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$$log" | tail -n 1)"; \
		[ -n "$$pos" ] || continue; \
		if [ "$$pos" != "$$last_pos" ]; then \
			if [ -n "$$last_pos" ] || [ "$$pos" != 0 ]; then saw_progress=1; fi; \
			last_pos=$$pos; \
			last_change=$$now; \
			echo "CUR_POS=$$pos at $$(( now - start ))s"; \
		elif [ "$$saw_progress" = 1 ] && [ $$(( now - last_change )) -ge "$(QEMU_DONE_STABLE_SECONDS)" ]; then \
			echo "OK: CUR_POS stable at $$pos for $(QEMU_DONE_STABLE_SECONDS)s"; \
			break; \
		fi; \
	done; \
	printf 'pmemsave 0xb8000 4000 %s\nquit\n' "$(FINAL_TEXT_BIN)" >&3; \
	exec 3>&-; \
	set +e; \
	wait "$$qemu_pid"; \
	status=$$?; \
	set -e; \
	qemu_pid=; \
	trap - EXIT INT TERM; \
	rm -f "$$fifo"; \
	if [ "$$status" != 0 ]; then echo "FAIL: qemu exited with status $$status; monitor log follows:" >&2; cat "$$log" >&2; exit "$$status"; fi; \
	if [ ! -s "$(FINAL_TEXT_BIN)" ]; then echo "FAIL: QEMU did not write $(FINAL_TEXT_BIN); monitor log follows:" >&2; cat "$$log" >&2; exit 1; fi
	@[ "$$(wc -c < "$(FINAL_TEXT_BIN)")" = 4000 ] || { echo "FAIL: VGA dump is not 4000 bytes" >&2; exit 1; }
	@xxd -p -c2 "$(FINAL_TEXT_BIN)" | sed 's/..$$//' | xxd -r -p | LC_ALL=C tr -c ' -~' ' ' | fold -w 80 | sed 's/[[:space:]]*$$//' >"$(FINAL_TEXT_TXT)"
	@echo "OK: captured final VGA text after generation stopped"

run: boot.img
	$(QEMU)

screenshot: $(SCREENSHOT_PPM)

$(SCREENSHOT_PPM): boot.img | check-qemu
	@mkdir -p "$(@D)" artifacts
	@png="$(patsubst %.ppm,%.png,$@)"; \
	log="$(SCREENSHOT_LOG)"; \
	rm -f "$@" "$$png"; \
	( sleep "$(SCREENSHOT_DELAY)"; printf 'screendump %s\nquit\n' "$@" ) | \
		timeout "$$(( $(SCREENSHOT_DELAY) + 10 ))" $(QEMU_MONITOR) >"$$log" 2>&1; \
	if [ ! -s "$@" ]; then echo "FAIL: QEMU did not write $@; monitor log follows:" >&2; cat "$$log" >&2; exit 1; fi; \
	echo "OK: wrote $@"; \
	if $(REQ) magick >/dev/null 2>&1; then magick "$@" "$$png"; echo "OK: wrote $$png"; else echo "SKIP: ImageMagick not installed; leaving PPM only"; fi

test: size boot.img
	@echo "OK: built boot.img"
	@if [ "$(RUN_FINAL_TEXT)" = 1 ]; then $(MAKE) final-text; fi
	@if [ "$(RUN_QEMU_SMOKE)" = 1 ]; then \
		$(MAKE) check-qemu || exit $$?; \
		mkdir -p artifacts; \
		status=0; \
		timeout "$(QEMU_TIMEOUT)" $(QEMU) -display none -monitor none -serial none -no-reboot >"$(SMOKE_LOG)" 2>&1 || status=$$?; \
		if [ "$$status" != 0 ] && [ "$$status" != 124 ]; then echo "FAIL: qemu exited with status $$status; log: $(SMOKE_LOG)" >&2; exit "$$status"; fi; \
		echo "OK: qemu smoke ran for $(QEMU_TIMEOUT)s"; \
	fi
clean:
	rm -rf boot.img boot.bin artifacts \
		models/stories260K.bin \
		models/stories260K_int.bin \
		models/tok512.bin
