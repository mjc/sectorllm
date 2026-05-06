.PHONY: all run screenshot final-text test clean
all: boot.img
boot.img: sectorllm.asm models/stories260K_int.bin
	nasm -f bin sectorllm.asm -o boot.bin
	cat boot.bin models/stories260K_int.bin | dd of=$@ bs=1024 conv=notrunc count=1440
	rm boot.bin
run: boot.img
	qemu-system-i386 -hda boot.img
screenshot: boot.img
	SHORT="$(SHORT)" QEMU_SCREENSHOT_DELAY="$(QEMU_SCREENSHOT_DELAY)" QEMU_SCREENSHOT="$(QEMU_SCREENSHOT)" ./tests/screenshot.sh
final-text: boot.img
	QEMU_FINAL_CHECKS="$(QEMU_FINAL_CHECKS)" QEMU_EXPECT_TEXT="$(QEMU_EXPECT_TEXT)" QEMU_TEXT_ONLY="$(QEMU_TEXT_ONLY)" ./tests/final-text.py
test:
	./tests/test.sh
clean:
	rm -rf boot.img boot.bin artifacts
