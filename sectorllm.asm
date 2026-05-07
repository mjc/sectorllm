;;                _             _ _
;; ___  ___  ___| |_ ___  _ __| | |_ __ ___
;; / __|/ _ \/ __| __/ _ \| '__| | | '_ ` _ \
;; \__ \  __/ (__| || (_) | |  | | | | | | | |
;; |___/\___|\___|\__\___/|_|  |_|_|_| |_| |_|
;;
;; The world's smallest llama2 inference engine.
;; This software is dedicated to the public domain.
;; It can be used, modified, and distributed without any restrictions.
;; Written by: rdmsr

bits 16
org 0x7c00


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Model parameters                                                           ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
%define DIM 64
%define DIM_LOG 6
%define HIDDEN 172
%define LAYERS 5
%define HEADS 8
%define KV_HEADS 4
%define VOCAB 512               ; Vocab size
%define SEQ 512                 ; Maximum input sequence length
%define HEAD_DIM (DIM / HEADS)
%define KV_DIM (KV_HEADS * HEAD_DIM)
%define TOKEN_COUNT 300         ; Maximum token count to generate


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Model layout                                                               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
%define MODEL_BASE 0x2000

; Sizes in paragraphs

; Precomputed lookup tables (exp and silu)
%define P_LUT       0x180       ; 6144 bytes
%define P_TOKEN_EMB (VOCAB * DIM * 4 / 16)
; Attention pre-RMSNorm weights
%define P_RMS_ATT   (LAYERS * DIM * 4 / 16)

; Attention QKV projection (concatenated WQ, WK, WV matrices)
; Stored as int8 weights (_Q) and a global int32 scale (_S)
%define P_WQKV_Q    (LAYERS * DIM * (DIM + 2*KV_DIM) / 16)
%define P_WQKV_S    1

; Attention output projection
%define P_WO_Q      (LAYERS * DIM * DIM / 16)
%define P_WO_S      1

; Feed-forward, Pre-RMSNorm weights
%define P_RMS_FFN   (LAYERS * DIM * 4 / 16)
; FFN down-projection
%define P_W2_Q_L    0x300       ; padded per-layer stride
%define P_W2_Q      (LAYERS * P_W2_Q_L)
%define P_W2_S      1

; FFN Gate and up-projection (W1 and W3 concatenated)
%define P_W13_Q     (LAYERS * DIM * 2 * HIDDEN / 16)
%define P_W13_S     1

; Final output RMSNorm weight
%define P_RMS_FINAL (DIM * 4 / 16)

; RoPE frequencies
; interleaved cos/sin pairs for sequence length up to 512
%define P_FREQ      (SEQ * (HEAD_DIM/2) * 2 * 4 / 16)

; Dynamic Segments
%define W_TOKEN_EMB (MODEL_BASE + P_LUT)
%define W_RMS_ATT   (W_TOKEN_EMB + P_TOKEN_EMB)
%define W_WQKV_Q    (W_RMS_ATT + P_RMS_ATT)
%define W_WQKV_S    (W_WQKV_Q + P_WQKV_Q)
%define W_WO_Q      (W_WQKV_S + P_WQKV_S)
%define W_WO_S      (W_WO_Q + P_WO_Q)
%define W_RMS_FFN   (W_WO_S + P_WO_S)
%define W_W2_Q      (W_RMS_FFN + P_RMS_FFN)
%define W_W2_S      (W_W2_Q + P_W2_Q)
%define W_W13_Q     (W_W2_S + P_W2_S)
%define W_W13_S     (W_W13_Q + P_W13_Q)
%define W_RMS_FINAL (W_W13_S + P_W13_S)
%define W_FREQ_CIS  (W_RMS_FINAL + P_RMS_FINAL)
%define VOCAB_PTR   (W_FREQ_CIS + P_FREQ)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Scratch data (ES)                                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
%define R_X       0x0000    ; FP16.16[DIM]
%define R_XB      0x0100    ; FP16.16[DIM]
%define R_HB      0x0200    ; FP16.16[2*HIDDEN], overlaps dead R_QKV
%define R_QKV     0x0200    ; 384 bytes (Q=64, K=16, V=16), overlaps dead R_HB
%define R_XB2     0x0300    ; FP16.16[DIM], overlaps dead K/V tail of R_QKV

; Global State Variables
%define R_MAX     0x09E0    ; dword
%define R_BEST    0x09E4    ; word
%define CUR_LAYER 0x09E6    ; word
%define CUR_POS   0x09E8    ; word

%define R_ATT     0x0A00    ; FP16.16[TOKEN_COUNT*HEADS]
    

; Cache Segments
%define KC_SEG 0x0840
%define KS_SEG 0x1C40
%define VC_SEG 0x8600
%define VS_SEG 0x9A00

; Best for hot code
%macro Q16_SHIFT_INLINE 0
    shrd eax, edx, 16
%endmacro
    

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Boot sector                                                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; The boot sector loads the second stage and model data from disk.
; It also contains the main inference loop and utility subroutines.
entry:
    ; Set up segments
    mov sp, 0x7BF0

    ; Load stage2
    mov bh, 0x7E
    mov ax, 0x0202              ; AH=02 read, AL=2 sectors
    mov cl, 2                   ; assume CH=0, sector 2
    int 0x13
    push 0x2000                 ; LUT segment
    pop fs
    ; Build the DAP on the stack using zero registers that survive the CHS read.
    push 3
    push fs
    push es
    push 64
    push 0x10
    mov si, sp

    ; Load the model at 0x2000:0
    mov cl, 12

.load_model:
    mov ah, 0x42
    int 0x13

    add dword [si + 6], 0x00400800 ; add 64 to lba, 0x800 to segment
    loop .load_model

    ; Recover ES=0x8000 and BX=0 from the final DAP segment:offset pair.
    les bx, [si + 4]

start_inference:
    inc bx                      ; BOS

.gen_loop:
    call forward
    cmp bx, 2                   ; check for BOS or EOS
    jbe $
    call print_token
    inc word [es:CUR_POS]
    jmp .gen_loop


; Inverse square root approximation
; in  EBP: x (FP16.16)
; out EBP: 1/sqrt(x) (FP16.16)
; Clobbers: ecx
inv_sqrt:
    ; Initial guess from bit position: y ~= 2^((48-bsr(x))/2)
    bsr ecx, ebp
    neg cl
    add cl, 48
    shr cl, 1
    xor eax, eax
    bts eax, ecx                ; eax = (1<<ecx), initial y
    mov cl, 5                   ; Should be good enough

; Newton-Raphson: y = y * (3 - x*y^2) / 2
.loop:
    mov esi, eax                ; esi = y
    mul eax                     ; edx:eax = y^2
    call q16_shift              ; convert to FP16.16
    mul ebp                     ; edx:eax = x*y^2
    call q16_shift              ; convert to FP16.16
    neg eax                     ; eax = -x*y^2
    add eax, 3*65536            ; eax = 3 - x*y^2 (FP16.16)
    mul esi
    call q16_shift              ; eax = y*(3 - x*y^2) (FP16.16)
    shr eax, 1                  ; / 2
    loop .loop
.done:
    xchg eax, ebp
    ret

; rmsnorm helper: sets up DS and BX before doing rmsnorm logic
; in AX:      weight segment base
; in ES:DI:   output buffer
do_rmsnorm:
        imul cx, word [es:CUR_LAYER], 16
        add ax, cx
        mov ds, ax
        xor bx, bx

; Compute RMSNorm: out[i] = x[i] * w[i] / sqrt(mean(x^2) + epsilon)
; in  ES:DI: output buffer (FP16.16[DIM])
; in  ES:0:  input x (R_X, FP16.16[DIM])
; in  DS:0:  weights w (FP16.16[DIM])
rmsnorm:
    xor ebp, ebp                ; ebp = sum of squares accumulator
    mov cl, DIM
    xor si, si                  ; SI=0, points to ES:R_X
    push si                     

; 1. Compute sum of squares
.sum:
    es lodsd                    ; eax = x[i], SI+=4
    imul eax                    ; edx:eax = x[i]^2
    call q16_shift              ; eax = x[i]^2 in FP16.16
    add ebp, eax                ; ebp += x[i]^2
    loop .sum

.eps:
    ; ss = (sum / DIM) + epsilon
    shr ebp, DIM_LOG            ; ebp = sum/DIM
    inc bp ; epsilon

    call inv_sqrt               ; ebp = 1/sqrt(ss) in FP16.16

    pop si                      ; restore SI to R_X
    mov cl, DIM

; 2. Normalize and apply weights
.norm:
    es lodsd              ; eax = x[i], SI += 4
    imul ebp              ; eax = x[i] * (1/sqrt(ss))
    call q16_shift
    imul dword [bx]       ; eax *= w[i]
    call q16_shift        ; eax = x[i] * w[i] / sqrt(ss)
    add bx, 4             ; advance weight pointer
    stosd                 ; write to output, DI += 4
    loop .norm
.done:
    xchg di, bx
    ret

; matmul helper:
; in AX:   base_Q
; in CX:   layer stride
; in EDX:  (rows<<16) | cols
; ES:DI: input vector
; ES:BX: output vector
do_matmul:
    imul si, cx, LAYERS        ; SI = scale segment delta = stride * layers
    add si, ax                 ; SI = scale segment base
    imul cx, [es:CUR_LAYER]    ; cx = layer * stride (paragraphs)
    add ax, cx                 ; ax = weight base + layer*stride

    ; Load single global scale from the scale segment for this layer
    mov ds, si
    xor si, si
    mov ebp, [si]              ; load scale for current layer

    mov ds, ax                 ; DS = this layer's int8 weight segment

; Multiply an int8 matrix by a FP16.16 vector
; in DS:SI:     int8 weight matrix (row-major)
; in ES:DI:     input vector (FP16.16[COLS])
; in ES:BX:     output vector (FP16.16[ROWS])
; in EDX:       (ROWS << 16) | COLS
; in EBP:       FP16.16 scale factor for dequantization
matmul:
    xchg ax, dx                 ; ax = ROWS
    shr edx, 16                 ; dx = COLS

; For each output element
.row:
    push ax                     ; save row count
    push dx                     ; save cols
    push di                     ; save input vector base
    push bx                     ; save out

    xor ebx, ebx                ; ebx = dot product accumulator
    mov cx, dx                  ; cx = cols (loop counter)

; dot product: sum(weight[col] * input[col])
.dot:
    lodsb                       ; al = int8 weight, SI += 1
    ; sign-extend
    cbw
    cwde
    imul dword [es:di]          ; edx:eax = weight * input[col]
    scasd                       ; di += 4
    add ebx, eax                ; accumulate low 32 bits (should be safe for DIM=64)
    loop .dot

; dequantize: (acc * scale) >> 16
    xchg eax, ebx
    imul ebp                    ; edx:eax = acc * scale
    call q16_shift              ; eax = result in FP16.16

; Store result
    pop di                      ; output pointer
    stosd                       ; store result and advance output pointer
    mov bx, di                  ; keep advanced output pointer for caller/next row
    pop di
    pop dx
    pop ax
    dec ax
    jnz .row

    ret

; Add the matmul output into R_X in-place
; in ES:BX: matmul output
; Convenience wrapper around vadd for post-matmul accumulation (saves bytes)
vadd_rx:
    dec bh                      ; matmul leaves BX one DIM vector past output
    xchg si, bx                 ; grab pointer from matmul
    xor di, di                  ; R_X

; Vector addition: ES:DI += ES:SI for DIM FP16.16 elements
; in ES:SI: src vector (FP16.16[DIM])
; in ES:DI: dest vector (FP16.16[DIM])
vadd:
    mov cl, DIM
.lp:
    es lodsd                    ; eax = *SI, SI += 4
    add [es:di], eax            ; *DI += eax
    scasd                       ; DI += 4
    loop .lp
    ret

; Apply RoPE to a vector in-place.
; Rotates each consecutive pair (x0, x1) by the angle for its position,
; using precomputed interleaved (cos, sin) pairs in the frequency table.
; in ES:DI: vector to rotate (FP16.16), modified in place
; in CX:     number of heads to process
apply_rope:
    imul bx, [es:CUR_POS], 32   ; bx = CUR_POS * 32 (8 bytes per pair * 4 pairs per head)
    push W_FREQ_CIS
    pop ds                      ; DS = freq table


.head_loop:
    push bx                     ; save freq table offset
    push cx                     ; save head counter
    mov cl, 4                   ; 4 pairs per head

.pair_loop:
    ; Load sin and cos values
    mov ebp, [bx]               ; ebp = cos
    mov esi, [bx+4]             ; esi = sin

    ; Rotate (x0, x1):
    ;   new_x0 = x0*cos - x1*sin
    ;   new_x1 = x0*sin + x1*cos
    mov eax, [es:di+4]          ; x1
    imul esi                    ; x1*sin
    call q16_shift
    push eax                    ; stack = x1*sin

    mov eax, [es:di]            ; x0
    imul ebp                    ; x0 * cos
    call q16_shift
    pop edx                     ; edx = x1*sin
    sub eax, edx                ; new_x0 = (x0*cos)-(x1*sin)
    push eax                    ; stack = new_x0

    mov eax, [es:di+4]          ; x1
    imul ebp                    ; x1*cos
    call q16_shift
    push eax                    ; stack = x1*cos, new_x0

    mov eax, [es:di]            ; x0
    imul esi                    ; x0*sin
    call q16_shift
    pop edx                     ; edx = x1*cos
    add eax, edx                ; eax = x0*sin+x1*cos

    mov [es:di+4], eax          ; store new_x1
    pop eax                     ; eax = new_x0
    mov [es:di], eax            ; store new_x0

    add bx, 8                   ; advance to next (cos, sin) pair
    add di, 8                   ; advance to next (x0, x1) pair
    loop .pair_loop

    pop cx
    pop bx
    loop .head_loop
    ret



; Print the token string from the corresponding number.
; in BX: Token number
print_token:
    push VOCAB_PTR              ; DS = VOCAB_PTR
    pop ds
    push bx
    shl bx, 1
    mov si, [bx]
    mov ah, 0x0E                ; teletype out
.print_str:
    lodsb                       ; c=VOCAB_PTR[SI++]
    test al, al
    jz .done                    ; stop at NULL
    int 0x10                    ; print c
    jmp .print_str
.done:
    pop bx
    ret

; Set DS to a segmented address for KV cache access.
; two entry points for different cache stride sizes.
; in DX:  base segment
; out DS: base + (CUR_LAYER * stride)
set_seg_1024:
    mov cl, 10
    db 0x3D                     ; Nice!
set_seg_128:
    mov cl, 7
.do_seg:
    push ax
    mov ax, [es:CUR_LAYER]
    shl ax, cl                  ; ax = CUR_LAYER * stride
    add dx, ax                  ; dx = base + layer offset
    mov ds, dx                  ; DS = target segment
    pop ax
    ret

quant_kv_cache:
    mov ax, KC_SEG
    call .quant
    mov ax, VC_SEG
 .quant:
    push ax
    add ah, 0x14
    push ax
    mov cl, KV_DIM
    push si
    xor ebx, ebx
    jmp quant_cache

; Compute the byte offset into the KV cache for a given token and KV head
; The cache layout is [token][kv_head][DIM] with each element being int8
; in DI:   t (token position)
; in BP:   h (attention head index)
; out BX:  t * KV_DIM + kvh * HEAD_DIM
call_set_seg_1024_jmp_get_kv_offset:
    call set_seg_1024
get_kv_offset:
    imul bx, di, 32 ; bx = t * 32 (KV_DIM bytes per token)
    imul cx, bp, 4
    and cl, 0x18 ; cx = (h / 2) * HEAD_DIM
    add bx, cx ; bx = offset of this token's KV head slice
    ret

; Compute a pointer into the attention score buffer.
; R_ATT layout is [head][token], each element being FP16.16
; in BP:  h (head index)
; in DI:  t (token position)
; out SI: &R_ATT[h][t]
get_att_ptr:
    imul si, bp, 2048      ; h * 2048 (SEG * 4 bytes)
    add si, R_ATT          ; SI = base of this head's attention scores
    imul cx, di, 4         ; cx = t * 4 (4 bytes per score)
    add si, cx             ; SI = &R_ATT[h][t]
    ret

; It is slower to always call this function but it saves two bytes each time!
q16_shift:
    shrd eax, edx, 16
    ret

set_ds_token_emb_tail:
    mov ds, ax
zero_si_zero_di_jmp_get_pos_count:
    xor si, si

zero_di_jmp_get_pos_count:
    xor di, di
get_pos_count:
    mov cx, [es:CUR_POS]
    inc cx
    ret

lm_best_tail:
    mov bx, [es:R_BEST]
    ret

inc_bx_q_lp_tail:
    inc bx

quant_cache_q_lp_tail:
    loop quant_cache.q_lp
    ret

_bootsector_end:
%assign bootsector_size _bootsector_end - $$
%warning boot sector is bootsector_size bytes.
times 510 - ($ - $$) db 0
dw 0xAA55



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Sector 1 and 2                                                             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

quant_cache:
    ; Find max absolute value
.max_lp:
    es lodsd
    cdq                         ; sign-extend into edx
    xor eax, edx
    sub eax, edx                ; eax = abs(eax)
    cmp ebx, eax
    cmovl ebx, eax              ; new max
    loop .max_lp
    pop si

; Compute scale (max / 127)
.do_scale:
    xchg eax, ebx
    cdq
    mov cl, 127
    idiv ecx                    ; eax = max / 127
    xchg ebp, eax               ; ebp = scale

    ; store scale
    pop dx
    call set_seg_128            ; DS = scale cache segment for this layer
    imul bx, [es:CUR_POS], 4
    mov [bx], bp                ; scale_cache[bx] = scale

    ; quantize and cache
    pop dx
    call set_seg_1024           ; DS = int8 cache segment for this layer
    shl bx, 3                   ; bx = CUR_POS * 32
    mov cl, KV_DIM
.q_lp:
    es lodsd
    cdq
    idiv ebp                    ; eax = round(x/scale), clamped to int8
    mov [bx], al                ; store quantized byte
    jmp inc_bx_q_lp_tail

set_ds_token_emb:
    imul ax, bx, 16
    add ax, W_TOKEN_EMB
    jmp set_ds_token_emb_tail

; Full forward pass of the transformer for one token.
; in BX:  input token index
; out BX: predicted token
forward:
    ; Load token embedding into R_X
    call set_ds_token_emb
    mov cl, DIM * 2             ; dword
    rep movsw                   ; R_X = embedding[token]

    mov [es:CUR_LAYER], cx      ; cx is 0

.layer:
    ; Normalize input before attention
    mov ax, W_RMS_ATT
    call do_rmsnorm             ; R_XB = rmsnorm(R_X, w_rms_att[layer])



    ; Project normalized input to Q, K, V simultaneously
    mov ax, W_WQKV_Q
    mov ch, 2
    mov edx, (DIM << 16) | (DIM + 2*KV_DIM) ; rows=96 (Q+K+V), cols=64
    call do_matmul              ; R_QKV = [Q | K | V] = w_wqkv * R_XB


    ; Apply RoPE to Q and K
    mov di, R_QKV
    mov cl, HEADS + KV_HEADS
    call apply_rope             ; rotate Q and K

    ; Quantize and cache K and V for this position
    mov si, R_QKV + DIM*4
    call quant_kv_cache         ; KC/VC[layer][pos] = quantize(K/V)


    ; Compute attention scores, softmax and weight sum of V
    call attention              ; R_XB = attention(Q, KC, VC)

    ; Project attention output back to DIM
    mov ax, W_WO_Q
    mov ch, 1
    mov edx, (DIM << 16) | DIM
    mov di, R_XB
    mov bh, R_XB2 >> 8
    call do_matmul              ; R_XB2 = w_wo * R_XB

    ; Residual connection
    call vadd_rx                ; R_X += R_XB2

    ; FFN

    ; Normalize before FFN
    mov ax, W_RMS_FFN
    call do_rmsnorm             ; R_XB = rmsnorm(R_X, w_rms_ffn[layer])

    ; Project up to hidden dim
    mov ax, W_W13_Q
    mov cx, 0x560
    mov edx, (DIM << 16) | (2*HIDDEN)
    call do_matmul              ; R_HB = [gate | up] = w_w13 * R_XB

    ; Apply SiLU gating
    call silu_gate

    ; Project back down to DIM
    mov ax, W_W2_Q
    mov ch, P_W2_Q_L >> 8
    mov edx, (HIDDEN << 16) | DIM
    call do_matmul              ; R_XB = w_w2 * R_HB

    ; Residual connection
    call vadd_rx                ; R_X + R_XB

    ; Go to next layer
    inc word [es:CUR_LAYER]
    cmp word [es:CUR_LAYER], LAYERS
    jl .layer

    ; Final normalization
    mov ax, W_RMS_FINAL - LAYERS * 16
    xor di, di
    call do_rmsnorm             ; R_X = rmsnorm(R_X, w_rms_final)

    ; Compute logits and pick best token (use greedy argmax)
    xor bx, bx                       ; BX = token index

; logit computation: dot(R_X, embedding[i])
; Since the model uses weight tying, the output projection reuses
; the token embedding table the logit for token i is just the
; dot product of the final hidden state with embedding[i].
.lm_loop:
    call set_ds_token_emb

    xor ebp, ebp                ; ebp dot accumulator
    mov cl, DIM
.dot:
    lodsd                       ; eax = embedding[i][j], SI += 4
    imul dword [es:di]          ; edx:eax = embedding[i][j] * R_X[j]
    call q16_shift              ; inline this for more perf, but it'll cost you two bytes!
    add ebp, eax                ; accumulate
    scasd                       ; DI += 4
    loop .dot

; argmax, just track the highest scoring token
    test bx, bx
    jz .set_max                 ; token 0 seeds the max for this pass
    cmp ebp, [es:R_MAX]
    jle .skip_max
.set_max:
    mov di, R_MAX               ; store new best score/token
    xchg eax, ebp
    stosd
    xchg ax, bx
    stosw
    xchg ax, bx
.skip_max:
    inc bx
    test bh, VOCAB >> 8
    jz .lm_loop                 ; next token

    jmp lm_best_tail

; Compute multi-head grouped-query attention for the current position.
; Reads Q from R_QKV, K/V from the quantized KV cache.
; Output is written into R_XB (one HEAD_DIM slice per head).
attention:
    mov bp, HEADS-1             ; bp = h (head index, HEADS-1..0)
.head_loop:

    ; 1.QK dot products
    ; For each past token t, compute a_t = dot(Q_h, K_t) * scale
    ; and store in R_ATT[h][t]
    call zero_di_jmp_get_pos_count ; process tokens t = 0..CUR_POS inclusive, DI = t
.t_loop:
    push cx                     ; save token counter

    ; load K vector for token T, KV head kvh = h/2
    mov dx, KC_SEG
    call call_set_seg_1024_jmp_get_kv_offset ; DS = K cache, BX = offset of K[t][kvh]

    ; Load Q vector for head h
    imul si, bp, 32             ; h * 32
    add si, R_QKV               ; SI = &Q[h]

    push bp                     ; save h
    mov cl, HEAD_DIM
    xor ebp, ebp                ; acc

; dot(Q_h, K_t), K is int8, Q is FP16.16
.dot_loop:
    movsx edx, byte [bx]        ; edx = K
    inc bx
    es lodsd                    ; eax = Q[h][i], SI += 4
    imul edx                    ; edx:eax = Q[h][i] * K[t][i]
    add ebp, eax                ; accumulate (low 32 bits enough for HEAD_DIM=8)
    loop .dot_loop

.dot_done:
    xchg eax, ebp
    pop bp                      ; restore h

    ; Dequantize: multiply by K scale for token t
    call get_att_ptr            ; SI = &R_ATT[h][t], CX = t * 4
    push si
    push cx
    mov dx, KS_SEG
    call set_seg_128            ; DS = K scale cache for this layer
    pop si
    mov esi, [si]               ; esi = scale_kt

    imul esi
    call q16_shift              ; eax = dot * scale_kt

    ; multiply by 1/sqrt(HEAD_DIM) ~= 23170
    mov si, 23170
    imul esi
    call q16_shift              ; eax = a_t (attention score, FP16.16)

    ; store score in R_ATT[h][t]
    pop si
    mov [es:si], eax

    inc di                      ; t++
    pop cx
    loop .t_loop

    ; 2. Softmax over attention scores
    ; Converts raw R_ATT[h][0..pos] to probabilities
.softmax:
    push di
    xor di, di
    call get_att_ptr            ; SI = &R_ATT[h][0]
    mov di, si
    pop cx

    ; Find max score
    push di
    push cx
    es lodsd                    ; max = first elem
.max:
    scasd                       ; DI += 4, compare eax
    cmovl eax, [es:di-4]        ; new max
    loop .max
    pop cx
    pop di

.max_found:
    ; Compute exp(x - max) for each score and accumulate sum
    push di
    push cx
    xor si, si                  ; sum = 0
.s_exp:
    push eax                    ; save max
    sub eax, [es:di]            ; diff = max - x
    shr eax, 10                 ; scale down for LUT index, diff / 64
    cmp ax, 511                 ; clamp to LUT range
    jle .s_ok
    mov ax, 511
.s_ok:
    xchg ax, bx
    shl bx, 2                   ; bx = index * 4
    mov eax, [fs:bx]            ; eax = exp_lut[diff]
    add esi, eax                ; sum += exp
    stosd                       ; replace score with exp, DI += 4
    pop eax                     ; restore max
    loop .s_exp

    ; Divide each exp by sum
    pop cx
    pop di
.s_div:
    mov eax, [es:di]
    movzx edx, word [es:di+2]   ; edx:eax = exp value as FP32.16
    shl eax, 16
    div esi                     ; eax = exp / sum (FP16.16)
    stosd                       ; store probability, DI += 4
    loop .s_div
    ; 3. Weighted sum of V
    ; out[h] = sum over t of (attention[h][t] * V[t])
.agg:
    ; Clear r_xb[h] before accumulating
    imul di, bp, 32             ; h * 32
    add di, R_XB
    xor eax, eax
    mov cl, HEAD_DIM * 2
    rep stosw                   ; zero out R_XB[h]

    call zero_di_jmp_get_pos_count ; DI = t
.v_loop:
    push cx

    ; Load V vector for token t, KV head kvh = h/2
    mov dx, VC_SEG
    call call_set_seg_1024_jmp_get_kv_offset ; DS = V cache, BX = offset of V[t][kvh]

    ; a_t = R_ATT[h][t]
    call get_att_ptr            ; SI = &R_ATT[h][t]
    es lodsd                    ; eax = a_t

    ; Dequantize V: multiply a_t by V scale for token t
    push ds
    push cx
    mov dh, VS_SEG >> 8
    call set_seg_128            ; DS = V scale cache for this layer
    pop si                      ; t * 4 from get_att_ptr
    imul dword [si]             ; multiply by scale_vt
    pop ds                      ; restore DS = VC_SEG

    call q16_shift              ; eax = a_scaled = a_t * scale_vt

    ; Accumulate: R_XB[h] += a_scale * V[t]
    xchg edx, eax               ; edx = a_scaled
    imul si, bp, 32             ; h * 32
    add si, R_XB                ; SI = &R_XB[h]

    mov cx, HEAD_DIM
.v_mac:
    movsx eax, byte [bx]        ; eax = V[t][i] (int8)
    inc bx
    imul eax, edx               ; eax = V[t][i] * a_scaled
    add [es:si], eax
    add si, 4
    loop .v_mac

    inc di                      ; t++
    pop cx
    loop .v_loop

    ; Next head
    dec bp
    jns .head_loop
.done:
    ret

; SiLU gating: out[i] = silu(gate[i]) * up[i]
; where silu(x) = x * sigmoid(x), looked up from a precomputed table
silu_gate:
    mov di, R_HB                ; DI = gate vector
    push di
    mov si, R_HB+HIDDEN*4       ; SI = up vector
    mov cl, HIDDEN
.lp:
    ; Compute silu_lut index from gate[i]
    mov eax, [es:di]            ; eax = gate[i] (FP16.16)
    sar eax, 10                 ; downscale to [-512, 511] 
    add ah, 2                   ; ax += 512, shift to [0, 1023]

    shl ax, 2
    xchg ax, bx                 ; bx = index * 4

    ; Multiply by up[i] and store in gate[i]
    es lodsd                    ; eax = up[i]
    imul dword [fs:bx+0x800]    ; eax = up[i] * silu(gate[i])
    call q16_shift              ; shift back to FP16.16
    stosd                       ; gate[i] = res, DI += 4
    loop .lp
    xchg di, bx
    pop di
    ret

_code_end:
%assign code_size _code_end - $$
%warning The total code is code_size bytes.
times 1536-($-$$) db 0
