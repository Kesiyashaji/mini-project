/**
 * Fixed-length ring buffer — holds the last SEQ_LENGTH frames.
 * Push a Float32Array(63) per frame.
 * Call toTensor() to get Float32Array[SEQ_LENGTH * N_FEATURES].
 */
export const SEQ_LENGTH = 16
export const N_FEATURES = 63

export class FrameBuffer {
    private buf: Float32Array[]
    private ptr: number
    private count: number
    private zeroCount: number

    constructor() {
        this.buf = Array.from({ length: SEQ_LENGTH }, () => new Float32Array(N_FEATURES))
        this.ptr = 0
        this.count = 0
        this.zeroCount = SEQ_LENGTH  // starts fully empty → all zeros
    }

    push(frame: Float32Array, isZero: boolean): void {
        const old = this.buf[this.ptr]
        const wasZero = isZeroFrame(old)

        this.buf[this.ptr] = frame
        this.ptr = (this.ptr + 1) % SEQ_LENGTH
        this.count = Math.min(this.count + 1, SEQ_LENGTH)

        if (wasZero && !isZero) this.zeroCount = Math.max(0, this.zeroCount - 1)
        if (!wasZero && isZero) this.zeroCount = Math.min(SEQ_LENGTH, this.zeroCount + 1)
        if (isZero && wasZero) { } // no change
        if (!isZero && !wasZero) { } // no change
    }

    /** Returns true if ≥ half the buffer is zero-frames (no hand) */
    isMostlyEmpty(): boolean {
        return this.zeroCount >= SEQ_LENGTH / 2
    }

    isFull(): boolean {
        return this.count >= SEQ_LENGTH
    }

    /** Returns ordered Float32Array[SEQ_LENGTH, N_FEATURES] */
    toTensor(): Float32Array {
        const out = new Float32Array(SEQ_LENGTH * N_FEATURES)
        for (let i = 0; i < SEQ_LENGTH; i++) {
            const src = this.buf[(this.ptr + i) % SEQ_LENGTH]
            out.set(src, i * N_FEATURES)
        }
        return out
    }

    reset(): void {
        this.buf = Array.from({ length: SEQ_LENGTH }, () => new Float32Array(N_FEATURES))
        this.ptr = 0
        this.count = 0
        this.zeroCount = SEQ_LENGTH
    }
}

function isZeroFrame(f: Float32Array): boolean {
    for (let i = 0; i < f.length; i++) {
        if (f[i] !== 0) return false
    }
    return true
}
