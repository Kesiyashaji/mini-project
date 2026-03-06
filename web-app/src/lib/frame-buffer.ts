/**
 * Frame Buffer for Sequence-based Sign Language Recognition
 *
 * Implements a sliding window buffer that collects N frames of normalized
 * hand landmarks. When the buffer is full, it returns the sequence as a
 * flat Float32Array ready for ONNX inference.
 *
 * The buffer operates in "sliding window" mode — each new frame pushes in
 * and the oldest is dropped, so inference can run continuously.
 */

export class FrameBuffer {
    private buffer: number[][];
    private readonly seqLength: number;
    private readonly nFeatures: number;

    constructor(seqLength: number = 16, nFeatures: number = 63) {
        this.seqLength = seqLength;
        this.nFeatures = nFeatures;
        this.buffer = [];
    }

    /** Push a single frame of normalized landmarks (63 floats). */
    push(frame: number[]): void {
        if (frame.length !== this.nFeatures) {
            console.warn(
                `FrameBuffer: expected ${this.nFeatures} features, got ${frame.length}`,
            );
            return;
        }
        this.buffer.push(frame);
        if (this.buffer.length > this.seqLength) {
            this.buffer.shift();
        }
    }

    /** Whether the buffer has enough frames for inference. */
    isFull(): boolean {
        return this.buffer.length >= this.seqLength;
    }

    /** Return the current fill level as a fraction (0 to 1). */
    fillLevel(): number {
        return this.buffer.length / this.seqLength;
    }

    /**
     * Get the buffered sequence as a flat Float32Array of shape
     * [seq_length, n_features] for ONNX input.
     * Returns null if the buffer is not full yet.
     */
    getSequence(): Float32Array | null {
        if (!this.isFull()) return null;

        const flat = new Float32Array(this.seqLength * this.nFeatures);
        for (let t = 0; t < this.seqLength; t++) {
            for (let f = 0; f < this.nFeatures; f++) {
                flat[t * this.nFeatures + f] = this.buffer[t][f];
            }
        }
        return flat;
    }

    /** Clear the buffer (e.g., when hand is lost). */
    clear(): void {
        this.buffer = [];
    }

    /** Current number of frames in the buffer. */
    get length(): number {
        return this.buffer.length;
    }
}
