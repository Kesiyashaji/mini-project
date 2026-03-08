/**
 * Landmark normalization — must exactly match Python training code:
 *
 *   wx, wy, wz = landmarks[0].x/y/z   (wrist)
 *   ref = landmarks[9]                 (middle-finger MCP)
 *   scale = sqrt((ref.x-wx)^2 + (ref.y-wy)^2 + (ref.z-wz)^2)
 *   out[i] = [(lm.x-wx)/scale, (lm.y-wy)/scale, (lm.z-wz)/scale]
 */
export interface Landmark {
    x: number
    y: number
    z: number
}

export function normalizeLandmarks(landmarks: Landmark[]): Float32Array {
    if (landmarks.length !== 21) {
        throw new Error(`Expected 21 landmarks, got ${landmarks.length}`)
    }

    const wrist = landmarks[0]
    const mcp9 = landmarks[9]          // middle-finger MCP

    const rx = mcp9.x - wrist.x
    const ry = mcp9.y - wrist.y
    const rz = mcp9.z - wrist.z
    let scale = Math.sqrt(rx * rx + ry * ry + rz * rz)
    if (scale < 1e-6) scale = 1.0

    const out = new Float32Array(63)    // 21 * 3
    for (let i = 0; i < 21; i++) {
        out[i * 3 + 0] = (landmarks[i].x - wrist.x) / scale
        out[i * 3 + 1] = (landmarks[i].y - wrist.y) / scale
        out[i * 3 + 2] = (landmarks[i].z - wrist.z) / scale
    }
    return out
}

/** Zero-frame used when no hand is detected */
export const ZERO_FRAME = new Float32Array(63)
