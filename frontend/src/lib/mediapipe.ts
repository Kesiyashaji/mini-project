import {
    HandLandmarker,
    FilesetResolver,
    type HandLandmarkerResult,
} from '@mediapipe/tasks-vision'

export type { HandLandmarkerResult }
export type { NormalizedLandmark } from '@mediapipe/tasks-vision'

let landmarker: HandLandmarker | null = null

export async function initHandLandmarker(): Promise<HandLandmarker> {
    if (landmarker) return landmarker

    const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    )

    landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,       // detect both so we can pick dominant
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
    })

    return landmarker
}

/**
 * Dominant hand = hand whose wrist is closest to horizontal center (x=0.5).
 * Works on raw MediaPipe coords — do NOT flip X for mirrored camera.
 */
export function selectDominantHand(
    result: HandLandmarkerResult
): { landmark: HandLandmarkerResult['landmarks'][0]; handedness: string } | null {
    if (!result.landmarks || result.landmarks.length === 0) return null

    let bestIdx = 0
    let bestDist = Math.abs(result.landmarks[0][0].x - 0.5)

    for (let i = 1; i < result.landmarks.length; i++) {
        const d = Math.abs(result.landmarks[i][0].x - 0.5)
        if (d < bestDist) {
            bestDist = d
            bestIdx = i
        }
    }

    return {
        landmark: result.landmarks[bestIdx],
        handedness: result.handedness[bestIdx]?.[0]?.displayName ?? 'Unknown',
    }
}
