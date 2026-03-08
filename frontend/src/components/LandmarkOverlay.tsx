import { useRef, useEffect, forwardRef } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'

interface Props {
    landmarks: HandLandmarkerResult['landmarks'] | null
    videoWidth: number
    videoHeight: number
}

const CONNECTIONS = [
    // thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // index
    [0, 5], [5, 6], [6, 7], [7, 8],
    // middle
    [0, 9], [9, 10], [10, 11], [11, 12],
    // ring
    [0, 13], [13, 14], [14, 15], [15, 16],
    // pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // palm
    [5, 9], [9, 13], [13, 17],
]

export default function LandmarkOverlay({ landmarks, videoWidth, videoHeight }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')!
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        if (!landmarks || landmarks.length === 0) return

        canvas.width = videoWidth || canvas.offsetWidth
        canvas.height = videoHeight || canvas.offsetHeight

        for (const hand of landmarks) {
            // Connections
            ctx.strokeStyle = 'rgba(45, 212, 191, 0.7)'
            ctx.lineWidth = 2
            for (const [a, b] of CONNECTIONS) {
                const la = hand[a], lb = hand[b]
                ctx.beginPath()
                // mirror x for display (video is CSS mirrored)
                ctx.moveTo((1 - la.x) * canvas.width, la.y * canvas.height)
                ctx.lineTo((1 - lb.x) * canvas.width, lb.y * canvas.height)
                ctx.stroke()
            }

            // Dots
            for (let i = 0; i < hand.length; i++) {
                const lm = hand[i]
                const x = (1 - lm.x) * canvas.width
                const y = lm.y * canvas.height
                ctx.beginPath()
                ctx.arc(x, y, i === 0 ? 5 : 3, 0, Math.PI * 2)
                ctx.fillStyle = i === 0 ? '#7c3aed' : '#2dd4bf'
                ctx.fill()
            }
        }
    }, [landmarks, videoWidth, videoHeight])

    return (
        <canvas
            ref={canvasRef}
            width={videoWidth || 640}
            height={videoHeight || 480}
            className="absolute inset-0 w-full h-full pointer-events-none"
        />
    )
}
