import { useRef, useEffect, useState, useCallback } from 'react'
import LandmarkOverlay from './LandmarkOverlay'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'

interface Props {
    onVideoReady: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void
    landmarks: HandLandmarkerResult['landmarks'] | null
    isRunning: boolean
}

export default function CameraView({ onVideoReady, landmarks, isRunning }: Props) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const streamRef = useRef<MediaStream | null>(null)
    const [dims, setDims] = useState({ w: 1280, h: 720 })
    const [camError, setCamError] = useState<string | null>(null)
    const notified = useRef(false)

    // Notify parent once refs are mounted
    useEffect(() => {
        if (videoRef.current && canvasRef.current && !notified.current) {
            notified.current = true
            onVideoReady(videoRef.current, canvasRef.current)
        }
    })

    // Webcam lifecycle
    useEffect(() => {
        let mounted = true
        if (isRunning && !streamRef.current) {
            navigator.mediaDevices
                .getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } })
                .then(stream => {
                    if (!mounted) { stream.getTracks().forEach(t => t.stop()); return }
                    streamRef.current = stream
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream
                        videoRef.current.play().catch(console.error)
                    }
                })
                .catch(err => {
                    if (!mounted) return
                    setCamError(err.message)
                    console.error('Camera error:', err)
                })
        }
        if (!isRunning && streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop())
            streamRef.current = null
            if (videoRef.current) videoRef.current.srcObject = null
        }
        return () => { mounted = false }
    }, [isRunning])

    const handleMetadata = useCallback(() => {
        const v = videoRef.current
        if (v) setDims({ w: v.videoWidth, h: v.videoHeight })
    }, [])

    return (
        <div className="cam-shell">
            {/* Hidden canvas — raw unmirrored feed for MediaPipe */}
            <canvas ref={canvasRef} className="mp-canvas" width={dims.w} height={dims.h} />

            {/* Visible mirrored video */}
            <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                onLoadedMetadata={handleMetadata}
                className="cam-video"
            />

            {/* Landmark skeleton overlay (mirrors X for display) */}
            <LandmarkOverlay landmarks={landmarks} videoWidth={dims.w} videoHeight={dims.h} />

            {/* Radial vignette */}
            <div className="cam-vignette" />

            {/* Status overlays */}
            {camError && (
                <div className="cam-overlay">
                    <p className="text-red-400 text-sm font-medium">⚠ {camError}</p>
                </div>
            )}
            {!isRunning && !camError && (
                <div className="cam-overlay">
                    <p className="text-slate-400 text-base font-medium">Camera paused</p>
                </div>
            )}
        </div>
    )
}
