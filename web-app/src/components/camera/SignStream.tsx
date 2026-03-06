"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { getHandLandmarker, getDrawingUtils, getHandConnections } from "@/lib/mediapipe";
import { normalizeLandmarks } from "@/lib/normalization";
import { runInference, runSequenceInference, getModelMeta } from "@/lib/onnx-inference";
import { MajorityVoteBuffer } from "@/lib/buffer-utils";
import { FrameBuffer } from "@/lib/frame-buffer";

interface SignStreamProps {
	onPrediction: (char: string) => void;
	onLivePrediction?: (char: string) => void;
	isStreaming: boolean;
}

// --- Pose stability helpers (used in static mode only) ---

/** Compute average movement between two sets of landmarks. */
function landmarkDelta(
	prev: { x: number; y: number; z: number }[],
	curr: { x: number; y: number; z: number }[],
): number {
	if (prev.length !== curr.length || prev.length === 0) return Infinity;
	let sum = 0;
	for (let i = 0; i < prev.length; i++) {
		const dx = curr[i].x - prev[i].x;
		const dy = curr[i].y - prev[i].y;
		sum += Math.sqrt(dx * dx + dy * dy);
	}
	return sum / prev.length;
}

// Thresholds (tune these)
const STABILITY_THRESHOLD = 0.012; // Max avg landmark movement to be "still"
const STABLE_FRAMES_NEEDED = 8; // Must be still for this many frames before predicting
const COOLDOWN_MS = 1500; // Minimum ms between predictions

// Sequence mode settings
const SEQUENCE_INFERENCE_INTERVAL_MS = 1000; // Run sequence inference every 1s for responsiveness
const CONFIDENCE_THRESHOLD = 0.4; // Minimum softmax confidence to accept a prediction
const IGNORED_LABELS = new Set(["nothing", "del", "space"]);

const SignStream: React.FC<SignStreamProps> = ({
	onPrediction,
	onLivePrediction,
	isStreaming,
}) => {
	const videoRef = useRef<HTMLVideoElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [loading, setLoading] = useState(true);
	const [modelMode, setModelMode] = useState<"static" | "sequence">("static");
	const buffer = useRef(new MajorityVoteBuffer(10));
	const frameBuffer = useRef<FrameBuffer | null>(null);
	const requestRef = useRef<number | null>(null);
	const lastVideoTimeRef = useRef<number>(-1);

	// Pose stability state (static mode)
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const prevLandmarksRef = useRef<any[] | null>(null);
	const stableFrameCount = useRef(0);
	const lastPredictionTime = useRef(0);
	const hasInferredForPose = useRef(false);

	// Sequence mode timing
	const lastSeqInferenceTime = useRef(0);

	// Cache for CDN-loaded drawing utilities
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const drawingUtilsCtorRef = useRef<any>(null);
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const handConnectionsRef = useRef<any>(null);

	useEffect(() => {
		const init = async () => {
			// Load model metadata to determine mode
			const meta = await getModelMeta();
			setModelMode(meta.mode);

			if (meta.mode === "sequence") {
				frameBuffer.current = new FrameBuffer(meta.seq_length, meta.n_features);
				console.log(`Sequence mode: buffer size=${meta.seq_length}, model=${meta.model_type}`);
			}

			const [DrawingUtils, connections] = await Promise.all([
				getDrawingUtils(),
				getHandConnections(),
			]);
			drawingUtilsCtorRef.current = DrawingUtils;
			handConnectionsRef.current = connections;

			if (videoRef.current) {
				try {
					const stream = await navigator.mediaDevices.getUserMedia({
						video: {
							width: 1280,
							height: 720,
							frameRate: { ideal: 30 },
						},
					});
					videoRef.current.srcObject = stream;
					videoRef.current.onloadedmetadata = () => {
						videoRef.current?.play();
						setLoading(false);
					};
				} catch (err) {
					console.error("Error accessing webcam", err);
					setLoading(false);
				}
			}
		};
		init();

		return () => {
			if (videoRef.current && videoRef.current.srcObject) {
				const stream = videoRef.current.srcObject as MediaStream;
				stream.getTracks().forEach((track) => track.stop());
			}
			cancelAnimationFrame(requestRef.current!);
		};
	}, []);

	// ---- Static mode inference (single frame, pose stability) ----
	const handleStaticInference = useCallback(
		async (landmarks: { x: number; y: number; z: number }[], now: number) => {
			const delta = prevLandmarksRef.current
				? landmarkDelta(prevLandmarksRef.current, landmarks)
				: Infinity;

			prevLandmarksRef.current = landmarks.map((lm) => ({
				x: lm.x, y: lm.y, z: lm.z,
			}));

			if (delta < STABILITY_THRESHOLD) {
				stableFrameCount.current++;
			} else {
				stableFrameCount.current = 0;
				hasInferredForPose.current = false;
			}

			const cooldownPassed = now - lastPredictionTime.current > COOLDOWN_MS;
			const isStable = stableFrameCount.current >= STABLE_FRAMES_NEEDED;

			if (isStable && cooldownPassed && !hasInferredForPose.current) {
				try {
					const normalized = normalizeLandmarks(landmarks);
					if (normalized.length > 0) {
						const char = await runInference(normalized);
						if (char) {
							buffer.current.add(char);
							const stable = buffer.current.getStablePrediction();
							if (stable) {
								onPrediction(stable);
								buffer.current.clear();
								lastPredictionTime.current = now;
								hasInferredForPose.current = true;
							}
						}
					}
				} catch (inferErr) {
					console.warn("Inference error (continuing loop):", inferErr);
				}
			}
		},
		[onPrediction],
	);

	// ---- Sequence mode inference (frame buffer → model) ----
	const handleSequenceInference = useCallback(
		async (landmarks: { x: number; y: number; z: number }[], now: number) => {
			const normalized = normalizeLandmarks(landmarks);
			if (normalized.length === 0) return;

			frameBuffer.current?.push(normalized);

			// Run inference periodically when buffer is full
			const timeSinceLastInference = now - lastSeqInferenceTime.current;
			if (
				frameBuffer.current?.isFull() &&
				timeSinceLastInference > SEQUENCE_INFERENCE_INTERVAL_MS
			) {
				const seqData = frameBuffer.current.getSequence();
				if (seqData) {
					try {
						const result = await runSequenceInference(seqData);
						if (result) {
							lastSeqInferenceTime.current = now;

							// Always show live prediction for feedback
							if (onLivePrediction) {
								onLivePrediction(`${result.label} (${Math.round(result.confidence * 100)}%)`);
							}

							// Only add to transcript if confident and not an ignored label
							if (
								result.confidence >= CONFIDENCE_THRESHOLD &&
								!IGNORED_LABELS.has(result.label.toLowerCase())
							) {
								onPrediction(result.label);
							}
						}
					} catch (inferErr) {
						console.warn("Sequence inference error:", inferErr);
					}
				}
			}
		},
		[onPrediction, onLivePrediction],
	);

	const predictLoop = async () => {
		const video = videoRef.current;
		const canvas = canvasRef.current;
		if (!video || !canvas || !isStreaming) return;

		try {
			const ctx = canvas.getContext("2d");

			if (video.currentTime !== lastVideoTimeRef.current) {
				lastVideoTimeRef.current = video.currentTime;

				if (ctx) {
					ctx.clearRect(0, 0, canvas.width, canvas.height);
				}

				const handLandmarker = await getHandLandmarker();
				if (handLandmarker) {
					const results = handLandmarker.detectForVideo(video, performance.now());

					// Add confidence filter: assume target hand score must be > 0.5
					let hasValidHand = false;
					if (results.landmarks && results.landmarks.length > 0) {
						if (results.handednesses && results.handednesses.length > 0) {
							hasValidHand = results.handednesses[0][0].score > 0.5;
						} else {
							hasValidHand = true;
						}
					}

					if (ctx && hasValidHand && drawingUtilsCtorRef.current && handConnectionsRef.current) {
						ctx.save();
						ctx.scale(-1, 1);
						ctx.translate(-canvas.width, 0);

						const DrawingUtils = drawingUtilsCtorRef.current;
						const drawingUtils = new DrawingUtils(ctx);
						for (const landmarks of results.landmarks) {
							// Draw landmarks
							drawingUtils.drawConnectors(
								landmarks,
								handConnectionsRef.current,
								{ color: "#00FF00", lineWidth: 5 },
							);
							drawingUtils.drawLandmarks(landmarks, {
								color: "#FF0000",
								lineWidth: 2,
							});

							// --- Mode-specific inference ---
							const now = performance.now();
							if (modelMode === "sequence") {
								await handleSequenceInference(landmarks, now);

								// Draw buffer fill indicator with a modern clean glassmorphism styling
								if (frameBuffer.current) {
									const fillPct = frameBuffer.current.fillLevel();
									const barWidth = 240;
									const barHeight = 6;
									const barX = (canvas.width - barWidth) / 2;
									const barY = canvas.height - 40;

									ctx.save();
									ctx.scale(-1, 1);
									ctx.translate(-canvas.width, 0);

									// Background track
									ctx.shadowColor = "rgba(0,0,0,0.5)";
									ctx.shadowBlur = 10;
									ctx.fillStyle = "rgba(255,255,255,0.1)";
									ctx.beginPath();
									ctx.roundRect(barX, barY, barWidth, barHeight, 4);
									ctx.fill();

									// Foreground fill
									ctx.shadowColor = fillPct >= 1 ? "rgba(16,185,129,0.8)" : "rgba(6,182,212,0.8)";
									ctx.shadowBlur = 15;
									const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
									gradient.addColorStop(0, fillPct >= 1 ? "#34d399" : "#22d3ee");
									gradient.addColorStop(1, fillPct >= 1 ? "#10b981" : "#06b6d4");
									ctx.fillStyle = gradient;
									ctx.beginPath();
									ctx.roundRect(barX, barY, barWidth * fillPct, barHeight, 4);
									ctx.fill();

									// Text label
									ctx.shadowBlur = 5;
									ctx.shadowColor = "rgba(0,0,0,0.8)";
									ctx.fillStyle = "rgba(255,255,255,0.9)";
									ctx.font = "bold 13px Inter, sans-serif";
									ctx.textAlign = "center";
									ctx.letterSpacing = "2px";
									ctx.fillText(
										fillPct >= 1 ? "ANALYZING..." : "BUFFERING SIGN...",
										barX + barWidth / 2,
										barY - 12,
									);
									ctx.restore();
								}
							} else {
								await handleStaticInference(landmarks, now);

								// Draw stability indicator with modern look
								const stabilityPct = Math.min(stableFrameCount.current / STABLE_FRAMES_NEEDED, 1);
								const barWidth = 240;
								const barHeight = 6;
								const barX = (canvas.width - barWidth) / 2;
								const barY = canvas.height - 40;

								ctx.save();
								ctx.scale(-1, 1);
								ctx.translate(-canvas.width, 0);

								// Background track
								ctx.shadowColor = "rgba(0,0,0,0.5)";
								ctx.shadowBlur = 10;
								ctx.fillStyle = "rgba(255,255,255,0.1)";
								ctx.beginPath();
								ctx.roundRect(barX, barY, barWidth, barHeight, 4);
								ctx.fill();

								// Foreground fill
								ctx.shadowColor = stabilityPct >= 1 ? "rgba(16,185,129,0.8)" : "rgba(245,158,11,0.8)";
								ctx.shadowBlur = 15;
								const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
								gradient.addColorStop(0, stabilityPct >= 1 ? "#34d399" : "#fbbf24");
								gradient.addColorStop(1, stabilityPct >= 1 ? "#10b981" : "#f59e0b");
								ctx.fillStyle = gradient;
								ctx.beginPath();
								ctx.roundRect(barX, barY, barWidth * stabilityPct, barHeight, 4);
								ctx.fill();

								// Text label
								ctx.shadowBlur = 5;
								ctx.shadowColor = "rgba(0,0,0,0.8)";
								ctx.fillStyle = "rgba(255,255,255,0.9)";
								ctx.font = "bold 13px Inter, sans-serif";
								ctx.textAlign = "center";
								ctx.letterSpacing = "2px";
								ctx.fillText(
									stabilityPct >= 1 ? "READY" : "HOLD STILL",
									barX + barWidth / 2,
									barY - 12
								);
								ctx.restore();
							}
						}
						ctx.restore();
					} else {
						// No hand — reset state
						prevLandmarksRef.current = null;
						stableFrameCount.current = 0;
						hasInferredForPose.current = false;
						if (modelMode === "sequence") {
							frameBuffer.current?.clear();
						}
					}
				}
			}
		} catch (err) {
			console.warn("PredictLoop error (continuing):", err);
		}

		requestRef.current = requestAnimationFrame(predictLoop);
	};

	useEffect(() => {
		if (isStreaming && !loading) {
			requestRef.current = requestAnimationFrame(predictLoop);
		} else {
			if (requestRef.current !== null) {
				cancelAnimationFrame(requestRef.current);
			}
		}
		return () => {
			if (requestRef.current !== null) {
				cancelAnimationFrame(requestRef.current);
			}
		};
	}, [isStreaming, loading]);

	return (
		<div className="relative w-full max-w-4xl aspect-video bg-black/40 rounded-3xl overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.5)] border border-white/10 ring-1 ring-white/5 z-10">
			{loading && (
				<div className="absolute inset-0 flex items-center justify-center text-white/80 bg-black/60 backdrop-blur-sm z-20">
					<div className="flex flex-col items-center gap-4">
						<div className="w-8 h-8 rounded-full border-4 border-cyan-500 border-t-transparent animate-spin" />
						<span className="font-mono tracking-widest text-sm uppercase">Loading AI...</span>
					</div>
				</div>
			)}
			{/* Mode indicator */}
			{!loading && (
				<div className="absolute top-4 right-4 z-20 px-3 py-1.5 rounded-full text-[10px] font-bold tracking-widest uppercase text-cyan-300 bg-cyan-900/40 border border-cyan-500/30 backdrop-blur-md shadow-[0_0_15px_rgba(6,182,212,0.2)]">
					{modelMode === "sequence" ? "🎬 Sequence Mode" : "📸 Static Mode"}
				</div>
			)}
			<video
				ref={videoRef}
				className="absolute inset-0 w-full h-full object-cover transform -scale-x-100 opacity-90"
				playsInline
				muted
			/>
			<canvas
				ref={canvasRef}
				width={1280}
				height={720}
				className="absolute inset-0 w-full h-full object-cover pointer-events-none"
			/>
		</div>
	);
};

export default SignStream;
