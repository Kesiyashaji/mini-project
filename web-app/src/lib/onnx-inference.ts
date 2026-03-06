// ONNX Runtime - loaded via CDN
// Uses cdnImport to bypass Turbopack's static import() analysis
//
// Supports two modes:
//   1. "static" (legacy) — single frame of 63 floats → letter classification
//   2. "sequence" (new) — sequence of T frames × 63 floats → word classification
//
// The mode is auto-detected from model_meta.json at load time.

import { cdnImport } from "./cdn-import";

const ORT_CDN =
	"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.mjs";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ortModule: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let session: any = null;

// Model metadata — loaded from model_meta.json
interface ModelMeta {
	mode: "static" | "sequence";
	sign_names: string[];
	seq_length: number;
	n_features: number;
	model_type: string;
}

let modelMeta: ModelMeta | null = null;

// Fallback labels for legacy static model (Sign MNIST: A-Y, no J/Z)
const SIGN_MNIST_LABELS = [
	"A", "B", "C", "D", "E", "F", "G", "H", "I",
	"K", "L", "M", "N", "O", "P", "Q", "R", "S",
	"T", "U", "V", "W", "X", "Y",
];

const getOrt = async () => {
	if (ortModule) return ortModule;
	ortModule = await cdnImport(ORT_CDN);
	return ortModule;
};

/**
 * Load model metadata from the JSON file exported by train_seq.py.
 * Falls back to "static" mode if the metadata file doesn't exist.
 */
const loadMeta = async (): Promise<ModelMeta> => {
	if (modelMeta) return modelMeta;

	try {
		const res = await fetch("/models/model_meta.json");
		if (res.ok) {
			const loadedMeta = await res.json();
			modelMeta = {
				mode: loadedMeta.mode || "static",
				sign_names: loadedMeta.sign_names || SIGN_MNIST_LABELS,
				seq_length: loadedMeta.seq_length || 1,
				n_features: loadedMeta.n_features || 63,
				model_type: loadedMeta.model_type || "xgboost",
			};
			console.log("Model metadata loaded:", modelMeta);
			return modelMeta!;
		}
	} catch {
		// metadata not found — fall back to legacy static mode
	}

	modelMeta = {
		mode: "static",
		sign_names: SIGN_MNIST_LABELS,
		seq_length: 16,
		n_features: 63,
		model_type: "xgboost",
	};
	console.log("No model_meta.json found, using legacy static mode");
	return modelMeta;
};

// Loading lock — prevents concurrent InferenceSession.create calls from the predict loop
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadingPromise: Promise<any> | null = null;
let loadFailed = false;

export const loadModel = async () => {
	if (session) return session;
	if (loadFailed) return null;
	if (loadingPromise) return loadingPromise;

	loadingPromise = (async () => {
		const ort = await getOrt();

		// Point WASM files to the CDN so the browser can fetch them
		ort.env.wasm.wasmPaths =
			"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";

		try {
			session = await ort.InferenceSession.create("/models/model.onnx", {
				executionProviders: ["webgl", "wasm"],
			});
			console.log("ONNX Session loaded");

			// Also load metadata
			await loadMeta();

			return session;
		} catch (e) {
			console.error("Failed to load ONNX session", e);
			loadFailed = true;
			return null;
		} finally {
			loadingPromise = null;
		}
	})();

	return loadingPromise;
};

/**
 * Get the current model mode and configuration.
 * Useful for the UI to know whether to use the frame buffer.
 */
export const getModelMeta = async (): Promise<ModelMeta> => {
	return loadMeta();
};

/**
 * Run inference with a single frame (legacy static model).
 * Input: 63 floats (one frame of normalized landmarks)
 */
export const runInference = async (
	flattenedLandmarks: number[],
): Promise<string | null> => {
	if (!session) {
		await loadModel();
	}
	if (!session) return null;

	const meta = await loadMeta();
	const ort = await getOrt();

	try {
		const data = Float32Array.from(flattenedLandmarks);
		const expectedLen = meta.n_features;
		if (data.length !== expectedLen) {
			console.warn(`runInference: expected ${expectedLen} features, got ${data.length}`);
			return null;
		}
		const tensor = new ort.Tensor("float32", data, [1, meta.n_features]);

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const feeds: Record<string, any> = {};
		const inputName = session.inputNames[0];
		feeds[inputName] = tensor;

		const results = await session.run(feeds);
		const outputName = session.outputNames[0];
		const output = results[outputName];

		if (output.type === "int64" || output.type === "int32") {
			const classIndex = Number(output.data[0]);
			return meta.sign_names[classIndex] || "?";
		}

		const label = Number(output.data[0]);
		return meta.sign_names[label] || "?";
	} catch (e) {
		console.error("Inference failed", e);
		return null;
	}
};

/**
 * Run inference with a sequence of frames (new LSTM/Transformer model).
 * Input: Float32Array of length (seq_length × 63), already flattened row-major.
 * Returns the predicted label and its softmax confidence.
 */
export const runSequenceInference = async (
	sequenceData: Float32Array,
): Promise<{ label: string; confidence: number } | null> => {
	if (!session) {
		await loadModel();
	}
	if (!session) return null;

	const meta = await loadMeta();
	const ort = await getOrt();

	try {
		// Validate input length
		const expectedLen = meta.seq_length * meta.n_features;
		if (sequenceData.length !== expectedLen) {
			console.warn(
				`runSequenceInference: expected ${expectedLen} elements (${meta.seq_length}×${meta.n_features}), got ${sequenceData.length}`,
			);
			return null;
		}

		// Shape: [batch=1, seq_length, n_features]
		const tensor = new ort.Tensor("float32", sequenceData, [
			1,
			meta.seq_length,
			meta.n_features,
		]);

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const feeds: Record<string, any> = {};
		const inputName = session.inputNames[0];
		feeds[inputName] = tensor;

		const results = await session.run(feeds);
		const outputName = session.outputNames[0];
		const output = results[outputName];

		// PyTorch CrossEntropy output is logits — softmax + argmax
		const logits = output.data;
		const numClasses = logits.length;

		// Compute softmax for confidence
		let maxLogit = -Infinity;
		for (let i = 0; i < numClasses; i++) {
			if (Number(logits[i]) > maxLogit) maxLogit = Number(logits[i]);
		}
		let sumExp = 0;
		const exps = new Float64Array(numClasses);
		for (let i = 0; i < numClasses; i++) {
			exps[i] = Math.exp(Number(logits[i]) - maxLogit);
			sumExp += exps[i];
		}

		let maxIdx = 0;
		let maxProb = 0;
		for (let i = 0; i < numClasses; i++) {
			const prob = exps[i] / sumExp;
			if (prob > maxProb) {
				maxProb = prob;
				maxIdx = i;
			}
		}

		return {
			label: meta.sign_names[maxIdx] || "?",
			confidence: maxProb,
		};
	} catch (e) {
		console.error("Sequence inference failed", e);
		return null;
	}
};
