export interface Landmark {
	x: number;
	y: number;
	z: number;
}

export const normalizeLandmarks = (landmarks: Landmark[]): number[] => {
	if (!landmarks || landmarks.length === 0) return [];

	// 1. Find the wrist (index 0)
	const wrist = landmarks[0];

	// 2. Translate all points relative to wrist
	const relativeLandmarks = landmarks.map((lm) => ({
		x: lm.x - wrist.x,
		y: lm.y - wrist.y,
		z: lm.z - wrist.z,
	}));

	// 3. Scale normalization — divide by wrist-to-middle-MCP distance
	//    Must match training code: scale = dist(wrist, landmark[9])
	const mcp = relativeLandmarks[9]; // middle finger MCP
	let scale = Math.sqrt(mcp.x * mcp.x + mcp.y * mcp.y + mcp.z * mcp.z);
	if (scale < 1e-6) scale = 1.0;

	// 4. Flatten to 1D array [x0, y0, z0, x1, y1, z1, ...]
	const flattened: number[] = [];
	relativeLandmarks.forEach((lm) => {
		flattened.push(lm.x / scale, lm.y / scale, lm.z / scale);
	});

	return flattened;
};
