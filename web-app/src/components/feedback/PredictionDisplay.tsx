import React from "react";
import { Card, CardContent } from "@/components/ui/card";

interface PredictionDisplayProps {
	currentResult: string;
	livePrediction?: string;
	sentence: string;
}

export const PredictionDisplay: React.FC<PredictionDisplayProps> = ({
	currentResult,
	livePrediction,
	sentence,
}) => {
	return (
		<div className="w-full max-w-4xl mt-6 grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
			<Card className="md:col-span-1 border-0 bg-white/5 backdrop-blur-lg shadow-[0_8px_32px_0_rgba(31,38,135,0.3)] ring-1 ring-white/10 rounded-2xl overflow-hidden relative">
				<div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-cyan-500/10 opacity-50" />
				<CardContent className="flex flex-col items-center justify-center h-48 p-6 relative z-10">
					<h3 className="text-white/60 text-xs font-semibold uppercase tracking-[0.2em] mb-3">
						Current Sign
					</h3>
					<div className="text-5xl md:text-6xl font-black bg-clip-text text-transparent bg-gradient-to-br from-green-300 to-emerald-500 drop-shadow-lg">
						{currentResult || "—"}
					</div>
					{livePrediction && (
						<div className="mt-3 text-sm font-mono text-cyan-300/70 tracking-wider">
							{livePrediction}
						</div>
					)}
				</CardContent>
			</Card>

			<Card className="md:col-span-2 border-0 bg-white/5 backdrop-blur-lg shadow-[0_8px_32px_0_rgba(31,38,135,0.3)] ring-1 ring-white/10 rounded-2xl overflow-hidden relative">
				<div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 opacity-50" />
				<CardContent className="flex flex-col justify-start h-48 p-6 relative z-10 overflow-y-auto custom-scrollbar">
					<h3 className="text-white/60 text-xs font-semibold uppercase tracking-[0.2em] mb-4 sticky top-0">
						Transcript
					</h3>
					<div className="text-2xl md:text-3xl lg:text-4xl font-light text-white/90 leading-relaxed tracking-wide">
						{sentence}
						<span className="inline-block w-3 h-8 ml-2 bg-gradient-to-t from-cyan-500 to-blue-500 rounded-full animate-pulse align-middle shadow-[0_0_15px_rgba(6,182,212,0.6)]" />
					</div>
				</CardContent>
			</Card>
		</div>
	);
};
