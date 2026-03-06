"use client";

import React, { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { PredictionDisplay } from "@/components/feedback/PredictionDisplay";
import { ControlPanel } from "@/components/controls/ControlPanel";

const SignStream = dynamic(
	() => import("@/components/camera/SignStream"),
	{
		ssr: false,
		loading: () => (
			<div className="relative w-full max-w-4xl aspect-video bg-black rounded-lg overflow-hidden shadow-2xl flex items-center justify-center text-white">
				Loading AI Models...
			</div>
		),
	},
);

export default function Home() {
	const [currentResult, setCurrentResult] = useState<string>("");
	const [livePrediction, setLivePrediction] = useState<string>("");
	const [sentence, setSentence] = useState<string>("");
	const [isStreaming, setIsStreaming] = useState<boolean>(true);

	const handlePrediction = useCallback((char: string) => {
		setCurrentResult(char);

		setSentence((prev) => {
			const prevWords = prev.trim().split(" ");
			const lastWord = prevWords[prevWords.length - 1];

			// Duplicate prevention: avoid spamming the exact same word
			if (char.toLowerCase() === lastWord?.toLowerCase()) {
				return prev;
			}

			// TTS logic for the new word
			if ("speechSynthesis" in window) {
				const utterance = new SpeechSynthesisUtterance(char);
				utterance.volume = 1;
				utterance.rate = 1.05;
				window.speechSynthesis.speak(utterance);
			}

			// Add a space to separate sequence words
			return prev ? `${prev} ${char}` : char;
		});
	}, []);

	const handleLivePrediction = useCallback((sign: string) => {
		setLivePrediction(sign);
	}, []);

	const handleClear = () => {
		setSentence("");
		setCurrentResult("");
	};

	const handleSpeak = () => {
		if ("speechSynthesis" in window) {
			const utterance = new SpeechSynthesisUtterance(sentence);
			window.speechSynthesis.speak(utterance);
		}
	};

	const handleToggleStream = () => {
		setIsStreaming(!isStreaming);
	};

	return (
		<main className="flex min-h-screen flex-col items-center justify-between p-8 bg-gradient-to-br from-[#0B0D17] via-[#10142A] to-[#160D22] text-white">
			<div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex mb-8">
				<p className="fixed left-0 top-0 flex w-full justify-center border-b border-purple-500/20 bg-black/40 pb-6 pt-8 backdrop-blur-3xl shadow-[0_0_20px_rgba(139,92,246,0.1)] lg:static lg:w-auto lg:rounded-2xl lg:border lg:p-4">
					<span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-cyan-400 font-bold tracking-widest uppercase">
						SignStream Live
					</span>
					&nbsp;
					<code className="font-bold text-slate-400">v1.2.0</code>
				</p>
			</div>

			<div className="relative flex flex-col items-center w-full max-w-5xl">
				<SignStream onPrediction={handlePrediction} onLivePrediction={handleLivePrediction} isStreaming={isStreaming} />

				<PredictionDisplay currentResult={currentResult} livePrediction={livePrediction} sentence={sentence} />

				<ControlPanel
					onClear={handleClear}
					onSpeak={handleSpeak}
					isStreaming={isStreaming}
					onToggleStream={handleToggleStream}
				/>
			</div>

			<div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-4 lg:text-left">
				{/* Footer content if needed */}
			</div>
		</main>
	);
}
