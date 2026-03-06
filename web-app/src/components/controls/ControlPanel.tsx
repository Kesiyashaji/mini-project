import React from "react";
import { Button } from "@/components/ui/button";
import { Volume2, Trash2, StopCircle, PlayCircle } from "lucide-react";

interface ControlPanelProps {
	onClear: () => void;
	onSpeak: () => void;
	isStreaming: boolean;
	onToggleStream: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
	onClear,
	onSpeak,
	isStreaming,
	onToggleStream,
}) => {
	return (
		<div className="flex items-center gap-6 mt-8 w-full max-w-4xl px-4 py-4 bg-white/5 backdrop-blur-md rounded-2xl shadow-[0_4px_24px_rgba(0,0,0,0.4)] border border-white/10 relative z-10">
			<Button
				variant={isStreaming ? "destructive" : "default"}
				onClick={onToggleStream}
				className={`w-40 h-12 rounded-xl font-bold tracking-wide transition-all duration-300 ${isStreaming ? "bg-red-500/80 hover:bg-red-500 text-white shadow-[0_0_15px_rgba(239,68,68,0.4)]" : "bg-emerald-500/80 hover:bg-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.4)]"}`}
			>
				{isStreaming ? (
					<>
						<StopCircle className="mr-2 h-5 w-5 animate-pulse" /> Stop Stream
					</>
				) : (
					<>
						<PlayCircle className="mr-2 h-5 w-5" /> Start Stream
					</>
				)}
			</Button>

			<div className="flex-1" />

			<Button
				variant="outline"
				onClick={onClear}
				className="h-12 px-6 rounded-xl border-white/20 hover:border-white/40 bg-white/5 hover:bg-white/10 transition-colors text-white/80 hover:text-white"
			>
				<Trash2 className="mr-2 h-4 w-4" /> Clear
			</Button>

			<Button
				onClick={onSpeak}
				className="h-12 px-8 rounded-xl bg-gradient-to-r from-cyan-500/80 to-blue-500/80 hover:from-cyan-400 hover:to-blue-400 text-white shadow-[0_0_15px_rgba(6,182,212,0.4)] transition-all duration-300 border-0"
			>
				<Volume2 className="mr-2 h-5 w-5" /> Speak
			</Button>
		</div>
	);
};
