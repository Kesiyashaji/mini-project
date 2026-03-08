interface Props {
    threshold: number
    onThresholdChange: (v: number) => void
    isRunning: boolean
    onToggle: () => void
}

export default function Settings({ threshold, onThresholdChange, isRunning, onToggle }: Props) {
    const pct = Math.round(threshold * 100)

    return (
        <div className="glass" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <span style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 500 }}>
                Settings
            </span>

            {/* Camera toggle */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, color: '#94a3b8' }}>Camera</span>
                <button className={`toggle ${isRunning ? 'on' : 'off'}`} onClick={onToggle}>
                    <span className="toggle-knob" />
                </button>
            </div>

            {/* Threshold slider */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: 13, color: '#94a3b8' }}>Min Confidence</span>
                    <span style={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace', color: '#2dd4bf' }}>{pct}%</span>
                </div>
                <input
                    type="range"
                    min={0.1} max={0.9} step={0.05}
                    value={threshold}
                    onChange={e => onThresholdChange(parseFloat(e.target.value))}
                    style={{
                        background: `linear-gradient(90deg, #2dd4bf ${pct}%, rgba(255,255,255,.08) ${pct}%)`,
                    }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#334155' }}>
                    <span>Sensitive</span>
                    <span>Strict</span>
                </div>
            </div>
        </div>
    )
}
