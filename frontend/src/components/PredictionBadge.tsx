interface Props {
    prediction: string | null
    confidence: number
    handedness: string | null
}

export default function PredictionBadge({ prediction, confidence, handedness }: Props) {
    if (!prediction) {
        return (
            <div className="glass pred-card">
                <div style={{ fontSize: 13, color: '#475569', letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: 500 }}>
                    Waiting for sign…
                </div>
                <div style={{ fontSize: 12, color: '#334155', marginTop: 6 }}>
                    Show your hand to the camera
                </div>
            </div>
        )
    }

    const pct = Math.round(confidence * 100)

    return (
        <div key={prediction} className="glass pred-card animate-badge">
            <div style={{ fontSize: 12, color: '#64748b', letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: 500, marginBottom: 8 }}>
                Detected Sign
            </div>
            <div className="pred-sign">{prediction}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
                <span style={{ fontSize: 14, fontFamily: 'JetBrains Mono, monospace', color: '#2dd4bf', fontWeight: 500 }}>
                    {pct}%
                </span>
                {handedness && (
                    <span style={{
                        fontSize: 11, padding: '2px 8px', borderRadius: 99, fontWeight: 500,
                        background: 'rgba(124,58,237,.15)', color: '#a78bfa',
                        border: '1px solid rgba(124,58,237,.25)'
                    }}>
                        {handedness}
                    </span>
                )}
            </div>
        </div>
    )
}
