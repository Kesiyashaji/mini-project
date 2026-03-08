interface TopKEntry { label: string; prob: number }
interface Props { confidence: number; topK: TopKEntry[] }

export default function ConfidenceBar({ confidence, topK }: Props) {
    const pct = Math.round(confidence * 100)

    return (
        <div className="glass" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 500 }}>
                    Confidence
                </span>
                <span style={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace', color: '#2dd4bf' }}>
                    {pct}%
                </span>
            </div>

            <div className="conf-track">
                <div className="conf-fill" style={{ width: `${pct}%` }} />
            </div>

            {/* Top-3 */}
            {topK.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                    {topK.map((entry, i) => (
                        <div key={entry.label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: 11, color: '#334155', width: 12 }}>{i + 1}.</span>
                            <span style={{
                                fontSize: 12, fontWeight: 500, flex: 1,
                                color: i === 0 ? '#2dd4bf' : '#64748b',
                                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            }}>
                                {entry.label}
                            </span>
                            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: '#475569' }}>
                                {Math.round(entry.prob * 100)}%
                            </span>
                            <div style={{ width: 56, height: 3, borderRadius: 99, background: 'rgba(255,255,255,.06)', overflow: 'hidden' }}>
                                <div style={{
                                    height: '100%', borderRadius: 99, width: `${Math.round(entry.prob * 100)}%`,
                                    background: i === 0 ? 'linear-gradient(90deg,#2dd4bf,#7c3aed)' : 'rgba(148,163,184,.2)',
                                    transition: 'width .3s ease',
                                }} />
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
