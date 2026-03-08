import type { PredEntry } from '@/hooks/useInference'

interface Props { history: PredEntry[]; onClear: () => void }

export default function HistoryFeed({ history, onClear }: Props) {
    return (
        <div className="glass" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 500 }}>
                    History
                </span>
                {history.length > 0 && (
                    <button
                        onClick={onClear}
                        style={{
                            fontSize: 12, color: '#475569', background: 'none', border: 'none',
                            cursor: 'pointer', padding: '2px 6px', borderRadius: 6,
                            transition: 'color .2s',
                        }}
                        onMouseEnter={e => (e.currentTarget.style.color = '#f87171')}
                        onMouseLeave={e => (e.currentTarget.style.color = '#475569')}
                    >
                        Clear
                    </button>
                )}
            </div>

            {history.length === 0 ? (
                <p style={{ fontSize: 12, color: '#334155', textAlign: 'center', padding: '8px 0' }}>
                    No predictions yet
                </p>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                    {history.map((entry, i) => (
                        <div key={entry.id} className={`history-item${i === 0 ? ' first' : ''}`}>
                            <span style={{ fontWeight: 600, fontSize: 13, color: i === 0 ? '#2dd4bf' : '#94a3b8', flex: 1 }}>
                                {entry.label}
                            </span>
                            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: '#475569' }}>
                                {Math.round(entry.prob * 100)}%
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
