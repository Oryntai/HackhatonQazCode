import { useState } from 'react'
import './App.css'
import AuroraBackground from './components/AuroraBackground'

const DIAGNOSE_ENDPOINT = import.meta.env.DEV
  ? 'http://127.0.0.1:8000/diagnose'
  : '/diagnose'

function normalizeDiagnoses(payload) {
  const rawDiagnoses = Array.isArray(payload?.diagnoses) ? payload.diagnoses : []
  const sorted = [...rawDiagnoses].sort((a, b) => {
    const left = Number(a?.rank) || Number.MAX_SAFE_INTEGER
    const right = Number(b?.rank) || Number.MAX_SAFE_INTEGER
    return left - right
  })

  const normalized = sorted.slice(0, 3).map((item, index) => ({
    rank: Number(item?.rank) || index + 1,
    icd10_code:
      typeof item?.icd10_code === 'string' && item.icd10_code.trim()
        ? item.icd10_code.trim()
        : 'R69',
    diagnosis:
      typeof item?.diagnosis === 'string' && item.diagnosis.trim()
        ? item.diagnosis.trim()
        : 'Неуточненный диагноз',
    explanation:
      typeof item?.explanation === 'string' && item.explanation.trim()
        ? item.explanation.trim()
        : 'Недостаточно данных.',
  }))

  while (normalized.length < 3) {
    normalized.push({
      rank: normalized.length + 1,
      icd10_code: 'R69',
      diagnosis: 'Неуточненный диагноз',
      explanation: 'Недостаточно данных.',
    })
  }

  return normalized
}

function App() {
  const [symptoms, setSymptoms] = useState('')
  const [diagnoses, setDiagnoses] = useState([])
  const [hasResult, setHasResult] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleAnalyze = async (event) => {
    event.preventDefault()

    const preparedSymptoms = symptoms.trim()
    if (!preparedSymptoms) {
      setError('Введите симптомы, чтобы выполнить анализ.')
      return
    }

    setLoading(true)
    setError('')
    setHasResult(false)
    setDiagnoses([])

    try {
      const response = await fetch(DIAGNOSE_ENDPOINT, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json; charset=utf-8',
        },
        body: JSON.stringify({ symptoms: preparedSymptoms }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const payload = await response.json()
      const topDiagnoses = normalizeDiagnoses(payload)

      setDiagnoses(topDiagnoses)
      setHasResult(true)
    } catch (requestError) {
      setDiagnoses([])
      setHasResult(false)
      setError(
        `Не удалось получить ответ от сервера (${requestError.message || 'network error'}).`,
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuroraBackground>
      <main className="glass-shell">
        <header className="hero">
          <p className="hero-tag">МКБ-10 классификатор</p>
          <h1>Клинический RAG-анализ</h1>
          <p className="hero-subtitle">
            Введите анамнез пациента, чтобы получить топ-3 диагноза на основе
            протокольного контекста.
          </p>
        </header>

        <form className="diagnose-form" onSubmit={handleAnalyze}>
          <label htmlFor="symptoms">Симптомы и анамнез</label>
          <textarea
            id="symptoms"
            value={symptoms}
            onChange={(event) => setSymptoms(event.target.value)}
            placeholder="Например: выраженная одышка, боль за грудиной, сатурация 90%, сухой кашель..."
            rows={6}
          />
          <button type="submit" disabled={loading}>
            {loading ? (
              <span className="loading-inline">
                <span className="spinner" />
                Анализ...
              </span>
            ) : (
              'Анализировать'
            )}
          </button>
        </form>

        {error ? <p className="error-message">{error}</p> : null}

        {!loading && !hasResult ? (
          <section className="empty-state">
            <p>Результат появится здесь после анализа.</p>
          </section>
        ) : null}

        {hasResult ? (
          <section className="cards-grid">
            {diagnoses.slice(0, 3).map((item) => (
              <article
                className={`diagnosis-card ${item.icd10_code === 'R69' ? 'diagnosis-card-muted' : ''}`.trim()}
                key={`${item.rank}-${item.icd10_code}-${item.diagnosis}`}
              >
                <p className="rank-label">Ранг {item.rank}</p>
                <h2>{item.icd10_code}</h2>
                <p className="diagnosis-title">{item.diagnosis}</p>
                <p className="diagnosis-explanation">{item.explanation}</p>
              </article>
            ))}
          </section>
        ) : null}
      </main>
    </AuroraBackground>
  )
}

export default App
