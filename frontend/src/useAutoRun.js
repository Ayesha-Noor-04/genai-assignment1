import { useEffect, useState } from 'react'
import { postForm } from './api'

export function useAutoRun(endpoint, source, params) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const paramsKey = JSON.stringify(params)

  useEffect(() => {
    if (!source) return
    const controller = new AbortController()
    setLoading(true)
    setError('')

    postForm(endpoint, { ...source, ...JSON.parse(paramsKey) }, controller.signal)
      .then(setResult)
      .catch((err) => {
        if (err.name === 'AbortError') return
        setResult(null)
        setError(err.message || 'Something went wrong.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [endpoint, source, paramsKey])

  return { result, loading, error }
}