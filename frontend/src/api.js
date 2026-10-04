export const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
export const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL ?? 'http://127.0.0.1:5000'

export async function postForm(path, fields, signal) {
  const body = new FormData()
  Object.entries(fields).forEach(([k, v]) => {
    if (v !== undefined && v !== null) body.append(k, v)
  })

  let res
  try {
    res = await fetch(`${API_URL}${path}`, { method: 'POST', body, signal })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new Error('Cannot reach the backend.')
  }

  if (!res.ok) {
    let message = `Request failed (${res.status}).`
    try {
      const data = await res.json()
      if (typeof data.detail === 'string') message = data.detail
    } catch { /* keep default */ }
    throw new Error(message)
  }
  return res.json()
}

export async function getHealth() {
  try {
    return (await fetch(`${API_URL}/api/health`)).ok
  } catch {
    return false
  }
}

export function downloadDataUrl(url, filename) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
}