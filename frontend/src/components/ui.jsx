import { useEffect, useRef, useState } from 'react'

/* ---------- icons ---------- */
const svg = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', viewBox: '0 0 24 24' }
export const UploadIcon = ({ className }) => (
  <svg {...svg} className={className}><path d="M12 16V4m0 0L7 9m5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></svg>
)
export const DownloadIcon = ({ className }) => (
  <svg {...svg} className={className}><path d="M12 4v12m0 0 5-5m-5 5-5-5M4 20h16" /></svg>
)
export const MenuIcon = ({ className }) => (
  <svg {...svg} className={className}><path d="M4 6h16M4 12h16M4 18h16" /></svg>
)
const SpinnerIcon = ({ className }) => (
  <svg {...svg} className={`animate-spin ${className}`}><path d="M12 3a9 9 0 1 0 9 9" /></svg>
)

/* ---------- controls ---------- */
export function Select({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-on-surface-variant">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full cursor-pointer rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2 text-sm focus:border-primary focus:outline-none"
      >
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  )
}

export function Segmented({ label, options, value, onChange }) {
  return (
    <div className="flex flex-col gap-1.5">
      {label && <span className="text-xs font-medium text-on-surface-variant">{label}</span>}
      <div className="flex gap-1 rounded-lg bg-surface-container-low p-1">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              value === o.value
                ? 'bg-secondary-container text-primary'
                : o.disabled
                  ? 'cursor-not-allowed text-outline'
                  : 'text-on-surface-variant hover:bg-surface-container'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function DownloadButton({ onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-on-primary transition-colors hover:bg-primary-container disabled:cursor-not-allowed disabled:opacity-40"
    >
      <DownloadIcon className="h-4 w-4" /> Download
    </button>
  )
}

/* ---------- feedback ---------- */
export function ErrorBanner({ message }) {
  if (!message) return null
  return (
    <div className="mb-4 rounded-lg bg-error-container px-4 py-2 text-sm text-on-error-container">
      {message}
    </div>
  )
}

export function Loading() {
  return (
    <div className="flex items-center gap-2 text-sm text-on-surface-variant">
      <SpinnerIcon className="h-4 w-4" /> Running...
    </div>
  )
}

/* ---------- display ---------- */
export function ImagePanel({ label, src }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-sm font-medium text-on-surface-variant">{label}</span>
      <div className="flex aspect-square w-full items-center justify-center overflow-hidden rounded-lg border border-outline-variant bg-surface-container">
        {src
          ? <img src={src} alt={label} className="h-full w-full object-contain" />
          : <span className="text-sm text-outline">No image</span>}
      </div>
    </div>
  )
}

export function InfoList({ rows }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg bg-surface-container-low p-4 text-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-4">
          <span className="text-on-surface-variant">{label}</span>
          <span className="font-medium">{value}</span>
        </div>
      ))}
    </div>
  )
}

/* kind="percent" for classifier probabilities, kind="weight" for MoE weights */
export function BarList({ title, items, kind }) {
  const max = Math.max(...items.map((i) => i.value))
  return (
    <div className="flex flex-col gap-3 rounded-lg bg-surface-container-low p-4">
      <span className="text-sm font-medium">{title}</span>
      {items.map(({ label, value }) => {
        const top = value === max
        return (
          <div key={label} className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-xs">
              <span className={`flex items-center gap-2 ${top ? 'font-semibold text-primary' : ''}`}>
                {label}
                {kind === 'weight' && top && (
                  <span className="rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-on-primary">
                    Strongest
                  </span>
                )}
              </span>
              <span className={top ? 'font-semibold text-primary' : 'text-on-surface-variant'}>
                {kind === 'percent' ? `${(value * 100).toFixed(0)}%` : value.toFixed(2)}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-surface-container-highest">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  kind === 'weight' || top ? 'bg-primary-container' : 'bg-outline'
                }`}
                style={{
                  width: `${value * 100}%`,
                  opacity: kind === 'weight' ? Math.max(0.25, value) : 1,
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ---------- inputs ---------- */
export function UploadBox({ onFile, onSample }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0])
      }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
        dragging ? 'border-primary bg-surface-container' : 'border-outline-variant bg-surface-container-low hover:bg-surface-container'
      }`}
    >
      <UploadIcon className="h-7 w-7 text-outline" />
      <p className="text-sm font-medium">Drop an image or click to browse</p>
      {onSample && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onSample() }}
          className="text-xs font-medium text-primary hover:underline"
        >
          Use a sample image
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        hidden
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          if (e.target.files[0]) onFile(e.target.files[0])
          e.target.value = ''
        }}
      />
    </div>
  )
}

export function Webcam({ onCapture }) {
  const videoRef = useRef(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let stream
    let cancelled = false
    navigator.mediaDevices
      ?.getUserMedia({ video: true })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop())
        stream = s
        videoRef.current.srcObject = s
      })
      .catch(() => setError('Camera access denied or unavailable.'))
    return () => {
      cancelled = true
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const capture = () => {
    const v = videoRef.current
    if (!v?.videoWidth) return
    const size = Math.min(v.videoWidth, v.videoHeight)
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = size
    canvas.getContext('2d').drawImage(
      v, (v.videoWidth - size) / 2, (v.videoHeight - size) / 2, size, size, 0, 0, size, size,
    )
    canvas.toBlob(
      (blob) => blob && onCapture(new File([blob], 'capture.jpg', { type: 'image/jpeg' })),
      'image/jpeg', 0.95,
    )
  }

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex aspect-square w-full max-w-sm items-center justify-center overflow-hidden rounded-lg border border-outline-variant bg-surface-container">
        {error
          ? <span className="px-4 text-center text-sm text-outline">{error}</span>
          : <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-cover" />}
      </div>
      <button
        type="button"
        onClick={capture}
        disabled={!!error}
        className="rounded-lg border border-outline-variant bg-surface-container-lowest px-4 py-2 text-sm font-medium hover:bg-surface-container disabled:opacity-40"
      >
        Capture
      </button>
    </div>
  )
}