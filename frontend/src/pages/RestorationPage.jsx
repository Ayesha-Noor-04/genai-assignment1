import { useState } from 'react'
import { downloadDataUrl } from '../api'
import { useAutoRun } from '../useAutoRun'
import {
  UploadBox, Select, Segmented, ErrorBanner, Loading, ImagePanel, DownloadButton, InfoList, BarList,
} from '../components/ui'

const VARIANTS = {
  universal: { title: 'Universal Restoration', endpoint: '/api/task1/restore' },
  hard: { title: 'Hard-Routed Restoration', endpoint: '/api/task2/restore' },
  soft: { title: 'Soft Mixture-of-Experts Restoration', endpoint: '/api/task3/restore' },
}

const CORRUPTIONS = [
  { value: 'none', label: 'None' },
  { value: 'salt', label: 'Salt-and-pepper' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Rectangular occlusion' },
]
const SEVERITIES = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
]
const HARD_LABELS = ['Clean', 'Salt-and-pepper', 'Gaussian blur', 'Occlusion']
const SOFT_LABELS = ['Clean identity', 'Salt-and-pepper expert', 'Blur expert', 'Occlusion expert']

export default function RestorationPage({ variant }) {
  const { title, endpoint } = VARIANTS[variant]
  const [source, setSource] = useState(null)
  const [corruption, setCorruption] = useState('salt')
  const [severity, setSeverity] = useState('medium')
  const [mode, setMode] = useState('predicted')

  // Oracle routing needs a known corruption label.
  const effectiveMode = corruption === 'none' ? 'predicted' : mode

  const { result, loading, error } = useAutoRun(endpoint, source, {
    corruption,
    severity,
    mode: variant === 'hard' ? effectiveMode : undefined,
  })

  const rows = []
  if (result) {
    if (variant === 'universal') rows.push(['Corruption settings', result.settings])
    if (variant === 'hard') {
      rows.push(['Predicted corruption', result.predicted])
      rows.push(['Selected expert', result.expert])
    }
    rows.push(['Inference time', `${result.inference_ms} ms`])
  }

  return (
    <div className="mx-auto w-full max-w-4xl">
      <h1 className="mb-4 text-2xl font-semibold">{title}</h1>
      <ErrorBanner message={error} />

      <div className="flex flex-col gap-6 rounded-lg border border-outline-variant bg-surface-container-lowest p-6">
        <UploadBox
          onFile={(file) => setSource({ file })}
          onSample={() => setSource({ sample_id: Math.floor(Math.random() * 1000) })}
        />

        <div className={`grid grid-cols-1 gap-4 ${variant === 'hard' ? 'md:grid-cols-3' : 'md:grid-cols-2'}`}>
          <Select label="Corruption" value={corruption} onChange={setCorruption} options={CORRUPTIONS} />
          <Select label="Severity" value={severity} onChange={setSeverity} options={SEVERITIES} />
          {variant === 'hard' && (
            <Segmented
              label="Routing mode"
              value={effectiveMode}
              onChange={setMode}
              options={[
                { value: 'predicted', label: 'Predicted' },
                { value: 'oracle', label: 'Oracle', disabled: corruption === 'none' },
              ]}
            />
          )}
        </div>

        {loading && <Loading />}

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <ImagePanel label="Input" src={result?.input_image} />
          <ImagePanel label="Restored" src={result?.restored_image} />
        </div>

        {result && variant === 'hard' && (
          <BarList
            title="Classifier probabilities"
            kind="percent"
            items={HARD_LABELS.map((label, i) => ({ label, value: result.probabilities[i] }))}
          />
        )}
        {result && variant === 'soft' && (
          <BarList
            title="Routing weights"
            kind="weight"
            items={SOFT_LABELS.map((label, i) => ({ label, value: result.weights[i] }))}
          />
        )}
        {result && <InfoList rows={rows} />}

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <DownloadButton
            disabled={!result}
            onClick={() => downloadDataUrl(result.restored_image, `${variant}_restored.png`)}
          />
        </div>
      </div>
    </div>
  )
}