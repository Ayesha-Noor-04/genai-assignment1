import { useState } from 'react'
import { downloadDataUrl } from '../api'
import { useAutoRun } from '../useAutoRun'
import { UploadBox, Webcam, Segmented, ErrorBanner, Loading, ImagePanel, DownloadButton } from '../components/ui'

const TABS = [
  { value: 'upload', label: 'Upload' },
  { value: 'webcam', label: 'Webcam' },
]
const STYLES = [
  { value: 0, label: 'Style 1' },
  { value: 1, label: 'Style 2' },
  { value: 2, label: 'Style 3' },
]

export default function SketchPage() {
  const [tab, setTab] = useState('upload')
  const [source, setSource] = useState(null)
  const [photoUrl, setPhotoUrl] = useState(null)
  const [style, setStyle] = useState(0)

  const { result, loading, error } = useAutoRun('/api/task4/generate', source, { style })

  const setPhoto = (file) => {
    if (photoUrl) URL.revokeObjectURL(photoUrl)
    setPhotoUrl(URL.createObjectURL(file))
    setSource({ file })
  }

  return (
    <div className="mx-auto w-full max-w-5xl">
      <h1 className="mb-4 text-2xl font-semibold">Face-to-Sketch Generator</h1>
      <ErrorBanner message={error} />

      <div className="flex flex-col gap-6 rounded-lg border border-outline-variant bg-surface-container-lowest p-6">
        <div className="flex flex-col gap-4">
          <div className="max-w-xs">
            <Segmented value={tab} onChange={setTab} options={TABS} />
          </div>
          {tab === 'upload' ? <UploadBox onFile={setPhoto} /> : <Webcam onCapture={setPhoto} />}
        </div>

        <div className="max-w-md">
          <Segmented label="Sketch style" value={style} onChange={(v) => setStyle(Number(v))} options={STYLES} />
        </div>

        {loading && <Loading />}

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <ImagePanel label="Photograph" src={photoUrl} />
          <ImagePanel label="Sketch" src={result?.sketch_image} />
        </div>

        <div className="flex justify-end border-t border-outline-variant pt-4">
          <DownloadButton
            disabled={!result}
            onClick={() => downloadDataUrl(result.sketch_image, `sketch_style${style + 1}.png`)}
          />
        </div>
      </div>
    </div>
  )
}