import { useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTask, setActiveTask] = useState('task1')

  const handleFile = (selectedFile) => {
    if (!selectedFile) return

    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError('')
  }

  const restoreImage = async () => {
    if (!file) {
      setError('Please select an image first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/api/task1/restore`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Restoration failed.')
      }

      const blob = await response.blob()
      setResult(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand">
          <div className="brand-mark">GA</div>
          <div>
            <strong>Generative AI</strong>
            <span>Assignment 01</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot" />
          Model API online
        </div>
      </header>

      <main>
        <section className="hero">
          <p className="eyebrow">GENERATIVE AI • IMAGE RESTORATION</p>
          <h1>Generative AI<br />Image Studio</h1>
          <p className="hero-copy">
            Explore the models developed for Generative AI Assignment 01.
            Upload an image, run a trained model, and inspect the result.
          </p>
        </section>

        <nav className="task-tabs">
          <button
            className={activeTask === 'task1' ? 'active' : ''}
            onClick={() => setActiveTask('task1')}
          >
            <span>01</span>
            Image Restoration
          </button>

          <button
            className={activeTask === 'task2' ? 'active' : ''}
            onClick={() => setActiveTask('task2')}
          >
            <span>02</span>
            Coming Soon
          </button>

          <button
            className={activeTask === 'task3' ? 'active' : ''}
            onClick={() => setActiveTask('task3')}
          >
            <span>03</span>
            Coming Soon
          </button>

          <button
            className={activeTask === 'task4' ? 'active' : ''}
            onClick={() => setActiveTask('task4')}
          >
            <span>04</span>
            Photo → Sketch
          </button>
        </nav>

        {activeTask === 'task1' && (
          <section className="workspace">
            <div className="section-heading">
              <div>
                <p className="eyebrow">TASK 1</p>
                <h2>Universal Image Restoration</h2>
              </div>
              <div className="model-badge">UDAE • ONNX Runtime</div>
            </div>

            <div className="upload-card">
              <label className="upload-zone">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={(e) => handleFile(e.target.files[0])}
                />

                {preview ? (
                  <img
                    src={preview}
                    alt="Selected input"
                    className="upload-preview"
                  />
                ) : (
                  <>
                    <div className="upload-icon">↑</div>
                    <h3>Upload an image</h3>
                    <p>PNG or JPEG • The model processes images at 128 × 128</p>
                  </>
                )}
              </label>

              <div className="controls">
                <div>
                  <span className="control-label">MODEL</span>
                  <strong>Task 1 — Final v2</strong>
                </div>

                <div>
                  <span className="control-label">ARCHITECTURE</span>
                  <strong>Spatial UDAE</strong>
                </div>

                <button
                  className="primary-button"
                  onClick={restoreImage}
                  disabled={!file || loading}
                >
                  {loading ? 'Restoring…' : 'Restore image'}
                </button>
              </div>
            </div>

            {error && <div className="error">{error}</div>}

            {result && (
              <div className="results">
                <div className="result-heading">
                  <div>
                    <p className="eyebrow">RESULT</p>
                    <h2>Restored Image</h2>
                  </div>

                  <a
                    className="download-button"
                    href={result}
                    download="task1_restored.png"
                  >
                    Download PNG
                  </a>
                </div>

                <div className="comparison">
                  <div className="image-card">
                    <span>INPUT</span>
                    <img src={preview} alt="Input" />
                  </div>

                  <div className="arrow">→</div>

                  <div className="image-card">
                    <span>RESTORED</span>
                    <img src={result} alt="Restored" />
                  </div>
                </div>
              </div>
            )}

            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-number">01</div>
                <div>
                  <h3>Universal Denoising Autoencoder</h3>
                  <p>
                    Upload a corrupted or degraded image to run the trained
                    Task 1 restoration model.
                  </p>
                </div>
              </div>
            )}
          </section>
        )}

        {(activeTask === 'task2' || activeTask === 'task3') && (
          <section className="placeholder">
            <span className="placeholder-number">
              {activeTask === 'task2' ? '02' : '03'}
            </span>
            <p className="eyebrow">COMING SOON</p>
            <h2>This task will be integrated here.</h2>
            <p>
              The application structure is ready for the remaining assignment
              tasks.
            </p>
          </section>
        )}

        {activeTask === 'task4' && (
          <section className="placeholder">
            <span className="placeholder-number">04</span>
            <p className="eyebrow">PHOTO → SKETCH</p>
            <h2>Task 4 integration coming next.</h2>
            <p>
              The conditional GAN interface will be connected here.
            </p>
          </section>
        )}
      </main>

      <footer>
        <span>Generative AI — Assignment 01</span>
        <span>Task 1 • Task 4 • Future integrations</span>
      </footer>
    </div>
  )
}

export default App
