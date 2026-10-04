import { MLFLOW_URL } from '../api'

const NAV = [
  { id: 'universal', label: 'Universal Restoration' },
  { id: 'hard', label: 'Hard-Routed Restoration' },
  { id: 'soft', label: 'Soft Mixture-of-Experts Restoration' },
  { id: 'sketch', label: 'Face-to-Sketch Generator' },
]

export default function Sidebar({ active, onSelect, online, open }) {
  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex w-[200px] flex-col justify-between border-r border-outline-variant bg-surface-container-lowest transition-transform duration-200 md:translate-x-0 ${
        open ? 'translate-x-0' : '-translate-x-full'
      }`}
    >
      <div>
        <div className="flex h-14 items-center border-b border-outline-variant px-4">
          <span className="text-base font-semibold">GenAI Studio</span>
        </div>
        <nav className="flex flex-col gap-1 p-2">
          {NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              className={`rounded-lg px-2 py-2 text-left text-sm font-medium transition-colors ${
                active === item.id
                  ? 'bg-secondary-container text-primary'
                  : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="flex flex-col gap-1 border-t border-outline-variant p-4">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${online ? 'bg-emerald-500' : 'bg-red-500'}`} />
          <span className="text-xs font-medium">{online ? 'Backend online' : 'Backend offline'}</span>
        </div>
        <span className="text-xs text-on-surface-variant">ONNX Runtime - CPU</span>
        <a href={MLFLOW_URL} target="_blank" rel="noreferrer" className="self-start text-xs font-medium text-primary hover:underline">
          Open MLflow
        </a>
      </div>
    </aside>
  )
}