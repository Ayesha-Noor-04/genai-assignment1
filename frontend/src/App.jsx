import { useEffect, useState } from 'react'
import { getHealth } from './api'
import Sidebar from './components/Sidebar'
import { MenuIcon } from './components/ui'
import RestorationPage from './pages/RestorationPage'
import SketchPage from './pages/SketchPage'

export default function App() {
  const [page, setPage] = useState('universal')
  const [online, setOnline] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    let alive = true
    const check = () => getHealth().then((ok) => alive && setOnline(ok))
    check()
    const id = setInterval(check, 10000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const select = (id) => { setPage(id); setMenuOpen(false) }

  return (
    <>
      <Sidebar active={page} onSelect={select} online={online} open={menuOpen} />
      {menuOpen && <div className="fixed inset-0 z-30 bg-black/20 md:hidden" onClick={() => setMenuOpen(false)} />}

      <div className="bg-primary text-on-primary p-4">TAILWIND TEST</div>

      
      <div className="md:pl-[200px]">
        <header className="flex h-14 items-center border-b border-outline-variant bg-surface-container-lowest px-4 md:hidden">
          <button type="button" onClick={() => setMenuOpen(true)} aria-label="Open menu">
            <MenuIcon className="h-6 w-6" />
          </button>
          <span className="ml-3 text-base font-semibold">GenAI Studio</span>
        </header>

        <main className="p-4 md:p-6">
          {page === 'sketch' ? <SketchPage /> : <RestorationPage key={page} variant={page} />}
        </main>
      </div>
    </>
  )
}