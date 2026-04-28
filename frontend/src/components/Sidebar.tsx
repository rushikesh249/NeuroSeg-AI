import { Database, Zap, Cpu, BarChart3, Layers, Download } from 'lucide-react'
import type { TabID } from '../types'

interface Props {
  modelInfo: any
  activeTab: TabID
  setActiveTab: (tab: TabID) => void
}

export default function Sidebar({ modelInfo, activeTab, setActiveTab }: Props) {
  const formatParams = (n: number) => {
    if (!n) return '—'
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
    return (n / 1000).toFixed(0) + 'K'
  }

  const sections = [
    { label: 'Navigation', items: [
      { id: 'dashboard', icon: Layers, label: 'Dashboard' },
      { id: 'gallery', icon: Database, label: 'Patient Gallery' },
      { id: 'stats', icon: BarChart3, label: 'Performance' },
    ]}
  ]

  return (
    <aside className="w-72 h-screen fixed left-0 top-0 bg-white border-r border-slate-200/60 shadow-soft flex flex-col p-8 z-50">
      
      {/* Brand */}
      <div className="flex items-center gap-4 mb-16">
        <div className="w-12 h-12 rounded-2xl bg-accent-blue shadow-apple flex items-center justify-center text-2xl">
          🧠
        </div>
        <div>
          <h2 className="text-sm font-black tracking-tight text-slate-900">NeuroSeg Lab</h2>
          <p className="text-[10px] text-accent-blue font-bold uppercase tracking-widest">Medical A.I.</p>
        </div>
      </div>

      {/* Nav */}
      {sections.map(section => (
        <div key={section.label} className="mb-10">
          <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-5 px-3">{section.label}</h3>
          <nav className="space-y-2">
            {section.items.map(item => (
              <button 
                key={item.id}
                onClick={() => setActiveTab(item.id as TabID)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-bold transition-all group ${
                  activeTab === item.id ? 'bg-accent-blue/5 text-accent-blue border border-accent-blue/10 shadow-sm' : 'text-slate-500 hover:text-slate-900 hover:bg-slate-50 border border-transparent'
                }`}
              >
                <item.icon className={`w-4 h-4 ${activeTab === item.id ? 'text-accent-blue' : 'text-slate-400 group-hover:text-slate-600'}`} />
                {item.label}
              </button>
            ))}
            
            {/* Sample Data Button */}
            <div className="pt-4 mt-4 border-t border-slate-100">
              <button
                onClick={() => window.open('/download-sample', '_blank')}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all text-accent-blue hover:bg-accent-blue/5 group"
              >
                <Download className="w-4 h-4 group-hover:scale-110 transition-transform" />
                <span className="text-sm font-bold">Sample Data</span>
              </button>
            </div>
          </nav>
        </div>
      ))}

      {/* Model Stats */}
      <div className="mt-auto space-y-6">
        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest px-3">System Integrity</h3>
        
        <div className="glass-card !p-5 !rounded-3xl !shadow-apple border-white/50 bg-slate-50/50">
          <div className="flex items-center justify-between mb-4">
             <div className="text-[10px] text-slate-500 font-bold flex items-center gap-2"><Zap className="w-3 h-3 text-accent-blue"/> ENGINE</div>
             {modelInfo ? (
               modelInfo.error ? (
                  <span className="flex items-center gap-1.5 text-[10px] text-amber-500 font-black tracking-tighter animate-pulse"><span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> LOADING</span>
               ) : (
                  <span className="flex items-center gap-1.5 text-[10px] text-accent-green font-black tracking-tighter"><span className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse" /> ONLINE</span>
               )
             ) : (
               <span className="flex items-center gap-1.5 text-[10px] text-accent-rose font-black tracking-tighter"><span className="w-1.5 h-1.5 rounded-full bg-accent-rose" /> OFFLINE</span>
             )}
          </div>
          <div className="space-y-5">
            <div>
              <p className="text-[9px] text-slate-400 font-bold mb-1 uppercase tracking-wider">Accuracy Metric</p>
              <p className="text-2xl font-black tracking-tight text-slate-900">82%</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
               <div>
                 <p className="text-[9px] text-slate-400 font-bold mb-1 uppercase tracking-wider">Parameters</p>
                 <p className="text-xs font-bold text-slate-700">{formatParams(modelInfo?.num_params)}</p>
               </div>
               <div>
                 <p className="text-[9px] text-slate-400 font-bold mb-1 uppercase tracking-wider">Epoch</p>
                 <p className="text-xs font-bold text-slate-700">250</p>
               </div>
            </div>
          </div>
        </div>

        <div className="text-[9px] text-slate-400 px-3 flex items-center gap-2 font-medium">
           <Cpu className="w-3 h-3"/> Processed via 
           <span className="text-slate-900 font-black tracking-tighter">LOCAL_CPU_THREAD</span>
        </div>
      </div>

    </aside>
  )
}
