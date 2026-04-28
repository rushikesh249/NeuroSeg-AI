import { useState, useEffect } from 'react'
import { Search, Calendar, Hash, Activity, ArrowRight, Download } from 'lucide-react'
import { toast } from 'react-hot-toast'

interface Props {
  onLoadResult: (jobId: string, stats: any) => void
}

export default function PatientGallery({ onLoadResult }: Props) {
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  useEffect(() => {
    fetch('/results')
      .then(res => res.json())
      .then(data => {
        setResults(data)
        setLoading(false)
      })
      .catch(() => {
        toast.error('Failed to load gallery')
        setLoading(false)
      })
  }, [])

  const filtered = results.filter(r => 
    r.job_id.toLowerCase().includes(search.toLowerCase())
  )

  const formatDate = (ts: number) => {
    return new Date(ts * 1000).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex justify-between items-end px-2">
        <div>
          <h2 className="text-2xl font-black text-slate-900 tracking-tight">Patient Gallery</h2>
          <p className="text-slate-500 text-sm font-medium">Historical diagnostic records and volumetric analysis</p>
        </div>
        <div className="relative">
          <Search className="w-4 h-4 absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
          <input 
            type="text"
            placeholder="Search by Job ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-11 pr-6 py-2.5 rounded-2xl bg-white border border-slate-200 focus:border-accent-blue focus:ring-4 focus:ring-accent-blue/5 outline-none transition-all text-sm font-medium w-64 shadow-soft"
          />
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-32 gap-4">
           <div className="w-8 h-8 rounded-full border-4 border-slate-100 border-t-accent-blue animate-spin" />
           <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">Scanning Repository...</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="glass-card !py-24 flex flex-col items-center justify-center text-center">
           <div className="w-16 h-16 rounded-3xl bg-slate-50 flex items-center justify-center mb-6">
             <Database className="w-8 h-8 text-slate-300" />
           </div>
           <h3 className="text-lg font-bold text-slate-900">No results found</h3>
           <p className="text-slate-500 text-sm max-w-xs mt-2 font-medium">Try a different search term or run a new analysis to populate the gallery.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-6">
          {filtered.map((res) => (
            <div 
              key={res.job_id} 
              className="glass-card group hover:scale-[1.02] hover:shadow-apple-xl transition-all duration-300 cursor-pointer border-slate-200/60"
              onClick={() => onLoadResult(res.job_id, res.stats)}
            >
              <div className="flex items-start justify-between mb-6">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                    <Calendar className="w-3 h-3" /> {formatDate(res.timestamp)}
                  </div>
                  <h3 className="text-sm font-black text-slate-800 tracking-tight flex items-center gap-2">
                    <Hash className="w-3 h-3 text-accent-blue opacity-50" /> {res.job_id.slice(0, 8)}...{res.job_id.slice(-4)}
                  </h3>
                </div>
                <div className="w-10 h-10 rounded-xl bg-slate-50 group-hover:bg-accent-blue/10 flex items-center justify-center transition-colors">
                  <ArrowRight className="w-5 h-5 text-slate-400 group-hover:text-accent-blue transition-colors" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                 <div className="p-3 rounded-2xl bg-slate-50/50 border border-slate-100/50">
                   <p className="text-[9px] text-slate-400 font-bold uppercase mb-1">Whole Tumor</p>
                   <p className="text-base font-black text-slate-900">{res.stats?.WT?.volume_mm3?.toLocaleString(undefined, {maximumFractionDigits: 0})} <span className="text-[9px] text-slate-400">mm³</span></p>
                 </div>
                 <div className="p-3 rounded-2xl bg-slate-50/50 border border-slate-100/50">
                   <p className="text-[9px] text-slate-400 font-bold uppercase mb-1">Location</p>
                   <p className="text-xs font-bold text-slate-700 truncate">{res.stats?.location?.description || 'Central'}</p>
                 </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-slate-100/60">
                <div className="flex gap-1">
                   {res.stats?.ET?.voxels > 0 && <span className="w-1.5 h-1.5 rounded-full bg-accent-magenta" />}
                   {res.stats?.ED?.voxels > 0 && <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan" />}
                   {res.stats?.NET?.voxels > 0 && <span className="w-1.5 h-1.5 rounded-full bg-accent-blue" />}
                </div>
                <button 
                  className="text-[10px] font-black text-accent-blue hover:text-blue-700 transition-colors flex items-center gap-1.5"
                  onClick={(e) => {
                    e.stopPropagation()
                    window.open(`/result/${res.job_id}/segmentation`, '_blank')
                  }}
                >
                  <Download className="w-3 h-3" /> EXPORT NIFTI
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

import { Database } from 'lucide-react'
