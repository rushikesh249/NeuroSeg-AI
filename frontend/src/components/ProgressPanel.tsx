import { useState, useEffect } from 'react'
import { Activity, Download, CheckCircle2, Loader2, Gauge, Target, BarChart3, Layers, FileText } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'

interface Props {
  jobId: string | null
  isUploading: boolean
  uploadProgress: number
  onStatusChange?: (status: string) => void
  onStatsChange?: (stats: any) => void
  onGenerateReport?: () => void
  onSyncFocus?: () => void
}

interface JobState {
  status: string
  progress: number
  error: string | null
  stats: any
}

export default function ProgressPanel({ jobId, isUploading, uploadProgress, onStatusChange, onStatsChange, onGenerateReport, onSyncFocus }: Props) {
  const [job, setJob] = useState<JobState>({
    status: 'idle',
    progress: 0,
    error: null,
    stats: null
  })

  useEffect(() => {
    if (!jobId) {
       setJob({ status: 'idle', progress: 0, error: null, stats: null })
       return
    }

    const sse = new EventSource(`/progress/${jobId}`)
    
    sse.onmessage = (e) => {
      const data = JSON.parse(e.data)
      setJob(prev => ({
        ...prev,
        status: data.status,
        progress: data.progress,
        error: data.error,
        stats: data.stats || prev.stats
      }))

      if (onStatusChange) onStatusChange(data.status)
      if (onStatsChange && data.stats) onStatsChange(data.stats)

      if (['done', 'error'].includes(data.status)) {
        sse.close()
      }
    }

    sse.onerror = () => sse.close()

    return () => sse.close()
  }, [jobId])

  const stages = [
    { id: 'upload',  label: 'Ingestion',     icon: Target,   done: job.progress > 0 || job.status === 'done' },
    { id: 'load',    label: 'Pre-process',   icon: Gauge,    done: job.progress > 5 || job.status === 'done' },
    { id: 'infer',   label: 'Inference',     icon: Activity, active: job.status === 'running' },
    { id: 'save',    label: 'Post-process',  icon: CheckCircle2, done: job.status === 'done' },
  ]

  return (
    <div className="glass-card !rounded-3xl flex flex-col h-full gap-8 sticky top-8 !p-8 shadow-apple">
      <div className="flex items-center justify-between">
         <h2 className="text-xl font-black text-slate-900 flex items-center gap-3 tracking-tight">
            <Activity className="w-6 h-6 text-accent-blue" /> System Status
         </h2>
         {job.status === 'running' && (
           <span className="flex items-center gap-2 text-[10px] text-accent-blue font-black uppercase tracking-widest bg-accent-blue/5 border border-accent-blue/10 px-3 py-1.5 rounded-full">
             <Loader2 className="w-3 h-3 animate-spin"/> Processing
           </span>
         )}
      </div>

      {/* Progress Bar Container */}
      <div className="space-y-4">
         <div className="flex justify-between items-end mb-1 px-1">
            <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
              {isUploading ? 'Uploading Data Packet' : 'Global Analysis Progress'}
            </span>
            <span className="text-3xl font-black text-slate-900">{isUploading ? uploadProgress : job.progress}%</span>
         </div>
         
         <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden p-0.5 shadow-inner">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${isUploading ? uploadProgress : job.progress}%` }}
              className={`h-full rounded-full relative shadow-soft transition-colors duration-500 ${isUploading ? 'bg-amber-500' : 'bg-accent-blue'}`}
            >
              <div className="absolute inset-0 bg-white/20 animate-pulse" />
            </motion.div>
         </div>
         {isUploading && (
           <p className="text-[9px] text-amber-600 font-bold uppercase tracking-tight animate-pulse text-center">
             Streaming high-resolution imaging data to secure cloud...
           </p>
         )}
      </div>

      {/* Analysis Legend */}
      <div className="p-5 rounded-[2rem] bg-slate-50 border border-slate-100 space-y-4 shadow-sm">
        <h3 className="text-[9px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2 border-b border-slate-200/60 pb-3">
          <Layers className="w-3 h-3"/> Analysis Legend
        </h3>
        <div className="space-y-3">
          <LegendItem color="bg-accent-green" label="GD-Enhancing Tumor (ET)" desc="High-grade tumor region" />
          <LegendItem color="bg-accent-magenta" label="Peritumoral Edema (ED)" desc="Swelling & fluid accumulation" />
          <LegendItem color="bg-accent-cyan" label="Necrotic Core (NCR/NET)" desc="Non-enhancing necrotic tissue" />
        </div>
      </div>

      {/* Stage Pipeline */}
      <div className="grid grid-cols-4 gap-2 py-4">
         {stages.map((stage, idx) => (
           <div key={stage.id} className="flex flex-col items-center gap-3 group">
              <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all border-2 ${
                stage.done ? 'bg-accent-blue text-white border-accent-blue shadow-apple' : 
                (stage.active ? 'bg-white border-accent-blue text-accent-blue animate-pulse shadow-soft' : 'bg-slate-50 border-slate-100 text-slate-300')
              }`}>
                <stage.icon className="w-5 h-5"/>
              </div>
              <span className={`text-[8px] font-black tracking-widest uppercase text-center transition-all ${
                stage.done || stage.active ? 'text-slate-900' : 'text-slate-400'
              }`}>{stage.label}</span>
           </div>
         ))}
      </div>

      {/* Stats Summary */}
      <AnimatePresence>
        {job.stats && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-auto space-y-5"
          >
             <div className="flex items-center gap-3 px-1">
                <BarChart3 className="w-5 h-5 text-slate-400"/>
                <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Volumetric Summary</h3>
             </div>
             
             <div className="grid grid-cols-1 gap-3">
                <div className="flex gap-3">
                   <StatItem label="Enhancing (ET)" value={job.stats.ET?.volume_mm3} color="border-accent-green" />
                   <StatItem label="ET/WT Ratio" value={job.stats.ET?.ratio_to_wt?.toFixed(1) + '%'} color="border-accent-rose" isCoord />
                </div>
                <div className="flex gap-3">
                   <StatItem label="Necrotic (NET)" value={job.stats.NET?.volume_mm3} color="border-accent-cyan" />
                   <StatItem label="Oedema (ED)" value={job.stats.ED?.volume_mm3} color="border-accent-magenta" />
                </div>
                <div className="flex gap-3">
                   <StatItem label="Spatial Scale" value={job.stats.location?.bbox_mm || 'N/A'} color="border-slate-300" isCoord />
                </div>
             </div>

             <div className="flex flex-col gap-3 pt-4">
                 <button 
                   onClick={onGenerateReport}
                   className="w-full flex items-center justify-center gap-3 py-3 rounded-2xl bg-accent-blue hover:bg-blue-700 shadow-apple text-[10px] font-black uppercase tracking-widest text-white transition-all group active:scale-95"
                 >
                    <FileText className="w-4 h-4 text-white"/> Generate Clinical Report
                 </button>

                 <button 
                   onClick={onSyncFocus}
                   className="w-full flex items-center justify-center gap-3 py-3 rounded-2xl bg-slate-900 hover:bg-black shadow-apple text-[10px] font-black uppercase tracking-widest text-white transition-all group active:scale-95"
                 >
                    <Target className="w-4 h-4 group-hover:scale-125 transition-all text-accent-cyan"/> Synchronize Focus
                 </button>
                 
                 <a 
                   href={`/result/${jobId}/segmentation`}
                   download={`NeuroSeg_${jobId}.nii.gz`}
                   className="w-full flex items-center justify-center gap-3 py-3 rounded-2xl bg-white hover:bg-slate-50 border border-slate-200 text-[10px] font-black uppercase tracking-widest text-slate-600 shadow-soft transition-all group active:scale-95"
                 >
                    <Download className="w-4 h-4 text-accent-blue group-hover:scale-110 transition-all"/> Export Data Packet (.nii.gz)
                 </a>
              </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function LegendItem({ color, label, desc }: { color: string, label: string, desc: string }) {
  return (
    <div className="flex items-start gap-3">
      <div className={`w-2.5 h-2.5 rounded-full mt-1 shrink-0 ${color} shadow-sm`} />
      <div className="flex flex-col">
        <span className="text-[10px] font-black text-slate-800 tracking-tight leading-none">{label}</span>
        <span className="text-[8px] text-slate-400 font-bold uppercase tracking-tighter mt-1">{desc}</span>
      </div>
    </div>
  )
}

function StatItem({ label, value, color, isCoord }: { label: string, value: any, color: string, isCoord?: boolean }) {
  return (
    <div className={`flex-1 p-4 rounded-3xl bg-white border border-slate-100 border-l-4 transition-all hover:shadow-soft ${color}`}>
       <p className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-1">{label}</p>
       <p className={`font-black text-slate-900 tracking-tight ${isCoord ? 'text-[10px]' : 'text-lg'}`}>
         {isCoord ? value : Number(value || 0).toLocaleString()} <span className="text-[8px] text-slate-400 font-bold uppercase tracking-widest ml-1">{isCoord ? "" : "mm³"}</span>
       </p>
    </div>
  )
}
