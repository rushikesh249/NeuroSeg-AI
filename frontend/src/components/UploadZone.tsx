import { useRef } from 'react'
import { Activity, Crosshair, Binary, Database, Image as ImageIcon, Zap, Target, Box } from 'lucide-react'
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

interface Props {
  files: Record<string, File | null>
  setFiles: React.Dispatch<React.SetStateAction<Record<string, File | null>>>
  fastMode: boolean
  setFastMode: (val: boolean) => void
  patchSize: string
  setPatchSize: (val: string) => void
}

export default function UploadZone({ 
  files, 
  setFiles, 
  fastMode, 
  setFastMode, 
  patchSize, 
  setPatchSize 
}: Props) {
  
  const modalities = [
    { id: 'flair', label: 'FLAIR', desc: 'Fluid-attenuated', icon: Binary, color: 'text-accent-rose' },
    { id: 't1', label: 'T1', desc: 'Native weighted', icon: ImageIcon, color: 'text-accent-violet' },
    { id: 't1ce', label: 'T1CE', desc: 'Contrast-enhanced', icon: Crosshair, color: 'text-accent-magenta' },
    { id: 't2', label: 'T2', desc: 'T2-weighted', icon: Activity, color: 'text-accent-green' },
  ]

  const patchSizes = [
    { label: '64³', value: '64,64,64' },
    { label: '96³', value: '96,96,96' },
    { label: '128³', value: '128,128,128' },
  ]

  const handleFileChange = (mod: string, file: File | null) => {
    if (file) {
      setFiles(prev => ({ ...prev, [mod]: file }))
    }
  }

  const handleDrop = (mod: string, e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files?.[0]) {
      handleFileChange(mod, e.dataTransfer.files[0])
    }
  }

  return (
    <div className="glass-card !rounded-3xl !p-8 space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-black text-slate-900 flex items-center gap-3 tracking-tight">
          <Database className="w-6 h-6 text-accent-blue" /> Modality Ingestion
        </h2>
        <span className="text-[10px] bg-slate-100 text-slate-500 font-black px-2.5 py-1 rounded-full uppercase tracking-widest border border-slate-200">
          {Object.values(files).filter(f => f !== null).length} / 4 Sync
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {modalities.map(mod => {
          const file = files[mod.id]
          const inputRef = useRef<HTMLInputElement>(null)

          return (
            <div 
              key={mod.id}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => handleDrop(mod.id, e)}
              className={cn(
                "relative group flex flex-col items-center justify-center gap-4 p-8 rounded-3xl border-2 border-dashed transition-all cursor-pointer min-h-[180px]",
                file 
                  ? "bg-accent-blue/5 border-accent-blue/30 shadow-sm" 
                  : "bg-slate-50/50 border-slate-200 hover:border-accent-blue/50 hover:bg-white hover:shadow-apple"
              )}
            >
               <input 
                 type="file" 
                 ref={inputRef} 
                 className="hidden" 
                 onChange={(e) => handleFileChange(mod.id, e.target.files?.[0] || null)}
                 accept=".nii,.gz"
               />

               {/* Badge */}
               <span className={cn(
                 "absolute top-4 right-4 text-[8px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full border",
                 file ? "bg-accent-blue text-white border-accent-blue" : "bg-white text-slate-400 border-slate-200"
               )}>
                 {mod.label}
               </span>

               <mod.icon className={cn(
                 "w-10 h-10 transition-all group-hover:scale-110",
                 file ? "text-accent-blue" : "text-slate-300 group-hover:text-accent-blue/40"
               )} />

               <div className="text-center">
                  <p className={cn(
                    "text-sm font-black tracking-tight transition-all truncate max-w-[120px]",
                    file ? "text-slate-900" : "text-slate-400 group-hover:text-slate-600"
                  )}>
                    {file ? (file.name.length > 15 ? file.name.slice(0, 13) + '…' : file.name) : "Select " + mod.label}
                  </p>
                  <p className="text-[9px] text-slate-400 mt-1 font-bold uppercase tracking-wider">{mod.desc}</p>
               </div>
            </div>
          )
        })}
      </div>

      {/* Inference Protocol Settings */}
      <div className="pt-8 border-t border-slate-100 flex flex-col md:flex-row gap-8">
        
        {/* Mode Toggle */}
        <div className="flex-1 space-y-4">
          <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 uppercase tracking-widest">
            <Zap className="w-3 h-3 text-accent-blue" /> Optimization Mode
          </div>
          <div className="flex p-1 bg-slate-100 rounded-2xl w-fit">
            <button 
              onClick={() => setFastMode(true)}
              className={cn(
                "px-6 py-2 rounded-xl text-xs font-black transition-all flex items-center gap-2",
                fastMode ? "bg-white text-accent-blue shadow-sm" : "text-slate-500 hover:text-slate-700"
              )}
            >
              <Zap className={cn("w-3.5 h-3.5", fastMode ? "fill-accent-blue" : "opacity-40")} /> FAST
            </button>
            <button 
              onClick={() => setFastMode(false)}
              className={cn(
                "px-6 py-2 rounded-xl text-xs font-black transition-all flex items-center gap-2",
                !fastMode ? "bg-white text-accent-magenta shadow-sm" : "text-slate-500 hover:text-slate-700"
              )}
            >
              <Target className={cn("w-3.5 h-3.5", !fastMode ? "text-accent-magenta" : "opacity-40")} /> PRECISION
            </button>
          </div>
        </div>

        {/* Patch Size Selector */}
        <div className="flex-1 space-y-4">
          <div className="flex items-center gap-2 text-[10px] font-black text-slate-400 uppercase tracking-widest">
            <Box className="w-3 h-3 text-accent-blue" /> Path Dimension (Px)
          </div>
          <div className="flex gap-3">
            {patchSizes.map(size => (
              <button 
                key={size.value}
                onClick={() => setPatchSize(size.value)}
                className={cn(
                  "px-6 py-2 rounded-2xl text-xs font-black transition-all border",
                  patchSize === size.value 
                    ? "bg-slate-900 text-white border-slate-900 shadow-apple" 
                    : "bg-white text-slate-500 border-slate-200 hover:border-slate-300"
                )}
              >
                {size.label}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
