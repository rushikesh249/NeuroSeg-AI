import { useState, useEffect, useRef } from 'react'
import { Play, RotateCcw, Binary, Brain, LayoutDashboard, Database, BarChart3, Loader2 } from 'lucide-react'
import Sidebar from './components/Sidebar'
import UploadZone from './components/UploadZone'
import NiiVueViewer from './components/NiiVueViewer'
import type { NiiVueViewerRef } from './components/NiiVueViewer'
import ProgressPanel from './components/ProgressPanel'
import MedicalReport from './components/MedicalReport'
import PatientGallery from './components/PatientGallery'
import PerformancePanel from './components/PerformancePanel'
import { Toaster, toast } from 'react-hot-toast'

import type { TabID } from './types'

export default function App() {
  const [activeTab, setActiveTab] = useState<TabID>('dashboard')
  const [files, setFiles] = useState<Record<string, File | null>>({
    flair: null,
    t1: null,
    t1ce: null,
    t2: null
  })
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<string>('idle')
  const [viewMode, setViewMode] = useState<'axial' | 'sagittal' | 'coronal' | '3d'>('axial')
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [stats, setStats] = useState<any>(null)
  const [isReportOpen, setIsReportOpen] = useState(false)
  const [snapshot, setSnapshot] = useState<string | null>(null)

  // Inference Settings
  const [fastMode, setFastMode] = useState(true)
  const [patchSize, setPatchSize] = useState('128,128,128')

  const viewerRef = useRef<NiiVueViewerRef>(null)

  const isReady = Object.values(files).every(f => f !== null)

  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  // Polling for model-info until loaded
  useEffect(() => {
    let timer: number
    const checkModel = async () => {
      try {
        const res = await fetch('/model-info')
        const data = await res.json()
        if (res.ok && !data.error) {
          setModelInfo(data)
          // Stop polling once loaded
        } else {
          // Keep polling if 503 or has error property
          timer = window.setTimeout(checkModel, 2000)
        }
      } catch (e) {
        // Network error - server might be starting
        timer = window.setTimeout(checkModel, 2000)
      }
    }
    checkModel()
    return () => clearTimeout(timer)
  }, [])

  const runInference = () => {
    if (!isReady || isUploading) return
    
    setIsUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    Object.entries(files).forEach(([mod, file]) => {
      if (file) formData.append(mod, file)
    })
    
    formData.append('fast', String(fastMode))
    formData.append('patch_size', patchSize)
    
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/predict')

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const pct = Math.round((event.loaded / event.total) * 100)
        setUploadProgress(pct)
      }
    }

    xhr.onload = () => {
      setIsUploading(false)
      if (xhr.status >= 200 && xhr.status < 300) {
        const data = JSON.parse(xhr.responseText)
        setJobId(data.job_id)
        toast.success('Upload complete! Inference starting...')
        setActiveTab('dashboard')
      } else {
        const errorData = JSON.parse(xhr.responseText || '{}')
        toast.error(errorData.detail || 'Upload failed')
      }
    }

    xhr.onerror = () => {
      setIsUploading(false)
      toast.error('Network error during upload')
    }

    xhr.send(formData)
  }

  const resetAll = () => {
    setFiles({ flair: null, t1: null, t1ce: null, t2: null })
    setJobId(null)
    setStats(null)
    setSnapshot(null)
    toast('Dashboard reset', { icon: '🔄' })
  }

  const handleGenerateReport = async () => {
    if (viewerRef.current) {
      const img = await viewerRef.current.getAxialSnapshot()
      setSnapshot(img)
      setIsReportOpen(true)
    }
  }

  const loadPastResult = (pastJobId: string, pastStats: any) => {
    setJobId(pastJobId)
    setJobStatus('done')
    setStats(pastStats)
    setActiveTab('dashboard')
    toast.success('Loaded historical scan')
  }

  return (
    <div className="flex min-h-screen bg-background-deep selection:bg-accent-blue/20 selection:text-black">
      <Toaster position="bottom-right" toastOptions={{
        className: 'glass !bg-white/80 border !border-white/50 !rounded-2xl !text-slate-800 !shadow-apple',
        style: { backdropFilter: 'blur(16px)' }
      }} />
      
      {/* ═══════════════════ SIDEBAR ═══════════════════ */}
      <div className="no-print">
        <Sidebar modelInfo={modelInfo} activeTab={activeTab} setActiveTab={setActiveTab} />
      </div>

      {/* ═══════════════════ MAIN ═══════════════════════ */}
      <main className="flex-1 ml-72 flex flex-col min-h-screen p-8 gap-8 overflow-y-auto no-print">
        
        {/* Header */}
        <header className="flex justify-between items-center mb-4 px-2">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 flex items-center gap-3">
              NeuroSeg AI <span className="text-[10px] bg-accent-blue/5 px-2 py-1 rounded-full border border-accent-blue/10 font-bold text-accent-blue tracking-widest uppercase">v1.2.0</span>
            </h1>
            <p className="text-slate-500 mt-2 text-sm font-medium">Precision 3D Brain Segmentation & Volumetric Analysis</p>
          </div>
          
          {activeTab === 'dashboard' && (
            <div className="flex gap-3">
              <button 
                onClick={resetAll}
                className="px-5 py-2.5 rounded-2xl bg-white hover:bg-slate-50 border border-slate-200 shadow-soft transition-all text-sm font-bold text-slate-600 flex items-center gap-2 active:scale-95"
              >
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
              <button 
                onClick={runInference}
                disabled={!isReady || isUploading || (jobId !== null && jobStatus !== 'done')}
                className="px-8 py-2.5 rounded-2xl bg-accent-blue hover:bg-blue-700 transition-all active:scale-95 disabled:opacity-30 disabled:grayscale shadow-apple text-white font-bold text-sm flex items-center gap-2"
              >
                {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                {isUploading ? 'Uploading...' : 'Run Analysis'}
              </button>
            </div>
          )}
        </header>

        {/* Dynamic Content */}
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="xl:col-span-2 flex flex-col gap-8">
              <UploadZone 
                files={files} 
                setFiles={setFiles} 
                fastMode={fastMode} 
                setFastMode={setFastMode} 
                patchSize={patchSize} 
                setPatchSize={setPatchSize} 
              />
              <NiiVueViewer 
                ref={viewerRef}
                jobId={jobId} 
                jobStatus={jobStatus}
                stats={stats}
                files={files} 
                viewMode={viewMode} 
                setViewMode={setViewMode} 
              />
            </div>
            <div className="flex flex-col gap-8">
              <ProgressPanel 
                jobId={jobId} 
                isUploading={isUploading}
                uploadProgress={uploadProgress}
                onStatusChange={setJobStatus} 
                onStatsChange={setStats} 
                onGenerateReport={handleGenerateReport}
                onSyncFocus={() => viewerRef.current?.jumpToTumor()}
              />
            </div>
          </div>
        )}

        {activeTab === 'gallery' && (
          <PatientGallery onLoadResult={loadPastResult} />
        )}

        {activeTab === 'stats' && (
          <PerformancePanel modelInfo={modelInfo} />
        )}

        {/* Footer */}
        <footer className="mt-auto pt-16 text-slate-400 text-[10px] flex justify-between items-center opacity-60">
          <div className="flex gap-8">
             <span className="flex items-center gap-2 font-bold uppercase tracking-tighter"><Binary className="w-4 h-4 text-accent-blue"/> BraTS 2021 Pipeline</span>
             <span className="flex items-center gap-2 font-bold uppercase tracking-tighter"><Brain className="w-4 h-4 text-accent-magenta"/> 3D Residual U-Net</span>
          </div>
          <span className="font-medium">Medical Intelligence Dashboard &middot; 2024</span>
        </footer>
      </main>

      <MedicalReport 
        isOpen={isReportOpen} 
        onClose={() => setIsReportOpen(false)}
        stats={stats}
        jobId={jobId || ''}
        snapshot={snapshot}
      />
    </div>
  )
}
