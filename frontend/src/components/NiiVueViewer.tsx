import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react'
import { Niivue } from '@niivue/niivue'
import { Maximize2, Monitor, LayoutGrid, Box, Layers, ToggleLeft, ToggleRight, Binary } from 'lucide-react'
import { toast } from 'react-hot-toast'

interface Props {
  jobId: string | null
  jobStatus: string
  stats: any
  files: Record<string, File | null>
  viewMode: 'axial' | 'sagittal' | 'coronal' | '3d'
  setViewMode: React.Dispatch<React.SetStateAction<'axial' | 'sagittal' | 'coronal' | '3d'>>
}

export interface NiiVueViewerRef {
  getSnapshot: () => string | null
  getAxialSnapshot: () => Promise<string | null>
}

const NiiVueViewer = forwardRef<NiiVueViewerRef, Props>(({ jobId, jobStatus, stats, files, viewMode, setViewMode }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nvRef = useRef<Niivue | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const [overlayOpacity, setOverlayOpacity] = useState(0.85)

  useEffect(() => {
    if (!canvasRef.current) return

    const nv = new Niivue({
      show3Dcrosshair: true,
      backColor: [0.05, 0.08, 0.13, 1], // background-deep
      crosshairColor: [0.22, 0.75, 0.98, 0.9], // accent-cyan
    })

    nv.attachToCanvas(canvasRef.current)

    // Add Neon LUT
    const neonLUT = {
      R: [0, 0, 180, 0],
      G: [0, 255, 0, 255],
      B: [0, 255, 255, 0],
      A: [0, 255, 255, 255],
      In: [0, 1, 2, 4]
    };
    // Save original colormap function to avoid infinite recursion
    const originalColormap = nv.colormapFromKey.bind(nv);
    (nv as any).colormapFromKey = (key: string) => {
      if (key === 'neon') return neonLUT;
      return originalColormap(key);
    };

    nvRef.current = nv

    return () => {
      // Cleanup handled by niivue usually or keep instance
    }
  }, [])

  useImperativeHandle(ref, () => ({
    getSnapshot: () => {
      if (!canvasRef.current) return null
      return canvasRef.current.toDataURL('image/png')
    },
    getAxialSnapshot: async () => {
      if (!nvRef.current || !canvasRef.current) return null
      
      const prevSlice = nvRef.current.sliceType;
      
      // 1. Switch to Axial
      nvRef.current.setSliceType(nvRef.current.sliceTypeAxial);
      
      // 2. Jump to COM if available
      if (stats?.location?.center_of_mass && nvRef.current.volumes.length > 1) {
        const com = stats.location.center_of_mass;
        const vol = nvRef.current.volumes[1];
        const d = (vol as any)?.dims || (vol as any)?.header?.dims;
        if (d && d[1] > 0) {
           nvRef.current.setCrosshairPos(com[0]/d[1], com[1]/d[2], com[2]/d[3]);
        }
      }
      
      // Force render
      nvRef.current.drawScene();
      
      // Capture
      const img = canvasRef.current.toDataURL('image/png');
      
      // Restore
      nvRef.current.setSliceType(prevSlice);
      
      return img;
    },
    jumpToTumor: () => {
      if (nvRef.current && stats?.location?.center_of_mass && nvRef.current.volumes.length > 1) {
        const com = stats.location.center_of_mass;
        const vol = nvRef.current.volumes[1];
        const d = (vol as any)?.dims || (vol as any)?.header?.dims;
        if (d && d[1] > 0) {
           nvRef.current.setCrosshairPos(com[0]/d[1], com[1]/d[2], com[2]/d[3]);
           toast.success('Centered on Tumor');
        }
      }
    }
  }))

  useEffect(() => {
    let active = true;
    const loadVolumes = async () => {
      if (!nvRef.current || !active) return

      // If no flair, we are "loaded" but in empty state
      if (!files.flair) {
        setIsLoaded(true)
        nvRef.current.volumes = []
        return
      }

      setIsLoaded(false)

      try {
        // Build volume list configuration for nv.loadVolumes
        const volumeList = [];

        // 1. Base Image
        volumeList.push({
          url: URL.createObjectURL(files.flair),
          name: 'flair.nii.gz',
          colormap: 'gray'
        });

        // 2. Segmentation only if status IS 'done'
        if (jobId && jobStatus === 'done') {
          volumeList.push({
            url: `/result/${jobId}/segmentation`,
            name: 'mask.nii.gz',
            colormap: 'jet',
            opacity: overlayOpacity,
            cal_min: 1,
            cal_max: 4,
          });
        }

        if (active) {
          await nvRef.current.loadVolumes(volumeList);

          const mode = viewMode;
          if (mode === '3d') {
            nvRef.current.setSliceType(nvRef.current.sliceTypeRender);
          } else {
            const types: Record<string, number> = {
              axial: nvRef.current.sliceTypeAxial,
              sagittal: nvRef.current.sliceTypeSagittal,
              coronal: nvRef.current.sliceTypeCoronal
            };
            nvRef.current.setSliceType(types[mode] ?? nvRef.current.sliceTypeMultiplanar);
          }

          // --- SURGICAL AUTO-FOCUS ---
          if (jobStatus === 'done' && stats?.location?.center_of_mass && nvRef.current.volumes.length > 1) {
            const com = stats.location.center_of_mass; 
            const vol = nvRef.current.volumes[1];
            const d = (vol as any)?.dims || (vol as any)?.header?.dims;
            
            if (d && d[1] > 0 && d[2] > 0 && d[3] > 0) {
                // Convert index to frac [0..1]
                (nvRef.current as any).setCrosshairPos(com[0]/d[1], com[1]/d[2], com[2]/d[3]);
            }
          }

          setIsLoaded(true)
          if (jobId && jobStatus === 'done') toast.success('Neuro-Map Synced')
        }
      } catch (e: any) {
        if (active) {
          console.error('NiiVue Load Error:', e)
          toast.error(`Viewer: ${e.message || 'Parsing error'}`)
          setIsLoaded(true) // Release spinner even on error
        }
      }
    };

    loadVolumes();

    // Add resize listener
    const handleResize = () => (nvRef.current as any)?.resizeCanvas?.();
    window.addEventListener('resize', handleResize);

    return () => {
      active = false;
      window.removeEventListener('resize', handleResize);
    };
  }, [files.flair, jobId, jobStatus, viewMode])

  const updateLayout = (mode: string) => {
    if (!nvRef.current) return
    setViewMode(mode as any)
    switch (mode) {
      case 'axial': nvRef.current.setSliceType(nvRef.current.sliceTypeAxial); break;
      case 'sagittal': nvRef.current.setSliceType(nvRef.current.sliceTypeSagittal); break;
      case 'coronal': nvRef.current.setSliceType(nvRef.current.sliceTypeCoronal); break;
      case '3d': nvRef.current.setSliceType(nvRef.current.sliceTypeRender); break;
    }
  }

  const toggleOverlay = () => {
    if (!nvRef.current || nvRef.current.volumes.length < 2) return
    const newOpacity = overlayOpacity > 0 ? 0 : 0.65
    setOverlayOpacity(newOpacity)
    nvRef.current.setOpacity(1, newOpacity)
  }

  const modes = [
    { id: 'axial', icon: Monitor, label: 'Axial' },
    { id: 'sagittal', icon: Layers, label: 'Sagittal' },
    { id: 'coronal', icon: LayoutGrid, label: 'Coronal' },
    { id: '3d', icon: Box, label: '3D Vol' },
  ]

  const takeSnapshot = () => {
    if (!nvRef.current) return
    nvRef.current.saveScene(`NeuroSeg_Result_${jobId || 'temp'}.png`)
    toast.success('Snapshot Saved')
  }

  return (
    <div className="glass-card !rounded-3xl flex flex-col h-full min-h-[550px] !p-8">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-6 border-b border-slate-100 pb-6">
        <div className="flex gap-1.5 p-1 bg-slate-100/50 rounded-2xl border border-slate-200/50">
          {modes.map(m => (
            <button
              key={m.id}
              onClick={() => updateLayout(m.id)}
              className={`flex items-center gap-2 px-5 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${viewMode === m.id ? 'bg-white text-accent-blue shadow-apple border border-slate-200/50' : 'text-slate-400 hover:text-slate-600 hover:bg-white/50'
                }`}
            >
              <m.icon className="w-3.5 h-3.5" /> {m.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-6">
          {jobId && (
            <button
              onClick={toggleOverlay}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] uppercase font-black tracking-widest transition-all border shadow-sm ${overlayOpacity > 0 ? 'bg-accent-blue/5 border-accent-blue/20 text-accent-blue' : 'bg-slate-50 border-slate-200 text-slate-400'
                }`}
            >
              {overlayOpacity > 0 ? <ToggleRight className="w-4 h-4" /> : <ToggleLeft className="w-4 h-4" />} Layer Sync
            </button>
          )}
          <Maximize2 className="w-5 h-5 text-slate-300 hover:text-accent-blue cursor-pointer transition-all" />
        </div>
      </div>

      {/* Canvas Wrapper */}
      <div className="relative flex-1 bg-[#0A0E14] rounded-[2.5rem] overflow-hidden group min-h-[400px] shadow-inner ring-8 ring-slate-50">
        {!isLoaded && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/80 backdrop-blur-md rounded-[2.5rem]">
            <div className="w-16 h-16 border-4 border-slate-100 border-t-accent-blue rounded-full animate-spin mb-6" />
            <p className="text-slate-900 font-black tracking-widest text-xs animate-pulse uppercase">Syncing Medical Engine...</p>
          </div>
        )}

        {!files.flair && isLoaded && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-50/50 rounded-[2.5rem] border-4 border-dashed border-slate-200/60 transition-all">
            <div className="w-24 h-24 bg-white rounded-[2rem] shadow-apple flex items-center justify-center mb-6">
              <Binary className="w-10 h-10 text-slate-200" />
            </div>
            <p className="text-slate-900 font-black tracking-tight text-lg">Imaging System Offline</p>
            <p className="text-slate-400 text-[10px] mt-2 font-bold uppercase tracking-widest">Awaiting primary MRI ingestion</p>
          </div>
        )}

        <canvas
          ref={canvasRef}
          id="niivue-canvas"
          className={`w-full h-full transition-opacity duration-700 ${isLoaded ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        />

        {isLoaded && (
          <div className="absolute bottom-6 left-6 flex gap-3">
            <span className="text-[9px] py-1.5 px-4 bg-white/90 backdrop-blur-xl text-slate-900 font-black shadow-apple rounded-full flex items-center gap-2.5 border border-white uppercase tracking-widest">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-blue animate-pulse" /> Live_Analysis_Active
            </span>
          </div>
        )}
      </div>
    </div>
  )
})

export default NiiVueViewer
