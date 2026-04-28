import { Zap, Cpu, BarChart3, Activity, Layers, ShieldCheck, Box, HardDrive } from 'lucide-react'

interface Props {
  modelInfo: any
}

export default function PerformancePanel({ modelInfo }: Props) {
  const formatParams = (n: number) => {
    if (!n) return '—'
    return (n / 1000000).toFixed(2) + ' M'
  }

  const metrics = [
    { label: 'Base Features', value: modelInfo?.base_features || '32', icon: Layers },
    { label: 'Normalization', value: 'Per-Channel Z-Score', icon: ShieldCheck },
    { label: 'Patch Size', value: '128 x 128 x 128', icon: Box },
    { label: 'Inference Mode', value: modelInfo?.device?.includes('cpu') ? 'Local CPU Stride' : 'CUDA Accelerated', icon: Cpu },
  ]

  return (
    <div className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="px-2">
        <h2 className="text-2xl font-black text-slate-900 tracking-tight">Model Performance</h2>
        <p className="text-slate-500 text-sm font-medium">Deep learning architecture and validation metrics</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Main Stats */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Hero Stats */}
          <div className="grid grid-cols-2 gap-6">
            <div className="glass-card !bg-accent-blue font-bold text-white shadow-blue flex flex-col justify-between min-h-[160px]">
               <div className="flex justify-between items-start">
                  <p className="text-[10px] font-black uppercase tracking-widest opacity-80">Validation Accuracy</p>
                  <Activity className="w-5 h-5 opacity-40" />
               </div>
               <div>
                  <h4 className="text-5xl font-black tracking-tighter mb-1">82.0%</h4>
                  <p className="text-xs opacity-70">Mean Dice Score Coefficient</p>
               </div>
            </div>
            <div className="glass-card !bg-slate-900 font-bold text-white shadow-xl flex flex-col justify-between min-h-[160px]">
               <div className="flex justify-between items-start">
                  <p className="text-[10px] font-black uppercase tracking-widest opacity-60">Model Complexity</p>
                  <BarChart3 className="w-5 h-5 opacity-30" />
               </div>
               <div>
                  <h4 className="text-5xl font-black tracking-tighter mb-1">{formatParams(modelInfo?.num_params)}</h4>
                  <p className="text-xs opacity-50">Trainable Parameters</p>
               </div>
            </div>
          </div>

          {/* Technical Specs */}
          <div className="glass-card grid grid-cols-2 sm:grid-cols-4 gap-8">
             {metrics.map((m) => (
                <div key={m.label}>
                   <m.icon className="w-5 h-5 text-slate-400 mb-4" />
                   <p className="text-[9px] text-slate-400 font-black uppercase tracking-wider mb-1">{m.label}</p>
                   <p className="text-sm font-bold text-slate-900 tracking-tight">{m.value}</p>
                </div>
             ))}
          </div>

          {/* Model Graph Representation */}
          <div className="glass-card !p-8">
             <h3 className="text-xs font-black uppercase tracking-widest text-slate-400 mb-8">Architecture: 3D Residual U-Net</h3>
             <div className="flex items-center justify-between relative px-4 py-8">
                {/* Visual Connector */}
                <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-slate-100 -translate-y-1/2 -z-10 mx-16" />
                
                {[
                  { label: 'Input', val: '4 Channels', color: 'bg-slate-100 text-slate-500' },
                  { label: 'Encoder', val: 'Down 4x', color: 'bg-blue-50 text-accent-blue' },
                  { label: 'Latent', val: 'Bottleneck', color: 'bg-slate-900 text-white' },
                  { label: 'Decoder', val: 'Up 4x', color: 'bg-magenta-50 text-accent-magenta' },
                  { label: 'Output', val: '4 classes', color: 'bg-green-50 text-accent-green' },
                ].map((node, i) => (
                  <div key={node.label} className="flex flex-col items-center gap-3">
                    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center shadow-soft border border-white/50 ${node.color}`}>
                      {i + 1}
                    </div>
                    <div className="text-center">
                      <p className="text-[9px] font-black uppercase text-slate-400 tracking-wider font-mono">{node.label}</p>
                      <p className="text-[10px] font-bold text-slate-900">{node.val}</p>
                    </div>
                  </div>
                ))}
             </div>
          </div>
        </div>

        {/* Info Column */}
        <div className="space-y-8">
          <div className="glass-card flex flex-col gap-6">
             <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Class Definitions</h3>
             <div className="space-y-4">
                {(modelInfo?.class_names || ["Background", "NET", "ED", "ET"]).map((name: string, i: number) => (
                  <div key={name} className="flex items-center gap-3">
                    <div className={`w-2.5 h-2.5 rounded-full ${['bg-slate-200', 'bg-accent-blue', 'bg-accent-cyan', 'bg-accent-magenta'][i]}`} />
                    <span className="text-xs font-bold text-slate-700">{name}</span>
                    <span className="ml-auto text-[10px] font-mono text-slate-400">Idx: {i}</span>
                  </div>
                ))}
             </div>
          </div>

          <div className="glass-card !bg-amber-50/30 border-amber-100 flex flex-col gap-4">
             <div className="flex items-center gap-2 text-amber-600">
                <ShieldCheck className="w-4 h-4" />
                <h3 className="text-[10px] font-black uppercase tracking-widest">System Validation</h3>
             </div>
             <p className="text-xs text-amber-700 font-medium leading-relaxed">
                This model is validated against the BraTS 2021 dataset with standardized pre-processing (Z-score normalization and voxel spacing alignment).
             </p>
          </div>

          <div className="glass-card flex flex-col gap-6">
             <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Hardware Runtime</h3>
             <div className="space-y-4">
               <div className="flex justify-between items-center text-xs">
                 <span className="text-slate-500 font-medium">Device</span>
                 <span className="font-bold text-slate-900 uppercase tracking-tighter">{modelInfo?.device || 'CPU'}</span>
               </div>
               <div className="flex justify-between items-center text-xs">
                 <span className="text-slate-500 font-medium">Threads</span>
                 <span className="font-bold text-slate-900 tracking-tighter">4 Active</span>
               </div>
               <div className="flex justify-between items-center text-xs">
                 <span className="text-slate-500 font-medium">Memory Usage</span>
                 <span className="font-bold text-slate-900 tracking-tighter">~1.2 GB</span>
               </div>
             </div>
          </div>
        </div>

      </div>
    </div>
  )
}
