import React, { useState } from 'react';
import { ShaderParam, UniformLayout } from '../types';

// --- Logic ---
export const calculateUniformLayout = (params: ShaderParam[], startOffset: number = 0): UniformLayout => {
  let currentOffset = startOffset;
  const offsetMap: Record<string, number> = {};

  params.forEach(param => {
    let alignment = 4;
    let size = 4;

    if (param.type === 'color' || param.type === 'vec3') {
      alignment = 16;
      size = 16;
    }

    const padding = (alignment - (currentOffset % alignment)) % alignment;
    currentOffset += padding;
    offsetMap[param.id] = currentOffset;
    currentOffset += size;
  });

  const totalPadding = (16 - (currentOffset % 16)) % 16;
  const totalSize = currentOffset + totalPadding;

  return { size: totalSize, offsetMap };
};

export const writeParamsToBuffer = (
  data: Float32Array, 
  params: ShaderParam[], 
  layout: UniformLayout
) => {
  params.forEach(param => {
    const floatOffset = layout.offsetMap[param.id] / 4;
    if (param.type === 'float') {
      data[floatOffset] = param.value;
    } else if (param.type === 'color' || param.type === 'vec3') {
      data[floatOffset] = param.value[0];
      data[floatOffset + 1] = param.value[1];
      data[floatOffset + 2] = param.value[2];
    }
  });
};

// --- UI Component ---
interface ParamsControlPanelProps {
  params: ShaderParam[];
  setParams: React.Dispatch<React.SetStateAction<ShaderParam[]>>;
  description?: string;
}

export const ParamsControlPanel: React.FC<ParamsControlPanelProps> = ({ params, setParams, description }) => {
  const [isOpen, setIsOpen] = useState(true);

  const handleFloatChange = (id: string, newVal: number) => {
    setParams(prev => prev.map(p => {
      if (p.id === id && p.type === 'float') {
        return { ...p, value: newVal };
      }
      return p;
    }));
  };

  const handleColorChange = (id: string, hex: string) => {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    const newColor: [number, number, number] = [r, g, b];

    setParams(prev => prev.map(p => {
      if (p.id === id && (p.type === 'color' || p.type === 'vec3')) {
        return { ...p, value: newColor };
      }
      return p;
    }));
  };

  const rgbToHex = (rgb: [number, number, number]) => {
    const toHex = (c: number) => {
      const hex = Math.round(c * 255).toString(16);
      return hex.length === 1 ? '0' + hex : hex;
    };
    return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;
  };

  return (
    <div className={`fixed right-0 top-1/2 -translate-y-1/2 z-40 transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] flex items-start ${isOpen ? 'translate-x-0' : 'translate-x-[calc(100%-40px)]'}`}>
        
        {/* Toggle / Label Vertical */}
        <button 
            onClick={() => setIsOpen(!isOpen)}
            className="w-10 h-64 bg-black/40 border-l border-y border-white/10 backdrop-blur-md flex flex-col items-center justify-center gap-4 hover:bg-white/5 transition-colors cursor-pointer group"
        >
            <div className="whitespace-nowrap -rotate-90 text-[10px] font-mono tracking-[0.3em] text-white/40 uppercase group-hover:text-acid transition-colors">
                {isOpen ? 'Close Parameters' : 'Open Parameters'}
            </div>
            <div className={`w-1 h-1 bg-acid rounded-full transition-opacity ${isOpen ? 'opacity-100' : 'opacity-20'}`} />
        </button>

        {/* Content Area */}
        <div className="w-80 bg-black/80 border-y border-l border-white/10 backdrop-blur-xl p-8 flex flex-col gap-8 shadow-2xl relative overflow-hidden h-[60vh] max-h-[600px] overflow-y-auto custom-scrollbar">
            {/* Background Decor */}
            <div className="absolute top-0 right-0 p-2 opacity-20 pointer-events-none">
                 <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M0 0H60V60H0V0Z" fill="none"/>
                    <path d="M60 0L0 0" stroke="white" strokeWidth="0.5"/>
                    <path d="M60 60L60 0" stroke="white" strokeWidth="0.5"/>
                 </svg>
            </div>

            <div className="shrink-0 space-y-4">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-acid animate-pulse"></div>
                    <h3 className="text-xl font-bold uppercase tracking-tighter">Analysis</h3>
                </div>
                
                {description && (
                    <div className="text-[11px] font-mono leading-relaxed text-gray-400 border-l border-white/10 pl-3">
                        {description}
                    </div>
                )}
                
                <div className="h-[1px] w-full bg-gradient-to-r from-white/30 to-transparent"></div>
            </div>

            <div className="shrink-0 space-y-6">
                <div className="text-[10px] font-mono uppercase tracking-widest text-white/30 mb-4">Parameter Control</div>
                {params.map(param => (
                <div key={param.id} className="group shrink-0">
                    <div className="flex justify-between items-baseline mb-3">
                        <label className="text-xs font-mono uppercase tracking-widest text-gray-400 group-hover:text-acid transition-colors select-none">
                            {param.label}
                        </label>
                        <span className="text-[10px] font-mono text-white">
                            {param.type === 'float' ? param.value.toFixed(2) : ''}
                        </span>
                    </div>

                    {param.type === 'float' && (
                        <div className="relative h-4 flex items-center">
                            <input
                                type="range"
                                min={param.min}
                                max={param.max}
                                step={param.step || 0.01}
                                value={param.value}
                                onChange={(e) => handleFloatChange(param.id, parseFloat(e.target.value))}
                                className="w-full z-10 opacity-0 absolute inset-0 cursor-pointer"
                            />
                            <div className="w-full h-[1px] bg-white/20 relative">
                                <div 
                                    className="absolute top-0 bottom-0 bg-white transition-all duration-75"
                                    style={{ width: `${((param.value - param.min) / (param.max - param.min)) * 100}%` }}
                                />
                            </div>
                            <div 
                                className="absolute w-2 h-2 bg-acid rotate-45 pointer-events-none transition-all duration-75"
                                style={{ left: `${((param.value - param.min) / (param.max - param.min)) * 100}%`, transform: 'translateX(-50%) rotate(45deg)' }}
                            />
                        </div>
                    )}

                    {(param.type === 'color' || param.type === 'vec3') && (
                        <div className="flex gap-2">
                            <div className="relative w-full h-8 border border-white/20 group-hover:border-white transition-colors cursor-pointer">
                                <input
                                    type="color"
                                    value={rgbToHex(param.value)}
                                    onChange={(e) => handleColorChange(param.id, e.target.value)}
                                    className="opacity-0 absolute inset-0 w-full h-full cursor-pointer z-10"
                                />
                                <div className="absolute inset-0.5" style={{ backgroundColor: rgbToHex(param.value) }}></div>
                            </div>
                            <div className="font-mono text-[10px] self-center text-white/50">{rgbToHex(param.value)}</div>
                        </div>
                    )}
                </div>
                ))}
            </div>
        </div>
    </div>
  );
};