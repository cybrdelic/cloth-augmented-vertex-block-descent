
import React, { useState, useEffect, useRef } from 'react';
import { ShaderError, VideoConfig, ShotType } from '../types';
import Editor, { useMonaco, Monaco } from '@monaco-editor/react';

// --- Types ---
export interface MenuItem {
    label: string;
    action: () => void;
    shortcut?: string;
}

export interface MenuGroup {
    label: string;
    items: MenuItem[];
}

// --- Components ---

interface ErrorDisplayProps {
  error: ShaderError | null;
  onClose: () => void;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, onClose }) => {
  const [copied, setCopied] = useState(false);
  if (!error) return null;

  const handleCopy = () => {
    const text = `${error.type.toUpperCase()} ERROR:\n${error.message}\n${error.lineNum ? `Line: ${error.lineNum}, Pos: ${error.linePos}` : ''}`;
    navigator.clipboard.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-2xl px-6 animate-fade-in-up">
      <div className="bg-black border border-red-600 shadow-[0_0_50px_rgba(220,38,38,0.3)] relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-red-600 animate-pulse-fast"></div>
        <div className="p-8">
            <div className="flex justify-between items-start mb-6">
                <div>
                    <h2 className="text-3xl font-bold text-red-600 tracking-tighter uppercase mb-1">System Error</h2>
                    <p className="font-mono text-xs text-red-600/60 uppercase tracking-widest">
                        Module: {error.type} // Critical Failure
                    </p>
                </div>
                <button onClick={onClose} className="w-10 h-10 flex items-center justify-center border border-red-900 hover:bg-red-900/20 text-red-600 transition-colors">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
            </div>
            <div className="bg-red-950/10 border border-red-900/30 p-4 font-mono text-xs text-red-400 overflow-x-auto whitespace-pre-wrap max-h-64 custom-scrollbar mb-6">
                {error.message}
            </div>
            <div className="flex justify-between items-center">
                 {error.lineNum ? (
                     <div className="text-xs font-mono text-red-500 bg-red-950/30 px-2 py-1">
                         AT LINE {error.lineNum} : COL {error.linePos}
                     </div>
                 ) : <div></div>}
                 
                 <button onClick={handleCopy} className="text-xs font-mono font-bold text-white uppercase hover:text-red-500 transition-colors flex items-center gap-2">
                     {copied ? '[ LOG COPIED ]' : '[ COPY DIAGNOSTICS ]'}
                 </button>
            </div>
        </div>
      </div>
    </div>
  );
};

interface VideoExportOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  onStartRecord: (config: VideoConfig) => void;
}

export const VideoExportOverlay: React.FC<VideoExportOverlayProps> = ({ isOpen, onClose, onStartRecord }) => {
    const [config, setConfig] = useState<VideoConfig>({
        duration: 5,
        fps: 60,
        bitrate: 25,
        shotType: 'orbit',
        orchestrate: false,
        postProcess: { grain: 0.1, aberration: 0.2 },
        format: 'webm'
    });

    if (!isOpen) return null;

    const shotTypes: { id: ShotType, label: string }[] = [
        { id: 'orbit', label: 'Classic Orbit' },
        { id: 'sweep', label: 'Low Sweep' },
        { id: 'dolly', label: 'Slow Zoom' },
        { id: 'breathing', label: 'Breathing' },
        { id: 'chaos', label: 'Handheld Chaos' },
    ];

    const previewStyles = `
      @keyframes preview-orbit { 0% { transform: rotateY(0deg); } 100% { transform: rotateY(360deg); } }
      @keyframes preview-sweep { 0%, 100% { transform: translateY(0) rotateX(0); } 50% { transform: translateY(20px) rotateX(15deg); } }
      @keyframes preview-dolly { 0%, 100% { transform: scale(0.6); } 50% { transform: scale(1.1); } }
      @keyframes preview-breathing { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
      @keyframes preview-chaos { 
          0% { transform: translate(0,0) rotate(0); } 
          25% { transform: translate(2px,2px) rotate(1deg); } 
          50% { transform: translate(-2px, -1px) rotate(-1deg); }
          75% { transform: translate(1px, -2px) rotate(1deg); }
          100% { transform: translate(0,0) rotate(0); }
      }
    `;

    const getPreviewAnimation = (type: ShotType) => {
        switch(type) {
            case 'orbit': return 'preview-orbit 4s linear infinite';
            case 'sweep': return 'preview-sweep 4s ease-in-out infinite';
            case 'dolly': return 'preview-dolly 4s ease-in-out infinite';
            case 'breathing': return 'preview-breathing 3s ease-in-out infinite';
            case 'chaos': return 'preview-chaos 0.5s linear infinite';
            default: return 'none';
        }
    };

    return (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-4 animate-fade-in-up">
            <style>{previewStyles}</style>
            <div className="bg-black border border-white/20 w-full max-w-2xl p-8 relative shadow-2xl flex flex-col md:flex-row gap-8">
                 <div className="flex-1 space-y-6">
                    <div>
                         <h2 className="text-2xl font-bold tracking-tighter mb-1">DIRECTOR MODE<span className="text-acid">.</span></h2>
                         <p className="text-xs font-mono text-gray-500 uppercase tracking-widest">Automated Orchestration</p>
                    </div>
                    <div className="w-full h-32 bg-gray-900 border border-white/10 relative overflow-hidden flex items-center justify-center perspective-[500px]">
                        <div className="absolute top-2 left-2 text-[10px] font-mono text-white/40 uppercase z-10">Preview</div>
                        <div className="absolute inset-0 opacity-20" style={{ 
                            backgroundImage: 'linear-gradient(white 1px, transparent 1px), linear-gradient(90deg, white 1px, transparent 1px)', 
                            backgroundSize: '20px 20px',
                            transform: 'rotateX(60deg) translateY(20px) scale(2)'
                        }}></div>
                        <div style={{ 
                                width: '40px', height: '40px', 
                                border: '1px solid #ccff00', 
                                backgroundColor: 'rgba(204, 255, 0, 0.1)',
                                animation: getPreviewAnimation(config.shotType),
                                boxShadow: `0 0 20px rgba(204, 255, 0, 0.2)`
                            }}
                        >
                             <div className="w-full h-full border border-acid/50 rotate-45 transform scale-75"></div>
                        </div>
                        <div className="absolute inset-0 pointer-events-none" style={{ filter: `contrast(${1 + config.postProcess.grain}) sepia(${config.postProcess.aberration * 0.5})`, opacity: 0.8 }}>
                            {config.postProcess.grain > 0 && (<div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPjxyZWN0IHdpZHRoPSI0IiBoZWlnaHQ9IjQiIGZpbGw9IiMwMDAiLz48cmVjdCB3aWR0aD0iMSIgaGVpZ2h0PSIxIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjEiLz48L3N2Zz4=')] opacity-50 mix-blend-overlay"></div>)}
                            {config.postProcess.aberration > 0 && (<div className="absolute inset-0 mix-blend-screen text-red-500/20 translate-x-[1px]"></div>)}
                        </div>
                    </div>
                    <div>
                        <div className="text-xs font-mono uppercase text-gray-400 mb-2">Camera Movement (B-Roll)</div>
                        <div className="grid grid-cols-2 gap-2">
                            {shotTypes.map(shot => (
                                <button key={shot.id} onClick={() => setConfig({...config, shotType: shot.id})} className={`px-3 py-2 text-xs font-mono uppercase border transition-colors ${config.shotType === shot.id ? 'border-acid text-acid bg-acid/10' : 'border-white/20 text-gray-400 hover:border-white/50'}`}>{shot.label}</button>
                            ))}
                        </div>
                    </div>
                    <div className="flex items-center justify-between border border-white/10 p-3 hover:bg-white/5 cursor-pointer transition-colors" onClick={() => setConfig({...config, orchestrate: !config.orchestrate})}>
                        <div><div className="text-sm font-bold">Scene Orchestration</div><div className="text-xs text-gray-500">Auto-animate lights & physics</div></div>
                        <div className={`w-3 h-3 border ${config.orchestrate ? 'bg-acid border-acid' : 'border-white/30'}`}></div>
                    </div>
                    <div className="space-y-4 pt-4 border-t border-white/10">
                        <div className="text-xs font-mono uppercase text-gray-400">Post-Processing</div>
                        <div className="space-y-1"><div className="flex justify-between text-[10px] uppercase"><span>Film Grain</span><span>{(config.postProcess.grain * 100).toFixed(0)}%</span></div><input type="range" min="0" max="1" step="0.05" value={config.postProcess.grain} onChange={(e) => setConfig({...config, postProcess: {...config.postProcess, grain: parseFloat(e.target.value)}})} /></div>
                        <div className="space-y-1"><div className="flex justify-between text-[10px] uppercase"><span>Chromatic Aberration</span><span>{(config.postProcess.aberration * 100).toFixed(0)}%</span></div><input type="range" min="0" max="1" step="0.05" value={config.postProcess.aberration} onChange={(e) => setConfig({...config, postProcess: {...config.postProcess, aberration: parseFloat(e.target.value)}})} /></div>
                    </div>
                 </div>
                 <div className="flex-1 space-y-6 flex flex-col justify-between border-l border-white/10 pl-0 md:pl-8">
                     <div className="space-y-6">
                        <div><div className="flex justify-between text-xs font-mono uppercase mb-2"><span className="text-gray-400">Duration</span><span className="text-acid">{config.duration}s</span></div><input type="range" min="1" max="20" step="1" value={config.duration} onChange={(e) => setConfig({...config, duration: parseInt(e.target.value)})} /></div>
                        <div><div className="flex justify-between text-xs font-mono uppercase mb-2"><span className="text-gray-400">Frame Rate</span><span className="text-acid">{config.fps} FPS</span></div><div className="flex gap-2">{[30, 60].map(fps => (<button key={fps} onClick={() => setConfig({...config, fps})} className={`flex-1 py-1 text-xs border ${config.fps === fps ? 'border-acid text-acid' : 'border-white/20 text-gray-500'}`}>{fps}</button>))}</div></div>
                        <div><div className="flex justify-between text-xs font-mono uppercase mb-2"><span className="text-gray-400">Bitrate</span><span className="text-acid">{config.bitrate} Mbps</span></div><input type="range" min="5" max="50" step="5" value={config.bitrate} onChange={(e) => setConfig({...config, bitrate: parseInt(e.target.value)})} /></div>
                    </div>
                    <div className="flex gap-4 pt-4">
                        <button onClick={onClose} className="flex-1 py-3 text-xs font-mono uppercase border border-white/20 hover:bg-white/5 transition-colors">Cancel</button>
                        <button onClick={() => { onStartRecord(config); onClose(); }} className="flex-1 py-3 text-xs font-bold font-mono uppercase bg-acid text-black hover:bg-white transition-colors flex items-center justify-center gap-2"><div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>ACTION!</button>
                    </div>
                 </div>
            </div>
        </div>
    );
};

export const RecordingIndicator: React.FC<{ isRecording: boolean, timeLeft: number, onStop?: () => void }> = ({ isRecording, timeLeft, onStop }) => {
    if (!isRecording) return null;
    return (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 flex items-center gap-4 z-50">
            <div className="bg-red-600/90 text-white px-4 py-1 flex items-center gap-3 rounded-full font-mono text-xs shadow-[0_0_20px_rgba(220,38,38,0.6)] animate-pulse">
                <div className="w-2 h-2 bg-white rounded-full"></div>REC // {timeLeft.toFixed(1)}s
            </div>
            {onStop && (
                <button onClick={onStop} className="bg-white text-black px-4 py-1 rounded-full font-mono text-xs font-bold hover:bg-gray-200 transition-colors shadow-lg flex items-center gap-1">
                    <div className="w-2 h-2 bg-black rounded-sm"></div>STOP
                </button>
            )}
        </div>
    );
};

export const DocumentationOverlay: React.FC<{ isOpen: boolean, onClose: () => void }> = ({ isOpen, onClose }) => {
    const [activeTab, setActiveTab] = useState<'guide' | 'uniforms' | 'troubleshooting'>('guide');
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 bg-black flex flex-col animate-fade-in-up text-white">
            <div className="h-20 border-b border-white/10 flex items-center justify-between px-8 md:px-12 bg-black/50 backdrop-blur-md">
                <h1 className="text-2xl font-bold tracking-tight">DOCUMENTATION<span className="text-acid">.</span></h1>
                <button onClick={onClose} className="group flex items-center gap-3 text-sm font-mono uppercase tracking-widest hover:text-acid transition-colors">Close [ESC]<div className="w-8 h-8 border border-white/20 group-hover:border-acid flex items-center justify-center rounded-full transition-colors"><svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" /></svg></div></button>
            </div>
            <div className="flex-1 flex overflow-hidden">
                <div className="w-64 border-r border-white/10 p-8 hidden md:block bg-black/50">
                    <div className="space-y-6">
                        {(['guide', 'uniforms', 'troubleshooting'] as const).map(tab => (
                            <button key={tab} onClick={() => setActiveTab(tab)} className={`block w-full text-left text-sm font-mono uppercase tracking-widest transition-all ${activeTab === tab ? 'text-acid pl-4 border-l-2 border-acid' : 'text-gray-500 hover:text-white pl-0 border-l-2 border-transparent'}`}>0{tab === 'guide' ? '1' : tab === 'uniforms' ? '2' : '3'} / {tab}</button>
                        ))}
                    </div>
                </div>
                <div className="flex-1 overflow-y-auto p-8 md:p-16 custom-scrollbar relative">
                    <div className="max-w-4xl mx-auto pb-24">
                        <div className="absolute top-8 right-8 text-[120px] font-bold text-white/5 pointer-events-none select-none leading-none -z-10">{activeTab === 'guide' ? '01' : activeTab === 'uniforms' ? '02' : '03'}</div>
                        
                        {activeTab === 'guide' && (
                            <div className="space-y-16 animate-slide-in-right">
                                <div><h2 className="text-5xl md:text-7xl font-bold mb-8 tracking-tighter">The Boilerplate.</h2><p className="text-xl text-gray-400 leading-relaxed font-light border-l-2 border-white/20 pl-6">Designed to eliminate the "Setup Fatigue" of WebGPU. Focus purely on shader art.</p></div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    <div className="border border-white/10 p-8 hover:bg-white/5 transition-colors group"><div className="text-acid font-mono text-xs mb-6 tracking-widest group-hover:underline">FEATURE_01</div><h3 className="text-2xl font-bold mb-4">Instant Hot-Reload</h3><p className="text-sm text-gray-500 leading-relaxed">Changes to WGSL compile instantly without losing context.</p></div>
                                    <div className="border border-white/10 p-8 hover:bg-white/5 transition-colors group"><div className="text-acid font-mono text-xs mb-6 tracking-widest group-hover:underline">FEATURE_02</div><h3 className="text-2xl font-bold mb-4">Error Guard</h3><p className="text-sm text-gray-500 leading-relaxed">Exact line number mapping for shader errors.</p></div>
                                </div>
                            </div>
                        )}
                        {activeTab === 'uniforms' && (
                             <div className="space-y-12 animate-slide-in-right">
                                <div><h2 className="text-5xl font-bold tracking-tighter mb-6">Standard Uniforms</h2><p className="text-gray-400">Pre-bound to group(0) binding(0).</p></div>
                                <div className="bg-gray-900 p-6 rounded-lg font-mono text-sm text-gray-300">
                                    struct Uniforms &#123;<br/>
                                    &nbsp;&nbsp;resolution: vec2f,<br/>
                                    &nbsp;&nbsp;time: f32,<br/>
                                    &nbsp;&nbsp;cameraPos: vec4f,<br/>
                                    &nbsp;&nbsp;mouse: vec4f,<br/>
                                    &nbsp;&nbsp;audio: vec4f <span className="text-gray-500">// Low, Mid, High, Vol</span><br/>
                                    &#125;;
                                </div>
                             </div>
                        )}
                        {activeTab === 'troubleshooting' && (
                            <div className="space-y-12 animate-slide-in-right">
                                <div><h2 className="text-5xl font-bold tracking-tighter mb-6">Troubleshooting</h2></div>
                                <ul className="list-disc pl-5 space-y-4 text-gray-400">
                                    <li><strong className="text-white">Device Lost:</strong> Usually due to an infinite loop in the shader (TDR). Refresh the page.</li>
                                    <li><strong className="text-white">Syntax Error:</strong> Check line numbers in the Error Overlay.</li>
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export const MenuBar: React.FC<{ menus: MenuGroup[] }> = ({ menus }) => {
    const [openMenu, setOpenMenu] = useState<string | null>(null);

    return (
        <div className="fixed top-0 left-0 right-0 h-10 bg-black/80 backdrop-blur-md border-b border-white/10 flex items-center px-4 z-50 select-none">
            <div className="font-bold tracking-tighter mr-6">WGSL<span className="text-acid">.STUDIO</span></div>
            <div className="flex h-full">
                {menus.map(menu => (
                    <div key={menu.label} className="relative h-full" onMouseEnter={() => openMenu && setOpenMenu(menu.label)}>
                        <button 
                            onClick={() => setOpenMenu(openMenu === menu.label ? null : menu.label)}
                            className={`h-full px-4 text-xs font-mono uppercase tracking-widest hover:bg-white/10 transition-colors ${openMenu === menu.label ? 'bg-white/10 text-white' : 'text-gray-400'}`}
                        >
                            {menu.label}
                        </button>
                        {openMenu === menu.label && (
                            <>
                                <div className="fixed inset-0 z-40" onClick={() => setOpenMenu(null)} />
                                <div className="absolute top-full left-0 min-w-[200px] bg-black border border-white/20 shadow-2xl py-2 z-50 animate-fade-in-up">
                                    {menu.items.map((item, i) => (
                                        <button 
                                            key={i} 
                                            onClick={() => { item.action(); setOpenMenu(null); }}
                                            className="w-full text-left px-4 py-2 text-xs font-mono text-gray-300 hover:bg-white/10 hover:text-white flex justify-between group"
                                        >
                                            <span>{item.label}</span>
                                            {item.shortcut && <span className="text-gray-600 group-hover:text-acid">{item.shortcut}</span>}
                                        </button>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

interface ShaderEditorProps {
    isOpen: boolean;
    onClose: () => void;
    code: string;
    onCodeChange: (code: string) => void;
    error: ShaderError | null;
}

export const ShaderEditor: React.FC<ShaderEditorProps> = ({ isOpen, onClose, code, onCodeChange, error }) => {
    const editorRef = useRef<any>(null);
    const monacoRef = useRef<Monaco | null>(null);

    const handleBeforeMount = (monaco: Monaco) => {
        // Register WGSL Language
        monaco.languages.register({ id: 'wgsl' });
        monaco.languages.setMonarchTokensProvider('wgsl', {
            keywords: [
                'fn', 'let', 'var', 'if', 'else', 'for', 'return', 'struct', 'texture_2d', 'sampler', 
                'uniform', 'group', 'binding', 'vertex', 'fragment', 'compute', 'builtin', 'location', 
                'true', 'false', 'switch', 'case', 'default', 'break', 'continue', 'discard'
            ],
            typeKeywords: [
                'f32', 'i32', 'u32', 'bool', 'vec2f', 'vec3f', 'vec4f', 'mat2x2f', 'mat3x3f', 'mat4x4f', 
                'array', 'ptr', 'atomic'
            ],
            tokenizer: {
                root: [
                    [/[a-z_$][\w$]*/, { cases: { '@keywords': 'keyword', '@typeKeywords': 'type', '@default': 'identifier' } }],
                    [/[0-9]+(\.[0-9]*)?/, 'number'],
                    [/\/\/.*/, 'comment'],
                ]
            }
        });

        // AUTOCOMPLETE PROVIDER
        monaco.languages.registerCompletionItemProvider('wgsl', {
            triggerCharacters: ['@', '.'], 
            provideCompletionItems: (model, position) => {
                const word = model.getWordUntilPosition(position);
                const range = {
                    startLineNumber: position.lineNumber,
                    endLineNumber: position.lineNumber,
                    startColumn: word.startColumn,
                    endColumn: word.endColumn
                };
                
                const lineContent = model.getLineContent(position.lineNumber);
                const textBeforeCursor = lineContent.substring(0, position.column - 1);
                
                // CHECK: Is triggered by '@' or currently typing an attribute?
                const isAttribute = textBeforeCursor.trim().endsWith('@') || 
                                   (textBeforeCursor.lastIndexOf('@') > textBeforeCursor.lastIndexOf(' '));

                if (isAttribute) {
                    const attributes = [
                        {
                            label: 'vertex',
                            kind: monaco.languages.CompletionItemKind.Keyword,
                            insertText: 'vertex',
                            detail: 'Vertex Stage',
                            documentation: { value: '**@vertex**\n\nEntry point for the vertex shader stage. This function must return a struct containing a `@builtin(position)` field.\n\n```wgsl\n@vertex\nfn vs_main() -> VertexOutput {\n    ...\n}\n```' }
                        },
                        {
                            label: 'fragment',
                            kind: monaco.languages.CompletionItemKind.Keyword,
                            insertText: 'fragment',
                            detail: 'Fragment Stage',
                            documentation: { value: '**@fragment**\n\nEntry point for the fragment shader stage. This function must return a color (usually `@location(0) vec4f`).\n\n```wgsl\n@fragment\nfn fs_main() -> @location(0) vec4f {\n    return vec4f(1.0, 0.0, 0.0, 1.0);\n}\n```' }
                        },
                        {
                            label: 'compute',
                            kind: monaco.languages.CompletionItemKind.Keyword,
                            insertText: 'compute',
                            detail: 'Compute Stage',
                            documentation: { value: '**@compute**\n\nEntry point for a compute shader. Must be accompanied by `@workgroup_size`.\n\n```wgsl\n@compute @workgroup_size(64)\nfn main() {\n    ...\n}\n```' }
                        },
                        {
                            label: 'group',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'group(${1:0})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'Resource Set (Folder)',
                            documentation: { value: '**@group(n)**\n\nThink of this as a **"Folder Number"** for your resources.\n\nWebGPU organizes external data (Uniforms, Textures) into sets so the CPU can send them to the GPU in batches.\n\nIn this boilerplate:\n- **Group 0**: Contains standard uniforms, time, mouse, and channel0.\n\n```wgsl\n@group(0) @binding(0) var<uniform> u: Uniforms;\n```' }
                        },
                        {
                            label: 'binding',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'binding(${1:0})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'Slot Number',
                            documentation: { value: '**@binding(n)**\n\nSpecifies the binding slot index *within* a specific Group.\n\n```wgsl\n@group(0) @binding(1) var myTexture: texture_2d<f32>;\n```' }
                        },
                        {
                            label: 'location',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'location(${1:0})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'IO Location Index',
                            documentation: { value: '**@location(n)**\n\nDefines the input/output location for variables passed between shader stages (Vertex -> Fragment).\n\n```wgsl\nstruct VertexOutput {\n    @location(0) uv: vec2f\n};\n```' }
                        },
                        {
                            label: 'builtin',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'builtin(${1:position})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'System Value',
                            documentation: { value: '**@builtin(name)**\n\nAccesses a system-generated value.\n\n**Common Values:**\n- `position`: (vec4f) Output clip-space position (Vertex)\n- `vertex_index`: (u32) Index of current vertex\n- `frag_coord`: (vec4f) Pixel coordinate (Fragment)\n- `global_invocation_id`: (vec3u) Thread ID (Compute)' }
                        },
                        {
                            label: 'workgroup_size',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'workgroup_size(${1:1}, ${2:1}, ${3:1})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'Compute Workgroup Dims',
                            documentation: { value: '**@workgroup_size(x, y, z)**\n\nDefines the dimensions of the local workgroup for compute shaders. Total threads = x * y * z.\n\n```wgsl\n@compute @workgroup_size(8, 8, 1)\nfn main() ...\n```' }
                        },
                         {
                            label: 'align',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'align(${1:16})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'Memory Alignment',
                            documentation: { value: '**@align(bytes)**\n\nForces a struct member to have a specific byte alignment. Useful for matching host-side memory layouts.\n\n```wgsl\nstruct MyUniforms {\n    a: f32,\n    @align(16) b: vec3f // Force start at 16-byte boundary\n}\n```' }
                        },
                        {
                            label: 'interpolate',
                            kind: monaco.languages.CompletionItemKind.Property,
                            insertText: 'interpolate(${1:perspective}, ${2:center})',
                            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                            detail: 'Interpolation Mode',
                            documentation: { value: '**@interpolate(mode, sampling)**\n\nControls how values are interpolated across the triangle surface.\n\n**Modes:**\n- `perspective`: Correct 3D perspective (default)\n- `linear`: Screen-space linear\n- `flat`: No interpolation\n\n**Sampling:** `center`, `centroid`, `sample`' }
                        }
                    ];
                    return { suggestions: attributes.map(a => ({...a, range})) };
                }

                // Standard Keywords if NOT an attribute
                const keywords = [
                    { label: 'fn', detail: 'Function', insertText: 'fn ${1:name}(${2}) -> ${3:void} {\n\t$0\n}', kind: monaco.languages.CompletionItemKind.Snippet, documentation: { value: 'Defines a GPU function.' } },
                    { label: 'struct', detail: 'Structure', insertText: 'struct ${1:Name} {\n\t${2:field}: ${3:type},\n};', kind: monaco.languages.CompletionItemKind.Snippet, documentation: { value: 'Defines a custom data structure.' } },
                    { label: 'if', detail: 'Control Flow', insertText: 'if (${1:condition}) {\n\t$0\n}', kind: monaco.languages.CompletionItemKind.Snippet, documentation: { value: 'Conditional block.' } },
                    { label: 'for', detail: 'Loop', insertText: 'for (var ${1:i} = 0; ${1:i} < ${2:10}; ${1:i}++) {\n\t$0\n}', kind: monaco.languages.CompletionItemKind.Snippet, documentation: { value: 'C-style for loop.' } },
                    { label: 'var', detail: 'Variable', insertText: 'var ${1:name}: ${2:type} = ${3:value};', kind: monaco.languages.CompletionItemKind.Variable, documentation: { value: 'Mutable variable.' } },
                    { label: 'let', detail: 'Constant', insertText: 'let ${1:name} = ${2:value};', kind: monaco.languages.CompletionItemKind.Constant, documentation: { value: 'Immutable constant.' } },
                    { label: 'vec2f', detail: 'Vector 2', insertText: 'vec2f(${1:0.0}, ${2:0.0})', kind: monaco.languages.CompletionItemKind.Constructor, documentation: { value: 'Constructs a vector of 2 floats.' } },
                    { label: 'vec3f', detail: 'Vector 3', insertText: 'vec3f(${1:0.0}, ${2:0.0}, ${3:0.0})', kind: monaco.languages.CompletionItemKind.Constructor, documentation: { value: 'Constructs a vector of 3 floats.' } },
                    { label: 'vec4f', detail: 'Vector 4', insertText: 'vec4f(${1:0.0}, ${2:0.0}, ${3:0.0}, ${4:1.0})', kind: monaco.languages.CompletionItemKind.Constructor, documentation: { value: 'Constructs a vector of 4 floats.' } },
                    { label: 'textureSample', detail: 'Texture', insertText: 'textureSample(${1:texture}, ${2:sampler}, ${3:uv})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Samples a texture at the given UV coordinates using implicit derivatives.' } },
                    { label: 'mix', detail: 'Math', insertText: 'mix(${1:x}, ${2:y}, ${3:a})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Linear interpolation: `x * (1-a) + y * a`.' } },
                    { label: 'smoothstep', detail: 'Math', insertText: 'smoothstep(${1:edge0}, ${2:edge1}, ${3:x})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Hermite interpolation between two edges.' } },
                    { label: 'dot', detail: 'Math', insertText: 'dot(${1:x}, ${2:y})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Calculates the dot product of two vectors.' } },
                    { label: 'cross', detail: 'Math', insertText: 'cross(${1:x}, ${2:y})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Calculates the cross product of two vec3s.' } },
                    { label: 'normalize', detail: 'Math', insertText: 'normalize(${1:x})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Returns a vector in the same direction with length 1.' } },
                    { label: 'length', detail: 'Math', insertText: 'length(${1:x})', kind: monaco.languages.CompletionItemKind.Function, documentation: { value: 'Calculates the length (magnitude) of a vector.' } },
                ];
                
                const suggestions = keywords.map(k => ({
                    ...k,
                    range,
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
                }));

                return { suggestions };
            }
        });

        // Define Dark Theme IMMEDIATELY to prevent flash
        monaco.editor.defineTheme('void-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                { token: 'comment', foreground: '6b7280' },
                { token: 'keyword', foreground: 'ccff00', fontStyle: 'bold' },
                { token: 'type', foreground: 'a78bfa' },
                { token: 'number', foreground: '60a5fa' },
                { token: 'identifier', foreground: 'e5e7eb' },
            ],
            colors: {
                'editor.background': '#000000',
                'editor.foreground': '#ffffff',
                'editor.lineHighlightBackground': '#111111',
                'editorCursor.foreground': '#ccff00',
                'editor.selectionBackground': '#ccff0033',
                'editor.inactiveSelectionBackground': '#ccff0011',
                'minimap.background': '#000000',
            }
        });
    };

    const handleOnMount = (editor: any, monaco: Monaco) => {
        editorRef.current = editor;
        monacoRef.current = monaco;
    };

    // Sync Error Markers
    useEffect(() => {
        if (monacoRef.current && editorRef.current) {
            const model = editorRef.current.getModel();
            if (model) {
                if (error && error.lineNum) {
                    monacoRef.current.editor.setModelMarkers(model, 'owner', [{
                        startLineNumber: error.lineNum,
                        startColumn: error.linePos || 1,
                        endLineNumber: error.lineNum,
                        endColumn: 1000,
                        message: error.message,
                        severity: monacoRef.current.MarkerSeverity.Error
                    }]);
                } else {
                    monacoRef.current.editor.setModelMarkers(model, 'owner', []);
                }
            }
        }
    }, [error]);

    return (
        <div className={`fixed top-10 left-0 bottom-0 w-[600px] bg-black border-r border-white/10 z-30 transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
            {/* Header */}
            <div className="h-10 border-b border-white/10 flex items-center justify-between px-4 bg-black">
                <div className="text-xs font-mono text-gray-500 uppercase">shader_module.wgsl</div>
                <div className="flex items-center gap-4">
                     {error && <div className="text-xs font-mono text-red-500 animate-pulse">COMPILATION_FAILED</div>}
                     <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                     </button>
                </div>
            </div>

            {/* AI Assistant Bar */}
            <div className="absolute bottom-6 left-6 right-6 z-20">
                <div className="bg-black/90 backdrop-blur-md border border-white/20 p-2 rounded-lg shadow-2xl flex gap-2">
                    <div className="w-8 h-8 bg-acid/10 rounded flex items-center justify-center text-acid">
                         <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                    </div>
                    <input 
                        type="text" 
                        placeholder="Ask AI to edit... (e.g. 'Make it rotate faster')" 
                        className="bg-transparent border-none outline-none text-xs font-mono text-white placeholder-gray-600 flex-1 min-w-0"
                    />
                    <button className="px-3 py-1 bg-white/10 hover:bg-white/20 text-[10px] font-mono uppercase rounded text-white transition-colors">
                        Generate
                    </button>
                </div>
            </div>

            {/* Editor */}
            <div className="w-full h-[calc(100%-40px)]">
                <Editor
                    height="100%"
                    defaultLanguage="wgsl"
                    theme="void-dark"
                    value={code}
                    onChange={(val) => val && onCodeChange(val)}
                    beforeMount={handleBeforeMount}
                    onMount={handleOnMount}
                    options={{
                        minimap: { enabled: false },
                        fontSize: 12,
                        fontFamily: "'JetBrains Mono', monospace",
                        lineHeight: 20,
                        padding: { top: 20, bottom: 20 },
                        scrollBeyondLastLine: false,
                        smoothScrolling: true,
                        cursorBlinking: 'smooth',
                        cursorSmoothCaretAnimation: 'on',
                        renderLineHighlight: 'all',
                        quickSuggestions: { other: true, comments: false, strings: false },
                    }}
                />
            </div>
        </div>
    );
};
