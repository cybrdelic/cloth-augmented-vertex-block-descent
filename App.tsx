
import React, { useState, useEffect, useRef } from 'react';
import WebGPURenderer, { WebGPURendererRef } from './components/FireRenderer';
import { ErrorDisplay, DocumentationOverlay, MenuBar, MenuGroup, VideoExportOverlay, RecordingIndicator, ShaderEditor } from './components/UIComponents';
import { ShaderError } from './types';
import { BOILERPLATE_SHADER_WGSL } from './constants';

const App: React.FC = () => {
  const [error, setError] = useState<ShaderError | null>(null);
  const [showDocs, setShowDocs] = useState(false);
  const [showVideoModal, setShowVideoModal] = useState(false);
  const [showEditor, setShowEditor] = useState(false);
  const [shaderCode, setShaderCode] = useState(BOILERPLATE_SHADER_WGSL);
  const [recordingStatus, setRecordingStatus] = useState({ isRecording: false, timeLeft: 0 });
  const [fps, setFps] = useState(0);
  const rendererRef = useRef<WebGPURendererRef>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Debounce Shader Updates
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleCodeChange = (newCode: string) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
          setShaderCode(newCode);
      }, 500); // 500ms debounce
  };

  useEffect(() => {
    let lastTime = performance.now();
    let frame = 0;
    const loop = () => {
      const now = performance.now();
      frame++;
      if (now - lastTime >= 1000) {
        setFps(frame);
        frame = 0;
        lastTime = now;
      }
      requestAnimationFrame(loop);
    };
    loop();
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
          rendererRef.current?.loadTexture(file);
      }
  };

  // Menu Configuration
  const menus: MenuGroup[] = [
    {
        label: 'File',
        items: [
            { label: 'Reset Sim', action: () => setShaderCode(BOILERPLATE_SHADER_WGSL + " "), shortcut: 'CMD+R' },
            { label: 'Load Texture...', action: () => fileInputRef.current?.click(), shortcut: 'CMD+O' },
        ]
    },
    {
        label: 'View',
        items: [
            { label: 'Toggle Code Editor', action: () => setShowEditor(!showEditor), shortcut: 'E' },
            { label: 'Documentation', action: () => setShowDocs(true), shortcut: 'F1' },
        ]
    },
    {
        label: 'Sim',
        items: [
            { label: 'Start Audio', action: () => rendererRef.current?.toggleAudio() },
        ]
    }
  ];

  const sceneDescription = "Augmented Vertex Block Descent (AVBD) Simulation. A 16,384 particle cloth simulated on the GPU using Compute Shaders. Solves distance constraints in parallel using a Checkerboard graph coloring strategy for stability.";

  return (
    <div className="w-screen h-screen relative bg-void overflow-hidden font-sans text-white select-none pt-10 antialiased">
      <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileSelect} />
      
      {/* Top Menu Bar */}
      <MenuBar menus={menus} />

      {/* 3D Canvas Layer */}
      <div className={`absolute inset-0 z-0 top-10 transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${showEditor ? 'left-[600px]' : 'left-0'}`}>
        <WebGPURenderer 
          ref={rendererRef}
          shaderCode={shaderCode}
          description={sceneDescription}
          onError={(e) => setError(e)}
          onClearError={() => setError(null)}
          onRecordProgress={(isRecording, timeLeft) => setRecordingStatus({ isRecording, timeLeft })}
        />
      </div>

      {/* HUD Layer (Non-Header parts) */}
      <div className={`absolute inset-0 z-10 pointer-events-none p-6 md:p-12 flex flex-col justify-end transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${showEditor ? 'left-[600px]' : 'left-0'}`}>
        <footer className="flex justify-between items-end">
            <div className="flex flex-col gap-1 pointer-events-auto opacity-50 hover:opacity-100 transition-opacity">
                 <div className="flex items-center gap-2 font-mono text-[10px] text-acid">
                    <span className="animate-pulse">●</span> PHYSICS_ENGINE_ACTIVE
                 </div>
                 <div className="font-mono text-[10px] text-white/40 tracking-widest">
                    {fps} FPS // {window.innerWidth}x{window.innerHeight}
                 </div>
            </div>
            <div className="text-right pointer-events-auto">
               <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase">
                   AVBD / WebGPU
               </span>
            </div>
        </footer>
      </div>
      
      {/* Modals & Overlays */}
      <div className="pointer-events-auto">
           <ErrorDisplay error={error} onClose={() => setError(null)} />
           <DocumentationOverlay isOpen={showDocs} onClose={() => setShowDocs(false)} />
           <ShaderEditor 
                isOpen={showEditor} 
                onClose={() => setShowEditor(false)} 
                code={shaderCode} 
                onCodeChange={handleCodeChange} 
                error={error}
           />
           <VideoExportOverlay 
                isOpen={showVideoModal} 
                onClose={() => setShowVideoModal(false)}
                onStartRecord={(config) => rendererRef.current?.startVideo(config)}
           />
           <RecordingIndicator 
                isRecording={recordingStatus.isRecording} 
                timeLeft={recordingStatus.timeLeft} 
                onStop={() => rendererRef.current?.stopVideo()}
           />
      </div>
    </div>
  );
};

export default App;
