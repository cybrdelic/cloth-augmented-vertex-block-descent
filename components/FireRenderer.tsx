
import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { ShaderError, ShaderParam, VideoConfig } from '../types';
import { calculateUniformLayout, writeParamsToBuffer, ParamsControlPanel } from './ShaderParams';

// --- WebGPU Type Stubs ---
type GPUDevice = any;
type GPUCanvasContext = any;
type GPURenderPipeline = any;
type GPUComputePipeline = any;
type GPUBuffer = any;
type GPUBindGroup = any;
declare const GPUBufferUsage: any;
declare const GPUShaderStage: any;

const getErrorMessage = (err: any): string => {
  if (err === undefined) return "Undefined Error";
  if (err === null) return "Null Error";
  if (typeof err === 'string') return err;
  if (err.reason !== undefined && err.message !== undefined) return `Device Lost (${err.reason}): ${err.message}`;
  if (err.error instanceof Event || err.error !== undefined) {
      const e = err.error || err;
      return `[${e.constructor.name}] ${e.message}`;
  }
  if (err.message !== undefined) return String(err.message);
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  try { const json = JSON.stringify(err); if (json !== '{}') return json; } catch (e) {}
  return String(err);
};

export interface WebGPURendererRef {
  capture: (quality?: number) => void;
  startVideo: (config: VideoConfig) => void;
  stopVideo: () => void;
  loadTexture: (file: File) => void;
  toggleAudio: () => Promise<void>;
}

interface WebGPURendererProps {
  shaderCode: string;
  description?: string;
  onError: (error: ShaderError) => void;
  onClearError: () => void;
  onRecordProgress: (isRecording: boolean, timeLeft: number) => void;
}

const WebGPURenderer = forwardRef<WebGPURendererRef, WebGPURendererProps>(({ shaderCode, description, onError, onClearError, onRecordProgress }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isSupported, setIsSupported] = useState<boolean>(true);
  
  const deviceRef = useRef<GPUDevice | null>(null);
  const contextRef = useRef<GPUCanvasContext | null>(null);
  
  // Pipelines
  const renderPipelineRef = useRef<GPURenderPipeline | null>(null);
  const computePredictRef = useRef<GPUComputePipeline | null>(null);
  const computeSolveRef = useRef<GPUComputePipeline | null>(null);
  const computeIntegrateRef = useRef<GPUComputePipeline | null>(null);

  // Layouts
  const computeBindGroupLayoutRef = useRef<any>(null);
  const renderBindGroupLayoutRef = useRef<any>(null);

  // Buffers
  const uniformBufferEvenRef = useRef<GPUBuffer | null>(null);
  const uniformBufferOddRef = useRef<GPUBuffer | null>(null);
  const particleBufferRef = useRef<GPUBuffer | null>(null);
  const indexBufferRef = useRef<GPUBuffer | null>(null);
  
  // Bind Groups
  const computeBindGroupEvenRef = useRef<GPUBindGroup | null>(null);
  const computeBindGroupOddRef = useRef<GPUBindGroup | null>(null);
  const renderBindGroupEvenRef = useRef<GPUBindGroup | null>(null); // Render needs its own BG because of layout diff
  
  const requestRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(performance.now());
  const isMountedRef = useRef<boolean>(true);
  
  // Audio State
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyzerRef = useRef<AnalyserNode | null>(null);
  const audioDataArrayRef = useRef<Uint8Array | null>(null);

  // Recording State
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const isRecordingRef = useRef<boolean>(false);
  const recordingStartTimeRef = useRef<number>(0);
  const recordingConfigRef = useRef<VideoConfig | null>(null);
  const recordedFramesRef = useRef<number>(0);
  const streamTrackRef = useRef<any>(null);
  
  // State for AVBD
  const GRID_W = 128;
  const GRID_H = 128;
  const NUM_PARTICLES = GRID_W * GRID_H;
  const [needsReset, setNeedsReset] = useState(true);

  // Params
  const [params, setParams] = useState<ShaderParam[]>([
    { id: 'gravity', label: 'Gravity', type: 'float', value: 1.0, min: -2.0, max: 2.0 },
    { id: 'stiffness', label: 'Flexibility', type: 'float', value: 0.05, min: 0.0, max: 0.5 },
    { id: 'mouseRadius', label: 'Mouse Radius', type: 'float', value: 1.0, min: 0.1, max: 2.0 },
    { id: 'damping', label: 'Damping', type: 'float', value: 0.99, min: 0.9, max: 1.0 },
    { id: 'baseColor', label: 'Cloth Color', type: 'color', value: [0.0, 0.8, 1.0] },
    { id: 'lightAz', label: 'Light Azimuth', type: 'float', value: 0.1, min: 0.0, max: 1.0 },
    { id: 'lightEl', label: 'Light Elevation', type: 'float', value: 0.6, min: 0.0, max: 1.0 },
  ]);

  const paramsRef = useRef(params);
  useEffect(() => { paramsRef.current = params; }, [params]);

  const STANDARD_HEADER_SIZE = 48;
  const layout = calculateUniformLayout(params, STANDARD_HEADER_SIZE);
  // Matches the struct in constants.ts (128 bytes)
  const TOTAL_BUFFER_SIZE = 128; 

  const cameraState = useRef({ theta: 0.5, phi: 0.3, radius: 8.0, isDragging: false, lastX: 0, lastY: 0 });
  const mouseState = useRef({ x: 0, y: 0, isDown: 0 });

  // Index Generation for Wireframe Grid
  const createIndexBuffer = (device: GPUDevice) => {
      const indices: number[] = [];
      for (let y = 0; y < GRID_H; y++) {
          for (let x = 0; x < GRID_W; x++) {
              const i = y * GRID_W + x;
              // Horizontal Line
              if (x < GRID_W - 1) { indices.push(i); indices.push(i + 1); }
              // Vertical Line
              if (y < GRID_H - 1) { indices.push(i); indices.push(i + GRID_W); }
          }
      }
      const buffer = device.createBuffer({
          size: indices.length * 4,
          usage: GPUBufferUsage.INDEX,
          mappedAtCreation: true,
      });
      new Uint32Array(buffer.getMappedRange()).set(indices);
      buffer.unmap();
      return { buffer, count: indices.length };
  };

  const stopVideo = () => {
    if (mediaRecorderRef.current && isRecordingRef.current) {
        mediaRecorderRef.current.stop();
    }
  };

  const startVideo = (config: VideoConfig) => {
    if (!canvasRef.current) return;
    isRecordingRef.current = true;
    recordingConfigRef.current = config;
    recordedChunksRef.current = [];
    recordedFramesRef.current = 0;
    
    // Force 1080p for recording
    canvasRef.current.width = 1920;
    canvasRef.current.height = 1080;
    
    // Setup MediaRecorder
    const stream = canvasRef.current.captureStream(0); // 0 FPS = Manual stepping
    const track = stream.getVideoTracks()[0];
    streamTrackRef.current = track; // Save track to call requestFrame

    const options = { mimeType: 'video/webm; codecs=vp9', videoBitsPerSecond: config.bitrate * 1000000 };
    const recorder = new MediaRecorder(stream, options);
    
    recorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunksRef.current.push(e.data); };
    recorder.onstop = () => {
         const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
         const url = URL.createObjectURL(blob);
         const a = document.createElement('a');
         a.href = url;
         a.download = `avbd_simulation_${Date.now()}.webm`;
         a.click();
         isRecordingRef.current = false;
         onRecordProgress(false, 0);
    };
    
    mediaRecorderRef.current = recorder;
    recorder.start();
    setNeedsReset(true); // Restart sim for video
  };

  useImperativeHandle(ref, () => ({
    capture: () => {},
    loadTexture: () => {},
    startVideo,
    stopVideo,
    toggleAudio: async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const ctx = new AudioContext();
            const source = ctx.createMediaStreamSource(stream);
            const analyzer = ctx.createAnalyser();
            analyzer.fftSize = 256;
            source.connect(analyzer);
            audioContextRef.current = ctx;
            analyzerRef.current = analyzer;
            audioDataArrayRef.current = new Uint8Array(analyzer.frequencyBinCount);
        } catch (e) { console.error(e); }
    }
  }));

  const compilePipeline = async (device: GPUDevice, code: string, context: GPUCanvasContext) => {
      if (!isMountedRef.current || !deviceRef.current) return;

      try {
        // Create layouts if they don't exist
        // COMPUTE LAYOUT (Binding 1 = Storage RW)
        if (!computeBindGroupLayoutRef.current) {
            computeBindGroupLayoutRef.current = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }},
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }}, 
                ]
            });
        }
        
        // RENDER LAYOUT (Binding 2 = Storage Read-Only)
        // Binding 1 skipped to avoid collision
        if (!renderBindGroupLayoutRef.current) {
             renderBindGroupLayoutRef.current = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' }},
                    { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' }}, 
                ]
            });
        }

        device.pushErrorScope('validation');

        const format = (navigator as any).gpu.getPreferredCanvasFormat();
        const shaderModule = device.createShaderModule({ label: 'Main', code });
        const compilationInfo = await shaderModule.getCompilationInfo();
        
        // Check Mount State after await
        if (!isMountedRef.current || !deviceRef.current) {
            // Siphon off the error scope to avoid warnings even if we're bailing
            device.popErrorScope().catch(() => {});
            return;
        }

        if (compilationInfo.messages.length > 0) {
            let hasError = false;
            for (const msg of compilationInfo.messages) {
                if (msg.type === 'error') {
                    hasError = true;
                    onError({ type: 'compilation', message: getErrorMessage(msg.message), lineNum: msg.lineNum, linePos: msg.linePos });
                }
            }
            if (hasError) {
                await device.popErrorScope().catch(() => {}); 
                return;
            }
        }
        onClearError();

        const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayoutRef.current] });
        const renderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayoutRef.current] });

        // Compute Pipelines
        const cPredict = device.createComputePipeline({ layout: computePipelineLayout, compute: { module: shaderModule, entryPoint: 'compute_predict' }});
        const cSolve = device.createComputePipeline({ layout: computePipelineLayout, compute: { module: shaderModule, entryPoint: 'compute_solve' }});
        const cIntegrate = device.createComputePipeline({ layout: computePipelineLayout, compute: { module: shaderModule, entryPoint: 'compute_integrate' }});

        if (!isMountedRef.current || !deviceRef.current) return;
        
        computePredictRef.current = cPredict;
        computeSolveRef.current = cSolve;
        computeIntegrateRef.current = cIntegrate;

        // Render Pipeline
        const renderPipeline = device.createRenderPipeline({
            layout: renderPipelineLayout,
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
            primitive: { topology: 'line-list' },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });
        renderPipelineRef.current = renderPipeline;

        // Re-create BindGroups with correct layouts
        bindGroupCreation(device);

        setNeedsReset(true);
        
        const valError = await device.popErrorScope();
        if (valError) {
            onError({ type: 'validation', message: getErrorMessage(valError) });
        }

      } catch (e) {
          // If we caught an error, it might be a validation error from popErrorScope OR a synchronous create error
          // Try to pop error scope if it was pushed
          try {
             const valError = await device.popErrorScope();
             if (valError) onError({ type: 'validation', message: getErrorMessage(valError) });
             else onError({ type: 'runtime', message: getErrorMessage(e) });
          } catch(inner) {
              // If popErrorScope fails, the device is likely lost
              onError({ type: 'runtime', message: getErrorMessage(e) });
          }
      }
  };

  const bindGroupCreation = (device: GPUDevice) => {
      // Compute Bind Groups (RW)
      computeBindGroupEvenRef.current = device.createBindGroup({
          layout: computeBindGroupLayoutRef.current,
          entries: [
              { binding: 0, resource: { buffer: uniformBufferEvenRef.current } },
              { binding: 1, resource: { buffer: particleBufferRef.current } },
          ]
      });

      computeBindGroupOddRef.current = device.createBindGroup({
          layout: computeBindGroupLayoutRef.current,
          entries: [
              { binding: 0, resource: { buffer: uniformBufferOddRef.current } },
              { binding: 1, resource: { buffer: particleBufferRef.current } },
          ]
      });

      // Render Bind Groups (Read-Only)
      // Uses Binding 2 for Particle Buffer
      renderBindGroupEvenRef.current = device.createBindGroup({
          layout: renderBindGroupLayoutRef.current,
          entries: [
              { binding: 0, resource: { buffer: uniformBufferEvenRef.current } },
              { binding: 2, resource: { buffer: particleBufferRef.current } },
          ]
      });
  }

  useEffect(() => {
    isMountedRef.current = true;
    const initWebGPU = async () => {
      try {
        const gpu = (navigator as any).gpu;
        if (!gpu) { setIsSupported(false); return; }
        
        const adapter = await gpu.requestAdapter();
        if (!isMountedRef.current) return;
        
        const device = await adapter.requestDevice();
        // Ensure we haven't unmounted while waiting for device
        if (!isMountedRef.current) {
            device.destroy();
            return;
        }
        
        deviceRef.current = device;

        // Handle Device Lost
        device.lost.then((info: any) => {
            console.error("Device lost", info);
            if (isMountedRef.current) {
                onError({ type: 'runtime', message: `GPU Device Lost: ${info.message || info.reason || 'Unknown Reason'}` });
            }
        });

        // Handle Uncaptured Errors (Validation Errors during Render)
        // This catches "Buffer too small" and other runtime pipeline issues
        device.addEventListener('uncapturederror', (event: any) => {
            if (isMountedRef.current) {
                onError({ type: 'validation', message: getErrorMessage(event.error) });
            }
        });

        const canvas = canvasRef.current;
        const context = canvas?.getContext('webgpu') as any;
        contextRef.current = context;
        const format = gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'opaque' });

        // 1. Uniform Buffers
        const uniformBufferEven = device.createBuffer({ size: TOTAL_BUFFER_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const uniformBufferOdd = device.createBuffer({ size: TOTAL_BUFFER_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        uniformBufferEvenRef.current = uniformBufferEven;
        uniformBufferOddRef.current = uniformBufferOdd;

        // 2. Particle Buffer
        const particleStride = 64;
        const particleBufferSize = NUM_PARTICLES * particleStride;
        const particleBuffer = device.createBuffer({
            size: particleBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, 
        });
        particleBufferRef.current = particleBuffer;

        // 3. Index Buffer
        const indexObj = createIndexBuffer(device);
        indexBufferRef.current = indexObj.buffer;

        await compilePipeline(device, shaderCode, context);
        
        if (isMountedRef.current) {
            requestRef.current = requestAnimationFrame(render);
        }
      } catch (e) {
          console.error("Init Error", e);
          if (isMountedRef.current) {
             onError({ type: 'runtime', message: `Initialization Failed: ${getErrorMessage(e)}` });
          }
      }
    };
    initWebGPU();
    
    return () => { 
        isMountedRef.current = false; 
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
        
        // Critical: Destroy device to release external instance reference
        if (deviceRef.current) {
            deviceRef.current.destroy();
            deviceRef.current = null;
        }
    };
  }, []);

  useEffect(() => {
      if (deviceRef.current && contextRef.current) {
          compilePipeline(deviceRef.current, shaderCode, contextRef.current);
      }
  }, [shaderCode]);

  const render = (time: number) => {
    // Safety check: Don't render if unmounted or device is gone
    if (!isMountedRef.current || !deviceRef.current) return;

    try {
        const device = deviceRef.current;
        const context = contextRef.current;
        if (!device || !context || !renderPipelineRef.current || !uniformBufferEvenRef.current) {
             requestRef.current = requestAnimationFrame(render);
             return;
        }

        // Determine Timing
        let elapsedTime = (time - startTimeRef.current) * 0.001;
        let dt = 0.016;
        
        // Recording Logic
        if (isRecordingRef.current && recordingConfigRef.current) {
            // Deterministic Step
            const fps = recordingConfigRef.current.fps;
            dt = 1.0 / fps;
            elapsedTime = recordedFramesRef.current * dt;
            
            // Update Time Limit UI
            const limit = recordingConfigRef.current.duration;
            onRecordProgress(true, limit - elapsedTime);
            
            if (elapsedTime >= limit) {
                 stopVideo();
            }
        } else {
            // Real-time Resize
            const canvas = canvasRef.current;
            if(canvas && !isRecordingRef.current) { 
                 const dw = canvas.clientWidth * window.devicePixelRatio;
                 const dh = canvas.clientHeight * window.devicePixelRatio;
                 if(canvas.width !== dw || canvas.height !== dh) {
                     canvas.width = dw; canvas.height = dh;
                 }
            }
        }
        const canvas = canvasRef.current;

        // --- Update Uniform Data ---
        const uniformData = new Float32Array(TOTAL_BUFFER_SIZE / 4);
        
        // Camera
        let cameraTheta = cameraState.current.theta;
        let cameraPhi = cameraState.current.phi;
        
        // Auto-Orbit for Video
        if (isRecordingRef.current && recordingConfigRef.current?.shotType === 'orbit') {
            cameraTheta = elapsedTime * 0.5;
        }

        const cx = cameraState.current.radius * Math.cos(cameraPhi) * Math.sin(cameraTheta);
        const cy = cameraState.current.radius * Math.sin(cameraPhi);
        const cz = cameraState.current.radius * Math.cos(cameraPhi) * Math.cos(cameraTheta);

        uniformData[0] = canvas?.width || 800; uniformData[1] = canvas?.height || 600; uniformData[2] = elapsedTime;
        uniformData[3] = dt;
        uniformData[4] = cx; uniformData[5] = cy; uniformData[6] = cz;
        uniformData[8] = mouseState.current.x / (canvas?.width || 1); uniformData[9] = mouseState.current.y / (canvas?.height || 1); uniformData[10] = mouseState.current.isDown;
        
        // Params
        writeParamsToBuffer(uniformData, paramsRef.current, layout);
        
        // Audio
        let vol = 0;
        if (analyzerRef.current && audioDataArrayRef.current) {
             analyzerRef.current.getByteFrequencyData(audioDataArrayRef.current);
             vol = audioDataArrayRef.current[0] / 255.0; 
        }
        uniformData[24] = vol; 

        // Reset Logic
        if (needsReset) {
            uniformData[22] = 1.0; 
            setNeedsReset(false);
        } else {
            uniformData[22] = 0.0;
        }

        // Write to Buffers
        // NOTE: If device is lost, queue operations might throw.
        uniformData[28] = 0.0; // Phase 0
        device.queue.writeBuffer(uniformBufferEvenRef.current, 0, uniformData);

        uniformData[28] = 1.0; // Phase 1
        device.queue.writeBuffer(uniformBufferOddRef.current, 0, uniformData);

        const commandEncoder = device.createCommandEncoder();

        // --- COMPUTE PASSES ---
        if (computePredictRef.current && computeBindGroupEvenRef.current) {
            const pass = commandEncoder.beginComputePass();
            
            // 1. Predict
            pass.setPipeline(computePredictRef.current);
            pass.setBindGroup(0, computeBindGroupEvenRef.current); 
            pass.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / 64));
            
            // 2. Solve (Iterations)
            if (computeSolveRef.current && !needsReset) { 
                pass.setPipeline(computeSolveRef.current);
                const iterations = 8;
                for(let i=0; i<iterations; i++) {
                    // RED STEP (Phase 0)
                    pass.setBindGroup(0, computeBindGroupEvenRef.current);
                    pass.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / 64));

                    // BLACK STEP (Phase 1)
                    pass.setBindGroup(0, computeBindGroupOddRef.current);
                    pass.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / 64));
                }
            }

            // 3. Integrate
            pass.setPipeline(computeIntegrateRef.current);
            pass.setBindGroup(0, computeBindGroupEvenRef.current);
            pass.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / 64));
            
            pass.end();
        }

        // --- RENDER PASS ---
        const depthTexture = device.createTexture({
            size: [canvas!.width, canvas!.height],
            format: 'depth24plus',
            usage: 0x10 // RENDER_ATTACHMENT
        });

        const textureView = context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: textureView, clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });
        
        renderPass.setPipeline(renderPipelineRef.current);
        renderPass.setBindGroup(0, renderBindGroupEvenRef.current); // Use Render Bind Group (RO)
        renderPass.setIndexBuffer(indexBufferRef.current, 'uint32');
        const indexCount = ((GRID_W - 1) * GRID_H + (GRID_H - 1) * GRID_W) * 2;
        renderPass.drawIndexed(indexCount);
        
        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);

        // Manual Frame Trigger for Recording
        if (isRecordingRef.current && streamTrackRef.current && streamTrackRef.current.requestFrame) {
            streamTrackRef.current.requestFrame();
            recordedFramesRef.current++;
        }

        requestRef.current = requestAnimationFrame(render);
    } catch (e) {
        console.error("Render Loop Error", e);
        // Only report runtime errors if mounted
        if (isMountedRef.current) {
            onError({ type: 'runtime', message: getErrorMessage(e) });
        }
    }
  };

  const handlePointerDown = (e: React.PointerEvent) => { 
      canvasRef.current?.setPointerCapture(e.pointerId); 
      cameraState.current.isDragging = true; 
      cameraState.current.lastX = e.clientX; 
      cameraState.current.lastY = e.clientY; 
      mouseState.current.isDown = 1.0; 
  };
  const handlePointerMove = (e: React.PointerEvent) => {
    if (cameraState.current.isDragging) {
      const dx = e.clientX - cameraState.current.lastX; const dy = e.clientY - cameraState.current.lastY;
      cameraState.current.lastX = e.clientX; cameraState.current.lastY = e.clientY;
      cameraState.current.theta -= dx * 0.01; cameraState.current.phi += dy * 0.01;
      cameraState.current.phi = Math.max(-1.5, Math.min(1.5, cameraState.current.phi));
    }
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
        mouseState.current.x = (e.clientX - rect.left) / rect.width;
        mouseState.current.y = (e.clientY - rect.top) / rect.height;
    }
  };
  const handlePointerUp = (e: React.PointerEvent) => { cameraState.current.isDragging = false; mouseState.current.isDown = 0.0; };
  const handleWheel = (e: React.WheelEvent) => { cameraState.current.radius += e.deltaY * 0.01; };

  if (!isSupported) return <div className="text-red-500">WebGPU Not Supported</div>;

  return (
    <>
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair touch-none bg-black" onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onWheel={handleWheel} />
        <ParamsControlPanel params={params} setParams={setParams} description={description} />
    </>
  );
});

export default WebGPURenderer;
