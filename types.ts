
export interface ShaderError {
  type: 'compilation' | 'validation' | 'runtime';
  message: string;
  lineNum?: number;
  linePos?: number;
}

export type ParamType = 'float' | 'color' | 'vec3';

export interface BaseParam {
  id: string;
  label: string;
  type: ParamType;
}

export interface FloatParam extends BaseParam {
  type: 'float';
  value: number;
  min: number;
  max: number;
  step?: number;
}

export interface ColorParam extends BaseParam {
  type: 'color';
  value: [number, number, number]; // RGB 0-1
}

export interface Vec3Param extends BaseParam {
  type: 'vec3';
  value: [number, number, number];
}

export type ShaderParam = FloatParam | ColorParam | Vec3Param;

export interface UniformLayout {
  size: number; // Total buffer size in bytes
  offsetMap: Record<string, number>; // Map of param ID to byte offset
}

export type ShotType = 'orbit' | 'sweep' | 'dolly' | 'breathing' | 'chaos';

export interface VideoConfig {
  duration: number; // Seconds
  fps: number;
  bitrate: number; // Mbps
  shotType: ShotType; // Camera Movement
  orchestrate: boolean; // Auto-animate scene params?
  postProcess: {
      grain: number;
      aberration: number;
  };
  format: 'webm' | 'mp4';
}
