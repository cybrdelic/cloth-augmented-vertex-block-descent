# WebGPU React Boilerplate

A robust, minimal, and reusable foundation for building high-performance 3D graphics applications using **WebGPU** and **React**.

## 🚀 Why Use This Boilerplate?

Starting with WebGPU is notoriously difficult. You often need 100+ lines of code just to clear the screen to black. This boilerplate solves the "Setup Fatigue" by abstracting the engine logic so you can focus purely on creativity.

### Distinguishing Features

1.  **⚡ Hot-Reloading Shader Pipeline**
    *   **The Problem:** In raw WebGPU, changing a shader usually requires reloading the page or writing complex recompilation logic.
    *   **Our Solution:** This project separates the *Engine* (React) from the *Shader* (`constants.ts`). When you edit the WGSL string, the app automatically recompiles *only* the shader module without destroying the WebGPU context, giving you instant feedback.

2.  **🛡️ Smart Error Overlay**
    *   **The Problem:** WebGPU errors in the browser console are often cryptic or generic (e.g., `GPUPipelineError`).
    *   **Our Solution:** We parse the compilation logs to find exact line numbers and error messages, displaying them in a copy-pasteable UI overlay directly on top of your canvas.

3.  **📦 Pre-Aligned Uniform Buffer**
    *   **The Problem:** Sending data from JS to WGSL requires strict byte alignment (16-byte chunks). One wrong byte causes the whole shader to break or data to drift.
    *   **Our Solution:** We provide a pre-configured `Uniforms` struct and a matching `Float32Array` layout for Time, Resolution, Camera, and Mouse. It "just works" out of the box.

4.  **⚛️ React Lifecycle Management**
    *   **The Problem:** WebGPU resources (Buffers, Textures) need to be manually destroyed. React's `useEffect` often creates race conditions where the GPU device is lost during hot-module replacement.
    *   **Our Solution:** Robust cleanup logic ensures the GPU device is properly destroyed and recreated when components unmount or re-render, preventing memory leaks and "Device Lost" crashes.

5.  **📷 Built-in Orbit Controls**
    *   **The Problem:** Writing a 3D camera from scratch involves complex matrix math.
    *   **Our Solution:** A touch-friendly, spherical orbit camera is built-in. It calculates the correct position vectors and sends them to the shader automatically.

---

## 📂 Project Structure

*   **`constants.ts`** (The Shader):
    *   **Edit this file to change visuals.**
    *   Contains the WGSL shader code string.
    *   Defines the Uniform structure.
*   **`components/FireRenderer.tsx`** (The Engine):
    *   Initializes the WebGPU Adapter and Device.
    *   Configures the Render Pipeline.
    *   Manages the Render Loop (`requestAnimationFrame`).
    *   Handles User Input (Mouse/Touch) and updates Uniforms.
*   **`components/UIComponents.tsx`**:
    *   Renders the error overlay and documentation panels.

---

## 🎮 How to Use

### 1. Writing Shaders
Open `constants.ts`. You will see the `BOILERPLATE_SHADER_WGSL` constant. This is your playground.

The boilerplate provides a standard **Raymarching** setup (Camera Ray generation) by default, but you can replace the contents of `fs_main` with any pixel shader logic you want.

### 2. The Uniform Buffer
The boilerplate automatically sends data to the GPU in a strict byte-aligned format. In your WGSL shader, it looks like this:

```wgsl
struct Uniforms {
  resolution: vec2f,      // The canvas width and height
  time: f32,              // Time in seconds since load
  _pad1: f32,             // Padding (WebGPU requires 16-byte alignment chunks)
  cameraPos: vec4f,       // Camera X, Y, Z coordinates
  mouse: vec4f,           // Mouse X, Y, ClickState (0/1), Scroll
};

@group(0) @binding(0) var<uniform> u: Uniforms;
```

**Usage Examples:**
*   **Animate based on time:** `sin(u.time)`
*   **Get corrected UVs:** `let uv = input.uv * u.resolution / u.resolution.y;`
*   **Interactive color:** `if (u.mouse.z > 0.0) { color = vec3f(1.0, 0.0, 0.0); }`

### 3. Adding New Uniforms
To add custom data (e.g., a "Speed" slider):
1.  **Update WGSL (`constants.ts`)**: Add the field to the `struct Uniforms`. **Note:** You must respect WebGPU padding rules (16-byte alignment is safest).
2.  **Update JS (`FireRenderer.tsx`)**:
    *   Increase the `device.createBuffer` size (currently 48 bytes).
    *   In the `render` loop, update the `Float32Array` writing logic to include your new value at the correct index.

---

## ⚠️ Common Errors

**1. "Device Lost"**
*   **Cause:** The GPU crashed, took too long to compute a frame (Timeout), or the browser tab was sleeping.
*   **Fix:** Refresh the page. Optimize your shader loops (reduce iterations).

**2. "Buffer size not multiple of 16"**
*   **Cause:** WebGPU uniform buffers prefer sizes divisible by 16 bytes.
*   **Fix:** Add `_pad` variables in your struct or Float32Array to align data.

**3. "Validation Error"**
*   **Cause:** Type mismatch in WGSL (e.g., multiplying a `vec3` by a `float` without explicit casting).
*   **Fix:** WGSL is strict! Use `vec3f(1.0)` instead of `1.0` when doing vector math. Check the Error Overlay for line numbers.
