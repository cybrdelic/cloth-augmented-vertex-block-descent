
export const BOILERPLATE_SHADER_WGSL = `
struct Uniforms {
  resolution: vec2f,
  time: f32,
  dt: f32,
  cameraPos: vec4f,
  mouse: vec4f,
  
  // Params
  gravity: f32,
  stiffness: f32, // Compliance (0 = stiff)
  mouseRadius: f32,
  damping: f32, 
  
  baseColor: vec3f,
  _pad_color: f32,
  
  lightAz: f32,
  lightEl: f32,
  isRendering: f32, // 0=Sim, 1=Reset
  aberrationStrength: f32,
  
  audio: vec4f,
  phase: f32, // 0.0 or 1.0 for Graph Coloring
  
  // Padding to reach 128 bytes exactly.
  // We use explicit f32s to avoid vec3 alignment (16 bytes) forcing us to 144 bytes.
  _pad_a: f32,
  _pad_b: f32,
  _pad_c: f32,
};

struct Particle {
  pos: vec4f,     // xyz, w=mass (0 = fixed)
  old_pos: vec4f, // xyz, w=padding
  vel: vec4f,     // xyz, w=padding
  uv: vec2f,      // texture coords
  _pad: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Split bindings for Read-Write (Compute) and Read-Only (Vertex)
// To avoid WGSL compilation errors (duplicate binding), we use different slots.
// Binding 1: Compute (RW)
// Binding 2: Vertex (RO)
@group(0) @binding(1) var<storage, read_write> particles_rw: array<Particle>;
@group(0) @binding(2) var<storage, read> particles_ro: array<Particle>;

// Constants for the Grid
const GRID_W: u32 = 128u;
const GRID_H: u32 = 128u;
const REST_DIST: f32 = 0.05;

fn get_idx(x: u32, y: u32) -> u32 {
    if (x >= GRID_W || y >= GRID_H) { return 0xFFFFFFFFu; }
    return y * GRID_W + x;
}

// --- COMPUTE STEP 1: PREDICTION ---
// Apply gravity and predict next position: x_pred = x + v * dt
@compute @workgroup_size(64)
fn compute_predict(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&particles_rw)) { return; }
    
    // Reset Logic
    if (u.isRendering > 0.5) {
        let x = f32(idx % GRID_W);
        let y = f32(idx / GRID_W);
        // Center the cloth
        let px = (x - f32(GRID_W)*0.5) * REST_DIST;
        let py = 4.0; 
        let pz = (y - f32(GRID_H)*0.5) * REST_DIST;
        
        particles_rw[idx].pos = vec4f(px, py, pz, 1.0);
        particles_rw[idx].old_pos = vec4f(px, py, pz, 1.0);
        particles_rw[idx].vel = vec4f(0.0);
        
        // Pin corners
        if ((x == 0.0 || x == f32(GRID_W)-1.0) && y == 0.0) {
             particles_rw[idx].pos.w = 0.0; // Infinite mass (Fixed)
        } else {
             particles_rw[idx].pos.w = 1.0; // Dynamic
        }
        return;
    }

    var p = particles_rw[idx];
    if (p.pos.w == 0.0) { return; } // Fixed particle

    // Apply External Forces (Gravity + Audio Wind)
    let gravity = vec3f(0.0, -9.8 * u.gravity, 0.0);
    // Simple wind noise based on position and time
    let noise = sin(p.pos.x * 2.0 + u.time * 3.0) * cos(p.pos.z * 1.5 + u.time);
    let wind = vec3f(0.0, 0.5, 0.5) * u.audio.x * 30.0 * noise;
    
    // Mouse Interaction (Sphere Repeller)
    // Project mouse ray roughly? For now just use mouse XY as a probe at z=0
    
    p.vel += vec4f((gravity + wind) * u.dt, 0.0);
    p.old_pos = p.pos; // Store current as old
    p.pos += p.vel * u.dt; // Prediction
    
    // Simple Floor Collision Prediction
    if (p.pos.y < -3.0) {
        p.pos.y = -3.0;
        p.vel.x *= 0.9; // Friction
        p.vel.z *= 0.9;
    }

    particles_rw[idx] = p;
}

// --- COMPUTE STEP 2: SOLVE (VBD) ---
@compute @workgroup_size(64)
fn compute_solve(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&particles_rw)) { return; }
    
    let x = idx % GRID_W;
    let y = idx / GRID_W;
    
    // Graph Coloring: Checkerboard Pattern
    // (x + y) % 2 == 0 -> Red
    // (x + y) % 2 == 1 -> Black
    let phase = u32(u.phase); 
    if ((x + y) % 2u != phase) { return; }

    var p = particles_rw[idx];
    if (p.pos.w == 0.0) { return; } // Fixed

    var gradient_sum = vec3f(0.0);
    var mass_sum = 0.0;
    
    // Neighbors: Right, Left, Down, Up
    let neighbors = array<vec2i, 4>(
        vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1)
    );

    // Solve Constraints
    for (var i = 0; i < 4; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        
        if (nx >= 0 && nx < i32(GRID_W) && ny >= 0 && ny < i32(GRID_H)) {
            let n_idx = u32(ny) * GRID_W + u32(nx);
            // Read neighbor position. Since we color the graph, neighbors are strictly "Read-Only"
            // (They belong to the other phase, so they aren't changing in this dispatch)
            let n_pos = particles_rw[n_idx].pos.xyz; 
            
            let dir = p.pos.xyz - n_pos;
            let d = length(dir);
            if (d > 0.0001) {
                let n_dir = dir / d;
                let constraint = d - REST_DIST;
                
                // VBD/XPBD Term
                // We want to pull p.pos towards (n_pos + rest_dist * n_dir)
                // With stiffness alpha = u.stiffness
                // Correction = -C * gradC / (w + alpha)
                // For distance constraint, gradC is n_dir.
                
                let alpha = u.stiffness; // Compliance
                let w = 1.0; // Mass inverse (simplified)
                
                let lambda = -constraint / (w + w + alpha); // w+w because 2 particles? VBD solves local.
                let correction = lambda * n_dir; 
                
                // Accumulate
                gradient_sum += correction;
                mass_sum += 1.0; // Count constraint
            }
        }
    }

    // Apply accumulated gradients
    // In strict VBD we solve the system. Here we average the corrections.
    if (mass_sum > 0.0) {
        // Fix for "cannot assign to swizzle":
        let newPos = p.pos.xyz + (gradient_sum * 1.5); // 1.5 is Over-relaxation factor
        p.pos = vec4f(newPos, p.pos.w);
    }
    
    // Mouse Interaction (Repel)
    let mPos = vec3f(u.mouse.x * 20.0 - 10.0, u.mouse.y * 20.0 - 10.0, 0.0);
    if (u.mouse.z > 0.0) {
       let mDir = p.pos.xyz - mPos;
       let md = length(mDir);
       let repelRadius = 3.0 * u.mouseRadius;
       if (md < repelRadius) {
           let push = normalize(mDir) * (repelRadius - md);
           p.pos = vec4f(p.pos.xyz + push, p.pos.w);
       }
    }

    // Floor Constraint (Hard)
    if (p.pos.y < -3.0) {
        p.pos = vec4f(p.pos.x, -3.0, p.pos.z, p.pos.w);
    }

    particles_rw[idx] = p;
}

// --- COMPUTE STEP 3: INTEGRATE ---
// Finalize velocity: v = (x - x_old) / dt
@compute @workgroup_size(64)
fn compute_integrate(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&particles_rw)) { return; }
    
    var p = particles_rw[idx];
    if (p.pos.w == 0.0) { return; }

    let vel_new = (p.pos.xyz - p.old_pos.xyz) / u.dt;
    p.vel = vec4f(vel_new * u.damping, 0.0);
    
    particles_rw[idx] = p;
}


// --- VISUALIZATION: VERTEX SHADER ---
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // Use Read-Only buffer for Vertex Stage (Binding 2)
  let p = particles_ro[vertexIndex];
  
  var output: VertexOutput;
  
  // Camera Transform
  let ro = u.cameraPos.xyz;
  let ta = vec3f(0.0, 0.0, 0.0);
  let ww = normalize(ta - ro); // Z Axis (Forward)
  let uu = normalize(cross(ww, vec3f(0.0, 1.0, 0.0))); // X Axis (Right)
  let vv = normalize(cross(uu, ww)); // Y Axis (Up)
  
  let pos = p.pos.xyz;
  
  // Manual ViewMatrix (LookAt)
  let camRel = pos - ro;
  let x = dot(camRel, uu);
  let y = dot(camRel, vv);
  let z = dot(camRel, ww); // Distance along view vector (positive is in front if camera looks at target)

  // Standard Projection (Perspective)
  // FOV ~ 60 degrees. Tan(30) ~ 0.577. 1/tan(30) ~ 1.732
  let fovScale = 1.7; 
  let aspect = u.resolution.x / u.resolution.y;
  
  // NOTE: In Clip Space, Z should be W.
  // x_ndc = x_clip / w_clip
  // So we set w_clip = z.
  
  if (z > 0.1) {
      // Perspective Divide happening automatically by GPU when we put z in w.
      // x_clip = x * fov / aspect
      // y_clip = y * fov
      // z_clip = z (or remapped z for depth buffer)
      // w_clip = z
      
      output.position = vec4f(x * fovScale / aspect, y * fovScale, z * 0.01, z); 
  } else {
      output.position = vec4f(0.0, 0.0, 0.0, 0.0); // Clip behind camera
  }
  
  // Color based on Velocity
  let speed = length(p.vel.xyz);
  let colorA = u.baseColor;
  let colorB = vec3f(1.0, 1.0, 1.0);
  let stress = min(speed * 0.2, 1.0);
  
  output.color = vec4f(mix(colorA, colorB, stress), 1.0);
  
  return output;
}

@fragment
fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
  return color;
}
`;