"""CarePriority Gov - 3D Command Center Hero Scene
=================================================
Procedural 3D hero scene built with Three.js (pinned cdnjs r128),
rendered exclusively in the Command Center hero banner area inside
Streamlit's isolated iframe component (st.components.v1.html).

Key Architecture & Guardrails:
1. Procedural 3D Geometry:
   - Floating translucent glass care-cross (MeshPhysicalMaterial / MeshStandardMaterial).
   - Thin glowing orbital rings (TorusGeometry).
   - Orbiting translucent teal/mint polyhedra with gentle sinusoidal bobbing.
   - Smooth continuous multi-axis rotation and subtle mouse parallax.
2. Resilience & Fallbacks (Never an empty box):
   - Polished static SVG/CSS vector hero rendered as baseline.
   - If Three.js CDN fails to load (offline / firewalled), WebGL context creation fails,
     or user has enabled OS reduced-motion ('prefers-reduced-motion: reduce'),
     the static SVG hero gracefully remains visible without layout shift.
3. Performance & Resource Lifecycle:
   - devicePixelRatio capped at 2 to conserve GPU cycles on 4K / retina screens.
   - Tab visibility detection (document.hidden) pauses rendering loop when inactive.
   - Clean disposal of WebGL renderer, geometries, materials, and listeners on unmount.
4. Method & Limits:
   - Method: Streamlit Component iframe embedding with client-side WebGL canvas.
   - Limits: Runs within sandboxed iframe; cannot project outside its component bounds;
     depends on client GPU / WebGL; gracefully falls back to SVG if offline or restricted.
"""

from typing import Optional
import streamlit as st
import streamlit.components.v1 as components


def get_hero_3d_html(height: int = 240) -> str:
    """Generates the self-contained HTML/JS bundle for the 3D Command Center hero scene."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CarePriority Gov 3D Hero</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }}
    html, body {{
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: transparent;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    #hero-container {{
      position: relative;
      width: 100%;
      height: {height}px;
      border-radius: 18px;
      overflow: hidden;
      background: radial-gradient(130% 120% at 50% 10%, #081B33 0%, #050E1F 75%, #030813 100%);
      border: 1px solid rgba(102, 227, 196, 0.22);
      box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 32px;
    }}

    /* Subtle Animated Perspective Grid Line in Background */
    #hero-container::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-image: 
        linear-gradient(rgba(102, 227, 196, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(102, 227, 196, 0.04) 1px, transparent 1px);
      background-size: 24px 24px;
      pointer-events: none;
      z-index: 1;
    }}

    /* Glow Vignette */
    #hero-container::after {{
      content: "";
      position: absolute;
      top: -40%;
      right: 5%;
      width: 450px;
      height: 450px;
      background: radial-gradient(circle, rgba(102, 227, 196, 0.12) 0%, rgba(56, 189, 248, 0.06) 40%, transparent 70%);
      border-radius: 50%;
      pointer-events: none;
      z-index: 2;
    }}

    /* Text & Status Overlay (Left Aligned, never obscured by 3D) */
    .hero-content {{
      position: relative;
      z-index: 10;
      max-width: 58%;
    }}
    .hero-badge-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      background: rgba(102, 227, 196, 0.12);
      border: 1px solid rgba(102, 227, 196, 0.35);
      border-radius: 20px;
      color: #66E3C4;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .hero-badge-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #00D68F;
      box-shadow: 0 0 8px #00D68F;
      animation: pulse-dot 2s infinite ease-in-out;
    }}
    @keyframes pulse-dot {{
      0%, 100% {{ transform: scale(1); opacity: 0.8; }}
      50% {{ transform: scale(1.3); opacity: 1; }}
    }}
    .hero-title {{
      color: #FFFFFF;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.02em;
      line-height: 1.2;
      margin-bottom: 6px;
    }}
    .hero-title-accent {{
      background: linear-gradient(135deg, #66E3C4 0%, #38BDF8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
      color: #94A3B8;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 12px;
    }}
    .hero-tech-strip {{
      display: flex;
      gap: 16px;
      align-items: center;
      font-size: 11px;
      color: #CBD5E1;
    }}
    .hero-tech-tag {{
      display: flex;
      align-items: center;
      gap: 5px;
      opacity: 0.85;
    }}

    /* 3D Canvas Viewport (Right Aligned) */
    #canvas-container {{
      position: absolute;
      right: 12px;
      top: 0;
      width: 360px;
      height: {height}px;
      z-index: 5;
      pointer-events: auto;
    }}
    #three-canvas {{
      width: 100%;
      height: 100%;
      display: block;
      opacity: 0;
      transition: opacity 0.6s ease;
    }}

    /* Polished Static SVG Hero Fallback */
    #static-hero-fallback {{
      position: absolute;
      right: 32px;
      top: 50%;
      transform: translateY(-50%);
      width: 220px;
      height: 200px;
      z-index: 4;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 1;
      transition: opacity 0.5s ease;
    }}
    .svg-cross-pulse {{
      animation: static-float 4s ease-in-out infinite;
      filter: drop-shadow(0 0 16px rgba(102, 227, 196, 0.45));
    }}
    @keyframes static-float {{
      0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
      50% {{ transform: translateY(-6px) rotate(2deg); }}
    }}
    .ring-rotate {{
      transform-origin: center;
      animation: ring-spin 16s linear infinite;
    }}
    @keyframes ring-spin {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <div id="hero-container">
    <!-- Left Text & Telemetry Strip -->
    <div class="hero-content">
      <div class="hero-badge-pill">
        <span class="hero-badge-dot"></span>
        <span id="hero-engine-label">3D Telemetry Active</span>
      </div>
      <div class="hero-title">
        CarePriority <span class="hero-title-accent">Gov</span> Command Center
      </div>
      <div class="hero-subtitle">
        Dynamic Clinical Severity OPD Triage for Overloaded Government Hospitals.<br>
        20x Asymmetric Cost Matrix &bull; Zero Missed Emergencies &bull; Hard Safety Guardrails.
      </div>
      <div class="hero-tech-strip">
        <div class="hero-tech-tag"><span>🛡️</span> 20x Guardrails</div>
        <div class="hero-tech-tag"><span>⚡</span> &lt;60s Rapid Intake</div>
        <div class="hero-tech-tag"><span>🔒</span> 100% Offline Privacy</div>
      </div>
    </div>

    <!-- Static Polished SVG Hero Fallback (Always present; never empty box) -->
    <div id="static-hero-fallback">
      <svg class="svg-cross-pulse" width="180" height="180" viewBox="0 0 180 180" fill="none" xmlns="http://www.w3.org/2000/svg">
        <!-- Orbital Rings -->
        <circle class="ring-rotate" cx="90" cy="90" r="76" stroke="rgba(56, 189, 248, 0.35)" stroke-width="1.5" stroke-dasharray="8 6" />
        <ellipse cx="90" cy="90" rx="82" ry="38" stroke="rgba(102, 227, 196, 0.4)" stroke-width="1.2" transform="rotate(-28 90 90)" />
        <ellipse cx="90" cy="90" rx="82" ry="38" stroke="rgba(102, 227, 196, 0.25)" stroke-width="1.2" transform="rotate(38 90 90)" />
        
        <!-- Central Glowing Care-Cross -->
        <defs>
          <linearGradient id="crossGrad" x1="20" y1="20" x2="160" y2="160" gradientUnits="userSpaceOnUse">
            <stop stop-color="#66E3C4" />
            <stop offset="0.6" stop-color="#38BDF8" />
            <stop offset="1" stop-color="#087F8C" />
          </linearGradient>
          <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        
        <!-- Glass Care Cross Body -->
        <g filter="url(#neonGlow)">
          <!-- Vertical bar -->
          <rect x="74" y="32" width="32" height="116" rx="10" fill="url(#crossGrad)" fill-opacity="0.88" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
          <!-- Horizontal bar -->
          <rect x="32" y="74" width="116" height="32" rx="10" fill="url(#crossGrad)" fill-opacity="0.88" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
          <!-- Cross inner highlight -->
          <rect x="78" y="36" width="8" height="108" rx="4" fill="rgba(255,255,255,0.35)" />
        </g>
        
        <!-- Floating Orbiting Particles -->
        <circle cx="28" cy="54" r="5" fill="#66E3C4" opacity="0.8" />
        <circle cx="152" cy="120" r="4" fill="#38BDF8" opacity="0.75" />
        <polygon points="145,45 152,58 138,58" fill="#66E3C4" opacity="0.7" />
        <polygon points="35,135 42,148 28,148" fill="#38BDF8" opacity="0.6" />
      </svg>
    </div>

    <!-- Three.js Canvas Container -->
    <div id="canvas-container">
      <canvas id="three-canvas"></canvas>
    </div>
  </div>

  <!-- Pinned Three.js from cdnjs (r128) -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    (function() {{
      var canvas = document.getElementById('three-canvas');
      var container = document.getElementById('canvas-container');
      var fallback = document.getElementById('static-hero-fallback');
      var engineLabel = document.getElementById('hero-engine-label');

      // Check 1: Reduced Motion Preference
      var prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (prefersReducedMotion) {{
        if (engineLabel) engineLabel.textContent = "Static Hero (Reduced Motion)";
        return; // Retain polished static SVG hero
      }}

      // Check 2: WebGL Support
      function checkWebGL() {{
        try {{
          var c = document.createElement('canvas');
          return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl')));
        }} catch(e) {{
          return false;
        }}
      }}

      if (!window.THREE || !checkWebGL()) {{
        if (engineLabel) engineLabel.textContent = "Static Vector Hero";
        return; // Retain polished static SVG hero
      }}

      // Initialize Procedural Three.js Scene
      var width = container.clientWidth || 360;
      var height = container.clientHeight || {height};
      var animId = null;
      var isPaused = false;

      var scene = new THREE.Scene();
      var camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
      camera.position.set(0, 0, 7.2);

      var renderer = null;
      try {{
        renderer = new THREE.WebGLRenderer({{
          canvas: canvas,
          alpha: true,
          antialias: true,
          powerPreference: 'low-power'
        }});
      }} catch (err) {{
        if (engineLabel) engineLabel.textContent = "Static Vector Hero";
        return;
      }}

      // Cap devicePixelRatio at 2
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(width, height);
      renderer.setClearColor(0x000000, 0);

      // --- LIGHTING ---
      var ambientLight = new THREE.AmbientLight(0x003344, 1.4);
      scene.add(ambientLight);

      var dirLight = new THREE.DirectionalLight(0x66E3C4, 2.2);
      dirLight.position.set(4, 5, 6);
      scene.add(dirLight);

      var pointLight1 = new THREE.PointLight(0x38BDF8, 2.5, 15);
      pointLight1.position.set(-4, -2, 4);
      scene.add(pointLight1);

      var pointLight2 = new THREE.PointLight(0x00D68F, 1.8, 12);
      pointLight2.position.set(2, 4, 3);
      scene.add(pointLight2);

      // --- 3D GLASS CARE-CROSS ---
      var crossGroup = new THREE.Group();

      // Translucent Glass Cross Material
      var glassMaterial = new THREE.MeshPhysicalMaterial({{
        color: 0x66E3C4,
        emissive: 0x002B26,
        roughness: 0.12,
        metalness: 0.1,
        transmission: 0.85, // Glass refractive transmission
        thickness: 0.5,
        transparent: true,
        opacity: 0.88,
        reflectivity: 0.85,
        clearcoat: 1.0,
        clearcoatRoughness: 0.1
      }});

      // Vertical Bar
      var vBarGeo = new THREE.BoxGeometry(0.72, 2.2, 0.45);
      var vBar = new THREE.Mesh(vBarGeo, glassMaterial);
      crossGroup.add(vBar);

      // Horizontal Bar
      var hBarGeo = new THREE.BoxGeometry(2.2, 0.72, 0.45);
      var hBar = new THREE.Mesh(hBarGeo, glassMaterial);
      crossGroup.add(hBar);

      // Inner Core Accent (faint glowing pulse)
      var coreGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
      var coreMat = new THREE.MeshBasicMaterial({{
        color: 0xFFFFFF,
        transparent: true,
        opacity: 0.45
      }});
      var coreMesh = new THREE.Mesh(coreGeo, coreMat);
      crossGroup.add(coreMesh);

      scene.add(crossGroup);

      // --- THIN ORBITAL RINGS ---
      var ringsGroup = new THREE.Group();

      var ringMat1 = new THREE.MeshBasicMaterial({{
        color: 0x38BDF8,
        transparent: true,
        opacity: 0.42,
        wireframe: false
      }});
      var ringGeo1 = new THREE.TorusGeometry(2.1, 0.022, 16, 120);
      var ring1 = new THREE.Mesh(ringGeo1, ringMat1);
      ring1.rotation.x = Math.PI / 3;
      ring1.rotation.y = Math.PI / 6;
      ringsGroup.add(ring1);

      var ringMat2 = new THREE.MeshBasicMaterial({{
        color: 0x66E3C4,
        transparent: true,
        opacity: 0.35,
        wireframe: false
      }});
      var ringGeo2 = new THREE.TorusGeometry(2.55, 0.018, 16, 120);
      var ring2 = new THREE.Mesh(ringGeo2, ringMat2);
      ring2.rotation.x = -Math.PI / 4;
      ring2.rotation.z = Math.PI / 5;
      ringsGroup.add(ring2);

      scene.add(ringsGroup);

      // --- FLOATING TRANSLUCENT TEAL/MINT SHAPES ---
      var floatingShapes = [];
      var shapeColors = [0x66E3C4, 0x38BDF8, 0x00D68F, 0x66E3C4];
      var shapeGeos = [
        new THREE.OctahedronGeometry(0.24, 0),
        new THREE.TetrahedronGeometry(0.26, 0),
        new THREE.IcosahedronGeometry(0.22, 0),
        new THREE.OctahedronGeometry(0.18, 0)
      ];

      for (var i = 0; i < 4; i++) {{
        var sMat = new THREE.MeshPhysicalMaterial({{
          color: shapeColors[i % shapeColors.length],
          roughness: 0.2,
          metalness: 0.2,
          transmission: 0.7,
          transparent: true,
          opacity: 0.72
        }});
        var mesh = new THREE.Mesh(shapeGeos[i % shapeGeos.length], sMat);
        var angle = (i / 4) * Math.PI * 2;
        var radius = 2.4 + (i * 0.2);
        mesh.position.set(
          Math.cos(angle) * radius,
          Math.sin(angle) * 1.2,
          (Math.sin(i) * 1.5)
        );
        scene.add(mesh);
        floatingShapes.push({{
          mesh: mesh,
          angle: angle,
          speed: 0.008 + (i * 0.003),
          radius: radius,
          yOffset: mesh.position.y,
          rotSpeed: 0.015 + (i * 0.005)
        }});
      }}

      // --- GENTLE MOUSE PARALLAX ---
      var targetX = 0;
      var targetY = 0;
      var windowWidth = window.innerWidth || 800;
      var windowHeight = window.innerHeight || 600;

      function onMouseMove(e) {{
        var clientX = e.clientX || 0;
        var clientY = e.clientY || 0;
        targetX = ((clientX / windowWidth) - 0.5) * 1.4;
        targetY = -((clientY / windowHeight) - 0.5) * 1.0;
      }}
      window.addEventListener('mousemove', onMouseMove, {{ passive: true }});

      // --- PAUSE ON HIDDEN TAB ---
      function onVisibilityChange() {{
        isPaused = document.hidden;
      }}
      document.addEventListener('visibilitychange', onVisibilityChange);

      // --- ANIMATION TICK ---
      var clock = new THREE.Clock();
      function animate() {{
        animId = requestAnimationFrame(animate);
        if (isPaused) return;

        var delta = clock.getDelta();
        var elapsed = clock.getElapsedTime();

        // Slow Rotation of Central Cross
        crossGroup.rotation.y += 0.006;
        crossGroup.rotation.x = Math.sin(elapsed * 0.4) * 0.12;

        // Counter-rotation of Rings
        ring1.rotation.z += 0.003;
        ring2.rotation.z -= 0.004;

        // Floating Bobbing Shapes
        for (var k = 0; k < floatingShapes.length; k++) {{
          var item = floatingShapes[k];
          item.angle += item.speed;
          item.mesh.position.x = Math.cos(item.angle) * item.radius;
          item.mesh.position.y = item.yOffset + Math.sin(elapsed * 1.5 + k) * 0.25;
          item.mesh.position.z = Math.sin(item.angle) * (item.radius * 0.6);
          item.mesh.rotation.x += item.rotSpeed;
          item.mesh.rotation.y += item.rotSpeed * 0.8;
        }}

        // Gentle Camera Parallax Interpolation
        camera.position.x += (targetX - camera.position.x) * 0.04;
        camera.position.y += (targetY - camera.position.y) * 0.04;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
      }}

      // Start animation and fade in canvas over static fallback
      animate();
      canvas.style.opacity = '1';
      if (fallback) {{
        fallback.style.opacity = '0.12'; // Keep soft ambient glow behind 3D scene
      }}
      if (engineLabel) {{
        engineLabel.textContent = "Three.js 3D Hero Active";
      }}

      // --- DISPOSE ON UNMOUNT ---
      function cleanup() {{
        if (animId) cancelAnimationFrame(animId);
        window.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('visibilitychange', onVisibilityChange);

        // Traverse scene and dispose
        scene.traverse(function(obj) {{
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) {{
            if (Array.isArray(obj.material)) {{
              obj.material.forEach(function(m) {{ m.dispose(); }});
            }} else {{
              obj.material.dispose();
            }}
          }}
        }});

        if (renderer) {{
          renderer.dispose();
          if (renderer.domElement && renderer.domElement.parentNode) {{
            renderer.domElement.parentNode.removeChild(renderer.domElement);
          }}
        }}
      }}

      window.addEventListener('beforeunload', cleanup);
      window.addEventListener('unload', cleanup);
    }})();
  </script>
</body>
</html>
"""


def render_hero_3d_banner(height: int = 240):
    """
    Renders the 3D procedural Three.js hero banner on the Command Center ONLY.
    Never rendered behind forms, tables, or patient cards.
    """
    html_markup = get_hero_3d_html(height=height)
    if hasattr(st, "iframe"):
        try:
            st.iframe(html_markup, height=height, width="stretch")
            return
        except Exception:
            pass
    components.html(html_markup, height=height, scrolling=False)
