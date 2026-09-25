/* Ambient Three.js particle field. Optional: chat works without it. */
const canvas = document.getElementById("bg");
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
if (!canvas || reduced) {
  canvas?.remove();
} else {
  try {
    const three = await import("three").catch(() => null);
    if (!three) throw new Error("three.js CDN unavailable");
    const { Scene, PerspectiveCamera, WebGLRenderer, BufferGeometry, PointsMaterial, Points, Float32BufferAttribute } = three;
    const renderer = new WebGLRenderer({ canvas, alpha: true, antialias: false });
    const scene = new Scene();
    const camera = new PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
    camera.position.z = 8;
    const N = innerWidth < 700 ? 250 : 600;
    const pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 22;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 14;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10;
    }
    const geo = new BufferGeometry();
    geo.setAttribute("position", new Float32BufferAttribute(pos, 3));
    const mat = new PointsMaterial({ color: 0x6da7ec, size: 0.045, transparent: true, opacity: 0.7 });
    const pts = new Points(geo, mat);
    scene.add(pts);
    let mx = 0, my = 0;
    addEventListener("pointermove", (e) => {
      mx = (e.clientX / innerWidth - 0.5) * 0.6;
      my = (e.clientY / innerHeight - 0.5) * 0.4;
    }, { passive: true });
    const fit = () => {
      renderer.setSize(innerWidth, innerHeight, false);
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
    };
    fit();
    addEventListener("resize", fit);
    const tick = () => {
      pts.rotation.y += 0.0006 + mx * 0.0004;
      pts.rotation.x += (my * 0.2 - pts.rotation.x) * 0.01;
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    };
    tick();
  } catch {
    // Offline-safe fallback: soft radial glow, no WebGL needed.
    canvas.style.background = "radial-gradient(60% 40% at 50% 0%, rgba(109,167,236,.14), transparent 70%)";
  }
}
