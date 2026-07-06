/**
 * 3D attack-flow visualization — packets move Internet → Tier-1 → Tier-2 → Tier-3 → verdict.
 * Uses Three.js (not playhtml — that library is for collaborative HTML, not 3D graphics).
 */
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

export const LABEL_COLORS = {
  BENIGN: "#3fb950",
  BRUTE_FORCE: "#ff6b6b",
  DDOS_HTTP_FLOOD: "#f85149",
  SLOW_HTTP: "#d29922",
  PORT_SCAN: "#bc8cff",
  DNS_TUNNELING: "#58a6ff",
  ANOMALY: "#ff79c6",
};

const NODES = {
  internet: new THREE.Vector3(-9, 0, 0),
  t1: new THREE.Vector3(-5.5, 0, 0),
  t2: new THREE.Vector3(-1.5, 0, 0),
  t3: new THREE.Vector3(2.5, 0, 0),
  allow: new THREE.Vector3(8, 3.5, 0),
  flag: new THREE.Vector3(8, 0, 0),
  block: new THREE.Vector3(8, -3.5, 0),
};

function hex(c) {
  return parseInt(c.replace("#", ""), 16);
}

function makeLabel(text, color = "#8b949e", scale = 0.55) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 256;
  canvas.height = 64;
  ctx.fillStyle = color;
  ctx.font = "bold 28px system-ui,sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, 128, 40);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sp = new THREE.Sprite(mat);
  sp.scale.set(scale * 4, scale, 1);
  return sp;
}

function makeGate(pos, color, label) {
  const g = new THREE.Group();
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(1.8, 2.2, 1.2),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, metalness: 0.3, roughness: 0.6 })
  );
  g.add(box);
  const wire = new THREE.LineSegments(
    new THREE.EdgesGeometry(box.geometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })
  );
  g.add(wire);
  const lbl = makeLabel(label, "#e6edf3", 0.45);
  lbl.position.set(0, 1.6, 0);
  g.add(lbl);
  g.position.copy(pos);
  return g;
}

function buildWaypoints(alert) {
  const pts = [NODES.internet.clone(), NODES.t1.clone()];
  const tiers = alert.tiers_used || [];
  const fastAllow = tiers.length === 1 && tiers[0] === "tier1_gate";
  if (fastAllow) {
    pts.push(NODES.allow.clone());
    return pts;
  }
  pts.push(NODES.t2.clone());
  if (tiers.includes("tier3_oneclass")) pts.push(NODES.t3.clone());
  const dest =
    alert.action === "BLOCK" ? NODES.block :
    alert.action === "FLAG" ? NODES.flag : NODES.allow;
  pts.push(dest.clone());
  return pts;
}

export function initFlow3D(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.Fog(0x0d1117, 14, 28);

  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(0, 2, 14);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0x404060, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(5, 8, 10);
  scene.add(key);

  // Internet source
  const src = new THREE.Mesh(
    new THREE.SphereGeometry(1.1, 24, 24),
    new THREE.MeshStandardMaterial({ color: 0x58a6ff, emissive: 0x1f6feb, emissiveIntensity: 0.5 })
  );
  src.position.copy(NODES.internet);
  scene.add(src);
  const srcLbl = makeLabel("NDN / Internet", "#58a6ff");
  srcLbl.position.set(NODES.internet.x, -1.8, 0);
  scene.add(srcLbl);

  scene.add(makeGate(NODES.t1, 0x238636, "Tier-1 Gate"));
  scene.add(makeGate(NODES.t2, 0x1f6feb, "Tier-2 CNN-GRU"));
  scene.add(makeGate(NODES.t3, 0x8957e5, "Tier-3 Zero-day"));

  for (const [name, pos, col] of [
    ["ALLOW", NODES.allow, "#3fb950"],
    ["FLAG", NODES.flag, "#d29922"],
    ["BLOCK", NODES.block, "#f85149"],
  ]) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.9, 0.12, 12, 32),
      new THREE.MeshStandardMaterial({ color: hex(col), emissive: hex(col), emissiveIntensity: 0.4 })
    );
    ring.position.copy(pos);
    scene.add(ring);
    const lbl = makeLabel(name, col);
    lbl.position.set(pos.x + 1.2, pos.y, 0);
    scene.add(lbl);
  }

  // Path guides
  const pathMat = new THREE.LineBasicMaterial({ color: 0x30363d, transparent: true, opacity: 0.5 });
  for (const dest of [NODES.allow, NODES.flag, NODES.block]) {
    const geo = new THREE.BufferGeometry().setFromPoints([NODES.internet, NODES.t1, NODES.t2, dest]);
    scene.add(new THREE.Line(geo, pathMat));
  }

  const packets = [];
  let lastAlertId = 0;

  function spawnPacket(alert) {
    const color = LABEL_COLORS[alert.label] || "#8b949e";
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 12, 12),
      new THREE.MeshStandardMaterial({
        color: hex(color),
        emissive: hex(color),
        emissiveIntensity: 0.6,
      })
    );
    const waypoints = buildWaypoints(alert);
    mesh.position.copy(waypoints[0]);
    scene.add(mesh);

    const trail = [];
    packets.push({
      mesh,
      waypoints,
      seg: 0,
      t: 0,
      speed: 0.55 + Math.random() * 0.2,
      alert,
      trail,
      done: false,
    });
  }

  function feedAlerts(alerts) {
    const sorted = [...alerts].sort((a, b) => a.id - b.id);
    for (const a of sorted) {
      if (a.id > lastAlertId) {
        spawnPacket(a);
        lastAlertId = a.id;
      }
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = 0.016;

    for (const p of packets) {
      if (p.done) continue;
      const from = p.waypoints[p.seg];
      const to = p.waypoints[p.seg + 1];
      if (!to) {
        p.done = true;
        p.mesh.scale.multiplyScalar(0.98);
        if (p.mesh.scale.x < 0.05) {
          scene.remove(p.mesh);
        }
        continue;
      }
      p.t += p.speed * dt;
      if (p.t >= 1) {
        p.t = 0;
        p.seg += 1;
        continue;
      }
      p.mesh.position.lerpVectors(from, to, p.t);
      p.mesh.rotation.y += 0.08;
    }

    src.rotation.y += 0.005;
    renderer.render(scene, camera);
  }
  animate();

  function onResize() {
    const w = container.clientWidth;
    const h = container.clientHeight || 420;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", onResize);

  return { feedAlerts, spawnPacket, seedLastId: (id) => { lastAlertId = id; }, renderer };
}

export function renderFlowFeed(alerts, el, max = 12) {
  if (!el) return;
  const recent = alerts.slice(0, max);
  if (!recent.length) {
    el.innerHTML = '<div class="empty">Waiting for traffic… click “Run demo”</div>';
    return;
  }
  el.innerHTML = recent.map(a => {
    const col = LABEL_COLORS[a.label] || "#8b949e";
    return `<div class="flow-item">
      <span class="flow-dot" style="background:${col}"></span>
      <span class="flow-label">${a.label.replace(/_/g, " ")}</span>
      <span class="badge ${a.action}">${a.action}</span>
      <span class="flow-conf">${(a.confidence * 100).toFixed(0)}%</span>
      <span class="flow-lat">${a.latency_ms != null ? Number(a.latency_ms).toFixed(1) + "ms" : ""}</span>
    </div>`;
  }).join("");
}

export function renderLegend(el) {
  if (!el) return;
  el.innerHTML = Object.entries(LABEL_COLORS).map(([k, c]) =>
    `<div class="legend-item"><span class="flow-dot" style="background:${c}"></span>${k.replace(/_/g, " ")}</div>`
  ).join("");
}
