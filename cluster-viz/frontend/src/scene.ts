import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PointData } from "./api";

// Six themed categories from the classification step, each gets a colour.
// Hex values chosen so they stay readable on a near-black background.
export const CATEGORY_COLORS: Record<string, string> = {
  "🌌 Starlit Wonders": "#8b5cf6", // violet
  "🌿 Wild Horizons": "#22c55e",   // green
  "⏳ Time in Motion": "#f59e0b",   // amber
  "🌆 Urban Glow": "#ec4899",      // pink
  "🌊 Ocean Whispers": "#06b6d4",  // cyan
  "🏞️ Liquid Cascades": "#14b8a6",  // teal
};
const FALLBACK_COLOR = "#a3a3a3";

const POINT_SIZE = 0.06;
const POINT_SIZE_HOVERED = 0.12;

// Click target in screen pixels. The raycaster threshold is recomputed every frame
// so that this stays roughly constant regardless of zoom level.
const PICK_RADIUS_PX = 14;

// If the pointer moves more than this many pixels between down and up, treat it as a
// drag (camera rotate/pan) and DON'T open the side panel.
const CLICK_DRAG_THRESHOLD_PX = 5;

export interface SceneCallbacks {
  onHover: (point: PointData | null, screenX: number, screenY: number) => void;
  onClick: (point: PointData) => void;
}

export class ClusterScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private pointer = new THREE.Vector2();
  // Re-used per pick to avoid per-call allocations.
  private projectedTmp = new THREE.Vector3();
  private points!: THREE.Points;
  private data: PointData[] = [];
  private sizes!: Float32Array;
  private hoveredIndex = -1;
  private callbacks: SceneCallbacks;
  private container: HTMLElement;
  private pointerDownAt: { x: number; y: number } | null = null;

  constructor(container: HTMLElement, callbacks: SceneCallbacks) {
    this.container = container;
    this.callbacks = callbacks;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(
      55,
      container.clientWidth / container.clientHeight,
      0.01,
      100
    );
    this.camera.position.set(2.5, 2.0, 2.5);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.6;

    // Picker is done in screen space (see pick()), so no raycaster setup needed.
    this.renderer.domElement.addEventListener("pointermove", (e) => this.onPointerMove(e));
    this.renderer.domElement.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    this.renderer.domElement.addEventListener("pointerup", (e) => this.onPointerUp(e));
    this.renderer.domElement.addEventListener("pointercancel", () => { this.pointerDownAt = null; });
    window.addEventListener("resize", () => this.onResize());

    this.renderLoop();
  }

  load(data: PointData[]) {
    this.data = data;

    const positions = new Float32Array(data.length * 3);
    const colors = new Float32Array(data.length * 3);
    this.sizes = new Float32Array(data.length);

    const colorObj = new THREE.Color();
    data.forEach((d, i) => {
      positions[i * 3 + 0] = d.x;
      positions[i * 3 + 1] = d.y;
      positions[i * 3 + 2] = d.z;

      const hex = (d.category && CATEGORY_COLORS[d.category]) || FALLBACK_COLOR;
      colorObj.set(hex);
      colors[i * 3 + 0] = colorObj.r;
      colors[i * 3 + 1] = colorObj.g;
      colors[i * 3 + 2] = colorObj.b;

      this.sizes[i] = POINT_SIZE;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(this.sizes, 1));

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
      uniforms: {},
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          // Soft circular disc with feathered edge.
          vec2 c = gl_PointCoord - 0.5;
          float d = length(c);
          if (d > 0.5) discard;
          float alpha = smoothstep(0.5, 0.1, d);
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
    });

    if (this.points) this.scene.remove(this.points);
    this.points = new THREE.Points(geometry, material);
    this.scene.add(this.points);
  }

  private onPointerMove(e: PointerEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const idx = this.pick();
    if (idx !== this.hoveredIndex) {
      // Reset previous hover size.
      if (this.hoveredIndex >= 0) {
        this.sizes[this.hoveredIndex] = POINT_SIZE;
      }
      this.hoveredIndex = idx;
      if (idx >= 0) {
        this.sizes[idx] = POINT_SIZE_HOVERED;
      }
      (this.points.geometry.getAttribute("size") as THREE.BufferAttribute).needsUpdate = true;
      this.renderer.domElement.style.cursor = idx >= 0 ? "pointer" : "default";
    }

    this.callbacks.onHover(
      idx >= 0 ? this.data[idx] : null,
      e.clientX,
      e.clientY
    );
  }

  private onPointerDown(e: PointerEvent) {
    // Remember where the press started. We decide click-vs-drag on pointerup.
    this.pointerDownAt = { x: e.clientX, y: e.clientY };
  }

  private onPointerUp(e: PointerEvent) {
    const start = this.pointerDownAt;
    this.pointerDownAt = null;
    if (!start) return;

    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    if (Math.sqrt(dx * dx + dy * dy) > CLICK_DRAG_THRESHOLD_PX) return; // it was a drag

    // Sync the pointer to the release position before picking.
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const idx = this.pick();
    if (idx >= 0) this.callbacks.onClick(this.data[idx]);
  }

  private pick(): number {
    if (!this.points) return -1;

    // Convert a screen-pixel radius into NDC units (NDC spans [-1, 1] over the
    // canvas, so the full screen height is 2 NDC units tall).
    const ndcRadius = (PICK_RADIUS_PX * 2) / this.renderer.domElement.clientHeight;
    const ndcRadiusSq = ndcRadius * ndcRadius;

    const positions = this.points.geometry.getAttribute("position") as THREE.BufferAttribute;

    let bestIndex = -1;
    let bestDistSq = ndcRadiusSq;
    let bestDepth = Infinity;

    for (let i = 0; i < positions.count; i++) {
      this.projectedTmp.set(positions.getX(i), positions.getY(i), positions.getZ(i));
      this.projectedTmp.applyMatrix4(this.points.matrixWorld);
      this.projectedTmp.project(this.camera);
      // After project(): x/y in NDC [-1, 1], z in NDC [-1, 1] (depth, smaller = nearer).
      if (this.projectedTmp.z < -1 || this.projectedTmp.z > 1) continue; // outside frustum

      const dx = this.projectedTmp.x - this.pointer.x;
      const dy = this.projectedTmp.y - this.pointer.y;
      const distSq = dx * dx + dy * dy;
      if (distSq > ndcRadiusSq) continue;

      // Within the pick radius. Prefer the closest dot to the cursor; if two are
      // essentially tied in screen space, prefer the one nearer the camera.
      if (
        distSq < bestDistSq - 1e-6 ||
        (Math.abs(distSq - bestDistSq) < 1e-6 && this.projectedTmp.z < bestDepth)
      ) {
        bestDistSq = distSq;
        bestDepth = this.projectedTmp.z;
        bestIndex = i;
      }
    }

    return bestIndex;
  }

  private onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private renderLoop = () => {
    requestAnimationFrame(this.renderLoop);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
