import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

function dispose(object: THREE.Object3D) {
  object.traverse((o: any) => {
    o.geometry?.dispose();
    if (o.material)
      Array.isArray(o.material)
        ? o.material.forEach((material: any) => material.dispose())
        : o.material.dispose();
  });
}

export default function GeometryView({
  config,
  derived,
  result,
  selected,
  onSelect,
  display = "geometry",
  playback,
}: any) {
  const ref = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const metalRef = useRef<THREE.Mesh[]>([]);
  const tracks = useRef<any[]>([]);
  const playbackRef = useRef(playback);
  playbackRef.current = playback;
  const [view, setView] = useState("3d");
  const geometryKey = JSON.stringify([config?.geometry, derived?.geometry_id]);
  useEffect(() => {
    if (playback?.playing) setView("3d");
  }, [playback?.playing]);
  useEffect(() => {
    if (!ref.current || !derived || view !== "3d") return;
    const element = ref.current;
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color("#f1f5f4");
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    const depth = derived.depth_nm / 1000;
    const h = derived.domain_half_width_nm / 1000;
    const extent = Math.max(depth, h * 2);
    camera.position.set(extent * 1.3, extent * 0.6, extent * 1.7);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    element.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -depth / 2, 0);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(4, 6, 4);
    scene.add(light);
    const metalMeshes: THREE.Mesh[] = [];
    const basic = (color: string, opacity = 1) =>
      new THREE.MeshStandardMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        roughness: 0.55,
        side: THREE.DoubleSide,
        depthWrite: opacity === 1,
      });
    const tc = config.geometry.carbon_thickness_nm / 1000,
      sub = derived.substrate_top_nm / 1000;
    const oxide = new THREE.Mesh(
      new THREE.BoxGeometry(2 * h, sub - tc, 2 * h),
      basic("#bed0d4", 0.13),
    );
    oxide.position.y = -(sub + tc) / 2;
    scene.add(oxide);
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(oxide.geometry),
      new THREE.LineBasicMaterial({
        color: "#b6c7ca",
        transparent: true,
        opacity: 0.4,
      }),
    );
    outline.position.copy(oxide.position);
    scene.add(outline);
    const points = derived.profile
      .map(
        (p: any) => new THREE.Vector2(p.radius_nm / 1000, -p.depth_nm / 1000),
      )
      .reverse();
    const channel = new THREE.Mesh(
      new THREE.LatheGeometry(points, 64),
      basic("#568b91", 0.14),
    );
    scene.add(channel);
    const shape = new THREE.Shape();
    shape.moveTo(-h, -h);
    shape.lineTo(h, -h);
    shape.lineTo(h, h);
    shape.lineTo(-h, h);
    shape.closePath();
    const hole = new THREE.Path();
    hole.absarc(
      0,
      0,
      config.geometry.channel_diameter_nm / 2000,
      0,
      Math.PI * 2,
      true,
    );
    shape.holes.push(hole);
    const carbon = new THREE.Mesh(
      new THREE.ExtrudeGeometry(shape, {
        depth: tc,
        bevelEnabled: false,
        curveSegments: 48,
      }),
      basic("#365064", 0.25),
    );
    carbon.rotation.x = -Math.PI / 2;
    carbon.position.y = -tc;
    scene.add(carbon);
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(h * 2, 0.045, h * 2),
      basic("#60798a", 0.55),
    );
    base.position.y = -sub - 0.023;
    scene.add(base);
    derived.dummies.forEach((d: any) => {
      const length = d.length_nm / 1000;
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(
          d.diameter_nm / 2000,
          d.diameter_nm / 2000,
          length,
          32,
        ),
        basic(
          d.index === selected ? "#efb749" : "#91a29b",
          d.index === selected ? 0.94 : 0.7,
        ),
      );
      mesh.position.set(
        d.x_nm / 1000,
        -(d.top_nm + d.bottom_nm) / 2000,
        d.y_nm / 1000,
      );
      mesh.userData.index = d.index;
      metalMeshes.push(mesh);
      scene.add(mesh);
    });
    metalRef.current = metalMeshes;
    const grid = new THREE.GridHelper(h * 2.8, 12, 0xc3d2d1, 0xdfe7e5);
    grid.position.y = -sub - 0.055;
    scene.add(grid);
    const resize = () => {
      const w = element.clientWidth,
        height = element.clientHeight;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    const ray = new THREE.Raycaster();
    const down = { x: 0, y: 0 };
    const pointerDown = (e: PointerEvent) => {
      down.x = e.clientX;
      down.y = e.clientY;
    };
    const pointerUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(
        new THREE.Vector2(
          ((e.clientX - r.left) / r.width) * 2 - 1,
          (-(e.clientY - r.top) / r.height) * 2 + 1,
        ),
        camera,
      );
      const hit = ray.intersectObjects(metalMeshes)[0];
      if (hit) onSelect(hit.object.userData.index);
    };
    renderer.domElement.addEventListener("pointerdown", pointerDown);
    renderer.domElement.addEventListener("pointerup", pointerUp);
    let frame = 0;
    const render = () => {
      const playback = playbackRef.current;
      for (const track of tracks.current) {
        if (!playback?.enabled) {
          track.line.geometry.setDrawRange(0, track.points.length);
          track.marker.visible = false;
          continue;
        }
        const knots = playback.timed ? track.times : track.distances;
        const value = playback.timed
          ? playback.progress * playback.endTime
          : playback.progress;
        track.marker.visible = value >= knots[0];
        let low = 0,
          high = knots.length - 1;
        while (low < high) {
          const middle = Math.ceil((low + high) / 2);
          if (knots[middle] <= value) low = middle;
          else high = middle - 1;
        }
        const next = Math.min(low + 1, track.points.length - 1);
        const fraction = Math.max(
          0,
          Math.min(1, (value - knots[low]) / (knots[next] - knots[low] || 1)),
        );
        track.marker.position
          .copy(track.points[low])
          .lerp(track.points[next], fraction);
        track.line.geometry.setDrawRange(0, value < knots[0] ? 0 : low + 1);
      }
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      dispose(scene);
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null;
      tracks.current = [];
    };
  }, [geometryKey, onSelect, view]);
  useEffect(() => {
    metalRef.current.forEach((mesh) => {
      const material = mesh.material as THREE.MeshStandardMaterial;
      const active = mesh.userData.index === selected;
      material.color.set(active ? "#efb749" : "#91a29b");
      material.opacity = active ? 0.94 : 0.7;
    });
  }, [selected, geometryKey, view]);
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !result || view !== "3d") return;
    const group = new THREE.Group();
    const newTracks: any[] = [];
    (result.trajectories ?? []).forEach((path: any) => {
      const points = path.points_nm.map(
        (p: number[]) =>
          new THREE.Vector3(p[0] / 1000, -p[2] / 1000, p[1] / 1000),
      );
      if (points.length < 2) return;
      const color = path.species.toLowerCase().includes("electron")
        ? "#179d90"
        : "#e68831";
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity: 0.75,
        }),
      );
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.013, 12, 8),
        new THREE.MeshBasicMaterial({ color, depthTest: false }),
      );
      marker.renderOrder = 5;
      const distances = [0];
      for (let i = 1; i < points.length; i++)
        distances.push(distances[i - 1] + points[i].distanceTo(points[i - 1]));
      const total = distances.at(-1) || 1;
      newTracks.push({
        line,
        marker,
        points,
        distances: distances.map((value) => value / total),
        times: path.times_s ?? distances.map((value) => value / total),
      });
      group.add(line, marker);
    });
    const source =
      display === "charge"
        ? result.surface.centers_nm
        : result.field_samples.xyz_nm;
    const values =
      display === "charge"
        ? result.surface.sigma_c_m2
        : result.field_samples.potential_v;
    if (display !== "geometry" && display !== "field" && source.length) {
      const min = Math.min(...values),
        max = Math.max(...values);
      const xyz: number[] = [],
        colors: number[] = [];
      source.forEach((p: number[], i: number) => {
        xyz.push(p[0] / 1000, -p[2] / 1000, p[1] / 1000);
        const color = new THREE.Color().setHSL(
          0.64 - (0.64 * (values[i] - min)) / (max - min || 1),
          0.65,
          0.52,
        );
        colors.push(color.r, color.g, color.b);
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(xyz, 3),
      );
      geometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(colors, 3),
      );
      group.add(
        new THREE.Points(
          geometry,
          new THREE.PointsMaterial({
            size: 0.012,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
          }),
        ),
      );
    }
    if (display === "field" && result.field_vectors) {
      const vectors = result.field_vectors.electric_field_v_m,
        centers = result.field_vectors.centers_nm;
      const magnitude = vectors.map((v: number[]) => Math.hypot(...v));
      const maximum = Math.max(...magnitude, 1e-30);
      centers.forEach((p: number[], i: number) => {
        if (i % 6 !== 0 || magnitude[i] < 1e-12) return;
        const v = vectors[i];
        const direction = new THREE.Vector3(v[0], -v[2], v[1]).normalize();
        const color = new THREE.Color().setHSL(
          0.64 - (0.64 * magnitude[i]) / maximum,
          0.65,
          0.52,
        );
        group.add(
          new THREE.ArrowHelper(
            direction,
            new THREE.Vector3(p[0] / 1000, -p[2] / 1000, p[1] / 1000),
            0.09,
            color,
            0.024,
            0.013,
          ),
        );
      });
    }
    scene.add(group);
    tracks.current = newTracks;
    return () => {
      scene.remove(group);
      dispose(group);
      if (tracks.current === newTracks) tracks.current = [];
    };
  }, [result, display, geometryKey, view]);
  if (!derived)
    return (
      <div className="empty-preview">
        形状検査の完了後にプレビューを表示します。
      </div>
    );
  const r = derived.domain_half_width_nm;
  const circleR = ((200 / r) * config.geometry.channel_diameter_nm) / 2;
  return (
    <div className="geometry-view">
      <div className="view-switch">
        <button
          className={view === "3d" ? "active" : ""}
          onClick={() => setView("3d")}
        >
          立体
        </button>
        <button
          className={view === "top" ? "active" : ""}
          disabled={playback?.playing}
          onClick={() => setView("top")}
        >
          上面
        </button>
      </div>
      {view === "3d" ? (
        <div className="three-canvas" ref={ref} />
      ) : (
        <svg
          viewBox="-230 -230 460 460"
          className="top-view"
          aria-label="Channelとdummyの上面配置"
        >
          <rect
            x="-200"
            y="-200"
            width="400"
            height="400"
            fill="#e0e9e7"
            stroke="#bdcfca"
          />
          <line x1="-210" x2="210" stroke="#a5bdb4" />
          <line y1="-210" y2="210" stroke="#a5bdb4" />
          <circle r={circleR} fill="#f8fcfb" stroke="#438b80" strokeWidth="2" />
          {derived.dummies.map((d: any) => (
            <g
              key={d.index}
              onClick={() => onSelect(d.index)}
              className="dummy-hit"
            >
              <circle
                cx={(d.x_nm / r) * 200}
                cy={(-d.y_nm / r) * 200}
                r={(d.diameter_nm / r) * 100}
                fill={d.index === selected ? "#e6b14d" : "#92a39c"}
                stroke={d.index === selected ? "#a57a24" : "#607a70"}
                strokeWidth="2"
              />
              <text
                x={(d.x_nm / r) * 200}
                y={(-d.y_nm / r) * 200 + 4}
                textAnchor="middle"
                fontSize="12"
                fill="#fff"
              >
                {d.index + 1}
              </text>
            </g>
          ))}
          <text x="-200" y="225" fontSize="12" fill="#637b71">
            領域幅 {((r * 2) / 1000).toFixed(2)} µm · z=Carbon下面
          </text>
        </svg>
      )}
      <div className="view-legend">
        <span>
          <i style={{ background: "#438b80" }} />
          Channel
        </span>
        <span>
          <i style={{ background: "#efb749" }} />
          選択dummy
        </span>
        <span>
          <i style={{ background: "#91a29b" }} />
          共通導体
        </span>
      </div>
      <div className="view-caption">
        {playback?.enabled
          ? "動く点は代表粒子 · 軌跡は保存点間を補間 · ドラッグで回転"
          : display === "geometry"
            ? "ドラッグで回転 · スクロールで拡大 · dummyをクリックして選択"
            : display === "field"
              ? "気相四面体の電場方向 · 矢印の色は強度、長さは一定（位相0°）"
              : display === "charge"
                ? "SiO₂表面パッチの電荷密度 · 代表軌道"
                : "FEM節点の電位 · 代表軌道（電位表示はRF位相0°）"}
      </div>
    </div>
  );
}
