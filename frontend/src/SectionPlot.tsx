import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine, Minus, Plus, RotateCcw } from "lucide-react";

type Obj = Record<string, any>;
type Point = [number, number];
const quantities: Obj = {
  potential_v: ["電位", "V"],
  e_norm_v_m: ["電場の大きさ |E|", "V/m"],
  e_x_v_m: ["電場 Ex", "V/m"],
  e_y_v_m: ["電場 Ey", "V/m"],
  e_z_v_m: ["電場 Ez", "V/m"],
  sigma_c_m2: ["表面電荷密度", "C/m²"],
};
const number = (value: number) =>
  !Number.isFinite(value)
    ? "—"
    : Math.abs(value) >= 1e4 || (Math.abs(value) < 0.001 && value !== 0)
      ? value.toExponential(3)
      : Number(value.toPrecision(5)).toString();
const palette = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
function rgb(value: number, low: number, high: number, signed: boolean) {
  const fraction =
    high === low
      ? 0.5
      : Math.max(
          0,
          Math.min(
            1,
            signed
              ? value < 0
                ? (0.5 * (value - low)) / -low
                : 0.5 + (0.5 * value) / high
              : (value - low) / (high - low),
          ),
        );
  const stops = signed
    ? [
        [41, 89, 170],
        [248, 249, 244],
        [192, 62, 59],
      ]
    : palette;
  const p = fraction * (stops.length - 1),
    index = Math.min(stops.length - 2, Math.floor(p));
  return stops[index].map((a, i) =>
    Math.round(a + (stops[index + 1][i] - a) * (p - index)),
  );
}
function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob),
    link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function distanceToSegment(p: Point, a: Point, b: Point) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const t = Math.max(
    0,
    Math.min(
      1,
      ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1),
    ),
  );
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}
function insideTriangle(p: Point, points: Point[]) {
  const signs = points.map((a, i) => {
    const b = points[(i + 1) % 3];
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  });
  return signs.every((v) => v >= 0) || signs.every((v) => v <= 0);
}

export default function SectionPlot({
  jobId,
  result,
  config,
}: {
  jobId: string;
  result: Obj;
  config: Obj;
}) {
  const [plane, setPlane] = useState("xz"),
    [position, setPosition] = useState("0"),
    [angle, setAngle] = useState("0");
  const [resolution, setResolution] = useState(241),
    [quantity, setQuantity] = useState("potential_v");
  const [data, setData] = useState<Obj | null>(null),
    [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  const [gasOnly, setGasOnly] = useState(false),
    [boundaries, setBoundaries] = useState(true);
  const [contours, setContours] = useState(true),
    [arrows, setArrows] = useState(false);
  const [rangeMin, setRangeMin] = useState(""),
    [rangeMax, setRangeMax] = useState("");
  const [zoom, setZoom] = useState(1),
    [center, setCenter] = useState<Point | null>(null);
  const [hover, setHover] = useState<Obj | null>(null),
    [size, setSize] = useState(600);
  const host = useRef<HTMLDivElement>(null),
    canvas = useRef<HTMLCanvasElement>(null);
  const mapping = useRef<Obj | null>(null),
    drag = useRef<Obj | null>(null);
  const step = result.history.at(-1)?.step;
  const normalRange: Point =
    plane === "xy"
      ? [-config.geometry.entrance_height_nm, result.derived.substrate_top_nm]
      : plane === "vertical"
        ? ([-1, 1].map(
            (sign) =>
              sign *
              result.derived.domain_half_width_nm *
              (Math.abs(Math.sin((Number(angle) * Math.PI) / 180)) +
                Math.abs(Math.cos((Number(angle) * Math.PI) / 180))),
          ) as Point)
        : [
            -result.derived.domain_half_width_nm,
            result.derived.domain_half_width_nm,
          ];
  const normalAxis =
    plane === "xy" ? "z" : plane === "xz" ? "y" : plane === "yz" ? "x" : "n";
  const [label, unit] = quantities[quantity];
  const surface = quantity === "sigma_c_m2";

  useEffect(() => {
    const observer = new ResizeObserver((entries) =>
      setSize(Math.max(280, entries[0].contentRect.width)),
    );
    if (host.current) observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    const activeAngle = plane === "vertical" ? angle : "0";
    setData(null);
    setHover(null);
    setError("");
    setLoading(true);
    if (
      !position.trim() ||
      !activeAngle.trim() ||
      !Number.isFinite(Number(position)) ||
      !Number.isFinite(Number(activeAngle))
    ) {
      setError("断面位置と角度を数値で指定してください。");
      setLoading(false);
      return () => controller.abort();
    }
    const timer = setTimeout(async () => {
      try {
        const query = new URLSearchParams({
          plane,
          position_nm: position,
          angle_deg: activeAngle,
          resolution: String(resolution),
        });
        if (step !== undefined) query.set("step", String(step));
        const response = await fetch(`/api/jobs/${jobId}/slice?${query}`, {
          signal: controller.signal,
        });
        const body = await response.json();
        if (!response.ok)
          throw new Error(
            typeof body.detail === "string"
              ? body.detail
              : "断面条件を確認してください。",
          );
        if (!controller.signal.aborted) setData(body);
      } catch (e: any) {
        if (!controller.signal.aborted) setError(e.message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 180);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [jobId, step, result.time_s, plane, position, angle, resolution]);

  const values = useMemo(
    () =>
      !data
        ? []
        : surface
          ? [...data.surface_segments, ...data.surface_polygons].map(
              (p: Obj) => p.sigma_c_m2,
            )
          : data.values[quantity].filter(
              (v: number | null, i: number) =>
                v !== null && (!gasOnly || data.material[i] === 0),
            ),
    [data, surface, quantity, gasOnly],
  );
  const autoRange = useMemo(() => {
    if (!values.length) return [0, 0];
    let min = Infinity,
      max = -Infinity;
    for (const value of values) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    return [min, max];
  }, [values]);
  const custom = rangeMin.trim() !== "" || rangeMax.trim() !== "";
  const rangeValid =
    !custom ||
    (rangeMin.trim() !== "" &&
      rangeMax.trim() !== "" &&
      Number.isFinite(Number(rangeMin)) &&
      Number.isFinite(Number(rangeMax)) &&
      Number(rangeMin) < Number(rangeMax));
  const [low, high] =
    custom && rangeValid ? [Number(rangeMin), Number(rangeMax)] : autoRange;
  const signed = low < 0 && high > 0 && quantity !== "potential_v";
  const texture = useMemo(() => {
    if (!data) return null;
    const bitmap = document.createElement("canvas");
    bitmap.width = data.width;
    bitmap.height = data.height;
    const ctx = bitmap.getContext("2d")!,
      pixels = ctx.createImageData(data.width, data.height);
    for (let i = 0; i < data.width * data.height; i++) {
      const material = data.material[i];
      if (material < 0 || (gasOnly && material !== 0)) continue;
      const color = surface
        ? material === 0
          ? [235, 242, 243]
          : [220, 231, 223]
        : rgb(data.values[quantity][i], low, high, signed);
      const row = Math.floor(i / data.width),
        col = i % data.width;
      const target =
        ((data.axes.vertical_increases_down ? row : data.height - 1 - row) *
          data.width +
          col) *
        4;
      pixels.data.set([...color, 255], target);
    }
    ctx.putImageData(pixels, 0, 0);
    return bitmap;
  }, [data, quantity, surface, gasOnly, low, high, signed]);

  useEffect(() => {
    if (!data || !canvas.current || !texture) {
      mapping.current = null;
      return;
    }
    const element = canvas.current,
      ratio = Math.min(2, window.devicePixelRatio || 1);
    const fullU = data.bounds_nm.u,
      fullV = data.bounds_nm.v;
    const extentU = fullU[1] - fullU[0],
      extentV = fullV[1] - fullV[0];
    const height = Math.max(
      350,
      Math.min(710, ((size - 90) * extentV) / extentU + 130),
    );
    element.style.height = `${height}px`;
    element.width = Math.round(size * ratio);
    element.height = Math.round(height * ratio);
    const ctx = element.getContext("2d")!;
    ctx.scale(ratio, ratio);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, height);
    const scale = Math.min((size - 90) / extentU, (height - 168) / extentV);
    const w = extentU * scale,
      h = extentV * scale,
      left = 60 + (size - 90 - w) / 2,
      top = 40;
    const focus = center ?? [
      (fullU[0] + fullU[1]) / 2,
      (fullV[0] + fullV[1]) / 2,
    ];
    const minU = focus[0] - extentU / zoom / 2,
      maxU = focus[0] + extentU / zoom / 2;
    const minV = focus[1] - extentV / zoom / 2,
      maxV = focus[1] + extentV / zoom / 2;
    const down = data.axes.vertical_increases_down;
    const project = (p: Point): Point => [
      left + ((p[0] - minU) / (maxU - minU)) * w,
      top + ((down ? p[1] - minV : maxV - p[1]) / (maxV - minV)) * h,
    ];
    mapping.current = {
      left,
      top,
      w,
      h,
      minU,
      maxU,
      minV,
      maxV,
      down,
      project,
      focus,
    };
    ctx.fillStyle = "#405952";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      `${plane.toUpperCase()} · ${normalAxis} = ${number(data.position_nm)} nm · ${number(data.time_s * 1e6)} µs`,
      size / 2,
      21,
    );
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, w, h);
    ctx.clip();
    ctx.fillStyle = "#e4e7e6";
    ctx.fillRect(left, top, w, h);
    const p0 = project([fullU[0], down ? fullV[0] : fullV[1]]);
    ctx.imageSmoothingEnabled = false;
    // Grid values are at nodes; place pixel centers at those same coordinates.
    const du = extentU / (data.width - 1),
      dv = extentV / (data.height - 1);
    const pixelOrigin = project([
      fullU[0] - du / 2,
      down ? fullV[0] - dv / 2 : fullV[1] + dv / 2,
    ]);
    ctx.save();
    ctx.beginPath();
    ctx.rect(p0[0], p0[1], w * zoom, h * zoom);
    ctx.clip();
    ctx.drawImage(
      texture,
      pixelOrigin[0],
      pixelOrigin[1],
      (extentU + du) * scale * zoom,
      (extentV + dv) * scale * zoom,
    );
    ctx.restore();
    if (contours && !surface && quantity === "potential_v" && high > low) {
      ctx.strokeStyle = "#ffffffaa";
      ctx.lineWidth = 0.65;
      for (let levelIndex = 1; levelIndex <= 9; levelIndex++) {
        const level = low + ((high - low) * levelIndex) / 10;
        ctx.beginPath();
        for (let row = 0; row < data.height - 1; row++)
          for (let col = 0; col < data.width - 1; col++) {
            const indices = [
              row * data.width + col,
              row * data.width + col + 1,
              (row + 1) * data.width + col + 1,
              (row + 1) * data.width + col,
            ];
            const m = indices.map((i) => data.material[i]);
            if (
              m.some((v) => v < 0 || (gasOnly && v !== 0)) ||
              m.some((v) => v !== m[0])
            )
              continue;
            const positions: Point[] = [
              [data.u_nm[col], data.v_nm[row]],
              [data.u_nm[col + 1], data.v_nm[row]],
              [data.u_nm[col + 1], data.v_nm[row + 1]],
              [data.u_nm[col], data.v_nm[row + 1]],
            ];
            const crossings: Point[] = [];
            for (let edge = 0; edge < 4; edge++) {
              const next = (edge + 1) % 4,
                a = data.values.potential_v[indices[edge]],
                b = data.values.potential_v[indices[next]];
              if (a < level === b < level || a === b) continue;
              const t = (level - a) / (b - a);
              crossings.push(
                project([
                  positions[edge][0] +
                    t * (positions[next][0] - positions[edge][0]),
                  positions[edge][1] +
                    t * (positions[next][1] - positions[edge][1]),
                ]),
              );
            }
            // Split ambiguous four-crossing squares according to the cell center.
            if (
              crossings.length === 4 &&
              indices.reduce((sum, i) => sum + data.values.potential_v[i], 0) /
                4 <
                level !==
                data.values.potential_v[indices[0]] < level
            )
              crossings.push(crossings.shift()!);
            for (let i = 0; i + 1 < crossings.length; i += 2) {
              ctx.moveTo(...crossings[i]);
              ctx.lineTo(...crossings[i + 1]);
            }
          }
        ctx.stroke();
      }
    }
    if (boundaries || surface) {
      ctx.strokeStyle = "#465c59a0";
      ctx.lineWidth = 0.85;
      ctx.beginPath();
      for (const segment of data.boundary_segments) {
        ctx.moveTo(...project(segment[0]));
        ctx.lineTo(...project(segment[1]));
      }
      ctx.stroke();
    }
    if (surface)
      for (const patch of [
        ...data.surface_polygons,
        ...data.surface_segments,
      ]) {
        const color = `rgb(${rgb(patch.sigma_c_m2, low, high, signed).join(",")})`;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        patch.points_nm.forEach((point: Point, i: number) => {
          const xy = project(point);
          i ? ctx.lineTo(...xy) : ctx.moveTo(...xy);
        });
        if (patch.points_nm.length === 3) {
          ctx.closePath();
          ctx.fill();
        } else ctx.stroke();
      }
    if (arrows && !surface) {
      const hopU = Math.max(
        1,
        Math.ceil(data.width / (zoom * Math.max(1, w / 28))),
      );
      const hopV = Math.max(
        1,
        Math.ceil(data.height / (zoom * Math.max(1, h / 28))),
      );
      ctx.strokeStyle = quantity === "potential_v" ? "#182f35a0" : "#ffffffc0";
      ctx.lineWidth = 1;
      for (let row = Math.floor(hopV / 2); row < data.height; row += hopV)
        for (let col = Math.floor(hopU / 2); col < data.width; col += hopU) {
          const i = row * data.width + col;
          if (data.material[i] < 0 || (gasOnly && data.material[i] !== 0))
            continue;
          const vector = [
            data.values.e_x_v_m[i],
            data.values.e_y_v_m[i],
            data.values.e_z_v_m[i],
          ];
          const eu = vector.reduce((sum, e, j) => sum + e * data.basis.u[j], 0),
            ev = vector.reduce((sum, e, j) => sum + e * data.basis.v[j], 0);
          const length = Math.hypot(eu, ev);
          if (length < 1e-12) continue;
          const point = project([data.u_nm[col], data.v_nm[row]]),
            dx = (eu / length) * 12,
            dy = ((down ? ev : -ev) / length) * 12;
          const a: Point = [point[0] - dx / 2, point[1] - dy / 2],
            b: Point = [point[0] + dx / 2, point[1] + dy / 2];
          ctx.beginPath();
          ctx.moveTo(...a);
          ctx.lineTo(...b);
          ctx.moveTo(
            b[0] - dx * 0.35 + dy * 0.22,
            b[1] - dy * 0.35 - dx * 0.22,
          );
          ctx.lineTo(...b);
          ctx.lineTo(
            b[0] - dx * 0.35 - dy * 0.22,
            b[1] - dy * 0.35 + dx * 0.22,
          );
          ctx.stroke();
        }
    }
    ctx.restore();
    ctx.strokeStyle = "#8eaaa0";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, w, h);
    ctx.fillStyle = "#617971";
    ctx.font = "10px sans-serif";
    for (let i = 0; i <= 4; i++) {
      const t = i / 4;
      ctx.textAlign = "center";
      ctx.fillText(
        number(minU + t * (maxU - minU)),
        left + t * w,
        top + h + 16,
      );
      ctx.textAlign = "right";
      ctx.fillText(
        number(down ? minV + t * (maxV - minV) : maxV - t * (maxV - minV)),
        left - 8,
        top + t * h + 3,
      );
    }
    ctx.textAlign = "center";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${data.axes.horizontal} [nm]`, left + w / 2, top + h + 33);
    ctx.save();
    ctx.translate(Math.max(14, left - 48), top + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${data.axes.vertical} [nm]${down ? " · 深さ" : ""}`, 0, 0);
    ctx.restore();
    const barW = Math.min(230, size - 100),
      barX = (size - barW) / 2,
      barY = height - 47;
    for (let x = 0; x < barW; x++) {
      ctx.fillStyle = `rgb(${rgb(low + ((high - low) * x) / barW, low, high, signed)})`;
      ctx.fillRect(barX + x, barY, 1.5, 9);
    }
    ctx.fillStyle = "#405952";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(number(low), barX, barY + 23);
    ctx.textAlign = "right";
    ctx.fillText(number(high), barX + barW, barY + 23);
    ctx.textAlign = "center";
    ctx.fillText(`${label} [${unit}]`, size / 2, barY - 7);
    ctx.font = "9px sans-serif";
    ctx.fillText(
      `RF位相 ${data.field_phase_deg}° · 更新 ${data.step}${plane === "vertical" ? ` · θ = ${number(data.angle_deg)}°` : ""}`,
      size / 2,
      height - 7,
    );
  }, [
    data,
    texture,
    size,
    center,
    zoom,
    plane,
    normalAxis,
    quantity,
    surface,
    low,
    high,
    signed,
    contours,
    boundaries,
    arrows,
    gasOnly,
    label,
    unit,
  ]);

  function pointer(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!data || !mapping.current || !canvas.current) return;
    const map = mapping.current,
      rect = canvas.current.getBoundingClientRect();
    const x = event.clientX - rect.left,
      y = event.clientY - rect.top;
    if (drag.current) {
      const old = drag.current;
      setCenter([
        old.focus[0] - ((x - old.x) / map.w) * (map.maxU - map.minU),
        old.focus[1] -
          ((y - old.y) / map.h) * (map.maxV - map.minV) * (map.down ? 1 : -1),
      ]);
      setHover(null);
      return;
    }
    if (
      x < map.left ||
      x > map.left + map.w ||
      y < map.top ||
      y > map.top + map.h
    ) {
      setHover(null);
      return;
    }
    const u = map.minU + ((x - map.left) / map.w) * (map.maxU - map.minU);
    const v = map.down
      ? map.minV + ((y - map.top) / map.h) * (map.maxV - map.minV)
      : map.maxV - ((y - map.top) / map.h) * (map.maxV - map.minV);
    if (surface) {
      let patch = data.surface_polygons.find((p: Obj) =>
        insideTriangle([u, v], p.points_nm),
      );
      if (!patch) {
        let best = 7;
        for (const p of data.surface_segments) {
          const d = distanceToSegment(
            [x, y],
            map.project(p.points_nm[0]),
            map.project(p.points_nm[1]),
          );
          if (d < best) {
            best = d;
            patch = p;
          }
        }
      }
      setHover({ u, v, value: patch?.sigma_c_m2, patch: patch?.patch });
      return;
    }
    const col = Math.round(
      ((u - data.u_nm[0]) / (data.u_nm.at(-1) - data.u_nm[0])) *
        (data.width - 1),
    );
    const row = Math.round(
      ((v - data.v_nm[0]) / (data.v_nm.at(-1) - data.v_nm[0])) *
        (data.height - 1),
    );
    const i = row * data.width + col;
    const within =
      col >= 0 && row >= 0 && col < data.width && row < data.height;
    const valid =
      within && data.material[i] >= 0 && (!gasOnly || data.material[i] === 0);
    setHover({
      u: within ? data.u_nm[col] : u,
      v: within ? data.v_nm[row] : v,
      value: valid ? data.values[quantity][i] : undefined,
      material: valid
        ? data.material[i] === 0
          ? "気相"
          : "SiO₂"
        : "FEM領域外・非表示",
    });
  }
  function exportCSV() {
    if (!data) return;
    const meta = [
      plane,
      data.position_nm,
      data.angle_deg,
      data.step,
      data.time_s,
    ];
    const xyz = (p: Point) =>
      data.basis.u.map(
        (u: number, i: number) =>
          u * p[0] +
          data.basis.v[i] * p[1] +
          data.basis.normal[i] * data.position_nm,
      );
    let rows: any[][];
    if (surface)
      rows = [
        [
          "plane",
          "position_nm",
          "angle_deg",
          "step",
          "time_s",
          "patch",
          "kind",
          "u1_nm",
          "v1_nm",
          "u2_nm",
          "v2_nm",
          "u3_nm",
          "v3_nm",
          "sigma_c_m2",
        ],
        ...[...data.surface_segments, ...data.surface_polygons].map(
          (p: Obj) => [
            ...meta,
            p.patch,
            p.points_nm.length === 3 ? "triangle" : "segment",
            ...p.points_nm.flat(),
            ...(p.points_nm.length === 2 ? ["", ""] : []),
            p.sigma_c_m2,
          ],
        ),
      ];
    else {
      rows = [
        [
          "plane",
          "position_nm",
          "angle_deg",
          "step",
          "time_s",
          "x_nm",
          "y_nm",
          "z_nm",
          "material",
          ...Object.keys(data.values),
        ],
      ];
      for (let row = 0; row < data.height; row++)
        for (let col = 0; col < data.width; col++) {
          const i = row * data.width + col,
            valid =
              data.material[i] >= 0 && (!gasOnly || data.material[i] === 0);
          rows.push([
            ...meta,
            ...xyz([data.u_nm[col], data.v_nm[row]]),
            valid ? (data.material[i] === 0 ? "gas" : "SiO2") : "outside",
            ...Object.values(data.values).map((values: any) =>
              valid ? values[i] : "",
            ),
          ]);
        }
    }
    download(
      new Blob(["\ufeff" + rows.map((row) => row.join(",")).join("\r\n")], {
        type: "text/csv;charset=utf-8",
      }),
      `${jobId}_slice_${plane}_${quantity}_step${data.step}.csv`,
    );
  }
  function changePlane(value: string) {
    setPlane(value);
    setPosition(value === "xy" ? String(result.derived.depth_nm / 2) : "0");
    setZoom(1);
    setCenter(null);
  }
  return (
    <section className="panel section-plot">
      <div className="section-header">
        <h2>2D断面プロット</h2>
        <span className="section-tag">RF位相 0° · 保存済みFEM解</span>
      </div>
      <div className="slice-controls">
        <label>
          断面の向き
          <select
            aria-label="断面の向き"
            value={plane}
            onChange={(e) => changePlane(e.target.value)}
          >
            <option value="xz">XZ · y位置を指定</option>
            <option value="yz">YZ · x位置を指定</option>
            <option value="xy">XY · 深さzを指定</option>
            <option value="vertical">縦断面 · 角度を指定</option>
          </select>
        </label>
        <label>
          {normalAxis}位置 [nm]
          <input
            aria-label="断面位置 nm"
            type="number"
            step="any"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
          />
        </label>
        <label>
          表示量
          <select
            aria-label="断面の表示量"
            value={quantity}
            onChange={(e) => {
              setQuantity(e.target.value);
              if (e.target.value === "sigma_c_m2") setGasOnly(false);
              setRangeMin("");
              setRangeMax("");
              setHover(null);
            }}
          >
            {Object.entries(quantities).map(([key, [name, units]]: any) => (
              <option key={key} value={key}>
                {name} [{units}]
              </option>
            ))}
          </select>
        </label>
        {plane === "vertical" && (
          <label>
            x軸からの角度 [°]
            <input
              aria-label="断面角度 deg"
              type="number"
              min="-180"
              max="180"
              step="any"
              value={angle}
              onChange={(e) => {
                setAngle(e.target.value);
                setCenter(null);
                setZoom(1);
              }}
            />
          </label>
        )}
        <label>
          表示格子
          <select
            aria-label="断面の解像度"
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
          >
            <option value={121}>軽量 · 長辺121点</option>
            <option value={241}>標準 · 長辺241点</option>
            <option value={401}>詳細 · 長辺401点</option>
          </select>
        </label>
        <div className="slice-position">
          <input
            aria-label="断面位置スライダー"
            type="range"
            min={normalRange[0]}
            max={normalRange[1]}
            step={(normalRange[1] - normalRange[0]) / 500}
            value={Number(position) || 0}
            onChange={(e) => setPosition(e.target.value)}
          />
          <button
            className="text-button"
            onClick={() =>
              setPosition(
                plane === "xy" ? String(result.derived.depth_nm / 2) : "0",
              )
            }
          >
            中央へ
          </button>
          <small>
            {number(normalRange[0])}～{number(normalRange[1])} nm
          </small>
        </div>
      </div>
      {plane === "vertical" && (
        <p className="hint">
          水平軸sは指定角度の方向、位置nはその方向に垂直な水平距離です。n=0はホール中心を通ります。
        </p>
      )}
      <div className="slice-toolbar">
        <label>
          <input
            type="checkbox"
            checked={boundaries}
            onChange={(e) => setBoundaries(e.target.checked)}
          />
          材料境界
        </label>
        <label>
          <input
            type="checkbox"
            checked={gasOnly}
            disabled={surface}
            onChange={(e) => setGasOnly(e.target.checked)}
          />
          気相のみ
        </label>
        <label>
          <input
            type="checkbox"
            checked={contours}
            disabled={quantity !== "potential_v"}
            onChange={(e) => setContours(e.target.checked)}
          />
          等電位線
        </label>
        <label>
          <input
            type="checkbox"
            checked={arrows}
            disabled={surface}
            onChange={(e) => setArrows(e.target.checked)}
          />
          電場の方向
        </label>
        <div className="slice-zoom">
          <button
            aria-label="断面を縮小"
            disabled={zoom === 1}
            onClick={() => setZoom((v) => Math.max(1, v / 2))}
          >
            <Minus size={14} />
          </button>
          <span>{zoom}×</span>
          <button
            aria-label="断面を拡大"
            disabled={zoom >= 16}
            onClick={() => setZoom((v) => Math.min(16, v * 2))}
          >
            <Plus size={14} />
          </button>
          <button
            aria-label="断面の表示範囲をリセット"
            onClick={() => {
              setZoom(1);
              setCenter(null);
            }}
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>
      <div className="slice-color-range">
        <label>
          色範囲の下限
          <input
            aria-label="色範囲の下限"
            type="number"
            step="any"
            placeholder={number(autoRange[0])}
            value={rangeMin}
            onChange={(e) => setRangeMin(e.target.value)}
          />
        </label>
        <label>
          上限
          <input
            aria-label="色範囲の上限"
            type="number"
            step="any"
            placeholder={number(autoRange[1])}
            value={rangeMax}
            onChange={(e) => setRangeMax(e.target.value)}
          />
        </label>
        <button
          className="text-button"
          onClick={() => {
            setRangeMin("");
            setRangeMax("");
          }}
        >
          自動範囲
        </button>
        <div className="slice-export">
          <button
            className="secondary"
            disabled={!data || loading}
            onClick={() =>
              canvas.current?.toBlob((blob) => {
                if (blob && data)
                  download(
                    blob,
                    `${jobId}_slice_${plane}_${quantity}_step${data.step}.png`,
                  );
              })
            }
          >
            <ArrowDownToLine size={14} />
            PNG保存
          </button>
          <button
            className="secondary"
            disabled={!data || loading}
            onClick={exportCSV}
          >
            <ArrowDownToLine size={14} />
            断面CSV
          </button>
        </div>
      </div>
      {!rangeValid && (
        <p role="alert" className="slice-error">
          色範囲は下限と上限を両方指定し、下限を上限より小さくしてください。
        </p>
      )}
      <div ref={host} className="slice-canvas-host" aria-busy={loading}>
        {data ? (
          <canvas
            ref={canvas}
            aria-label={`${label}の2D断面`}
            data-step={data.step}
            data-job-id={jobId}
            data-plane={data.plane}
            data-points={data.valid_points}
            onPointerMove={pointer}
            onPointerLeave={() => {
              if (!drag.current) setHover(null);
            }}
            onPointerDown={(e) => {
              if (!mapping.current || zoom === 1) return;
              const rect = e.currentTarget.getBoundingClientRect();
              drag.current = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
                focus: mapping.current.focus,
              };
              e.currentTarget.setPointerCapture(e.pointerId);
            }}
            onPointerUp={() => {
              drag.current = null;
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
          />
        ) : (
          <div className="slice-placeholder" role={error ? "alert" : "status"}>
            {error ||
              (loading
                ? "断面を計算しています…"
                : "断面条件を指定してください。")}
          </div>
        )}
      </div>
      <div className="slice-readout" aria-live="polite">
        {hover && data ? (
          <span>
            {surface ? "壁面" : "格子点"}: {data.axes.horizontal}=
            {number(hover.u)}, {data.axes.vertical}={number(hover.v)} nm ·{" "}
            {hover.material ??
              (hover.patch !== undefined
                ? `パッチ ${hover.patch}`
                : "帯電面なし")}{" "}
            · {label} {hover.value === undefined ? "—" : number(hover.value)}{" "}
            {hover.value === undefined ? "" : unit}
          </span>
        ) : (
          <span>
            プロット上にポインターを置くと座標と値を表示します。拡大後はドラッグで移動できます。
          </span>
        )}
      </div>
      {data && (
        <p className="hint slice-info">
          {data.width} × {data.height}格子 · 交差要素{" "}
          {data.intersected_elements.toLocaleString()} · 更新 {data.step} ·{" "}
          {number(data.time_s * 1e6)} µs
        </p>
      )}
      {data && !data.valid_points && (
        <p className="slice-error">
          この位置には面積を持つFEM断面がありません。位置を領域の内側へ変更してください。
        </p>
      )}
      <p className="hint">
        灰色はFEM領域外または非表示の領域です。電位は一次FEM補間、電場は要素内の保存値です。深さzは下向きに増加します。矢印は断面内の電場方向を規格化した長さで表示します。
      </p>
      <p className="hint">
        {surface
          ? "表面電荷密度は帯電面と断面の交線を着色します。帯電面が断面と一致する場合は面を着色します。体積内の電荷密度ではありません。CSVは交線・面パッチを保存します。"
          : "等電位線は表示格子上の補間です。CSVは断面全体の格子値を保存し、PNGは現在の表示範囲を保存します。"}
      </p>
    </section>
  );
}
