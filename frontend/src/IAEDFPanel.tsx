import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDownToLine, ArrowRight, Play, Square, Waves } from "lucide-react";

type Obj = Record<string, any>;
const statusLabels: Obj = {
  queued: "待機中",
  running: "実行中",
  completed: "完了",
  cancelled: "中止",
  failed: "失敗",
  unavailable: "状態不明",
};
async function api(path: string, data?: any) {
  const response = await fetch(
    "/api/iaedf" + path,
    data === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        },
  );
  const body = await response.json();
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  return body;
}
function Field({
  label,
  value,
  change,
  unit = "",
  step = "any",
  min,
  max,
}: any) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <div>
        <input
          type="number"
          aria-label={label}
          aria-description={unit}
          value={value ?? ""}
          step={step}
          min={min}
          max={max}
          onChange={(e) => {
            if (e.target.value !== "") change(Number(e.target.value));
          }}
        />
        <span>{unit}</span>
      </div>
    </label>
  );
}
function Panel({ title, children }: any) {
  return (
    <section className="panel">
      <div className="panel-title">
        <h3>{title}</h3>
      </div>
      {children}
    </section>
  );
}
function save(name: string, data: any) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}
function WaveEditor({ title, wave, change, ring, template, error }: any) {
  const set = (key: string, value: any) => change({ ...wave, [key]: value });
  return (
    <Panel title={title}>
      <label className="form-label">
        波形
        <select
          aria-label={title + "の種類"}
          value={wave.mode}
          onChange={(e) => set("mode", e.target.value)}
        >
          <option value="sinusoid">正弦波</option>
          <option value="csv">CSV周期波形</option>
          {ring && <option value="scaled_wafer">ウェハ波形から生成</option>}
        </select>
      </label>
      {wave.mode === "sinusoid" ? (
        <div className="fields two">
          <Field
            label="DC電位"
            value={wave.sinusoid_dc_V}
            change={(v: number) => set("sinusoid_dc_V", v)}
            unit="V"
          />
          <Field
            label="RF振幅"
            value={wave.sinusoid_amplitude_V}
            change={(v: number) => set("sinusoid_amplitude_V", v)}
            unit="V"
          />
          <Field
            label="位相オフセット"
            value={wave.sinusoid_phase_offset_deg}
            change={(v: number) => set("sinusoid_phase_offset_deg", v)}
            unit="°"
          />
        </div>
      ) : wave.mode === "scaled_wafer" ? (
        <div className="fields two">
          <Field
            label="ウェハ倍率"
            value={wave.wafer_scale}
            change={(v: number) => set("wafer_scale", v)}
          />
          <Field
            label="位相差"
            value={wave.wafer_phase_offset_deg}
            change={(v: number) => set("wafer_phase_offset_deg", v)}
            unit="°"
          />
          <Field
            label="DCオフセット"
            value={wave.dc_offset_V}
            change={(v: number) => set("dc_offset_V", v)}
            unit="V"
          />
        </div>
      ) : (
        <>
          <label className="form-label">
            波形CSV
            <input
              type="file"
              accept=".csv,.txt"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                try {
                  if (file.size > 1024 * 1024)
                    throw new Error("波形CSVは1MB以下にしてください。");
                  set("csv_text", (await file.text()).replace(/^\uFEFF/, ""));
                } catch (err: any) {
                  error(err.message);
                }
              }}
            />
          </label>
          <button
            className="secondary"
            onClick={() =>
              change({
                ...wave,
                csv_text: template,
                x_axis: "time_s",
                delimiter: ",",
                skip_header_rows: 1,
                x_column: 0,
                voltage_column: 1,
              })
            }
          >
            <Waves size={15} />
            検証用5高調波を使用
          </button>
          <div className="fields two">
            <label className="form-label">
              CSV横軸
              <select
                value={wave.x_axis}
                onChange={(e) => set("x_axis", e.target.value)}
              >
                {["time_s", "time_ns", "time_us", "phase_deg", "phase_rad"].map(
                  (k) => (
                    <option key={k}>{k}</option>
                  ),
                )}
              </select>
            </label>
            <Field
              label="位相オフセット"
              value={wave.phase_offset_deg}
              change={(v: number) => set("phase_offset_deg", v)}
              unit="°"
            />
            <Field
              label="電圧倍率"
              value={wave.voltage_scale}
              change={(v: number) => set("voltage_scale", v)}
            />
            <Field
              label="電圧オフセット"
              value={wave.voltage_offset_V}
              change={(v: number) => set("voltage_offset_V", v)}
              unit="V"
            />
          </div>
          <details>
            <summary>CSVの列・本文</summary>
            <div className="fields two">
              <Field
                label="ヘッダー行数"
                value={wave.skip_header_rows}
                change={(v: number) => set("skip_header_rows", v)}
                min={0}
                step={1}
              />
              <Field
                label="横軸の列（0始まり）"
                value={wave.x_column}
                change={(v: number) => set("x_column", v)}
                min={0}
                step={1}
              />
              <Field
                label="電圧の列（0始まり）"
                value={wave.voltage_column}
                change={(v: number) => set("voltage_column", v)}
                min={0}
                step={1}
              />
              <label className="form-label">
                区切り文字
                <input
                  value={wave.delimiter}
                  onChange={(e) => set("delimiter", e.target.value)}
                />
              </label>
            </div>
            <textarea
              aria-label={title + "CSV本文"}
              value={wave.csv_text ?? ""}
              onChange={(e) => set("csv_text", e.target.value)}
              rows={5}
            />
          </details>
        </>
      )}
    </Panel>
  );
}
function LineChart({ x, y, xlabel, ylabel, color = "#26705e" }: any) {
  const finite = x
    .map((v: number, i: number) => [v, y[i]])
    .filter((p: number[]) => p.every(Number.isFinite));
  if (!finite.length) return <p>データがありません。</p>;
  const xmin = Math.min(...x),
    xmax = Math.max(...x),
    ymin = Math.min(...finite.map((p: number[]) => p[1]), 0),
    ymax = Math.max(...finite.map((p: number[]) => p[1]), 0);
  const span = Math.max(ymax - ymin, 1e-12);
  const points = finite
    .map(
      ([a, b]: number[]) =>
        `${52 + ((a - xmin) / (xmax - xmin || 1)) * 500},${200 - ((b - ymin) / span) * 165}`,
    )
    .join(" ");
  return (
    <svg
      viewBox="0 0 600 250"
      role="img"
      aria-label={xlabel + "の分布"}
      className="iaedf-chart"
    >
      <path d="M52 25 V200 H552" stroke="#aebeb5" fill="none" />
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <path d={`M52 ${200 - f * 165} H552`} stroke="#e5ebe7" />
          <text x="47" y={204 - f * 165} textAnchor="end">
            {(ymin + f * span).toPrecision(2)}
          </text>
        </g>
      ))}
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={52 + f * 500} y="218" textAnchor="middle">
          {(xmin + f * (xmax - xmin)).toFixed(1)}
        </text>
      ))}
      <text x="300" y="242" textAnchor="middle">
        {xlabel}
      </text>
      <text x="16" y="125" transform="rotate(-90 16 125)" textAnchor="middle">
        {ylabel}
      </text>
    </svg>
  );
}
function JointChart({ data }: { data: Obj }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const rows: number[][] = data.density;
    const max = Math.max(...rows.flat(), 1e-30);
    const raster = context.createImageData(rows[0].length, rows.length);
    rows.forEach((row, y) =>
      row.forEach((value, x) => {
        const f = Math.sqrt(value / max),
          i = ((rows.length - 1 - y) * row.length + x) * 4;
        raster.data.set([245 - 219 * f, 248 - 130 * f, 246 - 160 * f, 255], i);
      }),
    );
    context.putImageData(raster, 0, 0);
  }, [data]);
  return (
    <div className="iaedf-joint">
      <div className="iaedf-joint-y">
        <span>{data.energy_edges_ev.at(-1).toFixed(1)}</span>
        <span>エネルギー [eV]</span>
        <span>0</span>
      </div>
      <div>
        <canvas
          ref={canvas}
          width={data.density[0].length}
          height={data.density.length}
          role="img"
          aria-label="角度とエネルギーの相関分布"
        />
        <div className="iaedf-axis">
          <span>−90°</span>
          <span>0°</span>
          <span>90°</span>
        </div>
        <p>符号付き接線角 [°] / 濃度は確率密度（平方根色尺度）</p>
      </div>
    </div>
  );
}
function GeometryEditor({ geo, change }: any) {
  const pts: number[][] = geo.points_m;
  const high = Math.max(...pts.map((p) => p[1]), 1e-6) * 1.4;
  return (
    <Panel title="2D表面形状・材質">
      <div className="fields two">
        <Field
          label="領域の横幅"
          value={geo.domain_length_m * 1e3}
          change={(v: number) => change({ ...geo, domain_length_m: v * 1e-3 })}
          unit="mm"
          min={0}
        />
        <Field
          label="表面平滑化幅"
          value={geo.smoothing_m * 1e3}
          change={(v: number) => change({ ...geo, smoothing_m: v * 1e-3 })}
          unit="mm"
          min={0}
        />
      </div>
      <svg
        viewBox="0 0 600 150"
        role="img"
        aria-label="2Dウェハとリングの表面形状"
        className="iaedf-geometry"
      >
        {pts.slice(0, -1).map((p, i) => (
          <line
            key={i}
            x1={20 + (p[0] / geo.domain_length_m) * 560}
            y1={130 - (p[1] / high) * 100}
            x2={20 + (pts[i + 1][0] / geo.domain_length_m) * 560}
            y2={130 - (pts[i + 1][1] / high) * 100}
            stroke={
              geo.segment_materials[i] === "wafer"
                ? "#23715e"
                : geo.segment_materials[i] === "ring"
                  ? "#c99446"
                  : "#8c91aa"
            }
            strokeWidth={5}
          />
        ))}
        {pts.map((p, i) => (
          <circle
            key={i}
            cx={20 + (p[0] / geo.domain_length_m) * 560}
            cy={130 - (p[1] / high) * 100}
            r={4}
            fill="#314e46"
          />
        ))}
        <text x="20" y="146">
          0 mm
        </text>
        <text x="580" y="146" textAnchor="end">
          {(geo.domain_length_m * 1e3).toFixed(2)} mm
        </text>
      </svg>
      <div className="iaedf-points">
        <div className="iaedf-point">
          <b>x [mm]</b>
          <b>高さ [mm]</b>
          <b>右側の材質</b>
        </div>
        {pts.map((p, i) => (
          <div className="iaedf-point" key={i}>
            {[0, 1].map((axis) => (
              <input
                key={axis}
                type="number"
                step="any"
                aria-label={`制御点${i + 1} ${axis ? "高さ" : "x"}`}
                value={p[axis] * 1e3}
                onChange={(e) => {
                  const next = structuredClone(geo);
                  next.points_m[i][axis] = Number(e.target.value) * 1e-3;
                  change(next);
                }}
              />
            ))}
            {i < pts.length - 1 ? (
              <select
                aria-label={`区間${i + 1}の材質`}
                value={geo.segment_materials[i]}
                onChange={(e) => {
                  const next = structuredClone(geo);
                  next.segment_materials[i] = e.target.value;
                  change(next);
                }}
              >
                <option value="wafer">ウェハ</option>
                <option value="ring">リング</option>
                <option value="insulator">絶縁体</option>
              </select>
            ) : (
              <span>右端</span>
            )}
          </div>
        ))}
      </div>
      <p className="hint">
        制御点の追加・削除は詳細JSONで編集できます。絶縁体はIAEDF側では非帯電です。
      </p>
    </Panel>
  );
}

export default function IAEDFPanel({
  holeConfig,
  onApply,
  active,
}: {
  holeConfig: Obj;
  onApply: (result: Obj) => void;
  active: boolean;
}) {
  const [presets, setPresets] = useState<Obj | null>(null),
    [models, setModels] = useState<Obj>({}),
    [model, setModel] = useState("1d");
  const [name, setName] = useState("IAEDFシース"),
    [jobs, setJobs] = useState<Obj[]>([]),
    [selected, setSelected] = useState("");
  const [job, setJob] = useState<Obj | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [json, setJson] = useState("");
  const [pressure, setPressure] = useState(0),
    [minimum, setMinimum] = useState(3.2),
    [maximum, setMaximum] = useState(12.8),
    [azimuth, setAzimuth] = useState(0),
    [plot, setPlot] = useState<Obj | null>(null);
  const [pressureText, setPressureText] = useState("0, 5, 20");
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const refresh = useCallback(async () => setJobs(await api("/jobs")), []);
  useEffect(() => {
    api("/defaults")
      .then((d) => {
        setPresets(d);
        setModels({ "1d": d["1d"], "2d": d["2d"] });
      })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!active) return;
    refresh().catch((e) => setError(e.message));
    const timer = setInterval(
      () => refresh().catch((e) => setError(e.message)),
      1000,
    );
    return () => clearInterval(timer);
  }, [active, refresh]);
  const status = jobs.find((j) => j.id === selected)?.status;
  const progress = jobs.find((j) => j.id === selected)?.progress;
  useEffect(() => {
    if (!selected || !active) return;
    let cancelled = false;
    api("/jobs/" + selected)
      .then((d) => {
        if (!cancelled) setJob(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selected, active, status, progress]);
  useEffect(() => {
    setPlot(null);
    if (!active || status !== "completed" || !job || job.id !== selected)
      return;
    const params = new URLSearchParams({ pressure_index: String(pressure) });
    if (job.request.model === "2d") {
      params.set("collector_min_m", String(minimum * 1e-3));
      params.set("collector_max_m", String(maximum * 1e-3));
    }
    let cancelled = false;
    const timer = setTimeout(
      () =>
        api(`/jobs/${selected}/distribution?${params}`)
          .then((d) => {
            if (!cancelled) {
              setPlot(d);
              setError("");
            }
          })
          .catch((e) => {
            if (!cancelled) setError(e.message);
          }),
      200,
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [active, status, selected, job, pressure, minimum, maximum]);
  const action = async (task: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const config = models[model];
  const update = (path: string, value: any) =>
    setModels((prev) => {
      const next = structuredClone(prev);
      const keys = path.split(".");
      let target = next[model];
      keys.slice(0, -1).forEach((k) => (target = target[k]));
      target[keys.at(-1)!] = value;
      return next;
    });
  const load = async (data: Obj) => {
    const c = data.config ?? data;
    const m = data.model ?? (c.wafer_waveform ? "2d" : "1d");
    if (!["1d", "2d"].includes(m))
      throw new Error("1dまたは2dのIAEDF設定を選んでください。");
    const normalized = await api("/validate", {
      name: data.name ?? name,
      model: m,
      config: c,
    });
    setModels((prev) => ({ ...prev, [m]: normalized.config }));
    setModel(m);
    setPressureText(normalized.config.gas.pressures_mTorr.join(", "));
    if (data.name) setName(data.name);
  };
  const select = (id: string) => {
    setSelected(id);
    setJob(null);
    setPlot(null);
    setPressure(0);
  };
  if (!presets || !config) return <p>IAEDFモデルを読み込み中…</p>;
  const chosen2D = job?.request.model === "2d";
  return (
    <div className="iaedf-workspace">
      <div className="iaedf-heading">
        <div>
          <h2>シースからホールへ</h2>
          <p>
            IAEDF-Simの1D /
            2Dモデルで分布を計算し、3成分速度とRF位相を入口へ渡します。
          </p>
        </div>
        <span className="pill">IAEDF: CPU / ホール: CPU・GPU</span>
      </div>
      {error && (
        <div className="message error" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="IAEDFエラーを閉じる">
            ×
          </button>
        </div>
      )}
      <div className="iaedf-layout">
        <div className="iaedf-settings">
          <Panel title="モデルと条件">
            <label className="form-label">
              ケース名
              <input
                value={name}
                maxLength={120}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <label className="form-label">
              IAEDFモデル
              <select
                aria-label="IAEDFモデル"
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  setPressureText(
                    models[e.target.value].gas.pressures_mTorr.join(", "),
                  );
                }}
              >
                <option value="1d">1Dシース / moving front</option>
                <option value="2d">2Dウェハ・エッジリング</option>
              </select>
            </label>
            <div className="fields two">
              <label className="form-label">
                イオン / 断面積
                <select
                  aria-label="イオン / 断面積"
                  value={config.gas.xsec_csv_name}
                  onChange={(e) => {
                    const isHe = e.target.value.includes("_he_");
                    update("plasma.ion_mass_amu", isHe ? 4.002602 : 39.948);
                    update("gas.xsec_csv_name", e.target.value);
                  }}
                >
                  <option value="xsec_ar_ion_phelps_lxcat.csv">Ar+ / Ar</option>
                  <option value="xsec_he_ion_phelps_lxcat.csv">He+ / He</option>
                </select>
              </label>
              <Field
                label="周波数"
                value={config.plasma.frequency_Hz / 1e6}
                change={(v: number) => update("plasma.frequency_Hz", v * 1e6)}
                unit="MHz"
                min={0}
              />
              <Field
                label="電子温度"
                value={config.plasma.electron_temperature_eV}
                change={(v: number) =>
                  update("plasma.electron_temperature_eV", v)
                }
                unit="eV"
                min={0}
              />
              <Field
                label="シース端密度"
                value={config.plasma.sheath_edge_density_m3}
                change={(v: number) =>
                  update("plasma.sheath_edge_density_m3", v)
                }
                unit="m⁻³"
                min={0}
              />
              <Field
                label="ガス温度"
                value={config.gas.gas_temperature_K}
                change={(v: number) => update("gas.gas_temperature_K", v)}
                unit="K"
                min={0}
              />
              <label className="form-label">
                断面積モデル
                <select
                  aria-label="断面積モデル"
                  value={config.gas.cross_section_source}
                  onChange={(e) =>
                    update("gas.cross_section_source", e.target.value)
                  }
                >
                  <option value="lxcat_phelps">LXCat / Phelps</option>
                  <option value="approximation">近似式</option>
                </select>
              </label>
            </div>
            <label className="form-label">
              圧力ケース [mTorr / カンマ区切り]
              <input
                value={pressureText}
                onChange={(e) => setPressureText(e.target.value)}
              />
            </label>
            <div className="iaedf-actions">
              <label className="secondary iaedf-upload">
                既存config.jsonを読込
                <input
                  type="file"
                  accept=".json"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (file)
                      await action(async () =>
                        load(
                          JSON.parse(
                            (await file.text()).replace(/^\uFEFF/, ""),
                          ),
                        ),
                      );
                    e.target.value = "";
                  }}
                />
              </label>
              <button
                className="secondary"
                onClick={() =>
                  save("iaedf-request.json", {
                    name,
                    model,
                    config: {
                      ...config,
                      gas: {
                        ...config.gas,
                        pressures_mTorr: pressureText.split(",").map(Number),
                      },
                    },
                  })
                }
              >
                <ArrowDownToLine size={15} />
                条件保存
              </button>
            </div>
          </Panel>
          <WaveEditor
            title={model === "1d" ? "駆動波形" : "ウェハ波形"}
            wave={model === "1d" ? config.waveform : config.wafer_waveform}
            change={(v: Obj) =>
              update(model === "1d" ? "waveform" : "wafer_waveform", v)
            }
            template={presets.tailored_waveform_csv}
            error={setError}
          />
          {model === "2d" && (
            <>
              <WaveEditor
                title="リング波形"
                wave={config.ring_waveform}
                change={(v: Obj) => update("ring_waveform", v)}
                ring
                template={presets.tailored_waveform_csv}
                error={setError}
              />
              <GeometryEditor
                geo={config.geometry}
                change={(v: Obj) => update("geometry", v)}
              />
            </>
          )}
          <Panel title="粒子追跡・回路">
            <div className="fields two">
              <Field
                label="粒子数 / 圧力ケース"
                value={config.tpmc.n_particles}
                change={(v: number) => update("tpmc.n_particles", v)}
                min={10}
                max={2000000}
                step={100}
              />
              <Field
                label="RF周期の時間分割"
                value={config.tpmc.steps_per_rf_period}
                change={(v: number) => update("tpmc.steps_per_rf_period", v)}
                min={10}
                step={1}
              />
              <Field
                label="最大追跡周期"
                value={config.tpmc.max_rf_periods}
                change={(v: number) => update("tpmc.max_rf_periods", v)}
                unit="周期"
                min={0}
              />
              <Field
                label="イオン温度"
                value={config.tpmc.ion_temperature_eV}
                change={(v: number) => update("tpmc.ion_temperature_eV", v)}
                unit="eV"
                min={0}
              />
              <Field
                label="乱数シード"
                value={config.tpmc.seed}
                change={(v: number) => update("tpmc.seed", v)}
                min={0}
                step={1}
              />
              <Field
                label="回路位相点数"
                value={config.circuit.phase_points}
                change={(v: number) => update("circuit.phase_points", v)}
                min={64}
                max={8192}
                step={1}
              />
              {model === "1d" ? (
                <Field
                  label="駆動 / 接地の面積比"
                  value={config.circuit.powered_to_grounded_area_ratio}
                  change={(v: number) =>
                    update("circuit.powered_to_grounded_area_ratio", v)
                  }
                />
              ) : (
                <>
                  <Field
                    label="ウェハ / 接地の面積比"
                    value={config.electrodes.wafer_to_ground_area_ratio}
                    change={(v: number) =>
                      update("electrodes.wafer_to_ground_area_ratio", v)
                    }
                  />
                  <Field
                    label="リング / 接地の面積比"
                    value={config.electrodes.ring_to_ground_area_ratio}
                    change={(v: number) =>
                      update("electrodes.ring_to_ground_area_ratio", v)
                    }
                  />
                  <Field
                    label="2D格子 nx"
                    value={config.field2d.nx}
                    change={(v: number) => update("field2d.nx", v)}
                    min={17}
                    step={1}
                  />
                  <Field
                    label="2D格子 ny"
                    value={config.field2d.ny}
                    change={(v: number) => update("field2d.ny", v)}
                    min={17}
                    step={1}
                  />
                </>
              )}
            </div>
            {model === "2d" && (
              <>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={config.space_charge.enabled}
                    onChange={(e) =>
                      update("space_charge.enabled", e.target.checked)
                    }
                  />
                  IAEDF側の空間電荷補正を計算
                </label>
                <div className="fields two">
                  <Field
                    label="空間電荷の外部反復"
                    value={config.space_charge.outer_iterations}
                    change={(v: number) =>
                      update("space_charge.outer_iterations", v)
                    }
                    step={1}
                    min={1}
                  />
                  <Field
                    label="密度堆積用粒子数"
                    value={config.space_charge.deposition_particles}
                    change={(v: number) =>
                      update("space_charge.deposition_particles", v)
                    }
                    step={100}
                    min={10}
                  />
                </div>
              </>
            )}
            <details>
              <summary>静磁場・詳細設定</summary>
              <p className="hint">
                磁場はIAEDF側のイオンに作用します。ホール内の磁場力は省略します。
              </p>
              <div className="fields two">
                {["bx_T", "by_T", "bz_T"].map((k) => (
                  <Field
                    key={k}
                    label={k.replace("_T", "")}
                    value={config.magnetic[k] * 1e3}
                    change={(v: number) => update("magnetic." + k, v * 1e-3)}
                    unit="mT"
                  />
                ))}
              </div>
              <p className="hint">
                1D: x=入射方向、y/z=横方向。2D: x=横方向、y=上向き、z=面外。
              </p>
              <button
                className="secondary"
                onClick={() => setJson(JSON.stringify(config, null, 2))}
              >
                現在の条件をJSONへ展開
              </button>
              <textarea
                rows={8}
                aria-label="IAEDF詳細JSON"
                value={json}
                onChange={(e) => setJson(e.target.value)}
              />
              <button
                className="secondary"
                disabled={!json}
                onClick={() =>
                  action(async () => load({ model, config: JSON.parse(json) }))
                }
              >
                JSONを条件へ適用
              </button>
            </details>
            <div className="iaedf-actions">
              <button
                className="primary"
                disabled={busy}
                onClick={() =>
                  action(async () => {
                    const parts = pressureText.split(",");
                    if (parts.some((p) => !p.trim()))
                      throw new Error("圧力をカンマ区切りで指定してください。");
                    const pressures = parts.map(Number);
                    if (pressures.some((p) => !Number.isFinite(p) || p < 0))
                      throw new Error("圧力は非負の数値にしてください。");
                    const next = {
                      ...config,
                      gas: { ...config.gas, pressures_mTorr: pressures },
                    };
                    update("gas.pressures_mTorr", pressures);
                    const result = await api("/jobs", {
                      name,
                      model,
                      config: next,
                    });
                    select(result.id);
                    await refresh();
                  })
                }
              >
                <Play size={16} />
                IAEDFを計算
              </button>
              <button
                className="secondary"
                onClick={() => {
                  update("tpmc.n_particles", 2000);
                  if (model === "2d") {
                    update("field2d.nx", 97);
                    update("field2d.ny", 48);
                    update("space_charge.deposition_particles", 1000);
                    update("space_charge.outer_iterations", 2);
                  }
                }}
              >
                小規模検証の設定
              </button>
            </div>
            <p className="hint">
              IAEDFとホール計算は順番に実行します。小規模設定は動作確認用です。
            </p>
          </Panel>
        </div>
        <div className="iaedf-results">
          <Panel title="IAEDFジョブ">
            <label className="form-label">
              計算結果
              <select
                aria-label="計算結果"
                value={selected}
                onChange={(e) => select(e.target.value)}
              >
                <option value="">ジョブを選択</option>
                {jobs.map((j) => (
                  <option key={j.id} value={j.id}>
                    {j.name} / {j.model.toUpperCase()} /{" "}
                    {statusLabels[j.status] ?? j.status}
                  </option>
                ))}
              </select>
            </label>
            {job && (
              <>
                <div className="iaedf-status">
                  <strong>{statusLabels[job.status] ?? job.status}</strong>
                  <span>{job.stage}</span>
                </div>
                <progress
                  max={1}
                  value={job.progress ?? 0}
                  aria-label="IAEDF計算の進捗"
                />
                {job.error && <p className="message error">{job.error}</p>}
                <div className="iaedf-actions">
                  {["queued", "running"].includes(job.status) && (
                    <button
                      className="secondary"
                      disabled={busy}
                      onClick={() =>
                        action(async () => {
                          await api(`/jobs/${selected}/cancel`, {});
                          await refresh();
                        })
                      }
                    >
                      <Square size={14} />
                      計算を中止
                    </button>
                  )}
                  <button
                    className="secondary"
                    onClick={() => action(async () => load({ ...job.request }))}
                  >
                    この条件を編集
                  </button>
                  {["failed", "cancelled"].includes(job.status) && (
                    <button
                      className="secondary"
                      disabled={busy}
                      onClick={() =>
                        action(async () => {
                          const r = await api("/jobs", job.request);
                          select(r.id);
                          await refresh();
                        })
                      }
                    >
                      同じ条件で再計算
                    </button>
                  )}
                </div>
              </>
            )}
          </Panel>
          {job?.status === "completed" && (
            <>
              <Panel title="ホール入口へ接続">
                <label className="form-label">
                  圧力ケース
                  <select
                    aria-label="圧力ケース"
                    value={pressure}
                    onChange={(e) => setPressure(Number(e.target.value))}
                  >
                    {job.request.config.gas.pressures_mTorr.map(
                      (p: number, i: number) => (
                        <option value={i} key={i}>
                          p{i}: {p} mTorr
                        </option>
                      ),
                    )}
                  </select>
                </label>
                {chosen2D && (
                  <div className="fields two">
                    <Field
                      label="コレクタ x下限"
                      value={minimum}
                      change={setMinimum}
                      unit="mm"
                    />
                    <Field
                      label="コレクタ x上限"
                      value={maximum}
                      change={setMaximum}
                      unit="mm"
                    />
                  </div>
                )}
                <Field
                  label="ホール内の方位角"
                  value={azimuth}
                  change={setAzimuth}
                  unit="°"
                  min={-180}
                  max={180}
                />
                <p className="hint">
                  接線方向をSCAの+xへ対応させ、指定角度だけ回転します。表面内向き法線はSCAの+zです。
                </p>
                {plot && (
                  <div className="iaedf-metrics">
                    <span>
                      入射サンプル<b>{plot.sample_count.toLocaleString()}</b>
                    </span>
                    <span>
                      平均エネルギー<b>{plot.mean_energy_ev.toFixed(2)} eV</b>
                    </span>
                    <span>
                      平均極角<b>{plot.mean_polar_angle_deg.toFixed(2)}°</b>
                    </span>
                  </div>
                )}
                <button
                  className="primary"
                  disabled={busy || !plot}
                  onClick={() =>
                    action(async () => {
                      const sourceId = selectedRef.current;
                      const result = await api(`/jobs/${sourceId}/apply`, {
                        config: holeConfig,
                        pressure_index: pressure,
                        collector_min_m: chosen2D ? minimum * 1e-3 : null,
                        collector_max_m: chosen2D ? maximum * 1e-3 : null,
                        azimuth_deg: azimuth,
                      });
                      if (selectedRef.current === sourceId) onApply(result);
                    })
                  }
                >
                  <ArrowRight size={16} />
                  ホール入口に適用
                </button>
                <p className="hint">
                  ウェハ波形・電子温度・プラズマ電位・ガス圧力も共通化します。絶対流束はモデル推定、入口位置は一様円盤です。
                </p>
              </Panel>
              {plot && (
                <>
                  <Panel title="IEDF / エネルギー分布">
                    <LineChart
                      x={plot.iedf.x}
                      y={plot.iedf.y}
                      xlabel="エネルギー [eV]"
                      ylabel="確率密度 [eV⁻¹]"
                    />
                  </Panel>
                  <Panel title="IADF / 接線角分布">
                    <LineChart
                      x={plot.iadf.x}
                      y={plot.iadf.y}
                      xlabel="符号付き接線角 [°]"
                      ylabel="確率密度 [deg⁻¹]"
                    />
                  </Panel>
                  <Panel title="IAEDF / 角度・エネルギー相関">
                    <JointChart data={plot.iaedf} />
                  </Panel>
                </>
              )}
              <Panel title="RF波形・数値検証">
                <LineChart
                  x={job.plots.vp_waveform.phase_deg}
                  y={job.plots.vp_waveform[chosen2D ? "V_w" : "V_e"]}
                  xlabel="RF位相 [°]"
                  ylabel="電位 [V]"
                  color="#b37c36"
                />
                <p
                  className={
                    job.summary.validation.passed ? "hint" : "message error"
                  }
                >
                  {job.summary.validation.passed
                    ? "IAEDFの数値検証を通過しました。"
                    : "未達の検証項目があります。下の数値を確認してください。"}
                </p>
                <details>
                  <summary>検証値と計算ログ</summary>
                  <pre>{JSON.stringify(job.summary.validation, null, 2)}</pre>
                  {job.summary.log.map((s: string, i: number) => (
                    <p key={i}>{s}</p>
                  ))}
                </details>
                <div className="iaedf-actions">
                  {["raw", "config", "summary"].map((k) => (
                    <a
                      className="secondary"
                      key={k}
                      href={`/api/iaedf/jobs/${selected}/export/${k}`}
                    >
                      <ArrowDownToLine size={15} />
                      {k === "raw" ? "raw.npz" : k + ".json"}
                    </a>
                  ))}
                </div>
              </Panel>
            </>
          )}
          {!selected && (
            <Panel title="計算の流れ">
              <ol>
                <li>
                  1Dまたは2Dを選び、プラズマ・波形・粒子追跡条件を設定します。
                </li>
                <li>IAEDFを計算し、圧力ケースと2Dコレクタ範囲を選びます。</li>
                <li>
                  ホール入口に適用し、「設定」「実行」でホール計算を進めます。
                </li>
              </ol>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
