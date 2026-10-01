import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  FlaskConical,
  Layers3,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Settings2,
  SlidersHorizontal,
  Upload,
  Waves,
  X,
} from "lucide-react";
import GeometryView from "./GeometryView";
import ResultAnimation from "./ResultAnimation";

type Obj = Record<string, any>;
const labels: Obj = {
  uncharged: "帯電なし",
  fixed_charge: "指定電荷",
  self_consistent: "自己無撞着帯電",
  queued: "待機中",
  running: "実行中",
  paused: "停止中",
  completed: "完了",
  failed: "失敗",
};
async function api(path: string, data?: any) {
  const response = await fetch(
    "/api" + path,
    data === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        },
  );
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : JSON.stringify(result.detail),
    );
  return result;
}
function fmt(v: any, digits = 3) {
  return v === null || v === undefined
    ? "—"
    : Number(v).toLocaleString("ja-JP", { maximumFractionDigits: digits });
}
function sci(v: any) {
  return v === null || v === undefined ? "未評価" : Number(v).toExponential(3);
}
function percent(v: any) {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(2)}%`;
}
function NumberField({
  label,
  value,
  onChange,
  unit,
  step = 1,
  min,
  max,
}: any) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <div>
        <input
          type="number"
          value={value ?? ""}
          step={step}
          min={min}
          max={max}
          onChange={(e) => {
            if (e.target.value !== "") onChange(Number(e.target.value));
          }}
        />
        <span>{unit}</span>
      </div>
    </label>
  );
}
function Panel({ title, tag, children }: any) {
  return (
    <section className="panel">
      <div className="panel-title">
        <h3>{title}</h3>
        {tag && <span>{tag}</span>}
      </div>
      {children}
    </section>
  );
}
function download(filename: string, data: any) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [config, setConfig] = useState<Obj | null>(null),
    [derived, setDerived] = useState<Obj | null>(null);
  const [system, setSystem] = useState<Obj | null>(null),
    [tab, setTab] = useState("設定"),
    [selected, setSelected] = useState(0);
  const [jobs, setJobs] = useState<Obj[]>([]),
    [job, setSelectedJob] = useState(""),
    [result, setResult] = useState<Obj | null>(null),
    [jobConfig, setJobConfig] = useState<Obj | null>(null);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [geometryError, setGeometryError] = useState(""),
    [busy, setBusy] = useState(false);
  const [display, setDisplay] = useState("potential"),
    [sweepPath, setSweepPath] = useState("geometry.gap_nm"),
    [sweepValues, setSweepValues] = useState("50, 100, 150");
  const [distribution, setDistribution] = useState<File | null>(null),
    [sourceConfig, setSourceConfig] = useState<File | null>(null),
    [pressure, setPressure] = useState("p0"),
    [projection, setProjection] = useState("reject"),
    [speciesIndex, setSpeciesIndex] = useState(0);
  const [collectorMin, setCollectorMin] = useState(""),
    [collectorMax, setCollectorMax] = useState(""),
    [compare, setCompare] = useState<string[]>([]),
    [comparisons, setComparisons] = useState<Obj[]>([]);
  const [advanced, setAdvanced] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const activeJobRef = useRef("");
  const setJob = useCallback((id: string) => {
    activeJobRef.current = id;
    setSelectedJob(id);
  }, []);
  const setOwnedResult = useCallback(
    (data: Obj) => {
      if (activeJobRef.current === job) setResult(data);
    },
    [job],
  );
  const [timeSelection, setTimeSelection] = useState(0);
  const refresh = useCallback(
    () =>
      api("/jobs")
        .then(setJobs)
        .catch((e) => setError(e.message)),
    [],
  );
  useEffect(() => {
    api("/defaults").then((v) => {
      setConfig(v.config);
      setDerived(v.derived);
    });
    api("/system").then(setSystem);
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [refresh]);
  useEffect(() => {
    if (!config) return;
    setDerived(null);
    setGeometryError("");
    const timer = setTimeout(
      () =>
        api("/geometry", config)
          .then(setDerived)
          .catch((e) => setGeometryError(e.message)),
      250,
    );
    return () => clearTimeout(timer);
  }, [config]);
  const active = jobs.find((j) => j.id === job);
  useEffect(() => {
    setResult(null);
    setJobConfig(null);
  }, [job]);
  useEffect(() => {
    if (!job) return;
    let cancelled = false;
    api("/jobs/" + job)
      .then((v) => {
        if (!cancelled && activeJobRef.current === job) setJobConfig(v.config);
      })
      .catch((e) => {
        if (!cancelled && activeJobRef.current === job) setError(e.message);
      });
    api("/jobs/" + job + "/results")
      .then((v) => {
        if (!cancelled && activeJobRef.current === job) setResult(v);
      })
      .catch(() => {
        if (!cancelled && activeJobRef.current === job) setResult(null);
      });
    return () => {
      cancelled = true;
    };
  }, [job, active?.status, active?.time_s]);
  useEffect(() => {
    Promise.all(
      compare.map((id) =>
        Promise.all([api("/jobs/" + id), api("/jobs/" + id + "/results")]).then(
          ([j, r]) => ({ ...j, result: r }),
        ),
      ),
    )
      .then(setComparisons)
      .catch((e) => setError(e.message));
  }, [compare]);
  const update = (path: string, value: any) =>
    setConfig((prev) => {
      const next = structuredClone(prev!);
      const keys = path.split(".");
      let target = next;
      keys.slice(0, -1).forEach((k) => (target = target[k]));
      target[keys.at(-1)!] = value;
      return next;
    });
  const updateDummy = (key: string, value: any) =>
    setConfig((prev) => {
      const next = structuredClone(prev!);
      while (next.geometry.dummies.length <= selected)
        next.geometry.dummies.push({});
      next.geometry.dummies[selected][key] = value;
      return next;
    });
  const selectDummy = useCallback((i: number) => setSelected(i), []);
  const action = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const launch = () =>
    action(async () => {
      if (!derived)
        throw new Error(geometryError || "形状検査が完了していません。");
      const v = await api("/jobs", config);
      setJob(v.id);
      setTab("実行");
      await refresh();
    });
  const addIon = () =>
    update("ions", [
      ...config!.ions,
      {
        name: "Ion " + (config!.ions.length + 1),
        mass_amu: 40,
        charge_number: 1,
        flux_m2_s: 1e19,
        energy_ev: 100,
        angular_sigma_deg: 2,
        distribution_id: null,
      },
    ]);
  const importCase = async (file: File) =>
    action(async () => {
      const data = JSON.parse(await file.text());
      const c = data.config ?? data;
      await api("/geometry", c);
      setConfig(c);
      setSelected(0);
      setNotice("設定を読み込みました。");
    });
  const importInlet = () =>
    action(async () => {
      if (!distribution) throw new Error("CSVまたはNPZを選択してください。");
      const form = new FormData();
      form.append("file", distribution);
      form.append("config_json", JSON.stringify(config));
      const isElectron = speciesIndex >= config!.ions.length;
      form.append(
        "mass_amu",
        String(
          isElectron ? 0.000548579909 : config!.ions[speciesIndex].mass_amu,
        ),
      );
      form.append("pressure_case", pressure);
      form.append("projection_policy", projection);
      if (sourceConfig) form.append("source_config", sourceConfig);
      if (collectorMin !== "" && collectorMax !== "") {
        form.append("collector_min_m", collectorMin);
        form.append("collector_max_m", collectorMax);
      }
      const pair =
        sourceConfig && distribution.name.toLowerCase().endsWith(".npz");
      const response = await fetch(
        pair ? "/api/iaedf-case" : "/api/distributions",
        { method: "POST", body: form },
      );
      const v = await response.json();
      if (!response.ok)
        throw new Error(
          typeof v.detail === "string" ? v.detail : JSON.stringify(v.detail),
        );
      if (v.config) {
        setConfig(v.config);
        setSpeciesIndex(0);
      } else {
        update(
          isElectron
            ? "electron.distribution_id"
            : `ions.${speciesIndex}.distribution_id`,
          v.id,
        );
      }
      setNotice(
        `${v.metadata.sample_count}サンプルを取り込みました。 ${v.metadata.warnings.join(" ")}`,
      );
    });
  if (!config)
    return (
      <div className="loading">
        <FlaskConical />
        研究環境を読み込んでいます…
      </div>
    );
  const g = config.geometry,
    n = config.numerics;
  const d = derived?.dummies[Math.min(selected, g.dummy_count - 1)];
  const figConfig = tab === "設定" ? config : (jobConfig ?? config);
  const figDerived = tab === "設定" ? derived : (result?.derived ?? derived);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">
            <Layers3 size={24} />
          </span>
          <div>
            MEMORY HOLE<span>TRANSPORT LAB</span>
          </div>
        </div>
        <div className="workspace-label">ワークスペース</div>
        <button className="project active">
          <Box size={18} />
          <div>
            ホール内輸送・帯電<span>固定形状シミュレーション</span>
          </div>
        </button>
        <div className="sidebar-block">
          <div className="workspace-label">
            計算ジョブ <span>{jobs.length}</span>
          </div>
          {jobs.length === 0 ? (
            <p className="sidebar-empty">
              まだジョブはありません。
              <br />
              条件を設定して計算を始めます。
            </p>
          ) : (
            jobs.slice(0, 9).map((j) => (
              <button
                key={j.id}
                className={"job-link " + (j.id === job ? "selected" : "")}
                onClick={() => {
                  setJob(j.id);
                  setTab(j.status === "completed" ? "場と軌道" : "実行");
                }}
              >
                <i className={"status-dot " + j.status} />
                <div>
                  {j.name}
                  <span>
                    {labels[j.status]} · {j.backend.toUpperCase()}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>
        <div className="sidebar-footer">
          <span className="version">仕様 v0.5 / 実装 v0.1</span>
          <p>固定形状の研究用モデル</p>
          <button
            onClick={() => {
              setNotice(
                "検証用仮値でのCPU/GPU計算ができます。実測校正、RF時間依存輸送、形状発展は後続工程です。",
              );
            }}
          >
            <CircleHelp size={16} />
            モデルと適用範囲
          </button>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <span>
            研究ツール <ChevronRight size={14} /> ホール内輸送
          </span>
          <div className="system-pill">
            <i className="online" />
            {system?.gpu.available ? system.gpu.name : "CPU参照環境"}{" "}
            <span>LOCAL</span>
          </div>
        </header>
        <div className="page">
          <div className="page-heading">
            <div>
              <div className="eyebrow">MEMORY DEVICE / FIXED GEOMETRY</div>
              <h1>ホール内輸送・帯電</h1>
              <p>
                形状、入口分布、電場をつないで、粒子の到達と非対称性を調べる。
              </p>
            </div>
            <button
              className="secondary"
              onClick={() =>
                download("memory_hole_case.json", {
                  config,
                  derived,
                  units: { length: "nm", energy: "eV" },
                })
              }
            >
              <ArrowDownToLine size={16} />
              条件を保存
            </button>
          </div>
          <div className="validation-banner">
            <FlaskConical size={17} />
            <strong>検証用ケース</strong>
            <span>
              {config.ions.some((s: any) => s.distribution_id)
                ? "IAEDF分布を使用。面外速度・絶対流束の近似と、実デバイスの予測精度は未検証です。"
                : "入口分布・流束・波形は仮値です。実デバイスの予測精度は未検証です。"}
            </span>
            <span className="pill">3D / 全周360°</span>
          </div>
          {error && (
            <div role="alert" className="message error">
              {error}
              <button aria-label="閉じる" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div role="status" className="message notice">
              {notice}
              <button aria-label="閉じる" onClick={() => setNotice("")}>
                <X size={16} />
              </button>
            </div>
          )}
          <nav className="tabs">
            {["設定", "実行", "場と軌道", "衝突統計", "比較・精度"].map(
              (t, i) => (
                <button
                  key={t}
                  className={tab === t ? "active" : ""}
                  onClick={() => setTab(t)}
                >
                  <span>0{i + 1}</span>
                  {t}
                </button>
              ),
            )}
          </nav>
          {tab === "設定" && (
            <>
              <div className="case-title">
                <input
                  aria-label="ケース名"
                  value={config.name}
                  onChange={(e) => update("name", e.target.value)}
                />
                <button
                  className="text-button"
                  onClick={() => fileRef.current?.click()}
                >
                  <Upload size={14} />
                  条件を読込
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  hidden
                  accept=".json"
                  onChange={(e) =>
                    e.target.files?.[0] && importCase(e.target.files[0])
                  }
                />
              </div>
              <div className="settings-layout">
                <div className="settings-column">
                  <Panel title="Channelと積層" tag="寸法は nm">
                    <div className="fields two">
                      <NumberField
                        label="Channel入口径"
                        value={g.channel_diameter_nm}
                        onChange={(v: number) =>
                          update("geometry.channel_diameter_nm", v)
                        }
                        unit="nm"
                        min={1}
                      />
                      <NumberField
                        label="底部径"
                        value={g.bottom_diameter_nm ?? g.channel_diameter_nm}
                        onChange={(v: number) =>
                          update("geometry.bottom_diameter_nm", v)
                        }
                        unit="nm"
                        min={1}
                      />
                      <NumberField
                        label="Carbon厚"
                        value={g.carbon_thickness_nm}
                        onChange={(v: number) =>
                          update("geometry.carbon_thickness_nm", v)
                        }
                        unit="nm"
                      />
                      <NumberField
                        label="SiO₂厚"
                        value={g.oxide_thickness_nm}
                        onChange={(v: number) =>
                          update("geometry.oxide_thickness_nm", v)
                        }
                        unit="nm"
                      />
                    </div>
                    <div className="linked-value">
                      <span>
                        <Check size={14} />{" "}
                        {g.depth_mode === "linked"
                          ? "基板までの深さを膜厚に連動"
                          : "加工途中の深さを直接指定"}
                      </span>
                      <strong>
                        {fmt(derived?.depth_nm, 0)} <small>nm</small>
                      </strong>
                    </div>
                    <details>
                      <summary>プロファイル・領域・メッシュ</summary>
                      <label className="form-label">
                        深さの指定
                        <select
                          value={g.depth_mode}
                          onChange={(e) =>
                            update("geometry.depth_mode", e.target.value)
                          }
                        >
                          <option value="linked">膜厚に連動</option>
                          <option value="direct">直接入力（加工途中）</option>
                        </select>
                      </label>
                      {g.depth_mode === "direct" && (
                        <NumberField
                          label="Channel深さ"
                          value={g.depth_nm}
                          onChange={(v: number) =>
                            update("geometry.depth_nm", v)
                          }
                          unit="nm"
                        />
                      )}
                      <div className="fields two">
                        {[
                          ["lateral_margin_nm", "横方向余白"],
                          ["entrance_height_nm", "入口上方高さ"],
                          ["substrate_thickness_nm", "基板側厚さ"],
                          ["mesh_size_nm", "最大メッシュ寸法"],
                          ["interface_mesh_size_nm", "界面メッシュ寸法"],
                          ["gap_evaluation_depth_nm", "間隔評価深さ"],
                        ].map(([key, label]) => (
                          <NumberField
                            key={key}
                            label={label}
                            value={g[key]}
                            onChange={(v: number) =>
                              update("geometry." + key, v)
                            }
                            unit="nm"
                          />
                        ))}
                      </div>
                      <p className="hint">
                        深さ別半径、注入面、領域の直接指定は下の「全設定JSON」で編集できます。
                      </p>
                    </details>
                  </Panel>
                  <Panel title="埋め込み金属 dummy" tag="共通導体">
                    <div className="fields two">
                      <NumberField
                        label="個数"
                        value={g.dummy_count}
                        onChange={(v: number) => {
                          setConfig((prev) => {
                            const next = structuredClone(prev!);
                            next.geometry.dummy_count = v;
                            if (Number.isInteger(v) && v >= 1 && v <= 200)
                              next.geometry.dummies =
                                next.geometry.dummies.slice(0, v);
                            return next;
                          });
                          setSelected(0);
                        }}
                        unit="本"
                      />
                      <NumberField
                        label="共通径"
                        value={g.dummy_diameter_nm}
                        onChange={(v: number) =>
                          update("geometry.dummy_diameter_nm", v)
                        }
                        unit="nm"
                      />
                      <NumberField
                        label="共通SiO₂間隔"
                        value={g.gap_nm}
                        onChange={(v: number) => update("geometry.gap_nm", v)}
                        unit="nm"
                      />
                      <NumberField
                        label="開始方位"
                        value={g.start_angle_deg}
                        onChange={(v: number) =>
                          update("geometry.start_angle_deg", v)
                        }
                        unit="°"
                      />
                    </div>
                    <label className="form-label">
                      配置モード
                      <select
                        value={g.placement_mode}
                        onChange={(e) => {
                          const mode = e.target.value;
                          setConfig((prev) => {
                            const next = structuredClone(prev!);
                            next.geometry.placement_mode = mode;
                            if (derived)
                              next.geometry.dummies = derived.dummies.map(
                                (m: any) => ({
                                  ...next.geometry.dummies[m.index],
                                  radius_nm: m.radius_nm,
                                  x_nm: m.x_nm,
                                  y_nm: m.y_nm,
                                }),
                              );
                            return next;
                          });
                        }}
                      >
                        <option value="gap">間隔から中心距離を計算</option>
                        <option value="radial">中心距離を直接指定</option>
                        <option value="xy">xy座標を直接指定</option>
                      </select>
                    </label>
                    <div className="dummy-editor">
                      <div className="dummy-editor-head">
                        <span>
                          <i />
                          dummy {selected + 1}
                        </span>
                        <select
                          aria-label="編集するdummy"
                          value={selected}
                          onChange={(e) => setSelected(Number(e.target.value))}
                        >
                          {Array.from(
                            {
                              length: Math.max(
                                0,
                                Math.min(200, Math.floor(g.dummy_count)),
                              ),
                            },
                            (_, i) => (
                              <option key={i} value={i}>
                                #{i + 1}
                              </option>
                            ),
                          )}
                        </select>
                        <button
                          title="個別上書きを解除"
                          onClick={() =>
                            setConfig((prev) => {
                              const next = structuredClone(prev!);
                              if (next.geometry.dummies[selected])
                                next.geometry.dummies[selected] = {};
                              return next;
                            })
                          }
                        >
                          <RotateCcw size={15} />
                        </button>
                      </div>
                      <div className="fields two">
                        {g.placement_mode === "gap" ? (
                          <NumberField
                            label="個別間隔"
                            value={d?.gap_nm ?? g.gap_nm}
                            onChange={(v: number) => updateDummy("gap_nm", v)}
                            unit="nm"
                          />
                        ) : g.placement_mode === "radial" ? (
                          <NumberField
                            label="中心距離"
                            value={d?.radius_nm}
                            onChange={(v: number) =>
                              updateDummy("radius_nm", v)
                            }
                            unit="nm"
                          />
                        ) : (
                          <>
                            <NumberField
                              label="x座標"
                              value={d?.x_nm}
                              onChange={(v: number) => updateDummy("x_nm", v)}
                              unit="nm"
                            />
                            <NumberField
                              label="y座標"
                              value={d?.y_nm}
                              onChange={(v: number) => updateDummy("y_nm", v)}
                              unit="nm"
                            />
                          </>
                        )}
                        {g.placement_mode !== "xy" && (
                          <NumberField
                            label="方位角"
                            value={d?.angle_deg}
                            onChange={(v: number) =>
                              updateDummy("angle_deg", v)
                            }
                            unit="°"
                          />
                        )}
                      </div>
                      <div className="dummy-derived">
                        <span>
                          中心距離 <b>{fmt(d?.radius_nm, 1)} nm</b>
                        </span>
                        <span>
                          基準から <b>{fmt(d?.displacement_nm, 1)} nm</b>
                        </span>
                      </div>
                    </div>
                  </Panel>
                  <Panel title="計算モード">
                    <div className="mode-options">
                      {[
                        ["uncharged", "帯電なし", "入口と形状の基準計算"],
                        ["fixed_charge", "指定電荷", "固定した表面電荷で追跡"],
                        [
                          "self_consistent",
                          "自己無撞着帯電",
                          "輸送・電荷・電場を更新",
                        ],
                      ].map(([value, label, hint]) => (
                        <button
                          key={value}
                          onClick={() => update("mode", value)}
                          className={config.mode === value ? "active" : ""}
                        >
                          <span className="radio" />
                          <div>
                            <strong>{label}</strong>
                            <small>{hint}</small>
                          </div>
                        </button>
                      ))}
                    </div>
                    <div className="fields two">
                      <NumberField
                        label="初期表面電荷密度"
                        value={config.initial_sigma_c_m2}
                        onChange={(v: number) =>
                          update("initial_sigma_c_m2", v)
                        }
                        unit="C/m²"
                        step={0.00001}
                      />
                      <NumberField
                        label="SiO₂比誘電率"
                        value={config.oxide_relative_permittivity}
                        onChange={(v: number) =>
                          update("oxide_relative_permittivity", v)
                        }
                        unit=""
                        step={0.1}
                      />
                    </div>
                  </Panel>
                </div>
                <div className="preview-column">
                  <Panel title="形状プレビュー" tag="寸法検査">
                    <GeometryView
                      config={config}
                      derived={derived}
                      selected={selected}
                      onSelect={selectDummy}
                    />
                    {geometryError ? (
                      <div role="alert" className="geometry-error">
                        {geometryError}
                      </div>
                    ) : (
                      <>
                        <div className="geometry-metrics">
                          <div>
                            <span>Channel深さ</span>
                            <b>
                              {fmt((derived?.depth_nm ?? 0) / 1000, 2)}{" "}
                              <small>µm</small>
                            </b>
                          </div>
                          <div>
                            <span>金属長さ</span>
                            <b>
                              {fmt(g.oxide_thickness_nm / 1000, 2)}{" "}
                              <small>µm</small>
                            </b>
                          </div>
                          <div>
                            <span>外接直径</span>
                            <b>
                              {fmt(derived?.envelope_diameter_nm, 1)}{" "}
                              <small>nm</small>
                            </b>
                          </div>
                        </div>
                        <div className="connection">
                          <Check size={16} /> Carbon下面 ↔ 金属 ↔ 基板{" "}
                          <span>同一ウェハ電位</span>
                        </div>
                        {derived?.warnings.map((w: string) => (
                          <p className="hint warning" key={w}>
                            {w}
                          </p>
                        ))}
                      </>
                    )}
                  </Panel>
                  <Panel title="ウェハ電位" tag="IAEDFと共通">
                    <div className="fields two">
                      <NumberField
                        label="DC電位"
                        value={config.waveform.dc_v}
                        onChange={(v: number) => update("waveform.dc_v", v)}
                        unit="V"
                      />
                      <NumberField
                        label="RF振幅"
                        value={config.waveform.amplitude_v}
                        onChange={(v: number) =>
                          update("waveform.amplitude_v", v)
                        }
                        unit="V"
                      />
                      <NumberField
                        label="周波数"
                        value={config.waveform.frequency_hz / 1e6}
                        onChange={(v: number) =>
                          update("waveform.frequency_hz", v * 1e6)
                        }
                        unit="MHz"
                        step={0.01}
                      />
                      <NumberField
                        label="位相原点"
                        value={config.waveform.phase_origin_deg}
                        onChange={(v: number) =>
                          update("waveform.phase_origin_deg", v)
                        }
                        unit="°"
                      />
                    </div>
                    <div className="note">
                      <Waves size={17} />
                      <span>
                        Carbon・金属・基板へ同じ波形を指定します。4900
                        pFは全ウェハの記録用容量です。
                      </span>
                    </div>
                  </Panel>
                  <Panel title="実行条件">
                    <div className="fields two">
                      <label className="form-label">
                        計算デバイス
                        <select
                          value={n.backend}
                          onChange={(e) =>
                            update("numerics.backend", e.target.value)
                          }
                        >
                          <option value="cpu">CPU参照 / 倍精度</option>
                          <option value="gpu" disabled={!system?.gpu.available}>
                            GPU / 倍精度
                          </option>
                        </select>
                      </label>
                      <NumberField
                        label="サンプル数 / 種 / 更新"
                        value={n.samples_per_species}
                        onChange={(v: number) =>
                          update("numerics.samples_per_species", v)
                        }
                        unit=""
                        step={100}
                      />
                      <NumberField
                        label="帯電時間"
                        value={n.duration_s * 1e6}
                        onChange={(v: number) =>
                          update("numerics.duration_s", v * 1e-6)
                        }
                        unit="µs"
                        step={0.1}
                      />
                      <NumberField
                        label="帯電更新数（目安）"
                        value={n.charging_steps}
                        onChange={(v: number) =>
                          update("numerics.charging_steps", v)
                        }
                        unit=""
                      />
                      <NumberField
                        label="アニメーション用の代表軌道数"
                        value={n.representative_trajectories}
                        onChange={(v: number) =>
                          update("numerics.representative_trajectories", v)
                        }
                        unit="本"
                        min={0}
                        max={1000}
                      />
                    </div>
                    <p className="note">
                      0～1000本を指定できます。保存する合計を粒子種に配分します。
                      例えば60本なら、イオン1種と電子で各30本が目安です。
                      変更後は再計算してください。粒子再生では選択したRF位相の軌道を表示します。
                    </p>
                    <div className="run-row">
                      <span>
                        <Cpu size={14} /> {n.backend.toUpperCase()} · seed{" "}
                        {n.seed}
                      </span>
                      <button
                        className="primary"
                        disabled={busy || !derived}
                        onClick={launch}
                      >
                        <Play size={16} />
                        計算を開始 <ArrowRight size={16} />
                      </button>
                    </div>
                  </Panel>
                </div>
              </div>
              <div className="lower-grid">
                <Panel title="入口の粒子種・分布" tag="絶対流束を指定">
                  {config.ions.map((s: any, i: number) => (
                    <div className="species-row" key={i}>
                      <div className="species-heading">
                        <input
                          aria-label="イオン種名"
                          value={s.name}
                          onChange={(e) =>
                            update(`ions.${i}.name`, e.target.value)
                          }
                        />
                        {config.ions.length > 1 && (
                          <button
                            title="イオン種を削除"
                            onClick={() =>
                              update(
                                "ions",
                                config.ions.filter(
                                  (_: any, k: number) => i !== k,
                                ),
                              )
                            }
                          >
                            <X size={14} />
                          </button>
                        )}
                      </div>
                      <div className="fields four">
                        {[
                          ["mass_amu", "質量", "u"],
                          ["charge_number", "電荷数", "e"],
                          ["flux_m2_s", "法線流束", "m⁻²s⁻¹"],
                          ["energy_ev", "入射エネルギー", "eV"],
                        ].map(([key, label, unit]) => (
                          <NumberField
                            key={key}
                            label={label}
                            value={s[key]}
                            onChange={(v: number) =>
                              update(`ions.${i}.${key}`, v)
                            }
                            unit={unit}
                          />
                        ))}
                      </div>
                      <p className="hint">
                        {s.distribution_id
                          ? "取込分布: " + s.distribution_id.slice(0, 12)
                          : `生成分布：エネルギー固定・接線角の幅 ${s.angular_sigma_deg}°`}
                      </p>
                    </div>
                  ))}
                  <button className="text-button" onClick={addIon}>
                    <Plus size={14} />
                    イオン種を追加
                  </button>
                  <div className="electron-row">
                    <label>
                      <input
                        type="checkbox"
                        checked={config.electron.enabled}
                        onChange={(e) =>
                          update("electron.enabled", e.target.checked)
                        }
                      />{" "}
                      電子を追跡
                    </label>
                    <div className="fields two">
                      <NumberField
                        label="電子温度"
                        value={config.electron.temperature_ev}
                        onChange={(v: number) =>
                          update("electron.temperature_ev", v)
                        }
                        unit="eV"
                        step={0.1}
                      />
                      <NumberField
                        label="電子法線流束"
                        value={config.electron.flux_m2_s}
                        onChange={(v: number) =>
                          update("electron.flux_m2_s", v)
                        }
                        unit="m⁻²s⁻¹"
                      />
                    </div>
                    <p className="hint">
                      法線速度で重み付けした熱分布。電子温度から絶対流束は決めません。
                    </p>
                  </div>
                </Panel>
                <Panel title="IAEDF / CSV取込" tag="相関を保持">
                  <label className="upload-zone">
                    <Upload size={24} />
                    <strong>
                      {distribution?.name ?? "raw.npz または分布CSVを選択"}
                    </strong>
                    <span>速度3成分・3D極角サンプル・相関ビン</span>
                    <input
                      type="file"
                      accept=".npz,.csv"
                      onChange={(e) =>
                        setDistribution(e.target.files?.[0] ?? null)
                      }
                    />
                  </label>
                  <div className="fields two">
                    <label className="form-label">
                      対象粒子種
                      <select
                        value={speciesIndex}
                        onChange={(e) =>
                          setSpeciesIndex(Number(e.target.value))
                        }
                      >
                        {config.ions.map((s: any, i: number) => (
                          <option key={i} value={i}>
                            {s.name}
                          </option>
                        ))}
                        <option value={config.ions.length}>電子</option>
                      </select>
                    </label>
                    <label className="form-label">
                      圧力ケース
                      <input
                        value={pressure}
                        onChange={(e) => setPressure(e.target.value)}
                      />
                    </label>
                  </div>
                  <label className="form-label">
                    旧NPZの投影角
                    <select
                      value={projection}
                      onChange={(e) => setProjection(e.target.value)}
                    >
                      <option value="reject">
                        速度3成分がなければ取込を停止
                      </option>
                      <option value="zero_out_of_plane">
                        面外速度ゼロの検証用近似
                      </option>
                    </select>
                  </label>
                  <details>
                    <summary>上流設定・2Dコレクタ</summary>
                    <label className="form-label">
                      IAEDF config.json（出典として保存）
                      <input
                        type="file"
                        accept=".json"
                        onChange={(e) =>
                          setSourceConfig(e.target.files?.[0] ?? null)
                        }
                      />
                    </label>
                    <div className="fields two">
                      <label className="form-label">
                        コレクタ x下限 [m]
                        <input
                          value={collectorMin}
                          onChange={(e) => setCollectorMin(e.target.value)}
                        />
                      </label>
                      <label className="form-label">
                        コレクタ x上限 [m]
                        <input
                          value={collectorMax}
                          onChange={(e) => setCollectorMax(e.target.value)}
                        />
                      </label>
                    </div>
                  </details>
                  <button
                    className="secondary full"
                    onClick={importInlet}
                    disabled={busy || !distribution}
                  >
                    <Upload size={16} />
                    分布を取り込む
                  </button>
                  <p className="hint">
                    評価面・波形は現在の設定を記録します。上流との整合を確認してください。規格化分布から絶対流束は算出しません。
                  </p>
                </Panel>
              </div>
              <Panel title="寸法掃引" tag="最大64ケース">
                <div className="sweep-row">
                  <label className="form-label">
                    寸法パス
                    <input
                      value={sweepPath}
                      onChange={(e) => setSweepPath(e.target.value)}
                      list="dimensions"
                    />
                    <datalist id="dimensions">
                      {[
                        "geometry.gap_nm",
                        "geometry.channel_diameter_nm",
                        "geometry.carbon_thickness_nm",
                        "geometry.oxide_thickness_nm",
                        "geometry.dummy_count",
                        "geometry.dummy_diameter_nm",
                      ].map((v) => (
                        <option key={v} value={v} />
                      ))}
                    </datalist>
                  </label>
                  <label className="form-label">
                    値リスト
                    <input
                      value={sweepValues}
                      onChange={(e) => setSweepValues(e.target.value)}
                      placeholder="50, 100, 150"
                    />
                  </label>
                  <button
                    className="secondary"
                    disabled={busy}
                    onClick={() =>
                      action(async () => {
                        const values = sweepValues.split(",").map(Number);
                        if (values.some((v) => !Number.isFinite(v)))
                          throw new Error("掃引値を数値で指定してください。");
                        const v = await api("/sweeps", {
                          config,
                          dimensions: [{ path: sweepPath, values }],
                        });
                        setNotice(
                          `${v.ids.length}ケースを実行待ちに追加しました。`,
                        );
                        setTab("実行");
                        await refresh();
                      })
                    }
                  >
                    <SlidersHorizontal size={16} />
                    掃引を開始
                  </button>
                </div>
              </Panel>
              <details className="advanced">
                <summary>
                  <Settings2 size={15} />{" "}
                  全設定JSON（表面反射・二次電子・漏れ・数値条件を含む）
                </summary>
                <button
                  className="text-button"
                  onClick={() => setAdvanced(JSON.stringify(config, null, 2))}
                >
                  現在の設定を表示
                </button>
                <textarea
                  value={advanced}
                  onChange={(e) => setAdvanced(e.target.value)}
                  spellCheck={false}
                />
                <button
                  className="secondary"
                  onClick={() =>
                    action(async () => {
                      const c = JSON.parse(advanced);
                      await api("/geometry", c);
                      setConfig(c);
                      setNotice("全設定を適用しました。");
                    })
                  }
                >
                  JSONを検査して適用
                </button>
              </details>
            </>
          )}
          {tab === "実行" && (
            <>
              <div className="section-header">
                <h2>計算ジョブ</h2>
                <button
                  className="primary"
                  disabled={busy || !derived}
                  onClick={launch}
                >
                  <Play size={15} />
                  現在の条件で新規実行
                </button>
              </div>
              <div className="job-grid">
                {jobs.length === 0 ? (
                  <div className="empty">
                    条件を設定して「計算を開始」を押してください。
                  </div>
                ) : (
                  jobs.map((j) => (
                    <section
                      className={
                        "panel job-card " + (job === j.id ? "chosen" : "")
                      }
                      key={j.id}
                      onClick={() => setJob(j.id)}
                    >
                      <div className="job-card-top">
                        <span className={"status-label " + j.status}>
                          {labels[j.status]}
                        </span>
                        <span>
                          {j.backend.toUpperCase()} · {labels[j.mode]}
                        </span>
                      </div>
                      <h3>{j.name}</h3>
                      <p>{j.stage}</p>
                      <div className="progress-track">
                        <div style={{ width: (j.progress ?? 0) * 100 + "%" }} />
                      </div>
                      <div className="job-metadata">
                        <span>{percent(j.progress ?? 0)}</span>
                        <span>{fmt(j.elapsed_s, 1)} s</span>
                        {j.particles_per_s && (
                          <span>{fmt(j.particles_per_s, 0)} 粒子/s</span>
                        )}
                      </div>
                      <div className="job-actions">
                        {["running", "queued"].includes(j.status) ? (
                          <button
                            className="secondary"
                            onClick={(e) => {
                              e.stopPropagation();
                              action(async () => {
                                await api("/jobs/" + j.id + "/pause", {});
                                await refresh();
                              });
                            }}
                          >
                            <Pause size={14} />
                            停止
                          </button>
                        ) : ["paused", "failed"].includes(j.status) ? (
                          <button
                            className="secondary"
                            onClick={(e) => {
                              e.stopPropagation();
                              action(async () => {
                                await api("/jobs/" + j.id + "/resume", {});
                                await refresh();
                              });
                            }}
                          >
                            <Play size={14} />
                            再開
                          </button>
                        ) : (
                          <button
                            className="secondary"
                            onClick={() => {
                              setJob(j.id);
                              setTab("場と軌道");
                            }}
                          >
                            結果を表示 <ArrowRight size={14} />
                          </button>
                        )}
                        <a href={"/api/jobs/" + j.id + "/export/config"}>
                          条件JSON
                        </a>
                      </div>
                    </section>
                  ))
                )}
              </div>
              <div className="note">
                <Pause size={18} />
                <span>
                  停止は完了済みの帯電更新点を保存します。途中の更新は破棄し、同じ乱数状態で再開時に再計算します。実行中の設定変更は新しいジョブへ適用されます。
                </span>
              </div>
            </>
          )}
          {["場と軌道", "衝突統計"].includes(tab) && (
            <>
              <div className="section-header">
                <h2>{tab}</h2>
                <select
                  aria-label="結果ジョブ"
                  value={job}
                  onChange={(e) => setJob(e.target.value)}
                >
                  <option value="">ジョブを選択</option>
                  {jobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.name} · {labels[j.status]}
                    </option>
                  ))}
                </select>
                <div className="export-links">
                  <a href={"/api/jobs/" + job + "/export/csv"}>CSV</a>
                  <a href={"/api/jobs/" + job + "/export/hdf5"}>HDF5</a>
                </div>
              </div>
              {result && result.saved_times?.length > 1 && (
                <label className="time-select">
                  表示時刻
                  <select
                    value={result.history.at(-1)?.step}
                    onChange={(e) => {
                      setTimeSelection((v) => v + 1);
                      action(async () => {
                        setOwnedResult(
                          await api(
                            "/jobs/" + job + "/results?step=" + e.target.value,
                          ),
                        );
                      });
                    }}
                  >
                    {result.saved_times.map((t: any) => (
                      <option key={t.step} value={t.step}>
                        {fmt(t.time_s * 1e6, 4)} µs · 更新 {t.step}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {!result ? (
                <div className="empty">
                  <Waves size={36} />
                  <h3>結果を待っています</h3>
                  <p>完了した帯電更新の結果をここに表示します。</p>
                </div>
              ) : tab === "場と軌道" ? (
                <div className="results-layout">
                  <Panel
                    title="3D場と代表軌道"
                    tag={`${fmt(result.time_s * 1e6, 3)} µs`}
                  >
                    <div className="field-options">
                      {[
                        ["potential", "節点電位"],
                        ["charge", "表面電荷"],
                        ["field", "電場ベクトル"],
                        ["geometry", "軌道"],
                      ].map(([key, label]) => (
                        <button
                          className={display === key ? "active" : ""}
                          key={key}
                          onClick={() => setDisplay(key)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                    <ResultAnimation
                      key={job}
                      jobId={job}
                      config={figConfig}
                      result={result}
                      selected={selected}
                      onSelect={selectDummy}
                      display={display}
                      onResult={setOwnedResult}
                      onError={setError}
                      resetKey={timeSelection}
                    />
                  </Panel>
                  <div>
                    <Panel title="計算情報">
                      <dl className="info-list">
                        <dt>メッシュ節点</dt>
                        <dd>{fmt(result.metadata.mesh.nodes, 0)}</dd>
                        <dt>四面体数</dt>
                        <dd>{fmt(result.metadata.mesh.tetrahedra, 0)}</dd>
                        <dt>表面パッチ数</dt>
                        <dd>{fmt(result.metadata.mesh.surface_patches, 0)}</dd>
                        <dt>電場残差</dt>
                        <dd>{sci(result.history.at(-1)?.field_residual)}</dd>
                        <dt>電荷保存残差</dt>
                        <dd>{sci(result.history.at(-1)?.charge_residual)}</dd>
                        <dt>最大表面電荷密度</dt>
                        <dd>
                          {sci(result.history.at(-1)?.max_sigma_c_m2)} C/m²
                        </dd>
                      </dl>
                    </Panel>
                    <Panel title="適用範囲と診断">
                      <div className="warnings">
                        {result.metadata.warnings.map((w: string) => (
                          <p key={w}>{w}</p>
                        ))}
                      </div>
                    </Panel>
                  </div>
                </div>
              ) : (
                <>
                  <div className="stat-cards">
                    {Object.entries(result.statistics).map(
                      ([name, s]: [string, any]) => (
                        <Panel
                          title={name}
                          key={name}
                          tag={`ESS ${fmt(s.bottom_arrival.effective_samples, 0)}`}
                        >
                          <span className="stat-label">底部到達率</span>
                          <div className="big-stat">
                            {percent(s.bottom_arrival.rate)}
                          </div>
                          <p className="ci">
                            95%区間 {percent(s.bottom_arrival.low)} –{" "}
                            {percent(s.bottom_arrival.high)}
                          </p>
                          <dl className="info-list">
                            <dt>底部平均エネルギー</dt>
                            <dd>{fmt(s.mean_bottom_energy_ev)} eV</dd>
                            <dt>エネルギー区間半幅</dt>
                            <dd>{fmt(s.energy_ci_half_ev)} eV</dd>
                            <dt>底部重心 [nm]</dt>
                            <dd>
                              {s.bottom_centroid_nm
                                ? s.bottom_centroid_nm
                                    .map((v: number) => fmt(v, 2))
                                    .join(", ")
                                : "未到達"}
                            </dd>
                            <dt>未解決粒子の重み</dt>
                            <dd>{sci(s.unresolved_weight)}</dd>
                            <dt>追跡サンプル</dt>
                            <dd>{fmt(s.samples, 0)}</dd>
                          </dl>
                        </Panel>
                      ),
                    )}
                  </div>
                  <div className="lower-grid">
                    <Panel title="深さ・方位別の衝突分布" tag="全種の重み合計">
                      <Heatmap data={result.depth_azimuth_weight} />
                      <p className="hint">
                        壁面衝突を集計。反射による複数衝突を含みます。
                      </p>
                    </Panel>
                    <Panel title="電荷収支" tag="C">
                      <dl className="info-list">
                        {[
                          ["injected_c", "入射"],
                          ["escaped_c", "逃走"],
                          ["deposited_oxide_c", "SiO₂への正味流入"],
                          ["conductor_c", "導体への正味流入"],
                          ["unresolved_c", "未解決粒子"],
                          ["emitted_c", "二次電子放出"],
                          ["leaked_c", "漏れ"],
                        ].map(([key, label]) => (
                          <div className="dl-row" key={key}>
                            <dt>{label}</dt>
                            <dd>{sci(result.ledger[key])}</dd>
                          </div>
                        ))}
                      </dl>
                      <p className="hint">
                        放出電荷は壁からの差引と再衝突の両方に含まれます。電荷保持と粒子流入は分けて記録します。
                      </p>
                    </Panel>
                  </div>
                  <Panel title="衝突エネルギー・角度" tag="全種の重み合計">
                    <div className="lower-grid">
                      <CollisionSpectrum
                        data={result.collision_energy_angle_weight}
                        energy
                      />
                      <CollisionSpectrum
                        data={result.collision_energy_angle_weight}
                      />
                    </div>
                    <p className="hint">
                      エネルギーは対数ビン。1000
                      eV以上も末尾のビンに保存します。反射粒子と二次電子を含む壁面衝突の分布です。
                    </p>
                  </Panel>
                </>
              )}
            </>
          )}
          {tab === "比較・精度" && (
            <>
              <div className="section-header">
                <h2>条件比較</h2>
                <span className="hint">
                  最初に選んだケースを基準に比較します。
                </span>
              </div>
              <div className="compare-selector">
                {jobs
                  .filter((j) => j.status === "completed")
                  .map((j) => (
                    <label key={j.id}>
                      <input
                        type="checkbox"
                        checked={compare.includes(j.id)}
                        onChange={(e) =>
                          setCompare(
                            e.target.checked
                              ? [...compare, j.id]
                              : compare.filter((id) => id !== j.id),
                          )
                        }
                      />
                      {j.name}
                    </label>
                  ))}
              </div>
              {comparisons.length ? (
                <div className="panel comparison-table">
                  <table>
                    <thead>
                      <tr>
                        <th>ケース / 粒子種</th>
                        <th>底部到達率 [95%区間]</th>
                        <th>基準からの差</th>
                        <th>平均エネルギー</th>
                        <th>底部重心 [nm]</th>
                        <th>電荷保存残差</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisons.flatMap((c, i) =>
                        Object.entries(c.result.statistics).map(
                          ([name, s]: [string, any]) => {
                            const baseline =
                              comparisons[0].result.statistics[name]
                                ?.bottom_arrival.rate;
                            const delta =
                              baseline == null || s.bottom_arrival.rate == null
                                ? null
                                : s.bottom_arrival.rate - baseline;
                            return (
                              <tr key={c.id + name}>
                                <td>
                                  <b>{c.config.name}</b>
                                  <span>
                                    {i === 0 ? "基準 · " : ""}
                                    {name}
                                  </span>
                                </td>
                                <td>
                                  {percent(s.bottom_arrival.rate)}
                                  <small>
                                    {percent(s.bottom_arrival.low)} –{" "}
                                    {percent(s.bottom_arrival.high)}
                                  </small>
                                </td>
                                <td>
                                  {delta === null
                                    ? "未定義"
                                    : `${(delta * 100).toFixed(2)} pt`}
                                </td>
                                <td>{fmt(s.mean_bottom_energy_ev)} eV</td>
                                <td>
                                  {s.bottom_centroid_nm
                                    ? s.bottom_centroid_nm
                                        .map((v: number) => fmt(v, 2))
                                        .join(", ")
                                    : "未到達"}
                                </td>
                                <td>
                                  {sci(c.result.history.at(-1).charge_residual)}
                                </td>
                              </tr>
                            );
                          },
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty">
                  <SlidersHorizontal size={34} />
                  <h3>比較する完了ジョブを選んでください</h3>
                  <p>径・膜厚・dummy間隔の掃引や、CPU/GPUの比較に使えます。</p>
                </div>
              )}
              <Panel title="精度確認の目標">
                <div className="acceptance-grid">
                  {[
                    ["静電場・解析軌道", "解析解との差 1%以内"],
                    ["電荷保存", "入射絶対電荷で規格化して 10⁻⁶以下"],
                    [
                      "数値収束",
                      "メッシュ・運動刻み・帯電刻みの変更で主要出力差 2%以内",
                    ],
                    ["統計精度", "主要出力の95%区間の相対半幅 2%以内"],
                    [
                      "CPU/GPU一致",
                      "同一粒子で衝突先が一致し、軌道・主要出力が数値許容差内",
                    ],
                    ["希少な底部到達", "ゼロ件でも上限を表示。絶対半幅を併記"],
                  ].map(([title, text]) => (
                    <div key={title}>
                      <strong>{title}</strong>
                      <p>{text}</p>
                    </div>
                  ))}
                </div>
                <p className="hint">
                  COMSOL比較は対象外です。目標値は実デバイスの予測誤差を保証するものではありません。帯電を含む統計には独立シードのジョブ比較が必要です。
                </p>
              </Panel>
            </>
          )}
          <footer>
            MEMORY HOLE LAB{" "}
            <span>SI単位で計算 · 固定形状 · 日本語ローカルアプリ</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function CollisionSpectrum({ data, energy = false }: any) {
  const count = energy ? 49 : 18;
  const values = Array.from({ length: count }, (_, i) =>
    data.reduce(
      (sum: number, s: any) =>
        sum +
        (energy
          ? s[i].reduce((a: number, b: number) => a + b, 0)
          : s.reduce((a: number, row: any) => a + row[i], 0)),
      0,
    ),
  );
  const max = Math.max(...values, 1e-30);
  return (
    <div>
      <p className="hint">
        {energy ? "衝突エネルギー [eV]" : "入射法線からの角度 [°]"}
      </p>
      <svg
        className="spectrum"
        viewBox="0 0 460 180"
        aria-label={energy ? "衝突エネルギー分布" : "衝突角度分布"}
      >
        {values.map((v, i) => (
          <rect
            key={i}
            x={25 + (i * 420) / count}
            y={145 - (v / max) * 120}
            width={420 / count - 1}
            height={(v / max) * 120}
            fill={energy ? "#4d9980" : "#d5ad58"}
          >
            <title>
              ビン {i + 1}: {sci(v)}
            </title>
          </rect>
        ))}
        <line x1="25" x2="445" y1="145" y2="145" stroke="#d2dfd3" />
        <text x="25" y="170" fontSize="10" fill="#8da18c">
          {energy ? "0.01" : "0"}
        </text>
        <text x="425" y="170" fontSize="10" fill="#8da18c">
          {energy ? "≥1000" : "90"}
        </text>
      </svg>
    </div>
  );
}

function Heatmap({ data }: any) {
  const sum = Array.from({ length: 24 }, (_, z) =>
    Array.from({ length: 36 }, (_, a) =>
      data.reduce((v: number, s: any) => v + s[z][a], 0),
    ),
  );
  const max = Math.max(...sum.flat(), 1e-30);
  return (
    <svg
      viewBox="0 0 490 280"
      className="heatmap"
      aria-label="深さと方位の壁面衝突分布"
    >
      {sum.flatMap((row, z) =>
        row.map((v, a) => (
          <rect
            key={z * 36 + a}
            x={38 + a * 12}
            y={12 + z * 10}
            width="11.5"
            height="9.5"
            fill={`hsl(168 45% ${97 - 65 * Math.sqrt(v / max)}%)`}
          >
            <title>
              深さビン{z + 1} 方位{a * 10}°: {sci(v)}
            </title>
          </rect>
        )),
      )}
      <text x="38" y="270" fontSize="11" fill="#6c7f79">
        0°
      </text>
      <text x="242" y="270" fontSize="11" fill="#6c7f79">
        180°
      </text>
      <text x="436" y="270" fontSize="11" fill="#6c7f79">
        360°
      </text>
      <text x="5" y="20" fontSize="11" fill="#6c7f79">
        入口
      </text>
      <text x="5" y="250" fontSize="11" fill="#6c7f79">
        底部
      </text>
    </svg>
  );
}
