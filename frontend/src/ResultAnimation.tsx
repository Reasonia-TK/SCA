import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import GeometryView from "./GeometryView";

type Obj = Record<string, any>;

export default function ResultAnimation({
  jobId,
  config,
  result,
  selected,
  onSelect,
  display,
  onResult,
  onError,
  resetKey,
}: any) {
  const timesKey = JSON.stringify(result.saved_times ?? []);
  const times: Obj[] = useMemo(() => JSON.parse(timesKey), [timesKey]);
  const [mode, setMode] = useState(times.length > 1 ? "field" : "particle");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(true);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState("");
  const [loading, setLoading] = useState(false);
  const cache = useRef(new Map<number, Obj>());
  const controller = useRef<AbortController | null>(null);
  const progressRef = useRef(0);
  const index = Math.max(
    0,
    times.findIndex((t) => t.step === result.history.at(-1)?.step),
  );
  const indexRef = useRef(index);
  indexRef.current = index;
  const selection = useRef(0);
  const phaseKeys = Object.keys(result.trajectory_fields ?? {});
  const selectedPhase = phaseKeys.includes(phase)
    ? phase
    : (phaseKeys[0] ?? "");
  const matchedField = result.trajectory_fields?.[selectedPhase];
  const paths = useMemo(
    () =>
      (result.trajectories ?? []).filter(
        (path: Obj) =>
          !matchedField || String(path.phase_index) === selectedPhase,
      ),
    [result, selectedPhase, matchedField],
  );
  const timed =
    paths.length > 0 &&
    paths.every(
      (path: Obj) =>
        path.times_s?.length === path.points_nm.length &&
        path.times_s.length >= 2 &&
        path.times_s.every(
          (t: number, i: number) =>
            Number.isFinite(t) && (i === 0 || t >= path.times_s[i - 1]),
        ),
    );
  const flightEnd = Math.max(
    1e-18,
    ...paths.map((path: Obj) => path.times_s?.at(-1) ?? 0),
  );
  const shownResult = useMemo(
    () =>
      mode === "particle" && matchedField
        ? { ...result, ...matchedField, trajectories: paths }
        : result,
    [result, matchedField, mode, paths],
  );
  const available = mode === "field" ? times.length > 1 : paths.length > 0;
  const colorValues = useMemo(
    () =>
      display === "charge"
        ? shownResult.surface.sigma_c_m2
        : display === "field"
          ? (shownResult.field_vectors?.electric_field_v_m ?? []).map(
              (v: number[]) => Math.hypot(...v),
            )
          : shownResult.field_samples.potential_v,
    [shownResult, display],
  );
  const colorMin = colorValues.length ? Math.min(...colorValues) : 0;
  const colorMax = colorValues.length ? Math.max(...colorValues) : 0;

  useEffect(() => {
    const abort = new AbortController();
    controller.current = abort;
    return () => {
      abort.abort();
      selection.current++;
    };
  }, []);
  useEffect(() => {
    cache.current.set(result.history.at(-1)?.step, result);
    if (cache.current.size > 32)
      cache.current.delete(cache.current.keys().next().value!);
  }, [result]);
  useEffect(() => {
    setPlaying(false);
    selection.current++;
    setLoading(false);
  }, [resetKey]);

  async function readFrame(frameIndex: number) {
    const step = times[frameIndex].step;
    if (cache.current.has(step)) return cache.current.get(step)!;
    const response = await fetch(`/api/jobs/${jobId}/results?step=${step}`, {
      signal: controller.current?.signal,
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "保存時刻を読み込めませんでした。",
      );
    cache.current.set(step, data);
    if (cache.current.size > 32)
      cache.current.delete(cache.current.keys().next().value!);
    return data;
  }
  function stop() {
    setPlaying(false);
    selection.current++;
    setLoading(false);
  }
  async function selectFrame(frameIndex: number) {
    stop();
    const request = selection.current;
    setLoading(true);
    try {
      const data = await readFrame(frameIndex);
      if (request === selection.current) {
        onResult(data);
        return true;
      }
    } catch (error: any) {
      if (error.name !== "AbortError" && request === selection.current)
        onError(error.message);
    } finally {
      if (request === selection.current) setLoading(false);
    }
  }
  useEffect(() => {
    if (!playing || !available) return;
    if (mode === "field") {
      let cancelled = false;
      const request = selection.current;
      let timer: ReturnType<typeof setTimeout>;
      const nextFrame = async () => {
        let next = indexRef.current + 1;
        if (next >= times.length) {
          if (!loop) {
            setPlaying(false);
            return;
          }
          next = 0;
        }
        setLoading(true);
        try {
          const data = await readFrame(next);
          if (cancelled || request !== selection.current) return;
          indexRef.current = next;
          onResult(data);
          timer = setTimeout(nextFrame, 1000 / speed);
        } catch (error: any) {
          if (!cancelled && error.name !== "AbortError") {
            onError(error.message);
            setPlaying(false);
          }
        } finally {
          if (!cancelled) setLoading(false);
        }
      };
      timer = setTimeout(nextFrame, 1000 / speed);
      return () => {
        cancelled = true;
        clearTimeout(timer);
        setLoading(false);
      };
    }
    let raf = 0;
    let previous = performance.now();
    let posted = 0;
    const animate = (now: number) => {
      let next =
        progressRef.current + (Math.min(now - previous, 100) * speed) / 6000;
      previous = now;
      if (next >= 1) {
        if (loop) next %= 1;
        else {
          progressRef.current = 1;
          setProgress(1);
          setPlaying(false);
          return;
        }
      }
      progressRef.current = next;
      if (now - posted >= 30) {
        setProgress(next);
        posted = now;
      }
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [playing, mode, speed, loop, timesKey, available]);

  const changeMode = (next: string) => {
    stop();
    setMode(next);
    progressRef.current = 0;
    setProgress(0);
  };
  const seekParticle = (fraction: number) => {
    stop();
    progressRef.current = fraction;
    setProgress(fraction);
  };
  const reset = () => {
    if (mode === "field") void selectFrame(0);
    else seekParticle(0);
  };
  const toggle = () => {
    if (playing) {
      stop();
      return;
    }
    if (mode === "particle" && progressRef.current >= 1) {
      progressRef.current = 0;
      setProgress(0);
    }
    if (mode === "field" && index === times.length - 1) {
      const request = selection.current + 1;
      void selectFrame(0).then((success) => {
        if (success && request === selection.current) setPlaying(true);
      });
    } else setPlaying(true);
  };
  return (
    <>
      <div className="animation-controls" aria-label="アニメーション再生">
        <div className="animation-modes" role="group" aria-label="再生モード">
          <button
            className={mode === "field" ? "active" : ""}
            aria-pressed={mode === "field"}
            onClick={() => changeMode("field")}
          >
            帯電の時間変化
          </button>
          <button
            className={mode === "particle" ? "active" : ""}
            aria-pressed={mode === "particle"}
            onClick={() => changeMode("particle")}
          >
            粒子の飛行
          </button>
        </div>
        <div className="animation-transport">
          <button
            className="animation-play"
            aria-label={playing ? "再生を一時停止" : "アニメーションを再生"}
            onClick={toggle}
            disabled={!available || (loading && !playing)}
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}{" "}
            {playing ? "一時停止" : "再生"}
          </button>
          <button
            className="animation-reset"
            aria-label="先頭へ戻る"
            onClick={reset}
            disabled={!available}
          >
            <RotateCcw size={15} />
          </button>
          <label className="animation-speed">
            速度
            <select
              aria-label="再生速度"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              {[0.25, 0.5, 1, 2, 4].map((value) => (
                <option key={value} value={value}>
                  {value}×
                </option>
              ))}
            </select>
          </label>
          <label className="animation-loop">
            <input
              type="checkbox"
              checked={loop}
              onChange={(e) => setLoop(e.target.checked)}
            />
            繰り返す
          </label>
          <output
            className="animation-time"
            data-mode={mode}
            data-progress={progress.toFixed(4)}
          >
            {loading
              ? "読込中…"
              : mode === "field"
                ? `${(result.time_s * 1e6).toLocaleString("ja-JP", { maximumFractionDigits: 4 })} µs`
                : timed
                  ? `${(progress * flightEnd * 1e9).toFixed(3)} / ${(flightEnd * 1e9).toFixed(3)} ns`
                  : `${(progress * 100).toFixed(0)}%`}
          </output>
        </div>
        <input
          className="animation-range"
          type="range"
          aria-label={mode === "field" ? "帯電時刻" : "粒子の再生位置"}
          min={0}
          max={mode === "field" ? Math.max(0, times.length - 1) : 1}
          step={mode === "field" ? 1 : 0.001}
          value={mode === "field" ? index : progress}
          disabled={!available}
          onChange={(e) =>
            mode === "field"
              ? void selectFrame(Number(e.target.value))
              : seekParticle(Number(e.target.value))
          }
        />
        {mode === "particle" && phaseKeys.length > 0 && (
          <label className="animation-phase">
            軌道のRF位相
            <select
              aria-label="軌道のRF位相"
              value={selectedPhase}
              onChange={(e) => {
                seekParticle(0);
                setPhase(e.target.value);
              }}
            >
              {phaseKeys.map((key) => (
                <option key={key} value={key}>
                  {result.trajectory_fields[key].phase_deg.toFixed(2)}°
                </option>
              ))}
            </select>
            <span>
              計算場：更新前{" "}
              {(matchedField.field_time_s * 1e6).toLocaleString("ja-JP")} µs
            </span>
          </label>
        )}
        <p className="animation-note">
          {mode === "field"
            ? times.length > 1
              ? "保存状態を順に表示 · 電場はRF位相0° · 1×＝1保存時刻/秒"
              : "保存時刻が1点です。帯電を有効にして複数時刻を保存すると再生できます。"
            : paths.length === 0
              ? "この保存時刻には代表軌道がありません。代表軌道数を指定して再計算してください。"
              : timed
                ? "代表粒子の飛行時刻で再生 · 保存点間を補間 · RF位相は飛行中固定"
                : "旧データ：経路に沿った表示再生（実時間ではありません）· 背景の電場はRF位相0°"}
        </p>
      </div>
      <GeometryView
        config={config}
        derived={result.derived}
        result={shownResult}
        selected={selected}
        onSelect={onSelect}
        display={display}
        playback={{
          enabled: mode === "particle",
          progress,
          timed,
          endTime: flightEnd,
          playing,
        }}
      />
      {mode === "particle" && paths.length > 0 && (
        <div className="particle-legend">
          <span>
            <i className="ion-dot" />
            イオン{" "}
            {
              paths.filter(
                (p: Obj) => !p.species.toLowerCase().includes("electron"),
              ).length
            }
          </span>
          <span>
            <i className="electron-dot" />
            電子{" "}
            {
              paths.filter((p: Obj) =>
                p.species.toLowerCase().includes("electron"),
              ).length
            }
          </span>
          <span>代表飛行区間・表示数</span>
        </div>
      )}
      {display !== "geometry" && (
        <div
          className="color-scale"
          title="色の範囲は表示中の状態に合わせて自動調整します。"
        >
          <span>{colorMin.toExponential(2)}</span>
          <i />
          <span>{colorMax.toExponential(2)}</span>
          <b>
            {display === "charge" ? "C/m²" : display === "field" ? "V/m" : "V"}
          </b>
        </div>
      )}
    </>
  );
}
