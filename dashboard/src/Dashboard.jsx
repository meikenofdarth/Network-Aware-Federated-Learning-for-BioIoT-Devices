import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── Color palette ─────────────────────────────────────────────────────
const C = {
  teal:       "#1D9E75",
  tealLight:  "#9FE1CB",
  tealFill:   "rgba(29,158,117,0.12)",
  red:        "#A32D2D",
  redLight:   "#F09595",
  redFill:    "rgba(163,45,45,0.1)",
  redBorder:  "rgba(163,45,45,0.3)",
  purple:     "#534AB7",
  purpleFill: "rgba(83,74,183,0.1)",
  amber:      "#854F0B",
  amberFill:  "rgba(133,79,11,0.1)",
  amberBorder:"rgba(133,79,11,0.3)",
  blue:       "#185FA5",
  blueFill:   "rgba(24,95,165,0.12)",
  blueBorder: "rgba(24,95,165,0.3)",
  gray:       "#888780",
  bg:         "#0f1117",
  bgCard:     "#161b27",
  bgSecondary:"#1a2030",
  border:     "rgba(255,255,255,0.08)",
  borderMed:  "rgba(255,255,255,0.14)",
  text:       "#e8e8e8",
  textMuted:  "#888780",
};

const MAX_EEG  = 80;
const MAX_FL   = 20;
const DP_LIMIT = 0.5;
const API_POLL = 2000; // ms between API polls

// ── API helpers ───────────────────────────────────────────────────────
async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}

// ── EEG simulation (no real-time HTTP surface for gRPC stream) ────────
function eegPoint(t, sz) {
  return sz
    ? 0.5 + 0.38 * Math.sin(2 * Math.PI * 6 * t) + (Math.random() - 0.5) * 0.08
    : 0.28 + 0.12 * Math.sin(2 * Math.PI * 1.1 * t) + (Math.random() - 0.5) * 0.06;
}
function specRatio(sz) {
  return sz ? 0.44 + Math.random() * 0.16 : 0.12 + Math.random() * 0.14;
}
function fmt(n, d = 4) { return Number(n).toFixed(d); }

// ── Shared components ─────────────────────────────────────────────────
function ChartTip({ active, payload, label, rows }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: C.bgCard, border: `0.5px solid ${C.borderMed}`,
      padding: "8px 12px", borderRadius: "6px",
      fontSize: "11px", fontFamily: "monospace",
    }}>
      {label !== undefined && (
        <div style={{ color: C.textMuted, marginBottom: "4px" }}>Round {label}</div>
      )}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.stroke || p.fill || C.gray }}>
          {rows?.[i] ?? p.name}: {fmt(p.value)}
        </div>
      ))}
    </div>
  );
}

function Legend({ items }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "16px", marginTop: "10px" }}>
      {items.map(({ color, label, dashed }) => (
        <div key={label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <div style={{
            width: "16px", height: "2px",
            background: dashed ? "none" : color,
            borderBottom: dashed ? `2px dashed ${color}` : "none",
          }} />
          <span style={{ fontSize: "11px", color: C.textMuted }}>{label}</span>
        </div>
      ))}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: C.bgCard, border: `0.5px solid ${C.border}`,
      borderRadius: "10px", padding: "18px", ...style,
    }}>
      {children}
    </div>
  );
}

function CardTitle({ title, sub, right }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "14px" }}>
      <div>
        <div style={{ fontSize: "13px", fontWeight: 500, color: C.text }}>{title}</div>
        {sub && <div style={{ fontSize: "11px", color: C.textMuted, marginTop: "2px" }}>{sub}</div>}
      </div>
      {right}
    </div>
  );
}

function Badge({ label, type = "success" }) {
  const map = {
    success: { bg: C.tealFill,   color: C.teal,   border: C.tealLight + "55" },
    danger:  { bg: C.redFill,    color: C.red,     border: C.redBorder },
    warning: { bg: C.amberFill,  color: C.amber,   border: C.amberBorder },
    info:    { bg: C.blueFill,   color: C.blue,    border: C.blueBorder },
    neutral: { bg: C.bgSecondary,color: C.textMuted,border: C.border },
  };
  const t = map[type] ?? map.neutral;
  return (
    <div style={{
      display: "inline-block", padding: "3px 9px",
      borderRadius: "5px", fontSize: "11px", fontWeight: 500,
      background: t.bg, color: t.color, border: `0.5px solid ${t.border}`,
    }}>
      {label}
    </div>
  );
}

// ── API status indicator ──────────────────────────────────────────────
function APIIndicator({ live }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
      <div style={{
        width: "6px", height: "6px", borderRadius: "50%",
        background: live ? C.teal : C.amber,
      }} />
      <span style={{ fontSize: "11px", color: C.textMuted, fontFamily: "monospace" }}>
        {live ? "API live" : "API offline — simulating"}
      </span>
    </div>
  );
}

// ── Header ────────────────────────────────────────────────────────────
function Header({ status, round, uptime, apiLive }) {
  const type  = status === "seizure" ? "danger" : status === "aggregating" ? "warning" : "success";
  const label = status === "seizure" ? "Seizure event" : status === "aggregating" ? "Aggregating" : "Nominal";
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "12px 20px",
      borderBottom: `0.5px solid ${C.border}`,
      background: C.bgCard,
      position: "sticky", top: 0, zIndex: 10,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{
            width: "20px", height: "20px", borderRadius: "4px",
            background: C.teal, display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: "white" }} />
          </div>
          <span style={{ fontSize: "15px", fontWeight: 500, color: C.text, letterSpacing: "-0.3px" }}>
            Bio-Sync HPC
          </span>
        </div>
        <span style={{ fontSize: "11px", color: C.textMuted, borderLeft: `0.5px solid ${C.border}`, paddingLeft: "14px" }}>
          Federated learning command center
        </span>
        <APIIndicator live={apiLive} />
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "18px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: C.teal }} />
          <span style={{ fontSize: "11px", color: C.textMuted }}>Live</span>
        </div>
        <span style={{ fontSize: "11px", color: C.textMuted }}>
          FL round <strong style={{ color: C.text, fontWeight: 500 }}>{round}</strong>
        </span>
        <span style={{ fontSize: "11px", fontFamily: "monospace", color: C.textMuted }}>
          {uptime}
        </span>
        <Badge label={label} type={type} />
      </div>
    </div>
  );
}

function SeizureBanner() {
  return (
    <div style={{
      background: C.redFill, border: `0.5px solid ${C.redBorder}`,
      borderRadius: "7px", padding: "10px 14px",
      display: "flex", alignItems: "center", gap: "10px",
      fontSize: "12px", fontWeight: 500, color: C.red,
    }}>
      <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: C.redLight, flexShrink: 0 }} />
      Seizure event detected — spectral ratio exceeded 0.40 — KEDA scaling to 4 pods — ADT mirror syncing
    </div>
  );
}

function StatRow({ pods, round, latestDisc, seizure }) {
  const stats = [
    { label: "Active pods",        value: pods,               unit: "/ 4",  color: C.blue   },
    { label: "FL rounds complete", value: round,              unit: null,   color: C.text   },
    { label: "Discrimination",     value: fmt(latestDisc),    unit: null,   color: C.teal   },
    { label: "System status",      value: seizure ? "Seizure" : "Nominal", unit: null, color: seizure ? C.red : C.teal },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
      {stats.map(({ label, value, unit, color }) => (
        <div key={label} style={{ background: C.bgSecondary, borderRadius: "7px", padding: "14px" }}>
          <div style={{ fontSize: "11px", color: C.textMuted, marginBottom: "6px" }}>{label}</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "5px" }}>
            <span style={{ fontSize: "22px", fontWeight: 500, color, fontFamily: "monospace" }}>{value}</span>
            {unit && <span style={{ fontSize: "12px", color: C.textMuted }}>{unit}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── EEG panel ─────────────────────────────────────────────────────────
function EEGPanel({ data, seizure }) {
  const stroke = seizure ? C.red : C.teal;
  const fill   = seizure ? C.redFill : C.tealFill;
  return (
    <Card>
      <CardTitle
        title="EEG signal stream"
        sub="CHB-MIT dataset · 100 Hz · simulated real-time"
        right={seizure ? <Badge label="Theta burst detected" type="danger" /> : null}
      />
      <div style={{ fontSize: "10px", color: C.textMuted, marginBottom: "2px", fontFamily: "monospace" }}>
        Amplitude (0–1 normalised)
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
          <YAxis domain={[0, 1]} hide />
          <XAxis dataKey="t" hide />
          <Tooltip content={<ChartTip rows={["Amplitude"]} />} />
          <Area type="monotone" dataKey="v" stroke={stroke} strokeWidth={1.5}
            fill={fill} fillOpacity={1} dot={false} animationDuration={0} />
        </AreaChart>
      </ResponsiveContainer>
      <div style={{ fontSize: "10px", color: C.textMuted, margin: "12px 0 2px", fontFamily: "monospace" }}>
        FFT spectral ratio (theta + alpha) · threshold = 0.40
      </div>
      <ResponsiveContainer width="100%" height={85}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
          <YAxis domain={[0, 1]} hide />
          <XAxis dataKey="t" hide />
          <ReferenceLine y={0.40} stroke={C.amber} strokeDasharray="4 3" strokeWidth={1} />
          <Tooltip content={<ChartTip rows={["Spectral ratio"]} />} />
          <Area type="monotone" dataKey="r" stroke={C.purple} strokeWidth={1.5}
            fill={C.purpleFill} fillOpacity={1} dot={false} animationDuration={0} />
        </AreaChart>
      </ResponsiveContainer>
      <Legend items={[
        { color: C.teal,   label: "EEG amplitude" },
        { color: C.purple, label: "Spectral ratio" },
        { color: C.amber,  label: "Burst threshold (0.40)", dashed: true },
      ]} />
    </Card>
  );
}

// ── ADT panel — real data ─────────────────────────────────────────────
function ADTPanel({ adt }) {
  const TwinCard = ({ name, hr, crit }) => (
    <div style={{
      background: C.bgSecondary,
      border: `0.5px solid ${crit ? C.redBorder : C.border}`,
      borderRadius: "7px", padding: "12px",
      transition: "border-color 0.3s",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
        <span style={{ fontSize: "11px", fontWeight: 500, color: C.textMuted }}>{name}</span>
        <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: crit ? C.red : C.teal }} />
      </div>
      <div style={{ fontFamily: "monospace", fontSize: "19px", fontWeight: 500, color: crit ? C.red : C.text, marginBottom: "8px" }}>
        {fmt(hr)} <span style={{ fontSize: "11px", color: C.textMuted, fontWeight: 400 }}>bpm</span>
      </div>
      <div style={{
        display: "inline-block", padding: "2px 7px", borderRadius: "4px",
        fontSize: "10px", fontFamily: "monospace",
        background: crit ? C.redFill : C.tealFill,
        color: crit ? C.red : C.teal,
      }}>
        IsCritical: {String(crit)}
      </div>
    </div>
  );
  return (
    <Card>
      <CardTitle title="Azure Digital Twins" sub="BioSyncTwin · Southeast Asia · real-time" />
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <TwinCard name="Silo-Alpha" hr={adt.a.hr} crit={adt.a.crit} />
        <TwinCard name="Silo-Beta"  hr={adt.b.hr} crit={adt.b.crit} />
      </div>
      <div style={{ marginTop: "10px", fontSize: "10px", color: C.textMuted, fontFamily: "monospace" }}>
        dtmi:com:biosync:Patient;1 · Azure SDK
      </div>
    </Card>
  );
}

// ── KEDA panel — real data ────────────────────────────────────────────
function KEDAPanel({ pods, seizure }) {
  return (
    <Card>
      <CardTitle title="KEDA autoscaling" sub="hpc-aggregator · CPU trigger · real pod count" />
      <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "12px" }}>
        <span style={{ fontFamily: "monospace", fontSize: "34px", fontWeight: 500, color: C.blue, transition: "color 0.3s" }}>
          {pods}
        </span>
        <span style={{ fontSize: "12px", color: C.textMuted }}>active pods</span>
      </div>
      <div style={{ display: "flex", gap: "5px", marginBottom: "10px" }}>
        {[1, 2, 3, 4].map(i => (
          <div key={i} style={{
            flex: 1, height: "24px", borderRadius: "5px",
            background: i <= pods ? C.blueFill : C.bgSecondary,
            border: `0.5px solid ${i <= pods ? C.blueBorder : C.border}`,
            transition: "all 0.3s ease",
          }} />
        ))}
      </div>
      <div style={{ fontSize: "10px", color: C.textMuted, fontFamily: "monospace", lineHeight: 1.7 }}>
        Min: 1 · Max: 4 · Poll: 5s · Threshold: 1% CPU
      </div>
      {seizure && (
        <div style={{ marginTop: "6px", fontSize: "11px", color: C.red, fontWeight: 500 }}>
          Burst detected — scaling up
        </div>
      )}
    </Card>
  );
}

// ── FL convergence panel — real data ─────────────────────────────────
function FLPanel({ fl }) {
  const latest = fl[fl.length - 1];
  return (
    <Card>
      <CardTitle
        title="FL convergence"
        sub="MLflow v2.8.0 · FedAvg-Weighted · real metrics"
        right={latest && (
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: "10px", color: C.textMuted }}>Discrimination</div>
            <div style={{ fontFamily: "monospace", fontSize: "17px", fontWeight: 500, color: C.teal }}>
              {fmt(latest.disc)}
            </div>
          </div>
        )}
      />
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={fl} margin={{ top: 5, right: 8, bottom: 5, left: 0 }}>
          <XAxis dataKey="rnd"
            tick={{ fontSize: 10, fill: C.textMuted, fontFamily: "monospace" }}
            axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]}
            tick={{ fontSize: 10, fill: C.textMuted, fontFamily: "monospace" }}
            axisLine={false} tickLine={false} width={24} />
          <Tooltip content={<ChartTip rows={["Discrimination", "Seizure conf.", "Normal conf."]} />} />
          <Line type="monotone" dataKey="disc" stroke={C.teal}   strokeWidth={2}   dot={false} />
          <Line type="monotone" dataKey="sc"   stroke={C.blue}   strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
          <Line type="monotone" dataKey="nc"   stroke={C.purple} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
      <Legend items={[
        { color: C.teal,   label: "Discrimination score" },
        { color: C.blue,   label: "Seizure confidence", dashed: true },
        { color: C.purple, label: "Normal confidence",  dashed: true },
      ]} />
    </Card>
  );
}

// ── Privacy budget panel — real data ──────────────────────────────────
function PrivacyPanel({ dp }) {
  const Gauge = ({ name, epsilon }) => {
    const pct   = Math.min((epsilon / DP_LIMIT) * 100, 100);
    const color = pct > 80 ? C.red : pct > 60 ? C.amber : C.teal;
    return (
      <div style={{ marginBottom: "14px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "5px" }}>
          <span style={{ fontSize: "11px", fontWeight: 500, color: C.textMuted }}>{name}</span>
          <span style={{ fontFamily: "monospace", fontSize: "11px", color }}>
            ε = {fmt(epsilon)} / {DP_LIMIT}
          </span>
        </div>
        <div style={{ height: "5px", background: C.bgSecondary, borderRadius: "3px", overflow: "hidden" }}>
          <div style={{
            height: "100%", width: `${pct.toFixed(1)}%`,
            background: color, borderRadius: "3px",
            transition: "width 0.4s ease",
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "3px" }}>
          <span style={{ fontSize: "10px", color: C.textMuted, fontFamily: "monospace" }}>
            δ = 1e-5 · Rényi DP
          </span>
          <span style={{ fontSize: "10px", fontFamily: "monospace", color }}>
            {pct.toFixed(1)}%
          </span>
        </div>
        {pct > 80 && (
          <div style={{ marginTop: "5px" }}>
            <Badge label="Budget at 80% — aggregation recommended" type={pct > 100 ? "danger" : "warning"} />
          </div>
        )}
      </div>
    );
  };
  return (
    <Card>
      <CardTitle title="Privacy budget" sub="Opacus v0.16.3 · formal (ε, δ)-DP · logged to MLflow" />
      <Gauge name="Silo-Alpha" epsilon={dp.a} />
      <Gauge name="Silo-Beta"  epsilon={dp.b} />
      <div style={{
        padding: "8px 10px", background: C.bgSecondary,
        borderRadius: "6px", fontSize: "10px",
        color: C.textMuted, fontFamily: "monospace", lineHeight: 1.7,
      }}>
        Mechanism: Gaussian on gradients · max norm: 1.0<br />
        Guarantee: formal (ε, δ)-DP · HIPAA / GDPR
      </div>
    </Card>
  );
}

function Footer() {
  return (
    <div style={{
      padding: "10px 20px",
      borderTop: `0.5px solid ${C.border}`,
      display: "flex", justifyContent: "space-between",
      fontSize: "10px", color: C.textMuted, fontFamily: "monospace",
    }}>
      <span>Bio-Sync HPC v2.0 · AKS BioSync-Cluster (Southeast Asia) · ACR biosyncregistry1772554412</span>
      <span>KEDA v2 · MLflow v2.8.0 · Opacus v0.16.3 · gRPC bidi-streaming</span>
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────
export default function Dashboard() {
  // ── EEG state (simulated — no HTTP surface for gRPC stream) ──────
  const [eeg, setEeg] = useState(() =>
    Array.from({ length: MAX_EEG }, (_, i) => ({
      t: i,
      v: +(0.28 + Math.random() * 0.08).toFixed(4),
      r: +(0.12 + Math.random() * 0.1).toFixed(4),
    }))
  );
  const [sz, setSz] = useState(false);
  const tRef     = useRef(0);
  const szTimer  = useRef(null);

  // ── Real API state ────────────────────────────────────────────────
  const [adt,    setAdt]    = useState({ a: { hr: 0.2845, crit: false }, b: { hr: 0.2763, crit: false } });
  const [pods,   setPods]   = useState(1);
  const [fl,     setFl]     = useState([
    { rnd: 1, disc: 0.18, sc: 0.57, nc: 0.46 },
    { rnd: 2, disc: 0.24, sc: 0.61, nc: 0.43 },
    { rnd: 3, disc: 0.31, sc: 0.65, nc: 0.40 },
  ]);
  const [dp,     setDp]     = useState({ a: 0.0, b: 0.0 });
  const [round,  setRound]  = useState(3);
  const [status, setStatus] = useState("nominal");
  const [uptime, setUptime] = useState("00:00:00");
  const [apiLive, setApiLive] = useState(false);
  const t0 = useRef(Date.now());
  const flRoundRef = useRef(3);

  // ── Uptime clock ──────────────────────────────────────────────────
  useEffect(() => {
    const iv = setInterval(() => {
      const e = Math.floor((Date.now() - t0.current) / 1000);
      const h = String(Math.floor(e / 3600)).padStart(2, "0");
      const m = String(Math.floor((e % 3600) / 60)).padStart(2, "0");
      const s = String(e % 60).padStart(2, "0");
      setUptime(`${h}:${m}:${s}`);
    }, 1000);
    return () => clearInterval(iv);
  }, []);

  // ── Poll real API ─────────────────────────────────────────────────
  const pollAPI = useCallback(async () => {
    const [adtData, podsData, flData, dpData] = await Promise.all([
      fetchJSON("/api/adt"),
      fetchJSON("/api/pods"),
      fetchJSON("/api/mlflow"),
      fetchJSON("/api/dp"),
    ]);

    let anyLive = false;

    if (adtData) {
      anyLive = true;
      setAdt({
        a: { hr: adtData["Silo-Alpha"]?.HeartRate ?? 0, crit: adtData["Silo-Alpha"]?.IsCritical ?? false },
        b: { hr: adtData["Silo-Beta"]?.HeartRate  ?? 0, crit: adtData["Silo-Beta"]?.IsCritical  ?? false },
      });
      const crit = adtData["Silo-Alpha"]?.IsCritical || adtData["Silo-Beta"]?.IsCritical;
      setSz(crit);
      setStatus(crit ? "seizure" : "nominal");
    }

    if (podsData) {
      anyLive = true;
      setPods(podsData.count ?? 1);
    }

    if (flData && flData.runs?.length) {
      anyLive = true;
      const runs = flData.runs;
      const mapped = runs.map((r, i) => ({
        rnd:  i + 1,
        disc: r.discrimination_score ?? 0,
        sc:   r.seizure_confidence   ?? 0,
        nc:   r.normal_confidence    ?? 0,
      }));
      setFl(mapped.slice(-MAX_FL));
      setRound(mapped.length);
      flRoundRef.current = mapped.length;
    }

    if (dpData) {
      anyLive = true;
      setDp({
        a: dpData["Silo_Alpha"] ?? dpData["Silo-Alpha"] ?? 0,
        b: dpData["Silo_Beta"]  ?? dpData["Silo-Beta"]  ?? 0,
      });
    }

    setApiLive(anyLive);
  }, []);

  useEffect(() => {
    pollAPI();
    const iv = setInterval(pollAPI, API_POLL);
    return () => clearInterval(iv);
  }, [pollAPI]);

  // ── EEG simulation tick (500ms) ───────────────────────────────────
  // When API is offline, also simulate ADT and pods locally
  useEffect(() => {
    const iv = setInterval(() => {
      tRef.current += 0.5;

      // Only trigger synthetic seizure if API is offline (no real data)
      if (!apiLive && !sz && Math.random() > 0.91) {
        setSz(true);
        setStatus("seizure");
        setPods(4);
        setAdt(p => ({ a: { ...p.a, crit: true }, b: { ...p.b, crit: true } }));
        setDp(p => ({
          a: Math.min(+(p.a + 0.055 + Math.random() * 0.02).toFixed(4), DP_LIMIT),
          b: Math.min(+(p.b + 0.048 + Math.random() * 0.02).toFixed(4), DP_LIMIT),
        }));
        if (szTimer.current) clearTimeout(szTimer.current);
        szTimer.current = setTimeout(() => {
          setSz(false);
          setStatus("nominal");
          setAdt(p => ({ a: { ...p.a, crit: false }, b: { ...p.b, crit: false } }));
          setPods(2);
          setTimeout(() => setPods(1), 6000);
        }, 4500);
      }

      const v = eegPoint(tRef.current, sz);
      const r = specRatio(sz);
      setEeg(p => [...p.slice(1), { t: +tRef.current.toFixed(1), v: +v.toFixed(4), r: +r.toFixed(4) }]);

      if (!apiLive) {
        setAdt(p => ({
          a: { ...p.a, hr: +(v + Math.random() * 0.003).toFixed(4) },
          b: { ...p.b, hr: +(v * 0.98 + Math.random() * 0.003).toFixed(4) },
        }));
      }
    }, 500);
    return () => { clearInterval(iv); if (szTimer.current) clearTimeout(szTimer.current); };
  }, [sz, apiLive]);

  // ── Simulated FL aggregation when offline (every 30s) ─────────────
  useEffect(() => {
    if (apiLive) return;
    const iv = setInterval(() => {
      flRoundRef.current += 1;
      const n = flRoundRef.current;
      setRound(n);
      setStatus(s => s !== "seizure" ? "aggregating" : s);
      setTimeout(() => setStatus(s => s === "aggregating" ? "nominal" : s), 2000);
      setFl(p => {
        const d  = +Math.min(0.14 + 0.62 * (1 - Math.exp(-n / 7)) + (Math.random() - 0.5) * 0.03, 0.96).toFixed(4);
        const sc = +Math.min(0.52 + d * 0.43 + (Math.random() - 0.5) * 0.02, 0.99).toFixed(4);
        const nc = +Math.max(0.48 - d * 0.38 + (Math.random() - 0.5) * 0.02, 0.01).toFixed(4);
        return [...p, { rnd: n, disc: d, sc, nc }].slice(-MAX_FL);
      });
    }, 30000);
    return () => clearInterval(iv);
  }, [apiLive]);

  const latest = fl[fl.length - 1]?.disc ?? 0;

  return (
    <div style={{
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      background: C.bg, minHeight: "100vh", display: "flex", flexDirection: "column",
      color: C.text,
    }}>
      <Header status={status} round={round} uptime={uptime} apiLive={apiLive} />

      <div style={{ padding: "14px 20px", flex: 1, display: "flex", flexDirection: "column", gap: "14px" }}>
        {sz && <SeizureBanner />}

        <StatRow pods={pods} round={round} latestDisc={latest} seizure={sz} />

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr minmax(0, 270px)", gap: "14px" }}>
          {/* Left col */}
          <EEGPanel data={eeg} seizure={sz} />

          {/* Middle col */}
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <FLPanel fl={fl} />
            <PrivacyPanel dp={dp} />
          </div>

          {/* Right col */}
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <ADTPanel adt={adt} />
            <KEDAPanel pods={pods} seizure={sz} />
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
}