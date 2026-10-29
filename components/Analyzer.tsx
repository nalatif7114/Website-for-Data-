"use client";
import { useMemo, useState } from "react";
import { Activity, ArrowRight, Check, ChevronDown, CircleAlert, Clock3, LoaderCircle, Network, RotateCcw, Sparkles } from "lucide-react";
import { api, type CompareResponse, type ModelId, type ModelStatus, type Prediction, type Sentiment } from "@/lib/api";
const choices: { id: ModelId; name: string; short: string }[] = [{ id: "indobert", name: "IndoBERT", short: "ID" }, { id: "indolem", name: "IndoLEM", short: "ID" }, { id: "finbert", name: "FinBERT", short: "EN · translated" }];
const sentimentClass: Record<Sentiment, string> = { Negatif: "negative", Netral: "neutral", Positif: "positive" };
function ResultCard({ result }: { result: Prediction }) {
  const [showTranslation, setShowTranslation] = useState(false);
  return <article className="result-card">
    <div className="result-heading"><div><span className="overline">MODEL OUTPUT</span><h3>{result.model_name}</h3></div><span className="latency"><Clock3 size={13} /> {result.inference_ms.toFixed(0)} ms</span></div>
    <div className="sentiment-row"><span className={`sentiment ${sentimentClass[result.sentiment]}`}><i />{result.sentiment}</span><span className="confidence">{(Math.max(...Object.values(result.probabilities)) * 100).toFixed(1)}% score</span></div>
    <div className="probabilities">{(["Negatif", "Netral", "Positif"] as Sentiment[]).map(label => <div className="prob-row" key={label}><span>{label}</span><div className="prob-track"><i className={sentimentClass[label]} style={{ width: `${result.probabilities[label] * 100}%` }} /></div><b>{(result.probabilities[label] * 100).toFixed(1)}%</b></div>)}</div>
    {result.translation && <div className="translation"><button onClick={() => setShowTranslation(!showTranslation)} aria-expanded={showTranslation}><span>FinBERT · GoogleTranslator</span><ChevronDown size={15} className={showTranslation ? "turned" : ""} /></button>{showTranslation && <p>{result.translation.translated_text}</p>}</div>}
  </article>;
}
export default function Analyzer({ statuses }: { statuses: ModelStatus[] }) {
  const [text, setText] = useState(""); const [selected, setSelected] = useState<ModelId>("indobert"); const [compare, setCompare] = useState(true);
  const [busy, setBusy] = useState(false); const [results, setResults] = useState<Prediction[]>([]); const [errors, setErrors] = useState<string[]>([]); const [notice, setNotice] = useState("");
  const wordCount = useMemo(() => text.trim() ? text.trim().split(/\s+/).length : 0, [text]);
  const groupedResults = new Map<Sentiment, string[]>();
  for (const result of results) groupedResults.set(result.sentiment, [...(groupedResults.get(result.sentiment) ?? []), result.model_name]);
  const comparisonSummary = groupedResults.size <= 1
    ? `All available models agree on ${results[0]?.sentiment ?? "the label"}.`
    : `${[...groupedResults].map(([sentiment, names]) => `${names.join(" + ")} → ${sentiment}`).join(" · ")}. No consensus label is inferred.`;
  const run = async () => {
    if (text.trim().length < 8) { setNotice("Enter at least 8 characters so the models have enough context."); return; }
    setBusy(true); setResults([]); setErrors([]); setNotice("");
    try {
      if (compare) { const response: CompareResponse = await api.compare(text.trim(), choices.map(m => m.id)); setResults(response.results); setErrors(response.errors.map(e => `${e.model_name}: ${e.message}`)); }
      else { const response = await api.predict(text.trim(), selected); setResults([response.prediction]); }
    } catch (e) { setNotice(e instanceof Error ? e.message : "The backend request failed."); }
    finally { setBusy(false); }
  };
  const selectedStatus = statuses.find(m => m.id === selected);
  return <section className="analyzer-section" id="analyzer">
    <div className="section-head"><div><span className="eyebrow"><span className="pulse-dot" /> LIVE INFERENCE</span><h2>Test the signal.</h2><p>Run a fine-tuned checkpoint against your Indonesian financial news.</p></div><span className="local-only"><span className="lock-dot" /> PRIVATE BY DESIGN</span></div>
    <div className="analyzer-grid"><div className="input-panel">
      <div className="input-top"><span>INDONESIAN NEWS INPUT</span><button className="example-link" onClick={() => setText("Harga saham perusahaan mengalami kenaikan setelah laporan keuangan menunjukkan pertumbuhan laba.")}>Load example <ArrowRight size={13} /></button></div>
      <textarea className="focus-visible:ring-2 focus-visible:ring-cyan-300" value={text} maxLength={8000} onChange={e => { setText(e.target.value); setNotice(""); }} placeholder="Tempel berita finansial berbahasa Indonesia di sini…" aria-label="Indonesian financial news input" />
      <div className="input-meta"><span>{text.length.toLocaleString()} / 8,000 characters</span><span>{wordCount} words</span></div>
      <div className="control-line"><div className="mode-control"><button onClick={() => setCompare(true)} className={compare ? "active" : ""}><Network size={14} /> Compare all</button><button onClick={() => setCompare(false)} className={!compare ? "active" : ""}>Single model</button></div>{!compare && <select value={selected} onChange={e => setSelected(e.target.value as ModelId)} aria-label="Choose a model">{choices.map(c => <option value={c.id} key={c.id}>{c.name}</option>)}</select>}</div>
      {!compare && <div className="selected-state">{selectedStatus?.ready ? <><Check size={13} /> Checkpoint loaded and ready</> : selectedStatus?.downloaded ? <><Check size={13} /> Checkpoint downloaded · loads on first run</> : selectedStatus?.configured ? <><CircleAlert size={13} /> Model source configured · first run may download weights</> : <><CircleAlert size={13} /> Checkpoint not configured</>}</div>}
      {notice && <p className="inline-error"><CircleAlert size={15} />{notice}</p>}
      <button className="analyze-button focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950" onClick={run} disabled={busy || !text.trim()}>{busy ? <><LoaderCircle size={17} className="spin" /> Running inference…</> : <><Sparkles size={16} /> Analyze sentiment <ArrowRight size={15} /></>}</button>
      {busy && <p className="loading-note"><Activity size={13} /> First request may include model download and loading time.</p>}
    </div><div className="results-panel">
      <div className="results-header"><span>COMPARISON</span>{results.length > 0 && <button onClick={() => { setResults([]); setErrors([]); }}><RotateCcw size={12} /> Clear</button>}</div>
      {results.length === 0 && errors.length === 0 && <div className="empty-results"><div className="empty-orbit"><Network size={23} /></div><h3>Your analysis appears here</h3><p>Three models examine the same input independently. Nothing is inferred until you run it.</p></div>}
      {errors.map((e, i) => <div className="model-error" key={i}><CircleAlert size={15} />{e}</div>)}
      {results.length > 0 && <><div className="result-cards">{results.map(r => <ResultCard key={r.model_id} result={r} />)}</div>{results.length > 1 && <div className="agreement-note"><Network size={14} /><span>{comparisonSummary}</span></div>}</>}
      <p className="confidence-note">Softmax values are model confidence scores; they are not necessarily calibrated probabilities or measures of empirical accuracy.</p>
    </div></div>
  </section>;
}
