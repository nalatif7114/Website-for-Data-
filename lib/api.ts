export type ModelId = "indobert" | "indolem" | "finbert";
export type Sentiment = "Negatif" | "Netral" | "Positif";
export type Probabilities = Record<Sentiment, number>;
export interface Prediction {
  model_id: ModelId; model_name: string; sentiment: Sentiment; probabilities: Probabilities;
  inference_ms: number; preprocessing: { input_language: string; model_input_language: string; max_length: number; truncated: boolean };
  translation?: { status: "translated"; translated_text: string; provider: string };
}
export interface ModelStatus { id: ModelId; name: string; configured: boolean; downloaded: boolean; loaded: boolean; ready: boolean; source: string | null; state: "unconfigured" | "configured" | "downloaded" | "loaded" | "error"; detail?: string }
export interface Health { status: "ok"; ready: boolean; device: string; models_configured: number }
export interface PredictResponse { prediction: Prediction }
export interface CompareResponse { results: Prediction[]; errors: { model_id: ModelId; model_name: string; message: string }[]; agreement: { unanimous: boolean; agreeing_models: ModelId[]; disagreeing_models: ModelId[] } | null }
const base = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 180_000);
  try {
    const response = await fetch(`${base}${path}`, { ...init, signal: controller.signal, headers: { "Content-Type": "application/json", ...init?.headers } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Backend returned ${response.status}`);
    return data as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new Error("The request timed out. Model loading or CPU inference may take several minutes; try again once the model is warm.");
    throw error;
  } finally { window.clearTimeout(timeout); }
}
export const api = {
  health: () => request<Health>("/health"),
  models: () => request<{ models: ModelStatus[] }>("/models"),
  predict: (text: string, model_id: ModelId) => request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify({ text, model_id }) }),
  compare: (text: string, model_ids: ModelId[]) => request<CompareResponse>("/compare", { method: "POST", body: JSON.stringify({ text, model_ids }) }),
};
