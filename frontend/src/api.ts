// TypeScript mirrors of the Pydantic models in app/models.py.
// Keep these in sync if the backend contract changes.

import { useCallback, useEffect, useState } from 'react';

export type Side = 'buy' | 'sell' | 'hold';

export interface MarketObservation {
  ticker: string;
  date: string;
  price: number;
  price_history: number[];
  fundamentals: Record<string, unknown>;
  cash: number;
  position: number;
  portfolio_value: number;
  task_id: string;
  step_number: number;
  total_steps: number;
  chart_path: string;
  headlines: Array<{ date: string; headline: string; source: string; sentiment_label: string }>;
  forum_excerpts: Array<{ date: string; subreddit: string; score: number; post_text: string }>;
  indicators: Record<string, number | null>;
  peers: {
    peers: Array<{ peer_ticker: string; peer_close: number | null }>;
    commodity?: string | null;
    commodity_price?: number | null;
  };
  macro: Array<{ date: string; country: string; headline: string; policy_signal: string }>;
}

export interface EnvironmentState {
  task_id: string;
  current_step: number;
  total_steps: number;
  done: boolean;
  cash: number;
  position: number;
  portfolio_value: number;
  action_history: Array<{ side: Side; quantity: number }>;
  reward_history: number[];
}

export interface SpecialistVote {
  name: string;
  signal: number;
  confidence: number;
  rationale: string;
}

export interface CouncilDecision {
  votes: SpecialistVote[];
  action: { side: Side; quantity: number };
  rationale: string;
}

export interface OhlcvBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  in_episode: boolean;
}

export interface OhlcvResponse {
  task_id: string;
  ticker: string;
  bars: OhlcvBar[];
}

export interface TrainingMetrics {
  status: 'completed' | 'no_runs';
  run_name?: string;
  summary: Array<{
    task_id: string;
    total_reward: number;
    final_portfolio: number;
    buy_and_hold: number;
    alpha_pct: number;
  }>;
  mean_alpha_pct: number;
  reward_curve_png?: string | null;
  portfolio_curve_png?: string | null;
}

// ---------- fetch helpers --------------------------------------------------

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
}

async function jpost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  meta: () => jget<{ tasks: string[]; name: string; version: string }>('/meta'),
  reset: (task_id: string) =>
    jpost<{ observation: MarketObservation; info: Record<string, unknown> }>(
      '/reset',
      { task_id },
    ),
  step: (side: Side, quantity: number) =>
    jpost<{ observation: MarketObservation; reward: number; done: boolean; info: Record<string, unknown> }>(
      '/step',
      { side, quantity },
    ),
  state: () => jget<EnvironmentState>('/state'),
  ohlcv: (task_id: string) => jget<OhlcvResponse>(`/ohlcv?task_id=${encodeURIComponent(task_id)}`),
  council: () => jget<CouncilDecision>('/council'),
  trainingMetrics: () => jget<TrainingMetrics>('/training/metrics'),
};

// ---------- shared state hook ---------------------------------------------

export interface StockerEnv {
  tasks: string[];
  taskId: string | null;
  observation: MarketObservation | null;
  envState: EnvironmentState | null;
  council: CouncilDecision | null;
  ohlcv: OhlcvResponse | null;
  loading: boolean;
  error: string | null;
  selectTask: (taskId: string) => Promise<void>;
  submitTrade: (side: Side, quantity: number) => Promise<void>;
}

export function useStockerEnv(): StockerEnv {
  const [tasks, setTasks] = useState<string[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [observation, setObservation] = useState<MarketObservation | null>(null);
  const [envState, setEnvState] = useState<EnvironmentState | null>(null);
  const [council, setCouncil] = useState<CouncilDecision | null>(null);
  const [ohlcv, setOhlcv] = useState<OhlcvResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const selectTask = useCallback(async (next: string) => {
    setLoading(true);
    setError(null);
    try {
      const reset = await api.reset(next);
      setTaskId(next);
      setObservation(reset.observation);
      const [bars, st] = await Promise.all([api.ohlcv(next), api.state()]);
      setOhlcv(bars);
      setEnvState(st);
      try {
        setCouncil(await api.council());
      } catch (e) {
        // council can fail silently — UI shows empty state
        setCouncil(null);
        console.warn('council fetch failed', e);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const submitTrade = useCallback(async (side: Side, quantity: number) => {
    setLoading(true);
    setError(null);
    try {
      const stepRes = await api.step(side, quantity);
      setObservation(stepRes.observation);
      const st = await api.state();
      setEnvState(st);
      if (!stepRes.done) {
        try {
          setCouncil(await api.council());
        } catch (e) {
          setCouncil(null);
          console.warn('council fetch failed', e);
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  // bootstrap: load tasks, then auto-select first
  useEffect(() => {
    let cancelled = false;
    api
      .meta()
      .then((m) => {
        if (cancelled) return;
        setTasks(m.tasks);
        if (m.tasks.length > 0) {
          void selectTask(m.tasks[0]);
        }
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      });
    return () => {
      cancelled = true;
    };
  }, [selectTask]);

  return {
    tasks,
    taskId,
    observation,
    envState,
    council,
    ohlcv,
    loading,
    error,
    selectTask,
    submitTrade,
  };
}

// Display-name + role keyword mapping for the seven specialists.
// Backend `name` values come from app/council/specialists.py.
export const SPECIALIST_DISPLAY: Record<string, string> = {
  chart_pattern: 'Chart Pattern',
  seasonal_trend: 'Seasonal',
  indicator: 'Indicator',
  news: 'News',
  forum_sentiment: 'Forum',
  peer_commodity: 'Peer',
  geopolitics: 'Geo',
};

export function statusFromSignal(signal: number): 'green' | 'red' | 'gray' {
  if (signal > 0.1) return 'green';
  if (signal < -0.1) return 'red';
  return 'gray';
}
