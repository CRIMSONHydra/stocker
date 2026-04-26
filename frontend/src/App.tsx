/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Monitor,
  Users,
  TrendingUp,
  Library,
  Wallet,
  Brain,
  FileText,
  LifeBuoy,
  Bell,
  CreditCard,
  User,
  Search,
  ChevronUp,
  Cpu,
  BarChart3,
  Globe,
  Newspaper,
  Zap,
  Activity,
  Calendar,
  LineChart,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import {
  CrosshairMode,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
} from 'lightweight-charts';

import {
  api,
  SPECIALIST_DISPLAY,
  statusFromSignal,
  useStockerEnv,
  type CouncilDecision,
  type EnvironmentState,
  type MarketObservation,
  type OhlcvBar,
  type OhlcvResponse,
  type Side,
  type StockerEnv,
  type TrainingMetrics,
} from './api';

// --------------------------------------------------------------------- atoms
const SidebarItem = ({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: any;
  label: string;
  active: boolean;
  onClick: () => void;
}) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 group relative ${
      active
        ? 'text-[#00FF41] bg-gradient-to-r from-[#00FF41]/10 to-transparent border-l-2 border-[#00FF41]'
        : 'text-neutral-400 hover:bg-white/5 hover:text-[#00FF41] hover:translate-x-1'
    }`}
  >
    <Icon className={`w-5 h-5 ${active ? 'text-[#00FF41]' : 'group-hover:text-[#00FF41]'}`} />
    <span className="font-medium text-[13px] tracking-tight">{label}</span>
  </button>
);

const AgentCard = ({
  title,
  icon: Icon,
  stat,
  color,
  children,
  active,
}: {
  title: string;
  icon: any;
  stat?: string;
  color: string;
  children: any;
  active?: boolean;
}) => (
  <div
    className={`glass-panel rounded-xl flex flex-col transition-all duration-300 group hover:border-${color}/40 ${
      active ? `border-${color}/30 shadow-[0_0_30px_rgba(115,31,255,0.1)]` : ''
    }`}
  >
    <div className="p-3 px-4 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 text-${color}`} />
        <h3 className="font-semibold text-sm text-[#e5e2e1]">{title}</h3>
      </div>
      {stat && <span className={`font-mono text-xs font-bold text-${color}`}>{stat}</span>}
    </div>
    <div className="p-4 flex-1 flex flex-col gap-3">{children}</div>
  </div>
);

// ---------------------------------------------------------------------- chart

const CandlestickChart = ({ bars }: { bars: OhlcvBar[] }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  // mount once
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#a3a3a3',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: false,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.05, bottom: 0.28 },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: 'rgba(0,255,65,0.3)', width: 1, style: 3, labelBackgroundColor: '#00FF41' },
        horzLine: { color: 'rgba(0,255,65,0.3)', width: 1, style: 3, labelBackgroundColor: '#00FF41' },
      },
      autoSize: false,
    });
    chartRef.current = chart;

    candleRef.current = chart.addCandlestickSeries({
      upColor: '#00FF41',
      downColor: '#ffb4ab',
      wickUpColor: '#00FF41',
      wickDownColor: '#ffb4ab',
      borderVisible: false,
    });

    volumeRef.current = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      color: 'rgba(0,255,65,0.3)',
    });
    volumeRef.current.priceScale().applyOptions({
      scaleMargins: { top: 0.78, bottom: 0 },
    });

    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) chart.applyOptions({ width: Math.floor(rect.width), height: Math.floor(rect.height) });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
    };
  }, []);

  // push data when bars change
  useEffect(() => {
    if (!candleRef.current || !volumeRef.current) return;
    if (!bars || bars.length === 0) {
      candleRef.current.setData([]);
      volumeRef.current.setData([]);
      return;
    }
    candleRef.current.setData(
      bars.map((b) => ({
        time: b.time,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    );
    volumeRef.current.setData(
      bars.map((b) => ({
        time: b.time,
        value: b.volume,
        color: b.close >= b.open ? 'rgba(0,255,65,0.35)' : 'rgba(255,180,171,0.35)',
      })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [bars]);

  return <div ref={containerRef} className="flex-1 min-h-[300px] w-full" />;
};

// ---------------------------------------------------------------- TerminalView

const TerminalView = ({ env }: { env: StockerEnv }) => {
  const { observation, envState, council, ohlcv, submitTrade, loading } = env;

  const [side, setSide] = useState<Side>('buy');
  const [qty, setQty] = useState<number>(10);

  const change = useMemo(() => {
    if (!ohlcv || ohlcv.bars.length < 2) return 0;
    const last = ohlcv.bars[ohlcv.bars.length - 1];
    const prev = ohlcv.bars[ohlcv.bars.length - 2];
    return ((last.close - prev.close) / prev.close) * 100;
  }, [ohlcv]);

  const lastVolume = ohlcv?.bars[ohlcv.bars.length - 1]?.volume ?? 0;
  const portfolioValue = envState?.portfolio_value ?? observation?.portfolio_value ?? 0;
  const startingCash = 10000;
  const alphaPct = portfolioValue ? ((portfolioValue - startingCash) / startingCash) * 100 : 0;

  const consensus = council
    ? council.votes.reduce((a, v) => a + v.signal, 0) / council.votes.length
    : 0;

  const submitDisabled = loading || !observation || envState?.done;

  return (
    <div className="flex-1 flex flex-col gap-4 min-h-0">
      {/* Top Ticker Bar */}
      <div className="h-14 shrink-0 bg-white/5 backdrop-blur-md border border-white/10 rounded-lg flex items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <div className="flex items-baseline gap-2">
            <span className="font-bold text-lg text-white">{observation?.ticker ?? '—'}</span>
            <span className="font-mono text-neutral-400">
              {observation ? `$${observation.price.toFixed(2)}` : '—'}
            </span>
            <span
              className={`font-mono text-xs ${change >= 0 ? 'text-[#00FF41]' : 'text-[#ffb4ab]'}`}
            >
              ({change >= 0 ? '+' : ''}
              {change.toFixed(2)}%)
            </span>
          </div>
          <div className="w-px h-6 bg-white/10" />
          <div className="flex items-center gap-3">
            <span className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest">Vol:</span>
            <span className="font-mono text-sm text-white">
              {lastVolume ? `${(lastVolume / 1_000_000).toFixed(1)}M` : '—'}
            </span>
          </div>
          <div className="w-px h-6 bg-white/10" />
          <div className="flex items-center gap-3">
            <span className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest">Step:</span>
            <span className="font-mono text-sm text-white">
              {envState ? `${envState.current_step}/${envState.total_steps}` : '—'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-8">
          <div className="flex flex-col items-end">
            <span className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest mb-0.5">Portfolio</span>
            <span className="font-mono font-bold text-white">
              ${portfolioValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="w-px h-8 bg-white/10" />
          <div className="flex flex-col items-end">
            <span className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest mb-0.5">Alpha</span>
            <span
              className={`font-mono font-bold ${alphaPct >= 0 ? 'text-[#00FF41]' : 'text-[#ffb4ab]'}`}
            >
              {alphaPct >= 0 ? '+' : ''}
              {alphaPct.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        {/* Chart Area */}
        <div className="flex-1 glass-panel rounded-lg flex flex-col relative overflow-hidden group">
          <div className="absolute inset-0 scanline z-0 pointer-events-none" />
          <div className="h-10 border-b border-white/5 flex items-center justify-between px-4 bg-white/5 z-10">
            <div className="flex gap-4 font-mono text-[10px] uppercase tracking-widest">
              <button className="text-[#00FF41] border-b border-[#00FF41] py-3">1D</button>
              <button className="text-neutral-500 hover:text-white py-3">1W</button>
              <button className="text-neutral-500 hover:text-white py-3">1M</button>
              <button className="text-neutral-500 hover:text-white py-3">3M</button>
            </div>
            <div className="flex gap-2">
              <button className="p-1 hover:bg-white/10 rounded transition-colors text-neutral-400">
                <Zap className="w-4 h-4" />
              </button>
              <button className="p-1 hover:bg-white/10 rounded transition-colors text-neutral-400">
                <Monitor className="w-4 h-4" />
              </button>
            </div>
          </div>
          <CandlestickChart bars={ohlcv?.bars ?? []} />
        </div>

        {/* Council Feed Sidebar */}
        <div className="w-80 glass-panel rounded-lg flex flex-col overflow-hidden shrink-0">
          <div className="h-10 border-b border-white/5 px-4 flex items-center justify-between bg-white/5">
            <span className="text-[10px] font-bold uppercase tracking-widest text-[#e5e2e1]">Council Live Feed</span>
            <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            {council?.votes.length ? (
              council.votes.map((v) => {
                const status = statusFromSignal(v.signal);
                return (
                  <div
                    key={v.name}
                    className="p-3 rounded border border-transparent hover:border-white/10 hover:bg-white/5 transition-all group cursor-default"
                  >
                    <div className="flex justify-between items-center mb-1.5">
                      <span className="text-xs font-semibold group-hover:text-[#00FF41] transition-colors">
                        {SPECIALIST_DISPLAY[v.name] ?? v.name}
                      </span>
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${
                          status === 'green'
                            ? 'bg-[#00FF41]'
                            : status === 'red'
                            ? 'bg-[#ffb4ab]'
                            : 'bg-neutral-500'
                        }`}
                      />
                    </div>
                    <p className="text-[11px] text-neutral-400 leading-relaxed font-mono">{v.rationale}</p>
                  </div>
                );
              })
            ) : (
              <div className="p-3 text-[11px] text-neutral-500 font-mono">
                {observation ? 'Polling council…' : 'Waiting for environment.'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Panel */}
      <div className="h-32 shrink-0 glass-panel rounded-lg flex overflow-hidden border-[#731fff]/20">
        <div className="w-48 border-r border-white/5 flex flex-col justify-center px-6 bg-[#731fff]/5 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-[#731fff]/10 to-transparent pointer-events-none" />
          <div className="flex items-center gap-1.5 mb-1">
            <Activity className="w-3 h-3 text-[#731fff]" />
            <span className="text-[9px] uppercase font-bold tracking-widest text-[#731fff]">Moderator</span>
          </div>
          <h3 className="font-bold text-lg text-white">Gemma-4</h3>
          <span className="text-[10px] text-neutral-500 mt-0.5">
            Consensus: {consensus >= 0 ? '+' : ''}
            {consensus.toFixed(2)}
          </span>
        </div>
        <div className="flex-1 p-6 relative flex flex-col justify-center">
          <p className="font-mono text-[13px] text-neutral-300 leading-relaxed max-w-3xl">
            <span className="text-[#731fff] mr-2 opacity-70">&gt;</span>
            {council?.rationale ?? (observation ? 'Awaiting moderator synthesis.' : 'Run a task to see live moderator output.')}
            <span className="inline-block w-2.5 h-4 bg-[#731fff] align-middle ml-1 animate-pulse" />
          </p>
        </div>
        <div className="w-72 border-l border-white/5 flex flex-col items-stretch justify-center bg-[#00FF41]/5 backdrop-blur-sm px-4 gap-2">
          <div className="flex items-center gap-2">
            <select
              value={side}
              onChange={(e) => setSide(e.target.value as Side)}
              className="bg-black/40 border border-white/10 rounded text-[11px] font-mono px-2 py-1 text-white"
            >
              <option value="buy">BUY</option>
              <option value="sell">SELL</option>
              <option value="hold">HOLD</option>
            </select>
            <input
              type="number"
              min={0}
              value={qty}
              onChange={(e) => setQty(Math.max(0, parseInt(e.target.value || '0', 10)))}
              className="w-20 bg-black/40 border border-white/10 rounded text-[11px] font-mono px-2 py-1 text-white"
            />
          </div>
          <button
            onClick={() => void submitTrade(side, qty)}
            disabled={submitDisabled}
            className="bg-[#00FF41] disabled:opacity-40 disabled:cursor-not-allowed text-black font-black text-sm px-6 py-2 rounded-md shadow-[0_0_20px_rgba(0,255,65,0.3)] hover:scale-105 transition-transform"
          >
            {loading ? 'EXECUTING…' : `${side.toUpperCase()} ${side === 'hold' ? '' : qty}`}
          </button>
        </div>
      </div>
    </div>
  );
};

// ----------------------------------------------------------------- CouncilView

const SPECIALIST_VISUALS: Record<
  string,
  { icon: any; color: string; render: () => any }
> = {
  chart_pattern: {
    icon: Zap,
    color: '[#00FF41]',
    render: () => (
      <div className="h-20 w-full bg-black/40 rounded border border-white/5 relative overflow-hidden flex items-end p-2 pb-0">
        <svg className="w-full h-full" viewBox="0 0 100 40" preserveAspectRatio="none">
          <path d="M0,35 Q10,32 20,38 T40,20 T60,25 T100,5" fill="none" stroke="#00FF41" strokeWidth="2" />
          <path d="M0,35 Q10,32 20,38 T40,20 T60,25 T100,5 V40 H0 Z" fill="url(#chart-grad)" />
          <defs>
            <linearGradient id="chart-grad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#00FF41" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#00FF41" stopOpacity="0" />
            </linearGradient>
          </defs>
        </svg>
      </div>
    ),
  },
  indicator: {
    icon: LineChart,
    color: '[#00FF41]',
    render: () => (
      <div className="space-y-2 py-2">
        {[
          { l: 'RSI', v: 62, c: '#00FF41' },
          { l: 'MACD', v: 70, c: '#00FF41' },
          { l: 'ATR', v: 35, c: 'white' },
        ].map((b, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-[8px] font-bold text-neutral-500 w-10">{b.l}</span>
            <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div className="h-full bg-current transition-all" style={{ width: `${b.v}%`, color: b.c }} />
            </div>
          </div>
        ))}
      </div>
    ),
  },
  news: {
    icon: Newspaper,
    color: '[#00FF41]',
    render: () => (
      <div className="space-y-2 max-h-20 overflow-hidden relative">
        <div className="flex gap-2 items-start opacity-90">
          <div className="w-1 h-1 rounded-full bg-[#00FF41] mt-1.5" />
          <span className="text-[10px] text-neutral-400">Latest sentiment-tagged headlines.</span>
        </div>
        <div className="flex gap-2 items-start opacity-70">
          <div className="w-1 h-1 rounded-full bg-[#00FF41] mt-1.5" />
          <span className="text-[10px] text-neutral-400">Tone aggregated by NewsAgent.</span>
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-[#1c1b1b]" />
      </div>
    ),
  },
  forum_sentiment: {
    icon: Users,
    color: 'neutral-400',
    render: () => (
      <div className="h-20 w-full bg-black/40 rounded border border-white/5 relative p-2 overflow-hidden">
        <div className="absolute top-[20%] left-[70%] w-3 h-3 bg-[#00FF41] rounded-full blur-[2px] opacity-60" />
        <div className="absolute top-[40%] left-[60%] w-2 h-2 bg-[#00FF41] rounded-full blur-[1px] opacity-40" />
        <div className="absolute top-[10%] left-[80%] w-4 h-4 bg-[#00FF41] rounded-full blur-[3px] opacity-80" />
        <div className="absolute bottom-2 left-2 right-2 flex justify-between text-[7px] font-bold text-neutral-700 border-t border-white/5 pt-1">
          <span>BEAR</span>
          <span>BULL</span>
        </div>
      </div>
    ),
  },
  peer_commodity: {
    icon: TrendingUp,
    color: '[#00FF41]',
    render: () => (
      <div className="grid grid-cols-2 gap-2 h-20">
        {[
          { l: 'Peer 1', v: '+1.2% ▲' },
          { l: 'Peer 2', v: '+0.4%' },
          { l: 'Peer 3', v: '-0.1%' },
          { l: 'Comm.', v: '+0.7%' },
        ].map((m, i) => (
          <div key={i} className="bg-white/5 rounded p-1.5 flex flex-col justify-center">
            <span className="text-[7px] font-bold text-neutral-600 uppercase">{m.l}</span>
            <span className="text-[10px] font-mono text-neutral-300">{m.v}</span>
          </div>
        ))}
      </div>
    ),
  },
  geopolitics: {
    icon: Globe,
    color: 'neutral-400',
    render: () => (
      <div className="grid grid-cols-2 gap-2 h-20">
        {[
          { l: 'DXY', v: '104.2 ▲' },
          { l: '10Y', v: '4.32% ▲' },
          { l: 'SPX', v: '-0.2%' },
          { l: 'VIX', v: '14.1' },
        ].map((m, i) => (
          <div key={i} className="bg-white/5 rounded p-1.5 flex flex-col justify-center">
            <span className="text-[7px] font-bold text-neutral-600 uppercase">{m.l}</span>
            <span className="text-[10px] font-mono text-neutral-300">{m.v}</span>
          </div>
        ))}
      </div>
    ),
  },
  seasonal_trend: {
    icon: Calendar,
    color: '[#00FF41]',
    render: () => (
      <div className="h-20 w-full bg-black/40 rounded border border-white/5 relative p-2 flex items-end gap-1">
        {[30, 45, 35, 60, 55, 70, 50, 65, 75, 60, 80, 72].map((h, i) => (
          <div key={i} className="flex-1 bg-[#00FF41]/40 rounded-t-sm" style={{ height: `${h}%` }} />
        ))}
        <div className="absolute top-1 right-2 text-[8px] font-bold text-neutral-600 uppercase tracking-widest">12mo avg</div>
      </div>
    ),
  },
};

const CouncilView = ({ env }: { env: StockerEnv }) => {
  const { observation, council } = env;
  const consensus = council
    ? council.votes.reduce((a, v) => a + v.signal, 0) / council.votes.length
    : 0;
  const consensusLabel =
    consensus > 0.2 ? 'BULLISH' : consensus < -0.2 ? 'BEARISH' : 'NEUTRAL';
  const consensusColor =
    consensus > 0.2 ? 'text-[#00FF41]' : consensus < -0.2 ? 'text-[#ffb4ab]' : 'text-neutral-400';

  return (
    <div className="flex-1 flex flex-col gap-6">
      <header>
        <h1 className="text-3xl font-bold tracking-tighter text-white mb-2">Council Deep-Dive</h1>
        <p className="text-sm text-neutral-400 max-w-2xl">
          Real-time analysis and consensus generation across 7 autonomous specialized agents.
          Target asset:{' '}
          <span className="text-[#00FF41] font-mono bg-[#00FF41]/10 px-1.5 py-0.5 rounded border border-[#00FF41]/20">
            {observation?.ticker ?? '—'}
          </span>
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 flex-1">
        {/* Alpha Moderator */}
        <div className="col-span-1 glass-panel rounded-xl flex flex-col relative overflow-hidden group border-[#731fff]/30 shadow-[0_0_30px_rgba(115,31,255,0.1)]">
          <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#731fff] to-transparent opacity-50" />
          <div className="p-4 border-b border-white/5 flex justify-between items-start z-10">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-[#731fff]" />
              <h3 className="text-lg font-bold text-white">Alpha Moderator</h3>
            </div>
            <span className="text-[10px] font-bold text-[#731fff] border border-[#731fff]/30 px-2 py-1 rounded bg-[#731fff]/10">
              CONSENSUS
            </span>
          </div>

          <div className="p-6 flex-1 flex flex-col justify-between z-10">
            <div className="flex flex-col items-center py-6">
              <div className="relative w-40 h-20 overflow-hidden flex items-end justify-center mb-2">
                <div className="absolute top-0 w-40 h-40 rounded-full border-[12px] border-white/5 border-b-transparent border-r-transparent rotate-45" />
                <div
                  className={`absolute top-0 w-40 h-40 rounded-full border-[12px] ${
                    consensus > 0.2
                      ? 'border-[#00FF41]'
                      : consensus < -0.2
                      ? 'border-[#ffb4ab]'
                      : 'border-neutral-400'
                  } border-b-transparent border-r-transparent opacity-80`}
                  style={{
                    transform: `rotate(${45 + Math.max(-1, Math.min(1, consensus)) * 135}deg)`,
                    clipPath: 'polygon(50% 50%, 0% 0%, 100% 0%)',
                  }}
                />
                <div className="flex flex-col items-center translate-y-2">
                  <span className="font-mono text-3xl text-white font-bold">
                    {consensus >= 0 ? '+' : ''}
                    {consensus.toFixed(2)}
                  </span>
                  <span className={`text-[10px] font-bold tracking-widest mt-1 uppercase ${consensusColor}`}>
                    {consensusLabel}
                  </span>
                </div>
              </div>
              <div className="w-full flex justify-between px-4 font-mono text-[9px] text-neutral-500">
                <span>-1.0 (BEAR)</span>
                <span>0.0</span>
                <span>+1.0 (BULL)</span>
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-widest">Moderator Synthesis</span>
              <p className="text-sm text-neutral-400 leading-relaxed">
                {council?.rationale ??
                  'Run a task in Gallery to populate moderator synthesis from the live council.'}
              </p>
              {council && (
                <p className="text-[10px] text-[#731fff] font-mono mt-2">
                  Action: {council.action.side.toUpperCase()}
                  {council.action.side !== 'hold' ? ` ${council.action.quantity}` : ''}
                </p>
              )}
            </div>
          </div>
          <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-[#731fff]/10 blur-[100px] rounded-full pointer-events-none" />
        </div>

        {/* Specialist cards */}
        {Object.entries(SPECIALIST_VISUALS).map(([roleName, visual]) => {
          const vote = council?.votes.find((v) => v.name === roleName);
          const stat = vote ? `${vote.signal >= 0 ? '+' : ''}${vote.signal.toFixed(2)}` : '—';
          const Icon = visual.icon;
          return (
            <AgentCard
              key={roleName}
              title={SPECIALIST_DISPLAY[roleName] ?? roleName}
              icon={Icon}
              stat={stat}
              color={visual.color}
            >
              {visual.render()}
              <p className="text-xs text-neutral-500 leading-snug">
                {vote?.rationale ?? 'Awaiting council vote — run a task in Gallery.'}
              </p>
            </AgentCard>
          );
        })}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------- TrainingView

const MetricCard = ({
  label,
  value,
  trend,
}: {
  label: string;
  value: string;
  trend?: string;
}) => (
  <div className={`glass-panel p-4 rounded-lg flex flex-col justify-center ${trend ? 'border-l-2 border-[#00FF41]/30' : ''}`}>
    <span className={`text-[10px] font-bold uppercase tracking-widest mb-1 ${trend ? 'text-[#00FF41]' : 'text-neutral-500'}`}>{label}</span>
    <span className={`text-xl font-bold flex items-center ${trend ? 'text-[#00FF41]' : 'text-white'}`}>
      {value}
      {trend && (
        <span className="text-xs ml-1 flex items-center">
          {trend} <ChevronUp className="w-3 h-3 ml-0.5" />
        </span>
      )}
    </span>
  </div>
);

const TrainingView = () => {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);

  useEffect(() => {
    let cancel = false;
    api
      .trainingMetrics()
      .then((m) => {
        if (!cancel) setMetrics(m);
      })
      .catch(() => {
        if (!cancel) setMetrics({ status: 'no_runs', summary: [], mean_alpha_pct: 0 });
      });
    return () => {
      cancel = true;
    };
  }, []);

  // Placeholder logs — there's no JSON log on disk, kept as decorative.
  const logs = [
    { type: 'SYS', msg: 'Initializing GRPO policy trainer…' },
    { type: 'SYS', msg: 'Loading historical state tensors [BATCH_SIZE=256]…' },
    { type: 'OPT', msg: 'Step 4480: Adv = 0.124, Kl_div = 0.003', color: 'text-purple-400' },
    { type: 'REWARD', msg: 'Epoch 44 evaluation: Mean R = +1.24%', color: 'text-[#00FF41]' },
    { type: 'SYS', msg: 'Checkpoint saved.', pulse: true },
  ];

  const best = metrics?.summary?.length
    ? metrics.summary.reduce((a, b) => (b.alpha_pct > a.alpha_pct ? b : a))
    : null;
  const worst = metrics?.summary?.length
    ? metrics.summary.reduce((a, b) => (b.alpha_pct < a.alpha_pct ? b : a))
    : null;

  return (
    <div className="flex-1 flex flex-col gap-4 overflow-hidden">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0">
        <MetricCard label="Run" value={metrics?.run_name ?? '—'} />
        <MetricCard
          label="Mean Alpha %"
          value={metrics ? `${metrics.mean_alpha_pct.toFixed(2)}%` : '—'}
          trend={metrics && metrics.mean_alpha_pct > 0 ? 'up' : undefined}
        />
        <MetricCard
          label="Best Task"
          value={best ? `${best.task_id} (${best.alpha_pct.toFixed(1)}%)` : '—'}
        />
        <MetricCard
          label="Worst Task"
          value={worst ? `${worst.task_id} (${worst.alpha_pct.toFixed(1)}%)` : '—'}
        />
      </div>

      <div className="flex-1 flex flex-col md:flex-row gap-4 overflow-hidden min-h-0">
        {/* Left: status + logs */}
        <div className="w-full md:w-1/3 flex flex-col gap-4 h-full min-h-0">
          <div className="glass-panel rounded-lg p-6 shrink-0 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-[#731fff]/10 to-transparent pointer-events-none" />
            <div className="flex items-center justify-between border-b border-white/5 pb-3 mb-4">
              <h3 className="text-lg font-bold text-white">GRPO Status</h3>
              <span
                className={`px-2 py-1 text-[10px] font-bold rounded border ${
                  metrics?.status === 'completed'
                    ? 'bg-[#00FF41]/20 text-[#00FF41] border-[#00FF41]/40'
                    : 'bg-neutral-500/20 text-neutral-400 border-neutral-500/40'
                }`}
              >
                {metrics?.status === 'completed' ? 'EVAL DONE' : 'NO RUNS'}
              </span>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between text-xs">
                <span className="text-neutral-500">Tasks evaluated</span>
                <span className="text-white font-mono">{metrics?.summary.length ?? 0}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-500">LoRA Adapters</span>
                <span className="text-[#731fff] font-mono">Base (no LoRA loaded)</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-500">Run directory</span>
                <span className="text-white font-mono truncate">{metrics?.run_name ?? '—'}</span>
              </div>
            </div>
          </div>

          <div className="glass-panel rounded-lg flex-1 flex flex-col overflow-hidden border-[#00FF41]/20">
            <div className="border-b border-white/5 px-4 py-2 bg-white/5 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-neutral-500">
              <span>Terminal Logs (decorative)</span>
              <div className="w-2 h-2 rounded-full bg-[#00FF41] animate-pulse" />
            </div>
            <div className="flex-1 p-4 overflow-y-auto space-y-1 font-mono text-[11px] text-neutral-400">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className={`${log.type === 'SYS' ? 'text-blue-400' : log.color || 'text-neutral-500'}`}>
                    [{log.type}]
                  </span>
                  <span className={log.pulse ? 'animate-pulse' : ''}>
                    {log.msg}
                    {log.pulse ? ' _' : ''}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: real curves from training/runs */}
        <div className="w-full md:w-2/3 flex flex-col gap-4 h-full overflow-y-auto pr-1">
          <AgentCard title="Reward Curve" icon={TrendingUp} color="[#00FF41]" active>
            <div className="flex-1 h-full min-h-[220px] relative mt-2 flex items-center justify-center bg-black/30 rounded">
              {metrics?.reward_curve_png ? (
                <img
                  src={metrics.reward_curve_png}
                  alt="Reward curve"
                  className="max-h-[260px] max-w-full object-contain"
                />
              ) : (
                <span className="text-[11px] text-neutral-500 font-mono">
                  No run yet — run <code>python training/eval_rollout.py --mock</code>.
                </span>
              )}
            </div>
          </AgentCard>

          <AgentCard title="Portfolio Curve" icon={BarChart3} color="[#731fff]">
            <div className="flex-1 h-full min-h-[220px] relative mt-2 flex items-center justify-center bg-black/30 rounded">
              {metrics?.portfolio_curve_png ? (
                <img
                  src={metrics.portfolio_curve_png}
                  alt="Portfolio curve"
                  className="max-h-[260px] max-w-full object-contain"
                />
              ) : (
                <span className="text-[11px] text-neutral-500 font-mono">No run yet.</span>
              )}
            </div>
          </AgentCard>

          {metrics?.summary?.length ? (
            <div className="glass-panel rounded-lg p-4">
              <h4 className="text-xs font-bold uppercase tracking-widest text-neutral-500 mb-3">Per-task summary</h4>
              <table className="w-full text-[11px] font-mono">
                <thead className="text-neutral-500 text-left">
                  <tr>
                    <th className="py-1">Task</th>
                    <th className="py-1 text-right">Alpha %</th>
                    <th className="py-1 text-right">Final</th>
                    <th className="py-1 text-right">Buy &amp; Hold</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.summary.map((row) => (
                    <tr key={row.task_id} className="border-t border-white/5">
                      <td className="py-1.5 text-white">{row.task_id}</td>
                      <td
                        className={`py-1.5 text-right ${
                          row.alpha_pct >= 0 ? 'text-[#00FF41]' : 'text-[#ffb4ab]'
                        }`}
                      >
                        {row.alpha_pct.toFixed(2)}%
                      </td>
                      <td className="py-1.5 text-right text-neutral-300">
                        ${row.final_portfolio.toLocaleString()}
                      </td>
                      <td className="py-1.5 text-right text-neutral-400">
                        ${row.buy_and_hold.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------- GalleryView

const TASK_META: Record<string, { difficulty: string; name: string; color: string }> = {
  task_easy: { difficulty: 'Easy', name: 'Steady Regime', color: '[#00FF41]' },
  task_medium: { difficulty: 'Medium', name: 'Choppy Sideways', color: '[#731fff]' },
  task_hard: { difficulty: 'Hard', name: 'Drawdown / Snapback', color: '[#ffb4ab]' },
};

const TaskCard = ({
  taskId,
  bars,
  onDeploy,
  active,
}: {
  taskId: string;
  bars: OhlcvBar[] | null;
  onDeploy: () => void;
  active: boolean;
}) => {
  const meta = TASK_META[taskId] ?? { difficulty: '?', name: taskId, color: '[#00FF41]' };
  const ticker = bars?.[0] ? '' : '';
  const inEpisode = bars?.filter((b) => b.in_episode) ?? [];
  const dates =
    inEpisode.length > 0
      ? `${inEpisode[0].time} → ${inEpisode[inEpisode.length - 1].time}`
      : 'Loading…';

  // Compute a 9-bucket sparkline from in-episode closes
  let sparkBars: number[] = [];
  if (inEpisode.length > 0) {
    const closes = inEpisode.map((b) => b.close);
    const min = Math.min(...closes);
    const max = Math.max(...closes);
    const range = max - min || 1;
    const buckets = 9;
    const step = closes.length / buckets;
    sparkBars = Array.from({ length: buckets }, (_, i) => {
      const slice = closes.slice(Math.floor(i * step), Math.floor((i + 1) * step) || (i + 1));
      const avg = slice.reduce((a, c) => a + c, 0) / Math.max(1, slice.length);
      return ((avg - min) / range) * 100;
    });
  }

  return (
    <div
      className={`group glass-panel rounded-xl p-6 flex flex-col relative overflow-hidden transition-all hover:bg-white/[0.08] hover:shadow-[0_0_30px_rgba(0,255,65,0.05)] ${
        active ? `border-${meta.color}/50` : 'border-white/5'
      } hover:border-${meta.color}/50`}
    >
      <div
        className={`absolute inset-0 bg-gradient-to-br from-${meta.color}/5 to-transparent pointer-events-none group-hover:from-${meta.color}/15 transition-all`}
      />

      <div className="flex justify-between items-start mb-4 relative z-10">
        <div
          className={`px-2 py-1 bg-${meta.color}/10 border border-${meta.color}/30 text-${meta.color} text-[10px] font-bold uppercase tracking-widest rounded flex items-center gap-1.5`}
        >
          <div className={`w-1.5 h-1.5 rounded-full bg-${meta.color}`} />
          {meta.difficulty}
        </div>
        <span className="text-lg font-bold text-white font-mono">{bars?.[0] ? bars[0].time.slice(0, 4) : ''}</span>
      </div>

      <div className="mb-6 relative z-10">
        <h3 className="text-xl font-bold text-white mb-1">{meta.name}</h3>
        <div className="flex items-center gap-2 text-[10px] text-neutral-500 font-mono">
          <Monitor className="w-3 h-3" />
          {dates}
        </div>
      </div>

      <div className="h-24 w-full bg-black/40 rounded-lg border border-white/5 mb-8 flex items-end px-2 pb-2 gap-1 overflow-hidden relative z-10">
        <div className="absolute inset-x-0 bottom-0 h-[1px] bg-white/5" />
        {sparkBars.length > 0 ? (
          sparkBars.map((h, i) => (
            <motion.div
              key={i}
              initial={{ height: 0 }}
              animate={{ height: `${h}%` }}
              className={`flex-1 rounded-t-[2px] transition-all bg-${meta.color}/40 group-hover:bg-${meta.color}/70`}
            />
          ))
        ) : (
          <div className="w-full text-center text-[10px] text-neutral-600 font-mono">Loading bars…</div>
        )}
      </div>

      <button
        onClick={onDeploy}
        className={`w-full py-3 rounded-lg border border-white/10 flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-widest transition-all relative z-10 hover:text-${meta.color} hover:border-${meta.color}/50 hover:bg-${meta.color}/5 group/btn`}
      >
        <Zap className="w-4 h-4 group-hover/btn:fill-current" />
        {active ? 'Active' : 'Deploy Agent'}
      </button>
    </div>
  );
};

const GalleryView = ({
  env,
  onDeploy,
}: {
  env: StockerEnv;
  onDeploy: (taskId: string) => void;
}) => {
  const [bars, setBars] = useState<Record<string, OhlcvResponse>>({});

  useEffect(() => {
    let cancel = false;
    Promise.all(env.tasks.map((t) => api.ohlcv(t).then((r) => [t, r] as const))).then((pairs) => {
      if (cancel) return;
      const next: Record<string, OhlcvResponse> = {};
      for (const [t, r] of pairs) next[t] = r;
      setBars(next);
    });
    return () => {
      cancel = true;
    };
  }, [env.tasks]);

  return (
    <div className="flex-1 flex flex-col gap-8">
      <header className="space-y-3">
        <div className="flex items-center gap-2 text-[#00FF41]">
          <Brain className="w-5 h-5" />
          <span className="text-[10px] uppercase font-bold tracking-[0.2em]">Agent Environment</span>
        </div>
        <h1 className="text-5xl font-black text-white tracking-tighter">Task Gallery</h1>
        <p className="text-sm text-neutral-400 max-w-2xl leading-relaxed">
          Select a historical market regime to deploy and evaluate your AI trading agent's performance in isolated, controlled conditions.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {env.tasks.map((t) => (
          <TaskCard
            key={t}
            taskId={t}
            bars={bars[t]?.bars ?? null}
            active={env.taskId === t}
            onDeploy={() => onDeploy(t)}
          />
        ))}
      </div>
    </div>
  );
};

// --------------------------------------------------------------- PortfolioView

const PortfolioView = ({ env }: { env: StockerEnv }) => {
  const { observation, envState } = env;

  if (!observation || !envState) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-neutral-500">Run a task in Gallery to populate your portfolio.</p>
      </div>
    );
  }

  const ticker = observation.ticker;
  const price = observation.price;
  const balance = observation.position;
  const value = balance * price;
  const portfolio = envState.portfolio_value;
  const cash = envState.cash;
  const startingCash = 10000;
  const pnl = portfolio - startingCash;
  const pnlPct = (pnl / startingCash) * 100;

  const equityShare = portfolio > 0 ? (value / portfolio) * 100 : 0;
  const cashShare = portfolio > 0 ? (cash / portfolio) * 100 : 0;
  const reservedShare = Math.max(0, 100 - equityShare - cashShare);

  // 24h change from the last two prices in price_history
  const ph = observation.price_history;
  const change24h =
    ph.length >= 2 ? ((ph[ph.length - 1] - ph[ph.length - 2]) / ph[ph.length - 2]) * 100 : 0;

  // Sparkline = last 7 closes
  const spark = ph.slice(-7);
  const sMin = Math.min(...spark);
  const sMax = Math.max(...spark);
  const sRange = sMax - sMin || 1;
  const sparkPct = spark.map((p) => ((p - sMin) / sRange) * 100);

  // Risk profile = action diversity
  const actions = envState.action_history;
  const totalActions = actions.length || 1;
  const tradeCount = actions.filter((a) => a.side !== 'hold').length;
  const riskScore = Math.min(100, Math.round((tradeCount / totalActions) * 100));
  const riskLabel =
    riskScore < 30 ? 'Conservative' : riskScore < 60 ? 'Moderate' : 'Aggressive';

  return (
    <div className="flex-1 flex flex-col gap-6">
      <header className="flex justify-between items-end">
        <div className="space-y-1">
          <span className="text-[10px] font-bold text-[#00FF41] uppercase tracking-widest">Active Position</span>
          <h1 className="text-4xl font-bold text-white tracking-tight">Holdings &amp; P&amp;L</h1>
        </div>
        <div className="flex gap-4">
          <div className="glass-panel p-3 px-5 rounded-lg border-l-2 border-[#00FF41]">
            <span className="text-[9px] text-neutral-500 uppercase font-bold block mb-1">Total Balance</span>
            <span className="text-xl font-bold text-white font-mono tracking-tight">
              ${portfolio.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
          </div>
          <div
            className={`glass-panel p-3 px-5 rounded-lg border-l-2 ${
              pnl >= 0 ? 'border-[#00FF41]' : 'border-[#ffb4ab]'
            }`}
          >
            <span className="text-[9px] text-neutral-500 uppercase font-bold block mb-1">Total P&amp;L</span>
            <span
              className={`text-xl font-bold font-mono tracking-tight ${
                pnl >= 0 ? 'text-[#00FF41]' : 'text-[#ffb4ab]'
              }`}
            >
              {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)} ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
            </span>
          </div>
        </div>
      </header>

      <div className="glass-panel rounded-xl overflow-hidden border-white/5">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-white/5 border-b border-white/10">
              <th className="px-6 py-4 text-[10px] font-bold text-neutral-500 uppercase tracking-widest">Asset</th>
              <th className="px-6 py-4 text-[10px] font-bold text-neutral-500 uppercase tracking-widest text-right">Price</th>
              <th className="px-6 py-4 text-[10px] font-bold text-neutral-500 uppercase tracking-widest text-right">24h Change</th>
              <th className="px-6 py-4 text-[10px] font-bold text-neutral-500 uppercase tracking-widest text-right">Balance</th>
              <th className="px-6 py-4 text-[10px] font-bold text-neutral-500 uppercase tracking-widest text-right">Value</th>
              <th className="px-6 py-4 text-[10px] font-bold text-neutral-500 uppercase tracking-widest text-center">7D Performance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 font-mono text-sm">
            <tr className="hover:bg-white/[0.04] transition-colors group">
              <td className="px-6 py-5">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded bg-white/5 border border-white/10 flex items-center justify-center font-black group-hover:text-[#00FF41] transition-colors">
                    {ticker[0]}
                  </div>
                  <div className="flex flex-col">
                    <span className="text-white font-bold">{ticker}</span>
                    <span className="text-[10px] text-neutral-500 font-sans tracking-tight">NASDAQ</span>
                  </div>
                </div>
              </td>
              <td className="px-6 py-5 text-right font-bold text-neutral-300">${price.toFixed(2)}</td>
              <td
                className={`px-6 py-5 text-right font-bold ${
                  change24h >= 0 ? 'text-[#00FF41]' : 'text-[#ffb4ab]'
                }`}
              >
                {change24h >= 0 ? '+' : ''}
                {change24h.toFixed(2)}%
              </td>
              <td className="px-6 py-5 text-right text-neutral-400">{balance} {ticker}</td>
              <td className="px-6 py-5 text-right text-white font-bold">
                ${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </td>
              <td className="px-6 py-5">
                <div className="h-6 w-24 mx-auto flex items-end gap-1 px-1">
                  {sparkPct.map((v, j) => (
                    <div
                      key={j}
                      className="flex-1 rounded-t-[1px] opacity-60"
                      style={{
                        height: `${v}%`,
                        backgroundColor: change24h >= 0 ? '#00FF41' : '#ffb4ab',
                      }}
                    />
                  ))}
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel rounded-xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <BarChart3 className="w-12 h-12" />
          </div>
          <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-4">Risk Profile</h4>
          <div className="flex items-center gap-4">
            <div
              className="w-16 h-16 rounded-full border-4 border-white/5 border-t-[#00FF41] relative"
              style={{ transform: `rotate(${(riskScore / 100) * 360 - 90}deg)` }}
            >
              <div
                className="absolute inset-0 flex items-center justify-center"
                style={{ transform: `rotate(${-((riskScore / 100) * 360 - 90)}deg)` }}
              >
                <span className="text-sm font-bold text-white">{riskScore}/100</span>
              </div>
            </div>
            <div>
              <p className="text-sm font-bold text-white mb-1 tracking-tight">{riskLabel}</p>
              <p className="text-[10px] text-neutral-400 leading-snug">
                {tradeCount} of {totalActions} steps were trades.
              </p>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-xl p-6 col-span-2 relative overflow-hidden group">
          <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-widest mb-4">Trade Allocation</h4>
          <div className="flex items-end h-16 gap-1.5 px-1 py-1">
            {[
              { l: 'Equities', v: equityShare, c: '#00FF41' },
              { l: 'Cash', v: cashShare, c: '#731fff' },
              { l: 'Reserved', v: reservedShare, c: 'rgba(255,255,255,0.1)' },
            ].map((a, i) => (
              <div key={i} className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-current opacity-30 rounded-t-sm"
                  style={{ height: `${a.v}%`, color: a.c }}
                />
                <span className="text-[8px] font-bold text-neutral-600 mt-1 uppercase tracking-tighter">
                  {a.l} ({a.v.toFixed(0)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ----------------------------------------------------------- IntelligenceView

const SENTIMENT_SCORE: Record<string, number> = {
  positive: 1,
  bullish: 1,
  neutral: 0,
  negative: -1,
  bearish: -1,
};

const IntelligenceView = ({ env }: { env: StockerEnv }) => {
  const { observation } = env;

  const positiveHeadlines = (observation?.headlines ?? [])
    .filter((h) => (SENTIMENT_SCORE[h.sentiment_label?.toLowerCase()] ?? 0) > 0);
  const negativeHeadlines = (observation?.headlines ?? [])
    .filter((h) => (SENTIMENT_SCORE[h.sentiment_label?.toLowerCase()] ?? 0) < 0);

  const peerEntries = observation?.peers?.peers ?? [];
  const macroEntries = observation?.macro ?? [];

  const signals = [
    positiveHeadlines[0] && {
      t: 'Top Headline',
      msg: positiveHeadlines[0].headline,
      time: positiveHeadlines[0].date,
      color: '#00FF41',
    },
    positiveHeadlines[1] && {
      t: 'Secondary',
      msg: positiveHeadlines[1].headline,
      time: positiveHeadlines[1].date,
      color: '#00FF41',
    },
    negativeHeadlines[0] && {
      t: 'Risk Note',
      msg: negativeHeadlines[0].headline,
      time: negativeHeadlines[0].date,
      color: '#ffb4ab',
    },
    peerEntries.length > 0 && {
      t: 'Peer Watch',
      msg: `${observation?.ticker} vs ${peerEntries
        .slice(0, 3)
        .map((p) => `${p.peer_ticker}=${p.peer_close?.toFixed(2) ?? '?'}`)
        .join(', ')}`,
      time: observation?.date ?? '',
      color: '#731fff',
    },
    macroEntries[0] && {
      t: 'Macro',
      msg: `${macroEntries[0].country} — ${macroEntries[0].headline}`,
      time: macroEntries[0].date,
      color: '#731fff',
    },
  ].filter(Boolean) as Array<{ t: string; msg: string; time: string; color: string }>;

  // Sentiment hub: mean of headline sentiment scores
  const headlines = observation?.headlines ?? [];
  const meanTone = headlines.length
    ? headlines.reduce((a, h) => a + (SENTIMENT_SCORE[h.sentiment_label?.toLowerCase()] ?? 0), 0) /
      headlines.length
    : 0;
  const toneScale = Math.round(((meanTone + 1) / 2) * 100); // 0..100
  const sentimentLabel =
    meanTone > 0.2 ? 'BULL' : meanTone < -0.2 ? 'BEAR' : 'NEUTRAL';
  const peerBeta = peerEntries.length ? '~1.00' : '—';

  return (
    <div className="flex-1 flex flex-col gap-6">
      <header className="space-y-1">
        <h1 className="text-4xl font-bold text-white tracking-tight flex items-center gap-3">
          Intelligence Nexus
          <div className="flex gap-1">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="w-1 h-4 bg-[#00FF41] animate-pulse"
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        </h1>
        <p className="text-sm text-neutral-400">
          Aggregating headlines, peer rotation, and macro signals from the current observation.
        </p>
      </header>

      <div className="flex-1 flex gap-6 min-h-0">
        {/* Neural map */}
        <div className="flex-1 glass-panel rounded-2xl relative overflow-hidden flex items-center justify-center bg-black/60 p-12">
          <div className="absolute inset-0 bg-aurora opacity-30 pointer-events-none" />
          <div className="absolute inset-0 scanline opacity-20 pointer-events-none" />

          <svg className="w-full h-full opacity-60" viewBox="0 0 400 300">
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <g stroke="white" strokeWidth="0.5" strokeOpacity="0.1">
              <line x1="200" y1="150" x2="100" y2="80" />
              <line x1="200" y1="150" x2="300" y2="80" />
              <line x1="200" y1="150" x2="100" y2="220" />
              <line x1="200" y1="150" x2="300" y2="220" />
              <line x1="100" y1="80" x2="50" y2="150" />
              <line x1="100" y1="220" x2="50" y2="150" />
              <line x1="300" y1="80" x2="350" y2="150" />
              <line x1="300" y1="220" x2="350" y2="150" />
            </g>
            <circle cx="200" cy="150" r="10" fill="#731fff" filter="url(#glow)">
              <animate attributeName="r" values="10;12;10" dur="3s" repeatCount="indefinite" />
            </circle>
            {[
              { x: 100, y: 80, c: '#00FF41' },
              { x: 300, y: 80, c: '#00FF41' },
              { x: 100, y: 220, c: '#ffb4ab' },
              { x: 300, y: 220, c: '#00FF41' },
              { x: 50, y: 150, c: '#731fff' },
              { x: 350, y: 150, c: '#731fff' },
            ].map((n, i) => (
              <g key={i}>
                <circle cx={n.x} cy={n.y} r="4" fill={n.c} filter="url(#glow)">
                  <animate attributeName="opacity" values="0.4;1;0.4" dur={`${2 + i}s`} repeatCount="indefinite" />
                </circle>
                <circle cx={n.x} cy={n.y} r="8" fill="none" stroke={n.c} strokeWidth="0.5" strokeOpacity="0.3">
                  <animate attributeName="r" values="8;15;8" dur={`${4 + i}s`} repeatCount="indefinite" />
                </circle>
              </g>
            ))}
          </svg>

          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <Brain className="w-16 h-16 text-[#731fff] opacity-20" />
            <span className="font-mono text-[10px] text-[#731fff] tracking-[0.5em] mt-4 uppercase">
              Neural Consensus Active
            </span>
          </div>
        </div>

        {/* Alpha signals — derived from observation */}
        <div className="w-96 flex flex-col gap-4 overflow-hidden shrink-0">
          <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-[0.2em] px-2 flex items-center justify-between">
            Alpha Signals
            <Activity className="w-3 h-3 text-[#00FF41]" />
          </h3>
          <div className="flex-1 overflow-y-auto space-y-4 pr-2">
            {signals.length > 0 ? (
              signals.map((s, i) => (
                <div
                  key={i}
                  className="glass-panel p-4 rounded-xl border-l-2 hover:bg-white/[0.04] transition-all cursor-default group"
                  style={{ borderColor: s.color }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold uppercase tracking-widest" style={{ color: s.color }}>
                      {s.t}
                    </span>
                    <span className="text-[10px] text-neutral-500">{s.time}</span>
                  </div>
                  <p className="text-xs text-neutral-400 leading-relaxed font-mono group-hover:text-neutral-300 transition-colors">
                    {s.msg}
                  </p>
                </div>
              ))
            ) : (
              <div className="text-[11px] text-neutral-500 font-mono px-2">
                {observation ? 'No headlines/peers/macro for this step.' : 'Run a task to populate signals.'}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="h-24 glass-panel rounded-xl flex items-center px-8 border-[#00FF41]/10 bg-[#00FF41]/[0.02]">
        <div className="flex-1 flex flex-col gap-1">
          <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-widest">Sentiment Hub</span>
          <div className="flex items-center gap-4">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-white">{toneScale}</span>
              <span
                className={`text-xs font-bold uppercase tracking-widest ${
                  sentimentLabel === 'BULL'
                    ? 'text-[#00FF41]'
                    : sentimentLabel === 'BEAR'
                    ? 'text-[#ffb4ab]'
                    : 'text-neutral-400'
                }`}
              >
                {sentimentLabel}
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-12">
          {[
            { l: 'News Tone', v: meanTone.toFixed(2) },
            { l: 'Sentiment', v: sentimentLabel },
            { l: 'Peer Beta', v: peerBeta },
          ].map((m, i) => (
            <div key={i} className="flex flex-col items-center">
              <span className="text-[9px] font-bold text-neutral-600 uppercase mb-1">{m.l}</span>
              <span className="text-xs font-mono font-bold text-neutral-300">{m.v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ------------------------------------------------------------------------- App

type Tab = 'Terminal' | 'Council' | 'Training' | 'Gallery' | 'Portfolio' | 'Intelligence';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('Terminal');
  const env = useStockerEnv();

  const onDeploy = (taskId: string) => {
    void env.selectTask(taskId);
    setActiveTab('Terminal');
  };

  const renderView = () => {
    switch (activeTab) {
      case 'Terminal':
        return <TerminalView env={env} />;
      case 'Council':
        return <CouncilView env={env} />;
      case 'Training':
        return <TrainingView />;
      case 'Gallery':
        return <GalleryView env={env} onDeploy={onDeploy} />;
      case 'Portfolio':
        return <PortfolioView env={env} />;
      case 'Intelligence':
        return <IntelligenceView env={env} />;
      default:
        return <TerminalView env={env} />;
    }
  };

  return (
    <div className="flex min-h-screen bg-[#050505] text-[#e5e2e1] font-sans selection:bg-[#00FF41]/20">
      <div className="bg-aurora fixed inset-0 pointer-events-none z-0" />
      <div className="fixed inset-0 pointer-events-none z-50 opacity-[0.03] scanline" />

      {/* Sidebar */}
      <nav className="w-64 fixed left-0 top-0 h-screen bg-black/40 backdrop-blur-3xl border-r border-white/5 flex flex-col pt-20 pb-8 z-40 transition-all">
        <div className="px-6 mb-8 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center relative overflow-hidden group">
            <Cpu className="w-6 h-6 text-[#00FF41] opacity-70 group-hover:scale-110 transition-transform" />
            <div className="absolute inset-0 bg-gradient-to-br from-[#00FF41]/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <div>
            <h2 className="font-bold text-sm tracking-widest text-[#e5e2e1]">ALPHA ENGINE</h2>
            <p className="text-[10px] text-[#00FF41] font-mono tracking-widest mt-0.5 uppercase">
              {env.taskId ?? 'Booting…'}
            </p>
          </div>
        </div>

        <div className="flex-1 px-3 space-y-1">
          <SidebarItem icon={Monitor} label="Terminal" active={activeTab === 'Terminal'} onClick={() => setActiveTab('Terminal')} />
          <SidebarItem icon={Users} label="Council" active={activeTab === 'Council'} onClick={() => setActiveTab('Council')} />
          <SidebarItem icon={TrendingUp} label="Training" active={activeTab === 'Training'} onClick={() => setActiveTab('Training')} />
          <SidebarItem icon={Library} label="Gallery" active={activeTab === 'Gallery'} onClick={() => setActiveTab('Gallery')} />
          <SidebarItem icon={Wallet} label="Portfolio" active={activeTab === 'Portfolio'} onClick={() => setActiveTab('Portfolio')} />
          <SidebarItem icon={Brain} label="Intelligence" active={activeTab === 'Intelligence'} onClick={() => setActiveTab('Intelligence')} />
        </div>

        <div className="px-3 mt-auto space-y-1 border-t border-white/5 pt-4">
          <button className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-neutral-500 hover:bg-white/5 hover:text-[#00FF41] transition-all">
            <FileText className="w-4 h-4" />
            <span className="text-xs font-medium">Docs</span>
          </button>
          <button className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-neutral-500 hover:bg-white/5 hover:text-[#00FF41] transition-all">
            <LifeBuoy className="w-4 h-4" />
            <span className="text-xs font-medium">Support</span>
          </button>
        </div>

        <div className="px-6 mt-6">
          <div
            className={`p-3 rounded-lg border flex items-center gap-3 ${
              env.error
                ? 'bg-[#ffb4ab]/5 border-[#ffb4ab]/20'
                : env.loading
                ? 'bg-[#731fff]/5 border-[#731fff]/20'
                : 'bg-[#00FF41]/5 border-[#00FF41]/20'
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full animate-pulse ${
                env.error ? 'bg-[#ffb4ab]' : env.loading ? 'bg-[#731fff]' : 'bg-[#00FF41]'
              }`}
            />
            <div className="flex flex-col">
              <span
                className={`text-[10px] font-bold uppercase tracking-widest ${
                  env.error ? 'text-[#ffb4ab]' : env.loading ? 'text-[#731fff]' : 'text-[#00FF41]'
                }`}
              >
                {env.error ? 'API Error' : env.loading ? 'Working…' : 'System Ready'}
              </span>
              <span className="text-[8px] text-neutral-500 font-mono truncate max-w-[160px]">
                {env.error ?? `task: ${env.taskId ?? '—'}`}
              </span>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 ml-64 flex flex-col relative z-10">
        <header className="h-14 fixed top-0 right-0 left-64 bg-black/40 backdrop-blur-xl border-b border-white/5 px-6 flex items-center justify-between z-50">
          <div className="flex items-center gap-4">
            <span className="text-xl font-black tracking-widest text-[#00FF41] drop-shadow-[0_0_8px_rgba(0,255,65,0.4)]">STOCKER AI</span>
          </div>

          <div className="flex items-center gap-6">
            <div className="relative group hidden md:block">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500 group-focus-within:text-[#00FF41] transition-colors" />
              <input
                type="text"
                placeholder="Search parameters..."
                className="bg-white/5 border border-white/10 rounded-full py-1.5 pl-10 pr-4 text-xs font-mono text-[#e5e2e1] focus:outline-none focus:border-[#00FF41]/50 w-64 transition-all"
              />
            </div>

            <div className="flex items-center gap-4">
              <button className="text-neutral-500 hover:text-white transition-colors relative">
                <Bell className="w-5 h-5" />
                <span className="absolute -top-1 -right-1 w-2 h-2 bg-[#00FF41] rounded-full" />
              </button>
              <button className="text-neutral-500 hover:text-white transition-colors">
                <CreditCard className="w-5 h-5" />
              </button>
              <button className="text-neutral-500 hover:text-white transition-colors">
                <User className="w-5 h-5" />
              </button>
            </div>
          </div>
        </header>

        <main className="pt-14 p-6 flex-1 flex flex-col min-h-screen">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="flex-1 flex flex-col"
            >
              {renderView()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
