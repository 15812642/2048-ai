/* ============================================================
   2048 AI 训练控制台 · 前端逻辑
   - 算法自适应：v1(DQN) / v2(Rainbow) / v3(N-Tuple) 显示不同指标与图表
   - 并行实验对比：卡片 + 分数曲线对比
   - 每 2s 轮询状态刷新，10s 刷新实验对比
   ============================================================ */
'use strict';

const $ = (id) => document.getElementById(id);
let currentAlgo = 'v1';
const q = (url) => url + (url.indexOf('?') >= 0 ? '&' : '?') + 'algo=' + currentAlgo;

const fmt = (n, d = 0) => (n === null || n === undefined || isNaN(n)) ? '–' : Number(n).toFixed(d);
const int = (n) => (n === null || n === undefined) ? '–' : Number(n).toLocaleString();

/** 滑动平均（前段用滚动均值填充，保证与原始序列等长） */
function MA(arr, w) {
  if (!arr || !arr.length) return [];
  w = Math.min(w, arr.length);
  const out = []; let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
    if (i >= w) sum -= arr[i - w];
    out.push(sum / Math.min(i + 1, w));
  }
  return out;
}
const TILE_CLASS = (v) => !v ? 'cell' : (v <= 2048 ? 'cell v' + v : 'cell vhigh');

/* ============================================================
   算法自适应配置表
   每个算法: 指标卡标签 / 图表标题 / 数据集定义
   ============================================================ */
const ALGO_META = {
  v1: {
    name: 'DQN v1', tag: '经典 DQN + MCTS', brand: 'DQN · 16维log2状态 · MCTS搜索增强',
    avgLabel: '近 50 局均分',
    third: { label: '探索率 ε', from: 'epsilon', fmt: (s) => fmt(s.epsilon, 3),
             sub: (s) => `Buffer ${int(s.buffer_size)}` },
    charts: {
      score: { title: '分数曲线', sub: '原始 + 50 局滑动平均' },
      second: { title: 'TD Loss', sub: '滑动平均 · 对数轴', log: true,
                series: (s) => ({ data: MA(s.losses_tail || [], 500), color: '#f67c5f' }),
                yLabel: 'MSE loss' },
      third: { title: 'ε 探索率衰减', series: (s) => ({ data: s.epsilons_tail || [], color: '#3ecf8e' }),
               yLabel: 'epsilon', yMin: 0, yMax: 1 },
    },
  },
  v2: {
    name: 'Rainbow v2', tag: 'Dueling-C51 + PER + NoisyNet',
    brand: 'Rainbow-Lite · Conv + C51(51) + n-step + PER',
    avgLabel: '近 50 局均分',
    third: { label: 'PER β / Buffer', from: 'per_beta',
             fmt: (s) => s.buffer_size ? int(s.buffer_size) : '–',
             sub: (s) => `PER β ${s.per_beta != null ? Number(s.per_beta).toFixed(2) : '–'}` },
    charts: {
      score: { title: '分数曲线', sub: '原始 + 50 局滑动平均' },
      second: { title: 'C51 损失', sub: '滑动平均 · 对数轴', log: true,
                series: (s) => ({ data: MA(s.losses_tail || [], 500), color: '#f67c5f' }),
                yLabel: 'loss' },
      third: { title: 'ε 探索率衰减', series: (s) => ({ data: s.epsilons_tail || [], color: '#3ecf8e' }),
               yLabel: 'epsilon', yMin: 0, yMax: 1 },
    },
  },
  v3: {
    name: 'N-Tuple v3', tag: 'N-Tuple + TD(λ) + Expectimax',
    brand: 'N-Tuple SOTA · 6-元组×8对称 · TD(λ) · 期望搜索',
    avgLabel: '近 200 局均分',
    third: { label: '网络规模', from: 'net_weights',
             fmt: (s) => s.net_weights ? `${(s.net_weights / 1e6).toFixed(0)}M` : '–',
             sub: (s) => s.net_weights
                 ? `${s.n_patterns} 图案 × ${s.tuple_len} 元组 · ${fmt(s.net_size_mb, 0)}MB`
                 : '权重 –' },
    charts: {
      score: { title: '分数曲线', sub: '原始 + 200 局滑动平均 · MA窗口更大' },
      second: { title: 'TD 误差收敛 |δ|', sub: '滑动平均 · 反映价值估计稳定性',
                log: false,
                series: (s) => ({ data: MA(s.losses_tail || [], 300), color: '#9e86c8' }),
                yLabel: '|δ|' },
      third: { title: '学习曲线（滑动平均对比）', sub: 'MA-200 vs MA-50',
               series: (s) => null },   // 单独处理
    },
  },
};

/* ---------------- 图表 ---------------- */
const CHART_OPTS = (yTitle, logScale = false) => ({
  responsive: true, maintainAspectRatio: false, animation: { duration: 180 },
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: '#8b95a5', boxWidth: 12, font: { size: 11 } } },
    tooltip: {
      backgroundColor: '#1b2028', borderColor: 'rgba(255,255,255,.12)', borderWidth: 1,
      callbacks: {
        label: (c) => `${c.dataset.label}: ${Number(c.parsed.y).toLocaleString(undefined, { maximumFractionDigits: 1 })}`
      }
    }
  },
  scales: {
    x: { ticks: { color: '#5d6675', maxTicksLimit: 8, font: { size: 10 } },
         grid: { color: 'rgba(255,255,255,.045)' } },
    y: { title: { display: true, text: yTitle, color: '#5d6675', font: { size: 10 } },
         ticks: { color: '#5d6675', font: { size: 10 },
                  callback: (v) => Number(v).toLocaleString() },
         grid: { color: 'rgba(255,255,255,.045)' },
         type: logScale ? 'logarithmic' : 'linear' }
  }
});

function mkChart(canvasId, datasets, yTitle, logScale = false) {
  return new Chart($(canvasId), {
    type: 'line', data: { labels: [], datasets }, options: CHART_OPTS(yTitle, logScale)
  });
}
const DS = (label, color, width = 1.6, fill = false) => ({
  label, data: [], borderColor: color, backgroundColor: color + '22',
  borderWidth: width, pointRadius: 0, pointHoverRadius: 3, tension: .28, fill
});

const charts = {};
const EXP_COLORS = ['#edc22e', '#58a6ff', '#3ecf8e', '#f67c5f', '#9e86c8', '#f2b179'];

function initCharts() {
  charts.score = mkChart('chartScore', [DS('raw', '#58a6ff', 1), DS('MA', '#edc22e', 2)], 'score');
  charts.loss = mkChart('chartLoss', [DS('value MA', '#f67c5f', 1.6, true)], 'loss', true);
  charts.eps = mkChart('chartEps', [DS('epsilon', '#3ecf8e', 2, true)], 'epsilon');
  charts.eps.options.scales.y.min = 0;
  charts.eps.options.scales.y.max = 1;
  charts.milestone = mkChart('chartMilestone', [
    DS('512', '#9e86c8', 1.8), DS('1024', '#58a6ff', 1.8),
    DS('2048', '#edc22e', 1.8), DS('4096', '#f67c5f', 1.8)
  ], 'ratio');
  charts.milestone.options.scales.y.min = 0;
  charts.milestone.options.scales.y.max = 1;
  charts.milestone.options.scales.y.ticks.callback = (v) => (v * 100).toFixed(0) + '%';

  charts.compare = mkChart('chartCompare', [], 'score');
  charts.compare.options.plugins.legend.labels.boxWidth = 14;
}

/* ---------------- 算法切换 ---------------- */
function applyAlgoUI(algo) {
  const m = ALGO_META[algo] || ALGO_META.v1;

  $('brandSub').textContent = m.brand;
  $('ctlAlgoTag').textContent = m.name;
  $('lbAvg').textContent = m.avgLabel;
  $('lbThird').textContent = m.third.label;

  // v1/v2 用 ctlV1，v3 用 ctlV3
  $('ctlV1').style.display = algo === 'v3' ? 'none' : 'flex';
  $('ctlV3').style.display = algo === 'v3' ? 'flex' : 'none';

  // 图表标题
  $('ttlScore').innerHTML = `${m.charts.score.title} <small id="subScore">${m.charts.score.sub}</small>`;
  $('ttlSecond').innerHTML = `${m.charts.second.title} <small id="subSecond">${m.charts.second.sub}</small>`;
  $('ttlThird').innerHTML = m.charts.third.title;

  // 第二张图: 重建数据集（v1/v2 用对数轴，v3 线性）
  rebuildChart(charts.loss, 'chartLoss',
    [DS(m.charts.second.yLabel + ' MA', m.charts.second.series({}).color, 1.6, true)],
    m.charts.second.yLabel, m.charts.second.log);

  // 第三张图: v3 特殊处理
  if (algo === 'v3') {
    rebuildChart(charts.eps, 'chartEps',
      [DS('MA-200', '#edc22e', 2, false), DS('MA-50', '#3ecf8e', 1.4, false)],
      'score', false);
  } else {
    rebuildChart(charts.eps, 'chartEps', [DS('epsilon', '#3ecf8e', 2, true)],
      'epsilon', false);
    charts.eps.options.scales.y.min = 0;
    charts.eps.options.scales.y.max = 1;
  }

  // 演示页的第三项参数含义不同
  $('lbSims').firstChild.textContent = algo === 'v3' ? '搜索深度 ' : '搜索/模拟 ';
  updateSimsLabel();
}

function rebuildChart(chart, canvasId, datasets, yTitle, logScale) {
  chart.data.datasets = datasets;
  chart.data.labels = [];
  chart.options.scales.y.title.text = yTitle;
  chart.options.scales.y.type = logScale ? 'logarithmic' : 'linear';
  if (!logScale) { delete chart.options.scales.y.min; delete chart.options.scales.y.max; }
  chart.update('none');
}

/* ---------------- 状态刷新 ---------------- */
let lastStatus = null;

async function refreshStatus() {
  let s;
  try {
    s = await (await fetch(q('/api/status'))).json();
  } catch (e) { setBadge(false, '服务离线'); return; }
  lastStatus = s;

  const running = !!s.proc_running;
  setBadge(running, running ? '训练中' : '空闲');

  // 空数据状态：该算法还没有训练记录
  const hasData = (s.episode || 0) > 0;
  const empty = $('emptyHint');
  if (empty) {
    empty.style.display = hasData ? 'none' : '';
    if (!hasData) {
      $('emptyHintText').textContent =
        `${(ALGO_META[currentAlgo] || {}).name || currentAlgo} 暂无训练数据`
        + `（可在下方「训练控制」中启动，或切换到 v3 查看并行实验）`;
    }
  }

  const meta = ALGO_META[currentAlgo] || ALGO_META.v1;
  const scores = s.recent_scores || [];
  const cfg = s.config || {};

  // ---- 指标卡 ----
  const ep = s.episode || 0;
  $('stEpisode').textContent = int(ep);
  const target = cfg.episodes || 0;
  const rate = s.episodes_per_sec;
  const eta = s.eta_sec;
  const etaStr = !eta ? null
    : (eta >= 86400 ? '> 1 天'
       : eta > 3600 ? (eta / 3600).toFixed(1) + ' 小时'
       : (eta / 60).toFixed(0) + ' 分钟');
  const fromExp = s._fallback_from_exp && s._exp_name ? ` · 来源 ${s._exp_name}` : '';
  $('stEpisodeSub').textContent = (target
      ? `目标 ${int(target)} 局 · ${(ep / target * 100).toFixed(2)}%`
      : '目标 –')
    + (rate ? ` · ${rate} 局/秒` : '')
    + (etaStr ? ` · 剩余 ${etaStr}` : '')
    + fromExp;
  $('stEpisodeSub').title = s._fallback_from_exp
    ? '主目录无训练数据，当前展示并行实验中进度最靠前的一个' : '';
  const pb = $('pbEpisode');
  pb.style.width = target ? Math.min(100, ep / target * 100).toFixed(1) + '%' : '0%';

  const w = currentAlgo === 'v3' ? 200 : 50;
  $('stAvg').textContent = scores.length ? fmt(MA(scores, w).slice(-1)[0], 0) : '–';
  $('stSteps').textContent = `总步数 ${int(s.total_steps)}`;

  $('stThird').textContent = meta.third.fmt(s);
  $('stThirdSub').textContent = meta.third.sub(s);

  $('stBest').textContent = s.best_eval_score != null ? int(Math.round(s.best_eval_score)) : '–';
  $('stVersion').textContent = currentAlgo === 'v3'
    ? `${s.n_patterns || '–'} 图案 × ${s.tuple_len || '–'} 元组`
    : `自我对弈版本 v${s.selfplay_version || 0}`;

  // 数据新鲜度（来自 status.json 的 updated_at）
  const fresh = $('pillFresh');
  if (fresh) {
    if (s.updated_at) {
      const t = new Date(s.updated_at.replace(/-/g, '/')).getTime();
      const age = Math.max(0, Math.round((Date.now() - t) / 1000));
      fresh.textContent = age <= 5 ? '● 实时更新' : `● ${age}s 前`;
      fresh.style.color = age <= 5 ? 'var(--green)'
                        : (age <= 30 ? 'var(--gold)' : 'var(--text-faint)');
    } else { fresh.textContent = '● 无数据'; }
  }

  $('pillAlgo').textContent = meta.name;
  $('pillDevice').textContent = `设备 ${s.device || '–'}`;
  $('pillBackend').textContent = currentAlgo === 'v3'
    ? `TD(λ) · ${int(s.net_weights)} 权重`
    : (currentAlgo === 'v2'
       ? 'C51 + PER + NoisyNet'
       : `MCTS ${s.mcts_backend || (s.use_mcts ? '待启用' : '已禁用')}`);

  $('btnStart').disabled = running;
  $('btnStop').disabled = !running;

  // ---- 图表 1: 分数 ----
  const start = s.recent_scores_start || 0;
  charts.score.data.labels = scores.map((_, i) => start + i);
  charts.score.data.datasets[0].data = scores;
  charts.score.data.datasets[1].data = MA(scores, w);
  charts.score.data.datasets[1].label = `MA-${w}`;
  charts.score.update('none');

  // ---- 图表 2: loss / |δ| ----
  const second = meta.charts.second.series(s);
  charts.loss.data.labels = second.data.map((_, i) => i);
  charts.loss.data.datasets[0].data = second.data;
  charts.loss.update('none');

  // ---- 图表 3 ----
  if (currentAlgo === 'v3') {
    charts.eps.data.labels = scores.map((_, i) => start + i);
    charts.eps.data.datasets[0].data = MA(scores, 200);
    charts.eps.data.datasets[1].data = MA(scores, 50);
  } else {
    charts.eps.data.labels = (s.epsilons_tail || []).map((_, i) => i);
    charts.eps.data.datasets[0].data = s.epsilons_tail || [];
  }
  charts.eps.update('none');

  // ---- 图表 4: 里程碑 ----
  const evals = s.evals || [];
  charts.milestone.data.labels = evals.map(e => e.episode);
  charts.milestone.data.datasets[0].data = evals.map(e => e.ms512 || 0);
  charts.milestone.data.datasets[1].data = evals.map(e => e.ms1024 || 0);
  charts.milestone.data.datasets[2].data = evals.map(e => e.ms2048 || 0);
  charts.milestone.data.datasets[3].data = evals.map(e => e.ms4096 || 0);
  charts.milestone.update('none');
}

function setBadge(running, text) {
  const b = $('statusBadge');
  b.className = 'status-badge ' + (running ? 'running' : 'idle');
  b.querySelector('span').textContent = text;
}

/* ---------------- 并行实验对比 ---------------- */
async function loadCompare() {
  let j;
  try { j = await (await fetch('/api/compare')).json(); }
  catch (e) { return; }
  const rows = (j.experiments || []).filter(r => r.episode > 0);
  if (!rows.length) { $('expSection').style.display = 'none'; return; }
  $('expSection').style.display = '';
  $('expHint').textContent = `${rows.length} 个并行实验 · 数据目录 ${j.base}/exp_*`;

  rows.sort((a, b) => (b.recent_avg || 0) - (a.recent_avg || 0));
  const leader = rows[0].name;

  // ---- 卡片 ----
  const box = $('expCards');
  box.innerHTML = '';
  rows.forEach((r, i) => {
    const ev = r.last_eval || {};
    const pct = (v) => v == null ? '–' : (v * 100).toFixed(0) + '%';
    const prog = r.target_episodes ? Math.min(100, r.episode / r.target_episodes * 100) : 0;
    const d = document.createElement('div');
    d.className = 'exp-card' + (r.name === leader ? ' leader' : '');
    d.innerHTML = `
      <div class="exp-top">
        <span class="exp-name">${r.name}${r.name === leader ? ' <span class="crown">领先</span>' : ''}</span>
        <span class="exp-state ${r.running ? 'on' : 'off'}">${r.running ? '运行中' : '已停止'}</span>
      </div>
      <div class="exp-cfg">
        图案 ${r.patterns || '–'} · λ=${r.lam ?? '–'} · α=${r.alpha ?? '–'}
      </div>
      <div class="exp-metric">
        <b>${int(Math.round(r.recent_avg))}</b><small>近 200 局均分</small>
      </div>
      <div class="exp-sub">
        <span>${int(r.episode)} 局</span>
        <span>${r.episodes_per_sec} 局/秒</span>
        <span>最佳 ${r.best_eval != null ? int(Math.round(r.best_eval)) : '–'}</span>
      </div>
      <div class="pbar thin"><i style="width:${prog.toFixed(1)}%"></i></div>
      <div class="exp-miles">
        <span>512 <b>${pct(ev.ms512)}</b></span>
        <span>1024 <b>${pct(ev.ms1024)}</b></span>
        <span>2048 <b>${pct(ev.ms2048)}</b></span>
        <span>4096 <b>${pct(ev.ms4096)}</b></span>
      </div>`;
    box.appendChild(d);
  });

  // ---- 对比曲线（按局数对齐，各实验按自身起点偏移绘制）----
  const keep = charts.compare.data.datasets.map(d => d.label);
  charts.compare.data.datasets = rows.map((r, i) => {
    const ds = DS(r.name, EXP_COLORS[i % EXP_COLORS.length], 2, false);
    ds.pointRadius = 0;
    // x 轴用局数，先记录 (episode, score) 对
    ds._pts = (r.score_series || []).map((v, k) => [r.score_start + k, v]);
    return ds;
  });
  // 用所有实验的最大局数范围构造 labels
  let maxEp = 0;
  charts.compare.data.datasets.forEach(d => d._pts.forEach(p => maxEp = Math.max(maxEp, p[0])));
  const step = Math.max(1, Math.floor(maxEp / 600));
  const labels = [];
  for (let e = 0; e <= maxEp; e += step) labels.push(e);
  charts.compare.data.labels = labels;
  charts.compare.data.datasets.forEach(d => {
    const m = new Map(d._pts.map(p => [Math.floor(p[0] / step) * step, p[1]]));
    d.data = labels.map(e => m.has(e) ? m.get(e) : null);
    delete d._pts;
  });
  charts.compare.update('none');
}

/* ---------------- 训练控制 ---------------- */
async function startTrain() {
  const body = { algo: currentAlgo };
  if (currentAlgo === 'v3') {
    body.episodes = parseInt($('v3Episodes').value) || 3000000;
    body.nt_patterns = $('v3Patterns').value;
    body.nt_lambda = parseFloat($('v3Lambda').value);
    body.nt_alpha = parseFloat($('v3Alpha').value);
    body.nt_vinit = parseFloat($('v3Vinit').value);
    body.resume = $('v3Resume').checked;
  } else {
    body.episodes = parseInt($('inpEpisodes').value) || 5000;
    body.parallel = parseInt($('inpParallel').value) || 0;
    body.mcts_interval = parseInt($('inpInterval').value) || 0;
    body.eps_decay = parseInt($('inpEpsDecay').value) || 200000;
    body.resume = $('inpResume').checked;
    body.no_mcts = $('inpNoMcts').checked;
  }
  $('btnStart').disabled = true;
  $('ctlCmd').textContent = '启动中…';
  try {
    const j = await (await fetch('/api/train/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    })).json();
    $('ctlCmd').textContent = j.ok ? `已启动 pid ${j.pid}` : (j.msg || '启动失败');
  } catch (e) { $('ctlCmd').textContent = '请求失败: ' + e; }
  setTimeout(refreshStatus, 1500);
}

async function stopTrain() {
  if (!confirm(`停止 ${currentAlgo.toUpperCase()} 训练？进度会自动保存到 checkpoint。`)) return;
  $('btnStop').disabled = true;
  $('ctlCmd').textContent = '正在停止并保存进度…';
  try {
    const j = await (await fetch('/api/train/stop', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algo: currentAlgo })
    })).json();
    $('ctlCmd').textContent = j.ok ? `已停止（退出码 ${j.exit_code ?? '–'}）` : (j.msg || '停止失败');
  } catch (e) { $('ctlCmd').textContent = '请求失败: ' + e; }
  setTimeout(refreshStatus, 1500);
}

/* ---------------- AI 演示（SSE） ---------------- */
let playSource = null;

function renderBoard(board, prevBoard) {
  const el = $('board');
  if (el.children.length !== 16) {
    el.innerHTML = '';
    for (let i = 0; i < 16; i++) el.appendChild(document.createElement('div'));
  }
  const flat = board.flat();
  const prev = prevBoard ? prevBoard.flat() : [];
  for (let i = 0; i < 16; i++) {
    const cell = el.children[i];
    const v = flat[i];
    const cls = TILE_CLASS(v);
    const changed = prev.length && prev[i] !== v;
    if (cell.className !== cls) cell.className = cls;
    if (cell.textContent !== String(v || '')) cell.textContent = v || '';
    if (changed) { cell.classList.remove('pop'); void cell.offsetWidth; cell.classList.add('pop'); }
  }
}

function updateSimsLabel() {
  const v = $('plSims').value;
  let t;
  if (currentAlgo === 'v3') {
    t = Number(v) <= 10 ? '贪心 1-ply' : `${Math.max(2, Math.round(Number(v) / 20))}-ply 搜索`;
  } else {
    t = v === '0' ? '关闭' : v + ' 次';
  }
  $('plSimsVal').textContent = t;
}

function startPlay() {
  if (playSource) playSource.close();
  const delay = parseFloat($('plDelay').value);
  const sims = parseInt($('plSims').value) || 0;
  const sel = $('plModel');
  const model = sel.value;
  const opt = sel.options[sel.selectedIndex];
  const src = (opt && opt.dataset && opt.dataset.src) || '';
  const url = `/api/play?algo=${currentAlgo}&model=${encodeURIComponent(model)}`
    + (src ? `&src=${encodeURIComponent(src)}` : '')
    + `&delay=${delay}&mcts=${sims > 0 ? 1 : 0}&sims=${sims || 50}`;

  $('moveList').innerHTML = '';
  $('plResult').textContent = '';
  $('plScore').textContent = '0';
  $('plMax').textContent = '–';
  $('plSteps').textContent = '0';
  $('btnPlay').disabled = true;
  $('btnPlayStop').disabled = false;
  $('boardHint').textContent = 'AI 自动游玩中…';

  let prev = null;
  playSource = new EventSource(url);

  playSource.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch (e) { return; }
    if (d.type === 'start') {
      renderBoard(d.board, null); prev = d.board;
      $('boardHint').textContent = `${(d.algo || '').toUpperCase()} 对局开始 · `
        + `${d.model}${d.src ? ' (来自 ' + d.src + ')' : ''}`;
    } else if (d.type === 'step') {
      renderBoard(d.board, prev); prev = d.board;
      $('plScore').textContent = int(d.score);
      $('plMax').textContent = int(d.max_tile);
      $('plSteps').textContent = d.step;
      const chip = document.createElement('span');
      chip.className = 'move-chip';
      chip.textContent = `${d.step}.${d.action_name}`;
      $('moveList').appendChild(chip);
      $('moveList').scrollTop = $('moveList').scrollHeight;
    } else if (d.type === 'end') {
      $('plResult').textContent = `✓ 终局：${int(d.score)} 分 · 最大方块 ${int(d.max_tile)} · 存活 ${d.steps} 步`;
      $('boardHint').textContent = '演示结束，可再次点击开始';
      endPlay();
    } else if (d.type === 'error') {
      $('plResult').textContent = '✗ ' + d.msg;
      $('boardHint').textContent = d.msg;
      endPlay();
    }
  };
  playSource.onerror = () => {
    $('boardHint').textContent = '连接中断（若模型不存在请先训练）';
    endPlay();
  };
}

function endPlay() {
  if (playSource) { playSource.close(); playSource = null; }
  $('btnPlay').disabled = false;
  $('btnPlayStop').disabled = true;
}

/* ---------------- 模型 / 评测 ---------------- */
async function loadModels() {
  // "仅当前算法" 时用单算法接口，否则拉取全部来源（含并行实验）
  const onlyCurrent = $('mdlOnlyCurrent') && $('mdlOnlyCurrent').checked;
  const url = onlyCurrent ? q('/api/models') : '/api/models?algo=all';
  let j;
  try { j = await (await fetch(url)).json(); }
  catch (e) { return; }

  const tb = $('tblModels').querySelector('tbody');
  tb.innerHTML = '';
  const files = j.files || [];
  const SRC_COLOR = { v1: '#58a6ff', v2: '#9e86c8', v3: '#edc22e' };

  files.forEach(f => {
    const src = f.source || j.algo || '?';
    const color = SRC_COLOR[src] || '#3ecf8e';
    const isBest = f.name.startsWith('best');
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><span class="src-badge" style="color:${color};background:${color}1a;border-color:${color}44">${src}</span></td>` +
      `<td class="mono">${f.name}${isBest ? ' <span class="crown">最佳</span>' : ''}</td>` +
      `<td>${f.size_mb} MB</td>` +
      `<td class="mono">${f.mtime}</td>` +
      `<td class="row-actions">` +
        `<button class="btn small" data-eval="${f.name}" data-src="${src}">评测</button>` +
        `<a class="btn small" href="/api/download?src=${encodeURIComponent(src)}&name=${encodeURIComponent(f.name)}">下载</a>` +
      `</td>`;
    tb.appendChild(tr);
  });
  if (!files.length) {
    const tip = onlyCurrent
      ? `「${currentAlgo}」目录下暂无模型。v3 的模型产出在并行实验目录中，`
        + `请在右上角<b>取消勾选</b>「仅当前算法」查看全部。`
      : '暂无模型。模型按 checkpoint 间隔产出（v3 约每 5000 局），请等待训练推进。';
    tb.innerHTML = `<tr><td colspan="5" class="dim">${tip}</td></tr>`;
  }
  if ($('modelsHint')) {
    $('modelsHint').textContent = onlyCurrent
      ? `当前算法 ${currentAlgo} · ${files.length} 个文件`
      : `全部来源 · ${files.length} 个文件` +
        ((j.dirs || []).length ? ` · ${j.dirs.map(d => d.source + '(' + d.count + ')').join(' / ')}` : '');
  }

  tb.querySelectorAll('[data-eval]').forEach(b =>
    b.addEventListener('click', () => runEval(b.dataset.eval, b.dataset.src)));

  // 演示页模型下拉：填充为「来源/文件（均分）」, 避免同名模型分不清
  const plModel = $('plModel');
  if (plModel && files.length) {
    const keep = plModel.value;
    const opts = files.filter(f => f.name.endsWith('.npz')
                                 || f.name.endsWith('.pt') || f.name.endsWith('.npz'));
    plModel.innerHTML = files.map(f => {
      const ep = f.episode ? `${(f.episode / 1000).toFixed(0)}k局` : '';
      const be = f.best_eval ? `${Math.round(f.best_eval / 1000)}k分` : '';
      const tag = [f.running ? '●' : '', ep, be].filter(Boolean).join(' ');
      return `<option value="${f.name}" data-src="${f.source}">` +
             `${f.source}/${f.name}${tag ? ' — ' + tag : ''}</option>`;
    }).join('');
    const same = [...plModel.options].some(o => o.value === keep);
    if (same) plModel.value = keep;
  }

  // 同步「用模型启动实验」的来源/文件下拉
  const srcSet = [...new Set(files.map(f => f.source))];
  const srcSel = $('mxSrc'), mdlSel = $('mxModel');
  if (srcSel && mdlSel) {
    const keepSrc = srcSel.value, keepMdl = mdlSel.value;
    srcSel.innerHTML = srcSet.map(s => `<option value="${s}">${s}</option>`).join('');
    if (srcSet.includes(keepSrc)) srcSel.value = keepSrc;
    const fillModels = () => {
      const src = srcSel.value;
      const list = files.filter(f => f.source === src);
      mdlSel.innerHTML = list.map(f =>
        `<option value="${f.name}">${f.name}${f.name.startsWith('best') ? '（最佳）' : ''}</option>`
      ).join('') || '<option value="">(无模型)</option>';
      if (list.some(f => f.name === keepMdl)) mdlSel.value = keepMdl;
    };
    fillModels();
    if (!srcSel._bound) {
      srcSel._bound = true;
      srcSel.addEventListener('change', fillModels);
    }
  }
}

/* ---------------- 用模型启动新实验 ---------------- */
async function startFromModel() {
  const src = $('mxSrc').value, model = $('mxModel').value;
  if (!model) { $('mxOut').textContent = '✗ 请先选择一个模型'; return; }
  const body = {
    src, model,
    name: ($('mxName').value || '').trim(),
    workers: parseInt($('mxWorkers').value) || 0,
    episodes: parseInt($('mxEpisodes').value) || 3000000,
  };
  $('btnStartFromModel').disabled = true;
  $('mxOut').textContent = `正在从 ${src}/${model} 启动新实验…`;
  try {
    const j = await (await fetch('/api/train/start_from_model', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })).json();
    if (!j.ok) {
      $('mxOut').textContent = '✗ ' + (j.msg || '启动失败');
    } else {
      $('mxOut').textContent =
        `✓ 实验「${j.experiment}」已启动\n` +
        `  pid ${j.pid} · 图案集 ${j.patterns} · ${j.workers || 1} 进程\n` +
        `  起点模型: ${j.from_model}\n` +
        `  目录: ${j.dir}\n\n` +
        `  约 10 秒后可在「实时监控 → 并行实验对比」看到它。`;
      setTimeout(loadCompare, 12000);
    }
  } catch (e) {
    $('mxOut').textContent = '✗ 请求失败: ' + e;
  }
  $('btnStartFromModel').disabled = false;
}

async function runEval(modelOverride, srcOverride) {
  const model = modelOverride || $('evModel').value;
  const games = parseInt($('evGames').value) || 10;
  const depth = parseInt(($('evDepth') && $('evDepth').value) || 0);
  // 模型来自并行实验时，评测要走对应的算法路径
  let algo = currentAlgo;
  if (srcOverride && (srcOverride.startsWith('exp_') || srcOverride === 'v3')) algo = 'v3';
  else if (srcOverride) algo = srcOverride;
  const est = depth >= 3 ? '（3-ply 很慢，每局约 3-5 分钟）'
            : depth === 2 ? '（2-ply，每局约 10-20 秒）' : '（约数秒）';
  $('evOut').textContent = `评测中：${model} · ${games} 局 ${est}…`;
  try {
    const j = await (await fetch('/api/eval', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, games, algo: currentAlgo,
                             search_depth: depth })
    })).json();
    if (!j.ok) { $('evOut').textContent = '✗ ' + (j.msg || '评测失败'); return; }
    const depthTag = j.search_depth > 1
      ? ` - ${j.search_depth}-ply 搜索` : ' - 1-ply 贪心';
    let txt =
      `模型     : ${j.model}  (${j.algo}${depthTag})\n` +
      `局数     : ${j.games}\n` +
      `平均分   : ${int(Math.round(j.avg))}\n` +
      `最高分   : ${int(j.max)}\n` +
      `中位数   : ${int(Math.round(j.median))}\n` +
      `平均步数 : ${j.avg_steps}\n`;
    if (j.ms512 != null) {
      txt += `里程碑   : 512 ${j.ms512}% | 1024 ${j.ms1024}% | 2048 ${j.ms2048}% | 4096 ${j.ms4096}%\n`;
    }
    if (j.scores && j.scores.length) txt += `各局分数 : ${j.scores.join(', ')}`;
    $('evOut').textContent = txt;
  } catch (e) { $('evOut').textContent = '✗ 请求失败: ' + e; }
}

/* ---------------- 日志 ---------------- */
async function loadLogs() {
  try {
    const j = await (await fetch(q('/api/logs?lines=300'))).json();
    const v = $('logView');
    const atBottom = v.scrollTop + v.clientHeight >= v.scrollHeight - 40;
    v.textContent = (j.lines || []).join('\n') || '（暂无日志）';
    if (atBottom) v.scrollTop = v.scrollHeight;
  } catch (e) { /* 忽略 */ }
}

/* ---------------- 事件绑定 ---------------- */
function bindEvents() {
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('panel-' + t.dataset.tab).classList.add('active');
    if (t.dataset.tab === 'logs') loadLogs();
    if (t.dataset.tab === 'models') loadModels();
  }));

  document.querySelectorAll('[data-algo]').forEach(b => b.addEventListener('click', () => {
    currentAlgo = b.dataset.algo;
    document.querySelectorAll('[data-algo]').forEach(x =>
      x.classList.toggle('active', x.dataset.algo === currentAlgo));
    $('ctlCmd').textContent = '';
    lastStatus = null;
    applyAlgoUI(currentAlgo);
    refreshStatus();
    loadModels();
    if ($('panel-logs').classList.contains('active')) loadLogs();
  }));

  $('btnStart').addEventListener('click', startTrain);
  $('btnStop').addEventListener('click', stopTrain);
  $('btnPlay').addEventListener('click', startPlay);
  $('btnPlayStop').addEventListener('click', endPlay);
  $('btnEval').addEventListener('click', () => runEval());
  $('btnLogRefresh').addEventListener('click', loadLogs);
  if ($('btnMdlRefresh')) $('btnMdlRefresh').addEventListener('click', loadModels);
  if ($('btnStartFromModel')) $('btnStartFromModel').addEventListener('click', startFromModel);
  if ($('mdlOnlyCurrent')) $('mdlOnlyCurrent').addEventListener('change', loadModels);

  $('plDelay').addEventListener('input', e => $('plDelayVal').textContent = e.target.value + 's');
  $('plSims').addEventListener('input', updateSimsLabel);

  setInterval(() => { if ($('logAuto').checked) loadLogs(); }, 5000);
}

/* ---------------- 启动 ---------------- */
/** 自动选择有数据的算法（优先 v3 训练主线，其次 v2、v1）*/
async function autoSelectAlgo() {
  for (const a of ['v3', 'v2', 'v1']) {
    try {
      const d = await (await fetch(`/api/status?algo=${a}`)).json();
      if ((d.episode || 0) > 0) {
        if (a !== currentAlgo) {
          currentAlgo = a;
          document.querySelectorAll('[data-algo]').forEach(x =>
            x.classList.toggle('active', x.dataset.algo === a));
          applyAlgoUI(a);
        }
        return;
      }
    } catch (e) { /* 继续尝试下一个 */ }
  }
}

window.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  bindEvents();
  applyAlgoUI(currentAlgo);
  renderBoard(Array.from({ length: 4 }, () => [0, 0, 0, 0]), null);
  await autoSelectAlgo();          // 避免默认落在无数据的 v1 上
  refreshStatus();
  loadModels();
  loadCompare();
  setInterval(refreshStatus, 2000);
  setInterval(loadCompare, 10000);
  setInterval(() => {
    if ($('panel-models').classList.contains('active')) loadModels();
  }, 15000);
});
