/* =========================================================================
 * js/pages/dashboard.js — Dashboard
 * split from web/static/app.js, source 4622-5877; behaviour unchanged
 * ========================================================================= */

import { buildLine, drawChart, movingAverage, setChartSeries } from '../components/chart.js';
import { jobSignal, renderRunStrip, updateJobUi, updateTrainMetrics, updateTrainRunState } from '../components/jobs.js';
import { loadRunLog } from '../components/logview.js';
import { defaultDashboardLayout, drawMetricCards, renderWidgetCards, saveDashboardLayout } from '../components/metriccards.js';
import { api, describeError } from '../core/api.js';
import { $, clamp, escapeHtml, fileUrl, setText, thumbUrl } from '../core/dom.js';
import { fmtDuration, fmtInt, fmtNum, fmtTime } from '../core/fmt.js';
import { localeCode, t } from '../core/i18n.js';
import { showAlert, showError } from '../core/notify.js';
import { state, workspaceParam } from '../core/state.js';
import { getCurrentYaml } from '../config/form.js';
import { renderDashboardConfigOptions, selectDashboardConfig } from '../config/list.js';

function dashboardSeries(rows) {
  const train = [];
  const avg = [];
  const evalPoints = [];
  for (const row of rows || []) {
    if (Number.isFinite(Number(row.step)) && Number.isFinite(Number(row.train)) && row.train !== null) {
      train.push({ x: Number(row.step), y: Number(row.train), row });
      if (Number.isFinite(Number(row.avg))) avg.push({ x: Number(row.step), y: Number(row.avg), row });
    }
    if (Number.isFinite(Number(row.step)) && Number.isFinite(Number(row.eval)) && row.eval !== null) {
      evalPoints.push({ x: Number(row.step), y: Number(row.eval), row });
    }
  }
  return { train, avg, eval: evalPoints };
}

function setTrend(selector, text, tone = '') {
  const element = $(selector);
  if (!element) return;
  element.textContent = text;
  element.className = `kpi-trend${tone ? ' is-' + tone : ''}`;
}

function formatDelta(value, digits = 4) {
  if (!Number.isFinite(value)) return '--';
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

const ESTIMATE_DEBOUNCE = 300;

const estimate = {
  timer: null,
  seq: 0,
  data: null,
  yaml: '',
  workspace: '',
};

function dashboardVisible() {
  return String(location.hash || '').replace(/^#/, '') === 'dashboard';
}

function estimateText(key, fallback, vars = {}) {
  const text = t(key, vars);
  if (text !== key) return text;
  // The key may not be in the installed packs yet; keep the Chinese copy here
  // so the card never renders a bare key.
  return String(fallback).replace(/\{(\w+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match
  ));
}

function formatGiB(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) : '--';
}

function cardTotalGiB() {
  const total = Number(state.system?.gpu?.total_gb);
  return Number.isFinite(total) && total > 0 ? total : null;
}

function measuredPeakGiB() {
  // Only a live training run describes the config currently in the form; a
  // viewed run's historical peak belongs to a different config.
  if (state.job?.status !== 'running' || state.job?.kind !== 'train') return null;
  let lastStep = -Infinity;
  let peak = null;
  for (const [step, meta] of state.series.meta.entries()) {
    const index = Number(step);
    if (!Number.isFinite(index) || index < lastStep) continue;
    const device = Number(meta?.vram_device);
    const allocated = Number(meta?.vram);
    // Mirror the peak card: device-wide when the log line carries it, else the
    // allocator peak from the live progress event.
    const value = Number.isFinite(device) && device > 0
      ? device
      : (Number.isFinite(allocated) && allocated > 0 ? allocated : null);
    if (value == null) continue;
    lastStep = index;
    peak = value;
  }
  return peak;
}

/** The shape renderDashboard expects when there is nothing to show yet. */
function emptyDashboard() {
  return {
    workspace: '',
    run: null,
    runs: [],
    metrics: [],
    samples: [],
    summary: {},
    compare: null,
    jobs: [],
    series: [],
  };
}

function blankEstimate() {
  const card = $('#dash-estimate-card');
  if (card) card.classList.remove('is-over');
  setText('#dash-estimate', '--');
  setText('#dash-estimate-sub', estimateText('dash.kpi.estimateBreakdown',
    '训练 {train} · 预览 {preview} · 上下文 {context} · 卡 {card}', {
      train: '--', preview: '--', context: '--', card: '--',
    }));
}

export function renderEstimateCard() {
  const card = $('#dash-estimate-card');
  if (!card) return;
  const data = estimate.data;
  // No loaded run config means there is no YAML to cost; an estimate of the
  // defaults would describe a run the user cannot start.
  if (!data || !state.configPath || estimate.workspace !== workspaceParam()) {
    blankEstimate();
    return;
  }
  const total = Number(data.total);
  if (!Number.isFinite(total)) {
    blankEstimate();
    return;
  }
  const capacity = cardTotalGiB();
  const over = capacity != null && total > capacity ? total - capacity : null;
  card.classList.toggle('is-over', over != null);
  setText('#dash-estimate', `${formatGiB(total)} GiB`);
  // The context term is why the total sits above the training and preview peaks;
  // without it the card reads as an over-estimate next to the measured one.
  const lines = [estimateText('dash.kpi.estimateBreakdown',
    '训练 {train} · 预览 {preview} · 上下文 {context} · 卡 {card}', {
      train: formatGiB(data.train),
      preview: formatGiB(data.preview),
      context: formatGiB(data.context),
      card: capacity == null ? '--' : formatGiB(capacity),
    })];
  if (over != null) {
    lines.push(estimateText('dash.kpi.estimateOver', '预计超出显卡 {over} GiB', { over: formatGiB(over) }));
  }
  const measured = measuredPeakGiB();
  if (measured != null) {
    lines.push(estimateText('dash.kpi.estimateMeasured', '实测峰值 {peak} GiB', { peak: formatGiB(measured) }));
  }
  setText('#dash-estimate-sub', lines.join('\n'));
}

async function refreshEstimate(seq) {
  if (!dashboardVisible()) return;
  if (!state.profileName || !state.configPath) {
    estimate.data = null;
    renderEstimateCard();
    return;
  }
  const workspace = workspaceParam();
  const yaml = getCurrentYaml();
  if (!yaml.trim()) {
    estimate.data = null;
    renderEstimateCard();
    return;
  }
  if (estimate.data && estimate.yaml === yaml && estimate.workspace === workspace) {
    renderEstimateCard();
    return;
  }
  try {
    const data = await api('/api/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml, workspace }),
    });
    if (seq !== estimate.seq) return;
    estimate.data = data;
    estimate.yaml = yaml;
    estimate.workspace = workspace;
    renderEstimateCard();
  } catch (error) {
    // A 404 means the workspace is gone and a 400 means the YAML is mid-edit;
    // both are normal here, so the card goes blank without a toast or a log.
    if (seq !== estimate.seq) return;
    estimate.data = null;
    estimate.yaml = '';
    estimate.workspace = workspace;
    renderEstimateCard();
  }
}

export function scheduleEstimate(delay = ESTIMATE_DEBOUNCE) {
  if (!dashboardVisible()) return;
  clearTimeout(estimate.timer);
  // Invalidate any answer still in flight: it was asked for an older form.
  estimate.seq += 1;
  const seq = estimate.seq;
  estimate.timer = setTimeout(() => {
    estimate.timer = null;
    refreshEstimate(seq).catch(() => {});
  }, Math.max(0, Number(delay) || 0));
}

export function initDashboardEstimate() {
  // The form marks dirty on every edit; follow that signal instead of polling.
  document.addEventListener('moss:config-dirty', () => scheduleEstimate());
  // The card total arrives with /api/system after boot. Re-render the warning
  // state from the answer already on screen, or ask once if there is none.
  document.addEventListener('moss:gpu-total-changed', () => {
    if (estimate.data) renderEstimateCard();
    else scheduleEstimate(0);
  });
}

function runRunState(data, activeOverride = null, liveStep = null) {
  if (!data || !data.run) return { text: '--', className: 'chip chip-muted' };
  const rows = data.rows || [];
  const trainRows = rows.filter((row) => Number.isFinite(row.train));
  const last = trainRows[trainRows.length - 1];
  const active = activeOverride == null ? Boolean(data.active) : Boolean(activeOverride);
  if (active) {
    // The heartbeat ticks every second; the metrics file is read on the slower
    // dashboard poll. Prefer the heartbeat so the chip and the header agree.
    const step = liveStep ?? last?.step ?? data.active?.step ?? '?';
    return { text: t('dash.runState.training', { step }), className: 'chip chip-run' };
  }
  if (last && last.total && last.step >= last.total) return { text: t('train.msg51'), className: 'chip chip-ok' };
  if (last) return { text: t('train.msg52'), className: 'chip chip-warn' };
  return { text: t('common.msg38'), className: 'chip chip-muted' };
}

const COMPARE_COLORS = ['#f472b6', '#a78bfa', '#fb923c', '#f87171', '#38bdf8', '#fbbf24'];

export function clearCompareState() {
  state.compareRuns.clear();
  state.compareData = [];
  state.sampleCompareSelection.clear();
  state.sampleCompareKnown = new Set();
  state.sampleCompareSignature = '';
  state.sampleCompareData = [];
  state.sampleCompareInitialized = false;
  state.sampleCompareLoading = false;
  state.compareVisible = {};
  if (state.charts.dash) state.charts.dash.compare = [];
  renderCompareLegend();
}

function compareRunsForMenu() {
  const runs = [...(state.dashboard.data?.runs || [])].sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  const names = new Set(runs.map((run) => run.name));
  state.compareRuns = new Set([...state.compareRuns].filter((name) => names.has(name)));
  return runs;
}

function renderCompareMenu(runs = null) {
  const menu = $('#compare-menu');
  if (!menu) return;
  const list = runs || compareRunsForMenu();
  const current = state.dashboard.data?.run || '';
  const selected = new Set(state.compareRuns);
  menu.innerHTML = list.length
    ? list.map((run) => `<label class="compare-option ${run.name === current ? 'is-current' : ''}">
        <input type="checkbox" data-compare-run="${escapeHtml(run.name)}" ${selected.has(run.name) ? 'checked' : ''}>
        <span class="compare-option-name mono">${escapeHtml(run.name)}</span>
        <span class="muted small">${fmtInt(run.step)}/${fmtInt(run.total)}${run.name === current ? escapeHtml(t('common.currentSuffix')) : ''}</span>
      </label>`).join('')
    : `<p class="compare-option-empty">${escapeHtml(t('dash.compare.empty'))}</p>`;
  const count = $('#compare-count');
  if (count) count.textContent = String(selected.size);
}

export function renderCompareLegend() {
  const box = $('#dash-compare-legend');
  if (!box) return;
  const groups = state.compareData || [];
  box.innerHTML = groups.map((run) => {
    const trainOn = run.visible?.train !== false;
    const evalOn = run.visible?.eval !== false;
    const label = String(run.label || run.name).split('/').pop();
    return `<div class="compare-legend-group">
      <span class="compare-run-name"><span class="legend-dot" style="background:${run.color}"></span>${escapeHtml(label)}</span>
      <button type="button" class="compare-legend-item ${trainOn ? 'is-on' : ''}" data-compare-run="${escapeHtml(run.name)}" data-compare-series="train" aria-pressed="${trainOn}">
        <span class="legend-line" style="color:${run.color}"></span>${escapeHtml(t('dash.chart.rawLoss'))}
      </button>
      <button type="button" class="compare-legend-item ${evalOn ? 'is-on' : ''}" data-compare-run="${escapeHtml(run.name)}" data-compare-series="eval" aria-pressed="${evalOn}">
        <span class="legend-line is-dashed" style="color:${run.color}"></span>${escapeHtml(t('dash.kpi.evalLoss'))}
      </button>
    </div>`;
  }).join('');
}

function sampleStepInfo(path) {
  const name = String(path || '').split('/').pop() || '';
  const match = name.match(/^step(\d+)_(\d+)\.png$/);
  if (!match) return null;
  return { step: Number(match[1]), index: Number(match[2]), path };
}

function sampleGroups(samples) {
  const groups = new Map();
  for (const path of samples || []) {
    const info = sampleStepInfo(path);
    if (!info) continue;
    if (!groups.has(info.step)) groups.set(info.step, []);
    groups.get(info.step).push({ ...info, path });
  }
  for (const items of groups.values()) items.sort((a, b) => a.index - b.index);
  return groups;
}

function samplePromptFor(run, index) {
  return (run.prompts || [])[index] || `Prompt ${index + 1}`;
}

function sampleNegativeFor(run) {
  return run.negative || '';
}

function runShortName(value) {
  return String(value || '').split('/').pop();
}

function sampleRunOptions() {
  const dashRuns = state.dashboard.data?.runs || [];
  const dashByShort = new Map();
  for (const run of dashRuns) {
    dashByShort.set(runShortName(run.name || run.label), run);
  }
  const seen = new Set();
  const options = [];
  const add = (entry, dash) => {
    const short = runShortName(entry?.name || entry?.label || dash?.name || dash?.label);
    if (!short || seen.has(short)) return;
    seen.add(short);
    const activeFallback = runShortName(state.dashboard.data?.run)
      && runShortName(state.dashboard.data?.run) === runShortName(dash?.name || dash?.label || entry?.name)
      ? (state.dashboard.data?.samples || []).length
      : 0;
    const samples = Math.max(
      Number(dash?.samples || 0),
      Number(entry?.samples || 0),
      Number(activeFallback || 0),
    );
    if (!Number.isFinite(samples) || samples <= 0) return;
    options.push({
      name: entry?.label || dash?.label || (state.profileName ? `${state.profileName}/${short}` : short),
      short,
      label: entry?.label || dash?.label || short,
      samples,
    });
  };
  for (const entry of state.runs || []) add(entry, dashByShort.get(runShortName(entry.name)));
  for (const dash of dashRuns) add(null, dash);
  return options;
}

function dashboardRunFor(name) {
  const short = runShortName(name);
  return (state.dashboard.data?.runs || []).find((run) => runShortName(run.name || run.label) === short) || null;
}

function sampleRunConfig(name) {
  const short = runShortName(name);
  const entry = (state.runs || []).find((item) => (
    item.name === short || runShortName(item.name) === short || item.label === name
  ));
  const config = entry?.config || {};
  return {
    prompts: Array.isArray(config.preview?.prompts) ? config.preview.prompts : [],
    negative: config.preview?.negative_prompt || '',
  };
}

export function initSampleCompareSelection() {
  if (!state.dashboard.data) return;
  const available = sampleRunOptions();
  const availableNames = new Set(available.map((run) => run.name));
  let known = state.sampleCompareKnown;
  if (!(known instanceof Set)) known = new Set(availableNames);
  let selection = state.sampleCompareSelection;
  if (!(selection instanceof Set)) selection = new Set();
  const first = !state.sampleCompareInitialized;
  let changed = false;

  if (first) {
    const saved = state.layout?.sampleRuns;
    selection = new Set(Array.isArray(saved)
      ? saved.filter((name) => availableNames.has(name))
      : [...availableNames]);
    known = new Set(availableNames);
    changed = true;
    state.sampleCompareInitialized = true;
  } else {
    for (const name of [...known]) {
      if (availableNames.has(name)) continue;
      known.delete(name);
      selection.delete(name);
      changed = true;
    }
    for (const name of availableNames) {
      if (known.has(name)) continue;
      known.add(name);
      // A newly discovered run starts selected; an explicitly deselected run
      // was already known, so it is never added back.
      selection.add(name);
      changed = true;
    }
  }
  state.sampleCompareKnown = known;
  state.sampleCompareSelection = selection;

  const signature = available.map((run) => `${run.name}:${run.samples}`).join('|');
  const dataChanged = signature !== state.sampleCompareSignature;
  state.sampleCompareSignature = signature;
  if (changed) {
    if (!state.layout) state.layout = defaultDashboardLayout();
    state.layout.sampleRuns = [...selection];
    if (state.layoutLoaded) saveDashboardLayout();
  }
  renderSampleRunMenu();
  // While the job's event stream is live it delivers each finished preview, so
  // the poll must not rebuild the grid: that rebuild is what listed a sample
  // file the trainer was still encoding.
  const streamOwnsSamples = Boolean(state.es) && state.job?.status === 'running' && state.job?.kind === 'train';
  if (changed || (dataChanged && !streamOwnsSamples)) loadSampleCompareData().catch(() => {});
}

export function applySampleEvent(event) {
  // The trainer logs a preview batch only after every file is written, so these
  // paths point at complete images and need no directory listing.
  const paths = (Array.isArray(event?.paths) ? event.paths : []).filter((path) => sampleStepInfo(path));
  if (!paths.length) return false;
  const run = runShortName(event.run || state.job?.label || '');
  if (!run) return false;
  const entry = (state.sampleCompareData || []).find((item) => runShortName(item.name) === run);
  if (!entry) return false;
  const known = new Set(entry.samples || []);
  const added = paths.filter((path) => !known.has(path));
  if (!added.length) return false;
  entry.samples = [...(entry.samples || []), ...added];
  if (state.dashboard.data && runShortName(state.dashboard.data.run) === run) {
    state.dashboard.data.samples = [...(state.dashboard.data.samples || []), ...added];
  }
  // Let the next full snapshot reconcile once the stream stops owning the grid.
  state.sampleCompareSignature = '';
  renderSampleCompareGrid();
  return true;
}

function renderSampleRunMenu() {
  const menu = $('#sample-run-menu');
  const button = $('#btn-sample-runs');
  if (!menu || !button) return;
  const runs = sampleRunOptions();
  const selected = state.sampleCompareSelection || new Set();
  menu.innerHTML = runs.length
    ? runs.map((run) => `<label class="compare-option ${selected.has(run.name) ? 'is-on' : ''}"><input type="checkbox" data-sample-run="${escapeHtml(run.name)}" ${selected.has(run.name) ? 'checked' : ''}> <span class="mono">${escapeHtml(run.label || run.name)}</span><span class="muted small">${escapeHtml(t('common.imagesCount', { count: fmtInt(run.samples) }))}</span></label>`).join('')
    : `<p class="compare-empty">${escapeHtml(t('dash.samples.noRuns'))}</p>`;
  setText('#sample-run-count', String(selected.size));
  button.setAttribute('aria-expanded', menu.classList.contains('hidden') ? 'false' : 'true');
}

function baselineIndex(path) {
  const match = String(path || '').split('/').pop().match(/^baseline_(\d+)\.png$/);
  return match ? Number(match[1]) : -1;
}

async function loadBaselineSamples() {
  if (!state.profileName || state.baselineLoading) return;
  state.baselineLoading = true;
  try {
    const dir = `${state.workspacePath}/baseline`;
    let found = [];
    let source = '';
    try {
      const data = await api('/api/fs/list?path=' + encodeURIComponent(dir));
      found = (data.entries || [])
        .filter((entry) => !entry.is_dir && baselineIndex(entry.name) >= 0)
        .sort((a, b) => baselineIndex(a.name) - baselineIndex(b.name));
      source = data.path || dir;
    } catch (error) {
      // missing baseline directory is the normal case before the first run
    }
    state.baselineSamples = found;
    state.baselineSource = source;
    renderSampleCompareGrid();
  } finally {
    state.baselineLoading = false;
  }
}

async function loadSampleCompareData() {
  if (state.sampleCompareLoading) return;
  loadBaselineSamples().catch(() => {});
  const names = [...(state.sampleCompareSelection || [])];
  if (!names.length) {
    state.sampleCompareData = [];
    renderSampleCompareGrid();
    return;
  }
  state.sampleCompareLoading = true;
  try {
    const data = await Promise.all(names.map(async (name) => {
      const dashRun = dashboardRunFor(name);
      const short = runShortName(name);
      const entry = (state.runs || []).find((run) => run.name === short
        || runShortName(run.name) === short || run.label === name);
      const payload = await api('/api/dashboard?workspace=' + encodeURIComponent(workspaceParam()) + '&run=' + encodeURIComponent(name));
      const config = sampleRunConfig(name);
      return {
        name,
        label: entry?.label || dashRun?.label || name,
        samples: payload.samples || [],
        prompts: config.prompts,
        negative: config.negative,
      };
    }));
    state.sampleCompareData = data.filter(Boolean);
    renderSampleCompareGrid();
  } catch (error) {
    showError(t('dash.msg6'), describeError(error));
  } finally {
    state.sampleCompareLoading = false;
  }
}

export function toggleSampleCompareRun(name, checked) {
  if (!state.sampleCompareSelection) state.sampleCompareSelection = new Set();
  if (checked) state.sampleCompareSelection.add(name);
  else state.sampleCompareSelection.delete(name);
  if (!state.layout) state.layout = defaultDashboardLayout();
  state.layout.sampleRuns = [...state.sampleCompareSelection];
  renderSampleRunMenu();
  loadSampleCompareData().catch(() => {});
  saveDashboardLayout();
}

function applySampleCollapsed() {
  const panel = $('#sample-compare-panel');
  const button = $('#btn-sample-collapse');
  if (!panel || !button) return;
  const collapsed = Boolean(state.layout?.sampleCollapsed);
  panel.classList.toggle('is-collapsed', collapsed);
  button.textContent = collapsed ? t('dash.msg7') : t('dash.samples.collapse');
  button.setAttribute('aria-expanded', String(!collapsed));
}

export function toggleSampleCollapsed() {
  if (!state.layout) state.layout = defaultDashboardLayout();
  state.layout.sampleCollapsed = !state.layout.sampleCollapsed;
  applySampleCollapsed();
  saveDashboardLayout();
}

function renderSampleCompareGrid() {
  const panel = $('#sample-compare-panel');
  const grid = $('#sample-compare-grid');
  if (!panel || !grid) return;
  applySampleCollapsed();
  panel.classList.remove('hidden');
  const runs = state.sampleCompareData || [];
  if (!state.sampleCompareSelection?.size) {
    setText('#sample-compare-count', t('dash.msg8'));
    grid.innerHTML = `<div class="sample-emptyState"><strong>${escapeHtml(t('dash.samples.selectTitle'))}</strong><p>${escapeHtml(t('dash.samples.selectText'))}</p></div>`;
    return;
  }
  if (!runs.length) {
    setText('#sample-compare-count', t('common.runsCount', { count: state.sampleCompareSelection.size }));
    grid.innerHTML = `<div class="sample-emptyState"><strong>${escapeHtml(t('dash.samples.reading'))}</strong></div>`;
    return;
  }
  const steps = new Set();
  const groupsByRun = runs.map((run) => {
    const groups = sampleGroups(run.samples);
    for (const step of groups.keys()) steps.add(step);
    return { run, groups };
  });
  const columns = [0, ...[...steps].filter((step) => step > 0).sort((a, b) => a - b)];
  setText('#sample-compare-count', t('dash.samples.gridCount', { runs: runs.length, steps: columns.length }));
  const head = `<thead><tr><th scope="col">Run</th>${columns.map((step) => `<th scope="col">step ${fmtInt(step)}</th>`).join('')}</tr></thead>`;
  const promptSource = runs[0];
  const baseline = state.baselineSamples || [];
  const stepZeroImages = baseline.length
    ? `<div class="sample-step-cell">${baseline.map((item) => {
        const index = baselineIndex(item.name);
        return `<button type="button" class="compare-thumb" data-compare-step="0" title="${escapeHtml(item.path)}"><img loading="lazy" src="${thumbUrl(item.path, 220)}" alt="${escapeHtml(samplePromptFor(promptSource, index))}"><span>P${index + 1}</span></button>`;
      }).join('')}</div>`
    : `<span class="sample-baseline-empty">${escapeHtml(t('dash.samples.noBaseline'))}<br><span class="muted small">${escapeHtml(t('dash.samples.baselineHint'))}</span></span>`;
  const body = groupsByRun.map(({ run, groups }) => `<tr><th scope="row">${escapeHtml(run.label || run.name)}</th>${columns.map((step) => {
    if (step === 0) return `<td class="sample-step-zero">${stepZeroImages}</td>`;
    const items = groups.get(step) || [];
    return `<td><div class="sample-step-cell">${items.map((item) => `<button type="button" class="compare-thumb" data-compare-step="${step}" title="${escapeHtml(item.path)}"><img loading="lazy" src="${thumbUrl(item.path, 220)}" alt="${escapeHtml(samplePromptFor(run, item.index))}"><span>P${item.index + 1}</span></button>`).join('') || '<span class="muted small">—</span>'}</div></td>`;
  }).join('')}</tr>`).join('');
  grid.innerHTML = `<table class="sample-compare-table" id="sample-compare-table">${head}<tbody>${body}</tbody></table>`;
}

export function openSampleCompare(step) {
  const dialog = $('#sample-compare-dialog');
  const body = $('#sample-compare-body');
  if (!dialog || !body) return;
  const runs = state.sampleCompareData || [];
  state.sampleCompareStep = step;
  if (!state.sampleCompareTrigger || !state.sampleCompareTrigger.isConnected) {
    state.sampleCompareTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  }
  if (step === 0) {
    const baseline = state.baselineSamples || [];
    const promptSource = runs[0];
    const images = baseline.map((item) => {
      const index = baselineIndex(item.name);
      return `<figure class="sample-prompt-card"><img loading="lazy" src="${fileUrl(item.path)}" alt="${escapeHtml(samplePromptFor(promptSource, index))}"><figcaption><b>P${index + 1}</b> ${escapeHtml(samplePromptFor(promptSource, index))}</figcaption></figure>`;
    }).join('');
    setText('#sample-compare-title', t('dash.msg9'));
    body.innerHTML = images ? `<section class="sample-run-block"><h3>${escapeHtml(t('dash.samples.baseTitle'))}</h3><div class="sample-prompt-grid">${images}</div></section>` : `<p class="muted">${escapeHtml(t('dash.samples.noBaselineShort'))}</p>`;
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
    else dialog.setAttribute('open', '');
    return;
  }
  const blocks = runs.map((run) => {
    const items = sampleGroups(run.samples).get(step) || [];
    if (!items.length) return '';
    return `<section class="sample-run-block"><h3 class="mono">${escapeHtml(run.label || run.name)}</h3><div class="sample-prompt-grid">${items.map((item) => `<figure class="sample-prompt-card">
      <img loading="lazy" src="${fileUrl(item.path)}" alt="${escapeHtml(samplePromptFor(run, item.index))}">
      <figcaption><b>P${item.index + 1}</b> ${escapeHtml(samplePromptFor(run, item.index))}${sampleNegativeFor(run) ? `<br><span class="neg">negative: ${escapeHtml(sampleNegativeFor(run))}</span>` : ''}</figcaption>
    </figure>`).join('')}</div></section>`;
  }).join('');
  setText('#sample-compare-title', t('dash.samples.stepCompare', { step: fmtInt(step), count: runs.length }));
  body.innerHTML = blocks || `<p class="muted">${escapeHtml(t('dash.samples.noStep'))}</p>`;
  if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
  else dialog.setAttribute('open', '');
}

export function closeSampleCompare() {
  const dialog = $('#sample-compare-dialog');
  if (!dialog) return;
  if (typeof dialog.close === 'function' && dialog.open) dialog.close();
  else dialog.removeAttribute('open');
}

export async function loadDashboardCompare() {
  const chart = state.charts.dash;
  const names = [...state.compareRuns];
  const current = state.dashboard.data?.run || '';
  if (!names.length) {
    state.compareData = [];
    if (chart) chart.compare = [];
    renderCompareLegend();
    drawChart('dash');
    return;
  }
  if (state.compareLoading) {
    state.comparePending = true;
    return;
  }
  state.compareLoading = true;
  try {
    if (!state.runs.length) {
      const runsData = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/runs');
      state.runs = Array.isArray(runsData.runs) ? runsData.runs : [];
    }
    const runs = state.dashboard.data?.runs || [];
    const selected = names.map((name) => runs.find((run) => run.name === name)).filter(Boolean);
    const data = await Promise.all(selected.map(async (run) => {
      const payload = await api('/api/dashboard?workspace=' + encodeURIComponent(workspaceParam()) + '&run=' + encodeURIComponent(run.name));
      const rows = payload.rows || [];
      return {
        name: run.name, label: run.label || run.name, rows,
      };
    }));
    state.compareData = data.filter((run) => run.name !== current).map((run, index) => ({
      ...run,
      color: COMPARE_COLORS[index % COMPARE_COLORS.length],
      visible: { train: true, eval: true },
      series: {
        train: buildLine(run.rows, 'train'),
        eval: buildLine(run.rows, 'eval'),
      },
    }));
    if (chart) chart.compare = state.compareData;
    renderCompareLegend();
    drawChart('dash');
  } catch (error) {
    showError(t('dash.msg10'), describeError(error));
  } finally {
    state.compareLoading = false;
    if (state.comparePending) {
      state.comparePending = false;
      loadDashboardCompare().catch(() => {});
    }
  }
}

export function openCompareMenu() {
  const menu = $('#compare-menu');
  const button = $('#btn-compare-toggle');
  if (!menu || !button) return;
  const open = menu.classList.toggle('hidden');
  button.setAttribute('aria-expanded', String(!open));
  if (!open) renderCompareMenu();
}

export async function loadActiveTraining() {
  if (!state.profileName) {
    state.activeTraining = null;
    updateJobUi();
    return;
  }
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/active');
    state.activeTraining = data.active || null;
    if (state.activeTraining && state.activeTraining.step == null) {
      try {
        const log = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/log?run=' + encodeURIComponent(state.activeTraining.run) + '&tail=20');
        const line = [...(log.lines || [])].reverse().find((item) => /step\s+\d+\/\d+/.test(item));
        const match = line?.match(/step\s+(\d+)\/(\d+)/);
        if (match) {
          state.activeTraining.step = Number(match[1]);
          state.activeTraining.total = Number(match[2]);
        }
      } catch (error) {}
    }
  } catch (error) {
    state.activeTraining = null;
  }
  updateJobUi();
}

let runsRefreshing = false;
let runsRefreshedAt = 0;

function refreshRunsInBackground(force = false) {
  if (runsRefreshing) return;
  if (!force && Date.now() - runsRefreshedAt < 15000) return;
  runsRefreshing = true;
  runsRefreshedAt = Date.now();
  // Loaded lazily: workspace.js imports jobsse.js, which imports this module.
  import('./workspace.js')
    .then(({ loadWorkspaceRuns }) => loadWorkspaceRuns())
    .then(async () => {
      const selectedExists = (state.runs || []).some((entry) => entry.name === state.activeRun);
      if (!selectedExists && state.runs.length && !state.dirty) {
        // The viewed run was deleted; fall back to the newest remaining config.
        const ok = await selectDashboardConfig(state.runs[0].name);
        if (ok) await loadDashboard(state.dashboard.run, { silent: true });
      } else {
        renderDashboardConfigOptions();
        initSampleCompareSelection();
      }
    })
    .catch(() => {})
    .finally(() => { runsRefreshing = false; });
}

export async function loadDashboard(run = null, options = {}) {
  if (state.dashboard.loading) return state.dashboard.data;
  // A fresh install has no workspace yet, and the endpoint answers 400 for one
  // it cannot resolve. Asking anyway turns the empty state into a console error
  // and a warning toast on first open.
  if (!workspaceParam()) {
    state.dashboard.data = null;
    state.dashboard.lastError = '';
    renderDashboard(emptyDashboard());
    return null;
  }
  state.dashboard.loading = true;
  const liveLabel = state.job?.status === 'running' && state.job.kind === 'train' ? state.job.label : '';
  const activeEntry = (state.runs || []).find((entry) => entry.name === state.activeRun);
  const requested = liveLabel || run || state.dashboard.run || activeEntry?.label || '';
  try {
    const query = new URLSearchParams({ workspace: workspaceParam() });
    if (requested) query.set('run', requested);
    const data = await api('/api/dashboard?' + query.toString());
    state.dashboard.data = data;
    // A config-only run has no rows yet, so the endpoint may not choose it.
    // Keep the run the user selected instead of falling back to the newest one.
    state.dashboard.run = data.run || requested || null;
    state.dashboard.lastError = '';
    renderDashboard(data);
    initSampleCompareSelection();
    const known = new Set((state.runs || []).map((entry) => runShortName(entry.name)));
    const missing = (data.runs || []).some((entry) => !known.has(runShortName(entry.name || entry.label)));
    refreshRunsInBackground(missing);
    loadRunLog(state.dashboard.run).catch(() => {});
    loadBaselineSamples().catch(() => {});
    if (!options.silent) setText('#dash-updated', t('dash.updated', { time: new Date().toLocaleTimeString(localeCode(), { hour12: false }) }));
    return data;
  } catch (error) {
    state.dashboard.lastError = error.message;
    if (!options.silent) {
      showAlert({ type: 'warn', title: t('dash.msg12'), text: error.message, timeout: 6000 });
    }
    return null;
  } finally {
    state.dashboard.loading = false;
  }
}

function gpuUsageSummary(gpu, peak = null) {
  const info = gpu || {};
  const parts = [];
  const used = Number(peak ?? info.used_gb);
  const total = Number(info.total_gb);
  if (Number.isFinite(used)) {
    parts.push(Number.isFinite(total)
      ? t('gpu.memoryPair', { used: used.toFixed(1), total: total.toFixed(1) })
      : t('gpu.memoryValue', { used: used.toFixed(1) }));
  }
  if (info.utilization != null && Number.isFinite(Number(info.utilization))) {
    parts.push(t('gpu.usage', { value: Math.round(Number(info.utilization)) }));
  }
  if (info.temperature != null) parts.push(t('gpu.temperature', { value: info.temperature }));
  return parts.join(' · ');
}

export function renderDashboard(data) {
  const runs = [...(data.runs || [])].sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  const rows = data.rows || [];
  const series = dashboardSeries(rows);
  const train = series.train;
  const evals = series.eval;
  const lastTrain = train[train.length - 1];
  const firstTrain = train[0];
  const lastEval = evals[evals.length - 1];
  const total = lastTrain?.row?.total || 0;
  const newestRun = runs[0]?.name;
  const activeName = state.currentRun?.label || state.currentRun?.name || newestRun;
  const active = Boolean(data.active) && runShortName(data.run) === runShortName(activeName);
  const liveLabel = state.job?.status === 'running' && state.job.kind === 'train' ? state.job.label : '';
  const liveMismatch = Boolean(liveLabel && runShortName(data.run) !== runShortName(liveLabel));
  if (!liveMismatch) setChartSeries('dash', series, total, rows);
  const smoothed = movingAverage(train, state.smoothWindow);
  const lastSmoothed = smoothed[smoothed.length - 1];
  renderCompareMenu(runs);
  renderCompareLegend();
  if (!state.layout) state.layout = defaultDashboardLayout();
  renderWidgetCards();
  drawMetricCards();
  if (state.layoutLoaded) initSampleCompareSelection();

  const progress = total > 0 && lastTrain ? clamp((lastTrain.x / total) * 100, 0, 100) : 0;
  setText('#dash-step', lastTrain ? `${fmtInt(lastTrain.x)} / ${total ? fmtInt(total) : '?'}` : '--');
  setText('#dash-step-sub', lastTrain
    ? (total > lastTrain.x ? t('dash.kpi.remaining', { count: fmtInt(total - lastTrain.x) }) : t('dash.msg13'))
    : t('dash.kpi.waiting'));
  setText('#dash-step-trend', lastTrain ? `${progress.toFixed(0)}%` : '');
  const dashProgress = $('#dash-progress');
  if (dashProgress) dashProgress.setAttribute('aria-valuenow', String(Math.round(progress)));
  const dashProgressBar = $('#dash-progress-bar');
  if (dashProgressBar) dashProgressBar.style.width = `${progress.toFixed(1)}%`;

  if (lastTrain) {
    setText('#dash-train', fmtNum(lastTrain.y, 4));
    const delta = firstTrain && firstTrain.y > 0 ? ((lastTrain.y - firstTrain.y) / firstTrain.y) * 100 : null;
    if (delta != null) {
      setText('#dash-train-sub', t('dash.kpi.trainSubRelative', { avg: fmtNum(lastSmoothed?.y, 4), sign: delta >= 0 ? '+' : '−', percent: Math.abs(delta).toFixed(1) }));
      setTrend('#dash-train-trend', `${delta <= 0 ? '↓' : '↑'} ${Math.abs(delta).toFixed(1)}%`, delta <= 0 ? 'good' : 'bad');
    } else {
      setText('#dash-train-sub', t('dash.kpi.trainSubAverage', { avg: fmtNum(lastSmoothed?.y, 4) }));
      setTrend('#dash-train-trend', '');
    }
  } else {
    setText('#dash-train', '--');
    setText('#dash-train-sub', t('dash.kpi.latestPoint'));
    setTrend('#dash-train-trend', '');
  }

  if (lastEval) {
    setText('#dash-eval', fmtNum(lastEval.y, 4));
    const firstEval = evals[0];
    const previous = evals.length >= 2 ? evals[evals.length - 2].y : null;
    let lastLine = t('dash.kpi.evalStep', { step: fmtInt(lastEval.x) });
    if (previous != null && Number.isFinite(previous)) {
      const diff = lastEval.y - previous;
      if (Math.abs(diff) < 0.00005) {
        lastLine += t('dash.msg14');
        setTrend('#dash-eval-trend', t('dash.msg15'));
      } else {
        lastLine += t('dash.kpi.evalVsPrevious', { delta: formatDelta(diff) });
        setTrend('#dash-eval-trend', `${diff <= 0 ? '↓' : '↑'} ${Math.abs(diff).toFixed(4)}`, diff <= 0 ? 'good' : 'bad');
      }
    } else {
      setTrend('#dash-eval-trend', '');
    }
    let firstLine = '';
    if (firstEval && evals.length > 1) {
      const firstDiff = lastEval.y - firstEval.y;
      const firstPct = firstEval.y > 0 ? (firstDiff / firstEval.y) * 100 : null;
      const pctText = firstPct == null ? '' : t('dash.kpi.evalFirstPct', { sign: firstPct >= 0 ? '+' : '−', percent: Math.abs(firstPct).toFixed(1) });
      firstLine = t('dash.kpi.evalFirst', { delta: formatDelta(firstDiff), pct: pctText });
    }
    setText('#dash-eval-sub', [lastLine, firstLine].filter(Boolean).join('\n'));
  } else {
    setText('#dash-eval', '--');
    setText('#dash-eval-sub', t('dash.kpi.noEval'));
    setTrend('#dash-eval-trend', '');
  }

  if (lastTrain && lastEval) {
    const gap = lastEval.y - lastTrain.y;
    const relative = lastTrain.y > 0 ? gap / lastTrain.y : 0;
    setText('#dash-gap', formatDelta(gap));
    setText('#dash-gap-sub', t('dash.kpi.gapSub', { eval: fmtNum(lastEval.y, 4), train: fmtNum(lastTrain.y, 4) }));
    setTrend('#dash-gap-trend', `${(relative * 100).toFixed(1)}%`, relative > 0.15 ? 'warn' : relative <= 0 ? 'good' : '');
  } else {
    setText('#dash-gap', '--');
    setText('#dash-gap-sub', t('dash.kpi.gapFormula'));
    setTrend('#dash-gap-trend', '');
  }

  if (lastTrain) {
    const sec = Number(lastTrain.row.sec);
    setText('#dash-speed', Number.isFinite(sec) ? `${sec.toFixed(2)}` : '--');
    let sub = t('dash.msg16');
    if (Number.isFinite(sec) && total > lastTrain.x) sub = t('dash.kpi.speedRemaining', { duration: fmtDuration((total - lastTrain.x) * sec), rate: (60 / sec).toFixed(1) });
    setText('#dash-speed-sub', sub);
  } else {
    setText('#dash-speed', '--');
    setText('#dash-speed-sub', t('dash.kpi.latestPoint'));
  }

  const gpuSummary = gpuUsageSummary(data.gpu, lastTrain?.row?.peak);
  if (lastTrain?.row?.peak != null) {
    setText('#dash-peak', `${fmtNum(lastTrain.row.peak, 1)} GiB`);
    setText('#dash-peak-sub', gpuSummary || t('dash.kpi.peak'));
  } else {
    setText('#dash-peak', '--');
    setText('#dash-peak-sub', gpuSummary || 'GPU --');
  }
  setText('#metric-gpu-usage', gpuSummary || '--');

  const liveStep = state.job?.status === 'running' && state.job.kind === 'train'
    && runShortName(state.job.label) === runShortName(data.run) ? state.job.last?.step : null;
  const runInfo = runs.find((run) => run.name === data.run || run.label === data.run
    || runShortName(run.name) === runShortName(data.run));
  const status = runRunState(data, active, liveStep);
  const selectedSignal = data.run && state.job?.status === 'error'
    && runShortName(state.job.label) === runShortName(data.run)
    ? jobSignal(state.job)
    : '';
  if (selectedSignal) {
    status.text = t('dash.runState.signal', { signal: selectedSignal });
    status.className = 'chip chip-err';
  }
  setText('#dash-run-status', status.text);
  setText('#dash-run-name', data.run || '--');
  setText('#dash-run-path', runInfo?.path || '--');
  // The run's own config file, or "no snapshot": never another config's.
  const runConfigName = String(data.run || '').split('/').pop();
  setText('#dash-run-config', data.run_config_path && runConfigName
    ? `runs/${runConfigName}/config.yaml`
    : t('train.noSnapshot'));
  setText('#dash-last-update', runInfo?.mtime ? fmtTime(runInfo.mtime) : '--');
  setText('#dash-points', t('dash.kpi.points', { train: train.length, eval: evals.length }));
  setText('#dash-samples-count', fmtInt((data.samples || []).length));
  const gpu = data.gpu || {};
  const gpuLine = gpu.name ? [gpu.name] : [];
  if (gpu.utilization != null) gpuLine.push(t('gpu.usage', { value: Math.round(Number(gpu.utilization)) }));
  if (gpu.used_gb != null && gpu.total_gb != null) {
    gpuLine.push(t('gpu.memoryPair', { used: Number(gpu.used_gb).toFixed(1), total: Number(gpu.total_gb).toFixed(1) }));
  }
  if (gpu.temperature != null) gpuLine.push(t('gpu.temperature', { value: gpu.temperature }));
  setText('#dash-gpu-line', gpuLine.length ? gpuLine.join(' · ') : t('gpu.unavailable'));
  if (liveMismatch) updateTrainMetrics();
  renderRunStrip();
  updateTrainRunState();
  // One picker, one run: the dashboard never rebuilds a second selector from
  // `data.runs`; it only keeps the training-config picker in sync.
  renderDashboardConfigOptions();
  // The measured peak lives in the live series, so refresh the suffix on each
  // dashboard poll without asking /api/estimate again.
  renderEstimateCard();
}
