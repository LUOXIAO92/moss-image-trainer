/* =========================================================================
 * js/components/jobs.js — Jobs + SSE
 * split from web/static/app.js, source 3212-4418 (12 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { movingAverage, setChartSeries } from './chart.js';
import { finishJob, subscribe } from './jobsse.js';
import { beginCacheProgress, beginCaptionProgress, cacheStageLabel, clearLog } from './logview.js';
import { hasDashboardConfigSelection } from '../config/list.js';
import { parseYaml } from '../config/yaml.js';
import { api } from '../core/api.js';
import { $, $$, clamp, setText } from '../core/dom.js';
import { fmtDuration, fmtInt, fmtNum } from '../core/fmt.js';
import { jobLabel, t } from '../core/i18n.js';
import { showAlert } from '../core/notify.js';
import { isCaptionJob, state, workspaceParam } from '../core/state.js';
import { currentTab } from '../main.js';
import { renderCacheJob, updateCacheStartState } from '../pages/cache.js';
import { updateCaptionStartState } from '../pages/dataset.js';
import { startSamplePolling } from '../pages/train.js';

export function resetTrainSeries() {
  state.series = { train: new Map(), avg: new Map(), eval: new Map(), meta: new Map() };
  state.trainTotal = 0;
  state.liveTrain = false;
  setChartSeries('dash', { train: [], eval: [] }, 0, []);
  updateTrainMetrics();
}

function seriesToChartSeries(series) {
  return {
    train: Array.from(series.train.entries()).map(([step, value]) => ({ x: step, y: value })),
    avg: Array.from(series.avg.entries()).map(([step, value]) => ({ x: step, y: value })),
    eval: Array.from(series.eval.entries()).map(([step, value]) => ({ x: step, y: value })),
  };
}

export function syncTrainChart() {
  const rows = Array.from(state.series.meta.entries()).map(([step, meta]) => ({ step, ...meta }));
  setChartSeries('dash', seriesToChartSeries(state.series), state.trainTotal, rows);
}

export function updateTrainMetrics() {
  const trainSteps = Array.from(state.series.train.keys()).sort((a, b) => a - b);
  const lastStep = trainSteps[trainSteps.length - 1];
  const meta = lastStep != null ? state.series.meta.get(lastStep) : null;
  const loss = lastStep != null ? state.series.train.get(lastStep) : null;
  const smoothed = movingAverage(
    trainSteps.map((step) => ({ x: step, y: state.series.train.get(step) })),
    state.smoothWindow,
  );
  const avg = smoothed[smoothed.length - 1]?.y;
  setText('#dash-train', fmtNum(loss, 4));
  setText('#dash-train-sub', lastStep != null ? t('dash.kpi.trainSub', { avg: fmtNum(avg, 4), step: fmtInt(lastStep) }) : t('train.msg47'));
  setText('#dash-speed', fmtNum(meta?.sec, 2));
  setText('#dash-peak', meta?.vram != null ? fmtNum(meta.vram, 1) : '--');
  const total = state.trainTotal || meta?.total || 0;
  const percent = total > 0 && lastStep != null ? clamp((lastStep / total) * 100, 0, 100) : 0;
  setText('#dash-step', total > 0 ? `${fmtInt(lastStep || 0)} / ${fmtInt(total)}` : '--');
  const progress = $('#dash-progress');
  if (progress) progress.setAttribute('aria-valuenow', String(Math.round(percent)));
  const bar = $('#dash-progress-bar');
  if (bar) bar.style.width = `${percent.toFixed(1)}%`;
  renderActivity();
}

export function updateTrainRunState() {
  const chip = $('#train-run-state');
  if (!chip) return;
  if (state.job && state.job.status === 'running' && state.job.kind === 'train') {
    chip.className = 'chip chip-run';
    chip.textContent = t('train.msg48');
    return;
  }
  if (state.job && state.job.status === 'error' && state.job.kind === 'train') {
    chip.className = 'chip chip-err';
    chip.textContent = t('train.msg49');
    return;
  }
  const run = state.dashboard.data && state.dashboard.data.run === (state.currentRun?.label || state.currentRun?.name)
    ? state.dashboard.data
    : null;
  const rows = run?.rows || [];
  const trainRows = rows.filter((row) => Number.isFinite(row.train));
  const lastSeriesStep = Array.from(state.series.train.keys()).sort((a, b) => a - b).pop();
  if (!trainRows.length && lastSeriesStep == null) {
    chip.className = 'chip chip-muted';
    chip.textContent = t('train.msg50');
    return;
  }
  const last = trainRows[trainRows.length - 1] || { step: lastSeriesStep, total: state.trainTotal };
  if (last.total && last.step >= last.total) {
    chip.className = 'chip chip-ok';
    chip.textContent = t('train.msg51');
  } else {
    chip.className = 'chip chip-warn';
    chip.textContent = t('train.msg52');
  }
}

export function renderRunStrip() {
  let name = '';
  let outputDir = '';
  const runningTrain = state.job && state.job.status === 'running' && state.job.kind === 'train';
  if (state.job && state.job.kind === 'train' && (state.job.label || state.job.name)) {
    name = state.job.label || state.job.name;
    outputDir = state.job.outputDir || '';
    if (!outputDir && state.job.workspace && state.job.name) {
      outputDir = `${state.workspacePath || ('workspaces/' + state.job.workspace)}/runs/${state.job.name}`;
    }
  }
  if (!name && state.currentRun) {
    name = state.currentRun.label || state.currentRun.name || '';
    outputDir = state.currentRun.outputDir || '';
  }
  if (!name) {
    const outputRoot = state.workspacePath || `${state.settings?.paths?.workspaces || 'workspaces'}/${state.profileName || '<workspace>'}`;
    name = t('common.msg12');
    outputDir = `${outputRoot}/runs/<run-id>`;
  }
  const stripLabel = $('#run-strip-label');
  if (stripLabel) {
    if (runningTrain) stripLabel.textContent = t('common.msg34');
    else if (state.currentRun && state.currentRun.profile && state.currentRun.profile !== state.profileName) {
      stripLabel.textContent = t('common.msg35');
    } else if (state.currentRun && !state.currentRun.profile && !state.job?.profile) {
      stripLabel.textContent = t('common.msg36');
    } else stripLabel.textContent = t('common.msg37');
  }
  setText('#train-run-name', name || '--');
  setText('#train-run-dir', outputDir || '--');
  const chip = $('#train-run-state');
  if (chip) {
    if (runningTrain) {
      chip.className = 'chip chip-run';
      chip.textContent = t('train.msg48');
    } else if (state.job && state.job.kind === 'train' && state.job.status === 'error') {
      chip.className = 'chip chip-err';
      chip.textContent = t('train.msg49');
    } else if (state.currentRun) {
      const rows = state.dashboard.data?.run === (state.currentRun.label || state.currentRun.name) ? state.dashboard.data.rows : [];
      const trainRows = rows.filter((row) => Number.isFinite(row.train));
      const lastSeriesStep = Array.from(state.series.train.keys()).sort((a, b) => a - b).pop();
      const last = trainRows[trainRows.length - 1] || (lastSeriesStep != null ? { step: lastSeriesStep, total: state.trainTotal } : null);
      if (last && last.total && last.step >= last.total) {
        chip.className = 'chip chip-ok';
        chip.textContent = t('train.msg51');
      } else if (last) {
        chip.className = 'chip chip-warn';
        chip.textContent = t('train.msg52');
      } else {
        chip.className = 'chip chip-muted';
        chip.textContent = t('common.msg38');
      }
    } else {
      chip.className = 'chip chip-muted';
      chip.textContent = t('train.msg50');
    }
  }
  const source = $('#train-chart-source');
  if (source) {
    if (runningTrain) source.textContent = t('dash.chart.liveSource', { name });
    else if (state.currentRun && state.series.train.size) source.textContent = t('dash.chart.historySource', { name, count: state.series.train.size });
    else source.textContent = t('common.msg39');
  }
}

export function updateJobUi() {
  const live = state.job && state.job.status === 'running' ? state.job.last : null;
  const sameRun = !state.activeTraining?.run || !state.job?.runName
    || state.activeTraining.run === state.job.runName;
  // The button label follows the heartbeat, not the last dashboard poll: a run
  // adopted from disk must tick at the same rate as one this tab started.
  const active = state.activeTraining
    ? {
      ...state.activeTraining,
      step: sameRun && live?.step != null ? live.step : state.activeTraining.step,
      total: sameRun && live?.total != null ? live.total : state.activeTraining.total,
    }
    : null;
  const running = Boolean(state.job && state.job.status === 'running') || Boolean(active);
  const blocked = !state.profileName;
  const missingDashboardConfig = currentTab() === 'dashboard' && !hasDashboardConfigSelection();
  const startDisabled = running || state.busy || blocked || missingDashboardConfig;
  for (const selector of ['#btn-start']) {
    const button = $(selector);
    if (button) button.disabled = startDisabled;
  }
  for (const button of $$('[data-caption-job]')) button.disabled = startDisabled;
  updateCaptionStartState();
  updateCacheStartState();
  const baselineToggle = $('#cfg-run-baseline');
  if (baselineToggle) baselineToggle.disabled = running || blocked;
  const activeProgress = active && (active.step != null || active.total != null)
    ? t('job.progressParen', { step: active.step ?? '?', total: active.total ?? '?' })
    : '';
  const activeText = active
    ? t('job.trainingActive', { run: active.run, progress: activeProgress })
    : (!running && missingDashboardConfig ? t('train.selectConfig') : t('dash.train.start'));
  setText('#btn-start-text', activeText);
  if (active && sameRun && state.job?.status === 'running' && state.job.kind === 'train'
      && active.step != null && $('#dash-run-status')) {
    // Same key the dashboard render uses, refreshed on the 1.2 s job poll so
    // the chip never lags behind the header.
    setText('#dash-run-status', t('dash.runState.training', { step: active.step }));
  }
  for (const selector of ['#btn-config-save', '#btn-config-new', '#btn-config-saveas', '#btn-config-rename', '#btn-config-delete']) {
    const button = $(selector);
    if (button) button.disabled = blocked;
  }
  const preflight = $('#btn-preflight');
  if (preflight) preflight.disabled = startDisabled;
  const stop = $('#btn-stop');
  if (stop) stop.disabled = !running;
  const cacheStop = $('#cache-stop');
  // A cache build published to disk is stoppable too; the in-memory registry
  // only knows jobs this server spawned.
  if (cacheStop) {
    cacheStop.classList.toggle('hidden', !(running && (state.job?.kind === 'cache' || state.cache?.jobRunning)));
  }
  const captionSource = $('#cfg-caption-source');
  if (captionSource) captionSource.disabled = running;

  const chip = $('#job-chip');
  const chipText = $('#job-chip-text');
  if (chip && chipText) {
    if (active && !(state.job && state.job.status === 'running')) {
      chip.className = 'chip chip-run';
      chipText.textContent = t('train.msg48');
      chip.title = t('job.trainingTitle', { run: active.run });
    } else if (running && state.job) {
      chip.className = 'chip chip-run';
      chipText.textContent = state.job.stage === 'baseline' ? t('job.msg3') : t('job.kindRunning', { kind: jobLabel(state.job.kind) });
      chip.title = state.job.name ? t('job.kindWithName', { kind: jobLabel(state.job.kind), name: state.job.name }) : t('job.msg4');
    } else if (state.job && state.job.status === 'error') {
      chip.className = 'chip chip-err';
      chipText.textContent = t('job.kindFailed', { kind: jobLabel(state.job.kind) });
      chip.title = t('job.msg5');
    } else if (state.job && state.job.status === 'done') {
      chip.className = 'chip chip-ok';
      chipText.textContent = t('job.kindDone', { kind: jobLabel(state.job.kind) });
      chip.title = t('job.msg6');
    } else if (state.job && state.job.status === 'stopped') {
      chip.className = 'chip chip-muted';
      chipText.textContent = t('job.msg7');
      chip.title = t('job.msg8');
    } else {
      chip.className = 'chip chip-idle';
      chipText.textContent = t('common.idle');
      chip.title = t('job.msg9');
    }
  }
  renderActivity();
  renderRunStrip();
  updateTrainRunState();
  renderCacheJob();
}

function setJobChipFromStatus(status, text) {
  const chip = $('#job-chip');
  const chipText = $('#job-chip-text');
  if (!chip || !chipText) return;
  chip.className = 'chip ' + ({ running: 'chip-run', done: 'chip-ok', error: 'chip-err' }[status] || 'chip-muted');
  chipText.textContent = text;
}

export function renderActivity() {
  const activity = $('#activity');
  if (!activity) return;
  const running = Boolean(state.job && state.job.status === 'running');
  activity.classList.toggle('hidden', !running);
  if (!running) {
    state.activityAnnounce = '';
    return;
  }
  const job = state.job;
  let percent = null;
  let label = '';
  let meta = '';
  if (job.kind === 'train' && job.stage === 'baseline') {
    label = t('job.baselineSampling', { name: job.name ? ` · ${job.name}` : '' });
    meta = t('job.msg10');
  } else if (job.kind === 'train') {
    const steps = Array.from(state.series.train.keys()).sort((a, b) => a - b);
    const lastStep = steps[steps.length - 1];
    const total = state.trainTotal;
    percent = total > 0 && lastStep != null ? (lastStep / total) * 100 : null;
    const metaPoint = lastStep != null ? state.series.meta.get(lastStep) : null;
    label = t('job.trainingRunning', { name: job.name ? ` · ${job.name}` : '' });
    meta = lastStep != null
      ? t('job.trainMeta', {
        step: fmtInt(lastStep),
        total: total ? t('job.trainTotal', { total: fmtInt(total) }) : '',
        loss: metaPoint?.loss != null ? t('job.trainLoss', { loss: fmtNum(metaPoint.loss, 4) }) : '',
        sec: metaPoint?.sec ? t('job.trainSec', { sec: metaPoint.sec.toFixed(2) }) : '',
      })
      : t('job.msg11');
  } else if (isCaptionJob(job.kind)) {
    const current = state.captionProgress.current;
    percent = current?.total > 0 ? (current.done / current.total) * 100 : null;
    label = t('job.kindRunning', { kind: jobLabel(job.kind) });
    meta = current?.total > 0
      ? t('job.captionMeta', { done: fmtInt(current.done), total: fmtInt(current.total), rate: current.rate ? t('job.captionRate', { rate: current.rate }) : '' })
      : t('job.msg11');
  } else if (job.kind === 'cache') {
    const progress = state.cache.progress;
    const active = state.cache.active || '';
    percent = progress && progress.total > 0 ? (progress.done / progress.total) * 100 : null;
    label = t('job.cacheRunning', { stage: active ? ` · ${cacheStageLabel(active)}` : '' });
    meta = progress
      ? (progress.total > 0
        ? t('job.cacheMeta', { done: fmtInt(progress.done), total: fmtInt(progress.total), rate: progress.rate ? t('job.cacheRate', { rate: progress.rate, unit: progress.unit }) : '' })
        : t('job.cachePending', { done: fmtInt(progress.done), unit: progress.unit || t('job.itemsUnit') }))
      : t('job.msg11');
  } else {
    label = t('job.kindRunning', { kind: jobLabel(job.kind) });
    meta = t('job.msg11');
  }
  const elapsed = job.created ? Math.max(0, Date.now() / 1000 - Number(job.created)) : null;
  if (elapsed != null) meta += `${meta ? ' · ' : ''}${t('job.elapsed', { duration: fmtDuration(elapsed) })}`;
  setText('#activity-text', label);
  setText('#activity-meta', meta);
  const announce = t('job.announce', { label, status: percent != null ? t('job.announceProgress', { percent: Math.floor(percent / 10) * 10 }) : t('job.announceStarting') });
  if (state.activityAnnounce !== announce) {
    state.activityAnnounce = announce;
    setText('#activity-live', announce);
  }
  const container = $('#activity-progress');
  const bar = $('#activity-bar');
  const value = percent == null ? 0 : clamp(percent, 0, 100);
  if (bar) bar.style.width = `${value.toFixed(1)}%`;
  if (container) {
    container.classList.toggle('is-indeterminate', percent == null);
    if (percent == null) container.removeAttribute('aria-valuenow');
    else container.setAttribute('aria-valuenow', String(Math.round(value)));
  }
}

export async function loadJobs() {
  const jobs = await api('/api/jobs');
  state.jobs = Array.isArray(jobs) ? jobs : [];
  const running = state.jobs.find((job) => job.status === 'running');
  if (running) {
    const isNew = !state.job || state.job.id !== running.id;
    if (isNew) {
      state.logs = [];
      clearLog();
      state.eventCount = 0;
      state.job = {
        id: running.id, kind: running.kind, status: 'running',
        name: running.run_name || '', runName: running.run_name || '',
        label: running.workspace && running.run_name ? `${running.workspace}/${running.run_name}` : (running.run_name || ''),
        profile: running.profile || '',
        workspace: running.workspace || '',
        outputDir: running.output_dir || '', configPath: running.config_path || '', logPath: running.log_path || '',
        stage: running.stage || '', baselineError: running.baseline_error || '',
        created: running.created || Date.now() / 1000,
      };
      state.jobInfoFor = null;
      if (running.kind === 'cache') beginCacheProgress();
      else if (running.kind !== 'train') state.cache.progress = null;
      if (running.kind === 'train') {
        state.dashboard.run = running.workspace && running.run_name ? `${running.workspace}/${running.run_name}` : null;
        resetTrainSeries();
        state.liveTrain = true;
        startSamplePolling();
      }
    } else {
      state.job.status = 'running';
      state.job.created = state.job.created || running.created;
      state.job.configPath = state.job.configPath || running.config_path || '';
      state.job.profile = state.job.profile || running.profile || '';
      state.job.workspace = state.job.workspace || running.workspace || '';
      state.job.runName = state.job.runName || running.run_name || '';
      state.job.label = state.job.label || (running.workspace && running.run_name ? `${running.workspace}/${running.run_name}` : (running.run_name || ''));
      state.job.outputDir = state.job.outputDir || running.output_dir || '';
      state.job.logPath = state.job.logPath || running.log_path || '';
      state.job.stage = running.stage || state.job.stage || '';
      state.job.baselineError = running.baseline_error || state.job.baselineError || '';
      // /api/jobs already carries the heartbeat's progress. Take it when it is
      // newer than what the stream delivered, so a run adopted from disk moves
      // at the same rate as one this tab started.
      const fresh = running.last;
      if (fresh && fresh.step != null) {
        const known = Number(state.job.last?.step ?? -1);
        if (!(known >= Number(fresh.step))) state.job.last = fresh;
      }
    }
    if (isCaptionJob(running.kind) && state.captionProgress.jobId !== running.id) {
      beginCaptionProgress(state.job);
    }
    if (!state.es || state.streamedJobId !== running.id) {
      subscribe(running.id, state.eventCount || 0);
    }
    applyJobRunInfo(state.job).catch(() => {});
  } else if (state.job && state.job.status === 'running') {
    const match = state.jobs.find((job) => job.id === state.job.id);
    if (match) finishJob(match.status === 'done' ? 'done' : match.status, '');
    else finishJob('stopped', t('job.msg55'));
  }
  updateJobUi();
}

export async function pollJobs() {
  try {
    await loadJobs();
    const running = state.job && state.job.status === 'running' && state.job.kind === 'train';
    if (running && state.currentRun?.name && Date.now() - state.lastEvalSync > 15000) {
      state.lastEvalSync = Date.now();
      syncLiveEval().catch(() => {});
    }
  } catch (error) {
    // Polling failures are transient; surface only the first one.
    if (!state.dashboard.lastError) {
      state.dashboard.lastError = error.message;
      showAlert({ type: 'warn', title: t('job.msg56'), text: error.message, timeout: 5000 });
    }
  }
}

async function syncLiveEval() {
  if (!state.currentRun?.name || !(state.job?.status === 'running') || state.job.kind !== 'train') return;
  const runLabel = state.currentRun.label || state.currentRun.name;
  const data = await api('/api/dashboard?run=' + encodeURIComponent(runLabel) + '&workspace=' + encodeURIComponent(workspaceParam()));
  if (!data || data.run !== (state.currentRun.label || state.currentRun.name)) return;
  let changed = false;
  for (const row of data.rows || []) {
    if (Number.isFinite(row.eval) && !state.series.eval.has(row.step)) {
      state.series.eval.set(row.step, Number(row.eval));
      changed = true;
    }
  }
  if (changed) {
    syncTrainChart();
  }
}

export async function applyJobRunInfo(job) {
  if (!job || state.jobInfoFor === job.id) return;
  state.jobInfoFor = job.id;
  let name = job.name || '';
  let outputDir = job.outputDir || '';
  if ((!name || !outputDir) && job.configPath) {
    try {
      const text = await api('/api/files?path=' + encodeURIComponent(job.configPath));
      const config = parseYaml(text);
      name = name || String(config.name || '');
      if (!outputDir) outputDir = `${config.output_dir || 'output'}/${name}`;
      job.name = name;
      job.outputDir = outputDir;
    } catch (_) { /* config path may not exist for an old job */ }
  }
  if (name && job.kind === 'train') {
    const workspace = job.workspace || job.profile || '';
    const label = workspace ? `${workspace}/${name}` : name;
    state.currentRun = { name, label, profile: workspace, workspace, outputDir: outputDir || `${workspace || 'output'}/${name}` };
    renderRunStrip();
    updateTrainRunState();
  }
}
