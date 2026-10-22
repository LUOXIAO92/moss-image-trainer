/* =========================================================================
 * js/pages/train.js — Train page / dataset linkage
 * split from web/static/app.js, source 5883-7085 (5 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { renderRunStrip, resetTrainSeries, syncTrainChart, updateTrainMetrics, updateTrainRunState } from '../components/jobs.js';
import { api } from '../core/api.js';
import { $, setHidden, setText } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state, workspaceParam } from '../core/state.js';
import { loadDashboard } from './dashboard.js';
import { createSampleCard, sampleMeta } from './samples.js';

function trainSeriesFromRows(rows) {
  for (const row of rows || []) {
    if (Number.isFinite(Number(row.step)) && Number.isFinite(Number(row.train)) && row.train !== null) {
      state.series.train.set(Number(row.step), Number(row.train));
      if (Number.isFinite(Number(row.avg))) state.series.avg.set(Number(row.step), Number(row.avg));
      state.series.meta.set(Number(row.step), {
        lr: row.lr, sec: row.sec, vram: row.peak,
        total: row.total, loss: row.train, avg: row.avg,
      });
      state.trainTotal = Number(row.total) || state.trainTotal;
    }
    if (Number.isFinite(Number(row.step)) && Number.isFinite(Number(row.eval)) && row.eval !== null) {
      state.series.eval.set(Number(row.step), Number(row.eval));
    }
  }
}

export async function loadTrainHistory() {
  const runningTrain = state.job && state.job.status === 'running' && state.job.kind === 'train';
  if (runningTrain) {
    state.liveTrain = true;
    renderRunStrip();
    updateTrainRunState();
    return;
  }
  if (state.series.train.size) {
    renderRunStrip();
    updateTrainRunState();
    return;
  }
  const data = await loadDashboard(null, { silent: true });
  if (!data || !data.run) {
    renderRunStrip();
    updateTrainRunState();
    return;
  }
  resetTrainSeries();
  const runInfo = data.runs.find((run) => run.name === data.run);
  state.currentRun = {
    name: runInfo?.run_id || data.run,
    label: data.run,
    profile: runInfo?.profile || '',
    outputDir: runInfo?.path || `output/${data.run}`,
  };
  trainSeriesFromRows(data.rows || []);
  state.liveTrain = false;
  syncTrainChart();
  updateTrainMetrics();
  renderRunStrip();
  updateTrainRunState();
  refreshTrainSamples().catch(() => {});
}

export async function refreshTrainSamples() {
  const box = $('#train-samples');
  if (!box) return;
  const run = state.currentRun?.label || state.currentRun?.name;
  if (!run) {
    box.innerHTML = '';
    box.classList.add('hidden');
    setHidden('#train-samples-empty', false);
    setText('#train-samples-empty p', t('dash.msg20'));
    return;
  }
  try {
    const list = await api('/api/samples?workspace=' + encodeURIComponent(workspaceParam()));
    box.innerHTML = '';
    let items = (Array.isArray(list) ? list : []).filter((item) => {
      const meta = sampleMeta(item.path, item);
      return meta.run === run && !/\/trajectory\.png$/i.test(item.path);
    });
    items = items.slice(-6).reverse();
    const empty = $('#train-samples-empty');
    setHidden(empty, items.length > 0);
    box.classList.toggle('hidden', !items.length);
    if (!items.length) setText(empty, t('dash.msg21'));
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      const meta = sampleMeta(item.path, item);
      fragment.append(createSampleCard(item.path, { run: meta.run, step: meta.step, small: true }));
    }
    box.append(fragment);
  } catch (_) { /* the samples panel is non-critical */ }
}

export function startSamplePolling() {
  stopSamplePolling();
  state.sampleTimer = setInterval(() => {
    if (state.job?.status === 'running' && state.job.kind === 'train') {
      refreshTrainSamples().catch(() => {});
    }
  }, 8000);
}

export function stopSamplePolling() {
  if (state.sampleTimer) {
    clearInterval(state.sampleTimer);
    state.sampleTimer = null;
  }
}
