/* =========================================================================
 * js/components/jobsse.js — Jobs + SSE
 * split from web/static/app.js, source 3507-4303 (13 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { applyJobRunInfo, loadJobs, renderActivity, resetTrainSeries, syncTrainChart, updateJobUi, updateTrainMetrics, updateTrainRunState } from './jobs.js';
import { beginCacheProgress, beginCaptionProgress, clearLog, completeCacheProgress, copyLog, finalizeCaptionProgress, logErrorMessage, pushLog } from './logview.js';
import { hidePreflight, runPreflight, setPreflight } from './preflight.js';
import { clearYamlError, focusYamlLine, getCurrentYaml, showYamlError } from '../config/form.js';
import { saveConfig } from '../config/file.js';
import { parseYaml } from '../config/yaml.js';
import { api, describeError } from '../core/api.js';
import { $ } from '../core/dom.js';
import { jobLabel, t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { isCaptionJob, state } from '../core/state.js';
import { currentTab, focusLogPanel } from '../main.js';
import { loadCache, readCacheSelection, stopCacheBuild, updateCacheStartState } from '../pages/cache.js';
import { loadActiveTraining, loadDashboard } from '../pages/dashboard.js';
import { readCaptionSelection, refreshDatasetAfterCaption, updateCaptionStartState } from '../pages/dataset.js';
import { loadLoras, updateExportStatus } from '../pages/export.js';
import { loadSamples } from '../pages/samples.js';
import { focusField, saveSettings } from '../pages/settings.js';
import { startSamplePolling, stopSamplePolling } from '../pages/train.js';
import { loadWorkspaceRuns } from '../pages/workspace.js';

function closeStream() {
  if (state.es) {
    state.es.close();
    state.es = null;
  }
  state.streamedJobId = null;
}

export function subscribe(jobId, fromIndex = 0) {
  closeStream();
  state.eventCount = fromIndex;
  state.lastSseAt = Date.now();
  state.lastSseLogAt = 0;
  state.lastProgressAt = 0;
  const source = new EventSource(`/api/jobs/${jobId}/events?from_index=${fromIndex}`);
  state.es = source;
  state.streamedJobId = jobId;
  source.onmessage = (event) => {
    let payload;
    try { payload = JSON.parse(event.data); } catch (_) { return; }
    state.eventCount += 1;
    state.lastSseAt = Date.now();
    handleJobEvent(payload);
  };
  source.onerror = () => {
    state.lastSseAt = 0;
    if (source.readyState === EventSource.CLOSED) {
      if (state.job?.id === jobId && state.job.status === 'running') {
        pushLog(t('common.msg40'), 'dim');
        state.es = null;
        state.streamedJobId = null;
      }
    }
  };
  startJobFallback(jobId);
}

function ingestProgress(event) {
  const step = Number(event.step);
  if (!Number.isFinite(step)) return false;
  state.series.train.set(step, Number(event.loss));
  state.series.avg.set(step, Number(event.avg));
  state.series.meta.set(step, {
    lr: Number(event.lr), sec: Number(event.sec), vram: Number(event.vram),
    total: Number(event.total), loss: Number(event.loss), avg: Number(event.avg),
  });
  state.trainTotal = Number(event.total) || state.trainTotal;
  state.liveTrain = true;
  updateTrainMetrics();
  syncTrainChart();
  return true;
}

function rebuildLogFromText(text) {
  state.logs = [];
  const box = $('#log');
  if (box) box.innerHTML = '';
  for (const line of String(text || '').split('\n')) {
    if (!line) continue;
    const cls = line.startsWith('$ ') ? 'sys'
      : /traceback|exception|error|failed|out of memory|CUDA error/i.test(line) ? 'err' : '';
    pushLog(line, cls);
  }
  state.logFileText = String(text || '');
}

async function fallbackTick(jobId) {
  if (state.fallbackBusy || !state.job || state.job.id !== jobId || state.job.status !== 'running') return;
  state.fallbackBusy = true;
  try {
    const now = Date.now();
    if (now - (state.lastProgressAt || 0) > 2400) {
      const jobs = await api('/api/jobs');
      const remote = jobs.find((item) => item.id === jobId);
      if (remote?.last && remote.last.type === 'progress') {
        const step = Number(remote.last.step);
        if (Number.isFinite(step) && !state.series.train.has(step)) {
          ingestProgress(remote.last);
          state.job.last = remote.last;
        }
      }
    }
    if (now - (state.lastSseLogAt || 0) > 2400) {
      const text = await api(`/api/jobs/${jobId}/log`);
      if (text && text !== state.logFileText) rebuildLogFromText(text);
    }
  } catch (_) {
    // transient while the file/app is starting
  } finally {
    state.fallbackBusy = false;
  }
}

function startJobFallback(jobId) {
  stopJobFallback();
  state.fallbackTimer = setInterval(() => fallbackTick(jobId), 1200);
}

function stopJobFallback() {
  if (state.fallbackTimer) {
    clearInterval(state.fallbackTimer);
    state.fallbackTimer = null;
  }
  state.fallbackBusy = false;
}

function stopJobFallbackOnDone() {
  stopJobFallback();
}

function handleJobEvent(event) {
  switch (event.type) {
    case 'start':
      state.lastSseLogAt = Date.now();
      pushLog(`$ ${event.command}`, 'sys');
      break;
    case 'log': {
      state.lastSseLogAt = Date.now();
      const line = String(event.line || '');
      const className = /traceback|exception|error|failed|out of memory|CUDA error/i.test(line) ? 'err' : '';
      pushLog(line, className);
      break;
    }
    case 'progress':
      state.lastProgressAt = Date.now();
      ingestProgress(event);
      // The stream is the fastest source there is, so it also owns the job's
      // step. Without this the fallback timer saw fresh progress, skipped its
      // fetch, and the panel showed a step from the last slow poll.
      if (state.job && event.step != null) {
        state.job.last = { ...event };
        updateJobUi();
      }
      break;
    case 'samples':
      pushLog(event.line || t('job.msg12'), 'ok');
      if (currentTab() === 'dashboard') loadDashboard(state.dashboard.run, { silent: true }).catch(() => {});
      break;
    case 'stage':
      if (state.job) state.job.stage = event.stage || '';
      if (event.stage === 'baseline' && event.status === 'running') {
        pushLog(t('job.msg13'), 'sys');
      } else if (event.stage === 'baseline' && event.status === 'done') {
        pushLog(t('job.msg14'), 'ok');
      } else if (event.stage === 'train' && event.status === 'running') {
        pushLog(t('job.msg15'), 'sys');
      }
      updateJobUi();
      renderActivity();
      break;
    case 'warning':
      if (state.job) state.job.baselineError = event.message || '';
      pushLog(t('job.baselineWarningLog', { message: event.message || t('job.baselineFailedContinue') }), 'err');
      showAlert({
        type: 'warn',
        title: event.title || t('job.msg16'),
        text: event.message || t('job.msg17'),
        timeout: 9000,
        key: 'baseline-warning',
      });
      break;
    case 'done':
      finishJob('done', event.message || '');
      break;
    case 'error':
      if (state.job?.status === 'stopped') finishJob('stopped', event.message || '');
      else finishJob('error', event.message || '');
      break;
    case 'closed':
      closeStream();
      stopJobFallback();
      break;
    default:
      break;
  }
}

export async function finishJob(status, message) {
  if (!state.job) return;
  const job = state.job;
  job.status = status;
  closeStream();
  stopJobFallback();
  stopSamplePolling();
  pushLog(`[job] ${status}${message ? ' · ' + message : ''}`, status === 'error' ? 'err' : 'ok');
  if (isCaptionJob(job.kind)) finalizeCaptionProgress(job, status);
  if (job.kind === 'cache' && status === 'done') completeCacheProgress();
  updateJobUi();
  const captionJob = isCaptionJob(job.kind);
  if (status === 'done') {
    const label = jobLabel(job.kind);
    showSuccess(t('job.kindDone', { kind: label }), captionJob ? t('job.msg18') : job.kind === 'train' && job.name ? t('job.trainFinished', { run: job.label || ((job.profile ? job.profile + '/' : '') + job.name) }) : job.kind === 'cache' ? t('job.msg19') : t('job.msg20'), [
      { label: job.kind === 'export' ? t('job.msg21') : job.kind === 'train' ? t('nav.dashboard') : captionJob ? t('job.msg22') : t('job.msg23'), onClick: () => {
        if (job.kind === 'train') { location.hash = '#dashboard'; }
        else if (job.kind === 'export') { location.hash = '#export'; }
        else if (captionJob) { location.hash = '#caption'; refreshDatasetAfterCaption(); }
        else { loadCache().catch(() => {}); }
      } },
    ], `job-done-${job.id}`);
  } else if (status === 'error') {
    const errorText = logErrorMessage() || message || t('job.msg24');
    showError(t('job.kindFailed', { kind: jobLabel(job.kind) }), errorText, [
      { label: t('job.viewLog'), onClick: focusLogPanel },
      { label: t('job.msg25'), onClick: copyLog },
    ], `job-error-${job.id}`);
  } else {
    showAlert({ type: 'info', title: t('job.kindStopped', { kind: jobLabel(job.kind) }), text: message || t('job.msg26'), timeout: 4000 });
  }
  if (job.kind === 'train') {
    state.dashboard.run = job.label || state.dashboard.run;
    loadDashboard(state.dashboard.run, { silent: true }).catch(() => {});
    loadWorkspaceRuns().catch(() => {});
  }
  if (job.kind === 'cache') {
    loadCache().catch(() => {});
  }
  if (captionJob) refreshDatasetAfterCaption();
  if (job.kind === 'export') {
    updateExportStatus(status === 'done' ? t('job.msg27') : status === 'error' ? t('job.msg28') : t('job.msg7'), status === 'done' ? 'chip-ok' : status === 'error' ? 'chip-err' : 'chip-muted');
    loadLoras().catch(() => {});
  }
  loadSamples().catch(() => {});
  if (job.kind !== 'export') loadLoras().catch(() => {});
}

export async function startJob(kind, options = {}) {
  if (!state.profileName) {
    showAlert({
      type: 'warn', title: t('job.msg29'),
      text: t('job.msg30'),
      timeout: 6000,
    });
    return false;
  }
  if (state.job?.status === 'running') {
    showAlert({
      type: 'warn', title: t('job.msg31'),
      text: t('job.msg32'),
      persistent: true,
    });
    return false;
  }
  if (state.busy) return false;
  const captionSelection = kind === 'caption' ? readCaptionSelection() : null;
  const cacheSelection = kind === 'cache' ? readCacheSelection() : null;
  if (kind === 'caption' && (!captionSelection.datasets.length || !captionSelection.styles.length)) {
    updateCaptionStartState();
    showAlert({
      type: 'warn',
      title: !captionSelection.datasets.length ? t('caption.selectDataset') : t('job.msg33'),
      text: !captionSelection.datasets.length
        ? t('job.msg34')
        : t('job.msg35'),
      timeout: 5000,
    });
    return false;
  }
  if (kind === 'cache' && !cacheSelection.parts.length) {
    updateCacheStartState();
    showAlert({
      type: 'warn',
      title: t('cache.selectOne'),
      text: t('job.msg36'),
      timeout: 5000,
    });
    return false;
  }
  state.busy = true;
  updateJobUi();
  try {
    const yamlText = getCurrentYaml();
    try {
      parseYaml(yamlText);
      clearYamlError();
    } catch (error) {
      showYamlError(error);
      showError(t('job.msg37'), error.line
        ? t('job.yamlSyntaxErrorWithLine', { error: error.message, line: error.line })
        : t('job.yamlSyntaxError', { error: error.message }));
      return false;
    }
    if (kind === 'train') {
      // The dashboard GC selector is persisted right before the job starts, so
      // the child process sees the same allocator the user selected.
      const saved = await saveSettings({ silent: true });
      if (!saved) return false;
      const preflight = await runPreflight({ yamlText, forStart: true });
      if (!preflight.ok) {
        const nextTab = preflight.data?.next_tab;
        if (nextTab) {
          showAlert({
            type: 'warn', title: t('job.msg38'),
            text: nextTab === 'train' ? t('job.msg39')
              : nextTab === 'dataset' ? t('job.msg40')
              : nextTab === 'cache' ? t('job.msg41')
              : t('job.msg42'),
            timeout: 6000, key: 'start-missing',
          });
          location.hash = `#${nextTab}`;
          if (preflight.data?.focus_field) {
            setTimeout(() => focusField(preflight.data.focus_field), 150);
          }
        }
        return false;
      }
    }

    if (state.activeRun) {
      // With a run selected the server reads runs/<run>/config.yaml from disk,
      // so a value typed on a page and never saved would be ignored by the job
      // that was just started. Persist it first, and stop if that fails.
      const persisted = await saveConfig({ silent: true });
      if (!persisted) return false;
    }

    const data = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind,
        // The run names the config; the server reads runs/<run>/config.yaml.
        // Caption jobs carry their own settings and need no run.
        ...(kind !== 'caption' && state.activeRun ? { run: state.activeRun } : { yaml: yamlText }),
        workspace: state.profileName,
        ...(kind === 'train' ? { baseline: Boolean($('#cfg-run-baseline')?.checked) } : {}),
        ...(kind === 'caption' ? {
          datasets: captionSelection.datasets,
          styles: captionSelection.styles,
          overwrite: captionSelection.overwrite,
        } : {}),
        ...(kind === 'cache' ? {
          parts: cacheSelection.parts,
          overwrite: cacheSelection.overwrite,
        } : {}),
      }),
    });
    state.logs = [];
    clearLog();
    state.eventCount = 0;
    state.cache.progress = null;
    const runLabel = data.workspace && data.name ? `${data.workspace}/${data.name}` : (data.name || data.id);
    if (kind === 'train') state.dashboard.run = runLabel;
    state.job = {
      id: data.id,
      kind,
      status: 'running',
      name: data.name || '',
      runName: data.run_id || data.name || '',
      label: runLabel,
      profile: data.profile || state.profileName,
      workspace: data.workspace || state.profileName,
      outputDir: data.output_dir || '',
      logPath: data.log_path || '',
      configPath: data.config_path || '',
      baseline: Boolean(data.baseline),
      baselineError: '',
      stage: '',
      created: Date.now() / 1000,
    };
    state.jobInfoFor = null;
    if (isCaptionJob(kind)) beginCaptionProgress(state.job, captionSelection);
    if (kind === 'cache') beginCacheProgress(cacheSelection.parts);
    if (kind === 'train') {
      resetTrainSeries();
      state.currentRun = {
        name: data.run_id || data.name || data.id,
        label: runLabel,
        profile: data.profile || state.profileName,
        workspace: data.workspace || state.profileName,
        outputDir: data.output_dir || '',
      };
      state.liveTrain = true;
      hidePreflight();
      startSamplePolling();
    }
    updateJobUi();
    updateTrainRunState();
    subscribe(data.id, 0);
    applyJobRunInfo(state.job).catch(() => {});
    const runHint = kind === 'train' && data.name ? t('job.trainRunHint', { run: `${data.workspace ? data.workspace + '/' : ''}${data.name}` }) : t('job.msg43');
    showSuccess(t('job.kindStarted', { kind: jobLabel(kind) }), runHint, [
      { label: t('job.msg44'), onClick: () => { location.hash = '#dashboard'; } },
    ], `job-start-${data.id}`);
    if (kind === 'train') {
      setTimeout(() => { if (currentTab() === 'train') location.hash = '#dashboard'; }, 450);
    }
    return true;
  } catch (error) {
    if (error.status === 409) {
      showError(t('job.msg31'), describeError(error), [
        { label: t('job.msg45'), onClick: () => loadJobs().catch(() => {}) },
      ]);
      await loadJobs().catch(() => {});
    } else {
      const actions = [];
      if (['err.cache.not_built', 'err.cache.legacy', 'err.cache.missing_text'].includes(error.code)
          || /index\.json|preview\.safetensors/i.test(describeError(error))) {
        actions.push({ label: t('job.msg46'), onClick: () => { location.hash = '#cache'; } });
      }
      if (error.line) {
        showYamlError({ message: describeError(error), line: error.line, column: error.column });
        actions.push({ label: t('job.msg47'), onClick: () => focusYamlLine(error.line) });
      }
      if (kind === 'train') {
        setPreflight({
          level: 'error',
          title: t('job.msg48'),
          text: describeError(error),
          actions,
        });
      }
      showError(t('job.kindStartFailed', { kind: jobLabel(kind) }), describeError(error), actions);
    }
    return false;
  } finally {
    state.busy = false;
    updateJobUi();
  }
}

async function stopActiveTraining() {
  const active = state.activeTraining;
  if (!active) return false;
  if (!window.confirm(t('train.stopConfirm', { run: active.run }))) return false;
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/stop', { method: 'POST' });
    if (!data.ok) {
      showError(t('train.msg53'), t('train.msg54'));
      await loadActiveTraining().catch(() => {});
      return false;
    }
    pushLog(t('train.msg55'), 'ok');
    await loadActiveTraining().catch(() => {});
    showAlert({ type: 'info', title: t('train.msg55'), text: t('train.stoppedProcesses', { count: data.stopped?.length || 0 }), timeout: 4000 });
    setTimeout(async () => {
      await loadActiveTraining().catch(() => {});
      pushLog(t('train.msg55'), 'ok');
    }, 1200);
    return true;
  } catch (error) {
    showError(t('train.msg56'), describeError(error));
    return false;
  }
}

export async function stopJob() {
  if (state.activeTraining) { await stopActiveTraining(); return; }
  if (!state.job || state.job.status !== 'running') {
    // A cache build published to disk is not in this page's registry, but it is
    // still a job with a pid; stopping it must not fall through to "nothing to
    // stop".
    if (state.cache.jobRunning) { await stopCacheBuild(); return; }
    showAlert({ type: 'info', title: t('job.msg49'), text: t('job.msg50'), timeout: 3000 });
    return;
  }
  try {
    const data = await api(`/api/jobs/${state.job.id}/stop`, { method: 'POST' });
    if (data.ok) {
      state.job.status = 'stopped';
      pushLog(t('job.msg51'), 'err');
      updateJobUi();
      showAlert({ type: 'info', title: t('train.msg55'), text: t('job.msg52'), timeout: 4000 });
    } else {
      showAlert({ type: 'warn', title: t('job.msg53'), text: t('job.msg50'), timeout: 4000 });
      await loadJobs().catch(() => {});
    }
  } catch (error) {
    showError(t('job.msg54'), describeError(error), [
      { label: t('common.retry'), onClick: () => stopJob() },
    ]);
  }
}
