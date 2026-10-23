/* =========================================================================
 * js/components/logview.js — Log console
 * split from web/static/app.js, source 3088-3965 (25 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { renderActivity } from './jobs.js';
import { api, describeError } from '../core/api.js';
import { copyText } from '../core/clipboard.js';
import { $, clamp, escapeHtml, setHidden, setText } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { MAX_LOG, isCaptionJob, state } from '../core/state.js';
import { currentTab } from '../main.js';
import { renderCacheJob } from '../pages/cache.js';

export function isLogNearBottom(box) {
  return box.scrollHeight - box.scrollTop - box.clientHeight < 48;
}

export function updateLogCount() {
  setText('#log-count', t('common.lines', { count: state.logs.length }));
}

function appendLog(line, className = '') {
  const box = $('#log');
  if (!box) return;
  const nearBottom = isLogNearBottom(box);
  const row = document.createElement('div');
  row.className = `line${className ? ' ' + className : ''}`;
  row.textContent = line;
  box.append(row);
  while (box.childElementCount > MAX_LOG) box.removeChild(box.firstElementChild);
  if (!state.logPaused && nearBottom) box.scrollTop = box.scrollHeight;
  setHidden('#log-jump', nearBottom || state.logPaused);
}

export function pushLog(line, className = '') {
  const text = String(line == null ? '' : line);
  state.logs.push({ line: text, className });
  if (state.logs.length > MAX_LOG) state.logs.splice(0, state.logs.length - MAX_LOG);
  appendLog(text, className);
  updateLogCount();
  if (state.job && state.job.status === 'running' && state.job.kind && state.job.kind !== 'train') {
    if (isCaptionJob(state.job.kind)) syncCaptionProgressFromLogs();
    else if (state.job.kind === 'cache') parseCacheProgress(text);
  }
}

function rebuildLog(pinBottom = true) {
  const box = $('#log');
  if (!box) return;
  const nearBottom = isLogNearBottom(box);
  box.innerHTML = '';
  const fragment = document.createDocumentFragment();
  for (const entry of state.logs) {
    const row = document.createElement('div');
    row.className = `line${entry.className ? ' ' + entry.className : ''}`;
    row.textContent = entry.line;
    fragment.append(row);
  }
  box.append(fragment);
  if (pinBottom && nearBottom) box.scrollTop = box.scrollHeight;
  updateLogCount();
  setHidden('#log-jump', nearBottom && !state.logPaused);
}

export function shortRunName(value) {
  return String(value || '').split('/').pop();
}

export function currentLogRun() {
  return shortRunName(state.dashboard.data?.run || $('#dash-run')?.value || state.activeTraining?.run || '');
}

export async function loadRunLog(run, { force = false } = {}) {
  const name = shortRunName(String(run || '').trim());
  if (!name || !state.profileName) return;
  if (state.runLog.loading) return;
  state.runLog.loading = true;
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/log?run=' + encodeURIComponent(name) + '&tail=200');
    const lines = Array.isArray(data.lines) ? data.lines : [];
    const text = lines.join('\n');
    if (!force && state.runLog.run === name && state.runLog.text === text) return;
    const first = state.runLog.run !== name;
    state.runLog.run = name;
    state.runLog.text = text;
    state.logs = lines.map((line) => ({
      line,
      className: /traceback|exception|error|failed|out of memory|CUDA error/i.test(line) ? 'err' : line.startsWith('$ ') ? 'cmd' : '',
    }));
    if (first) {
      const box = $('#log');
      if (box) box.scrollTop = 0;
    }
    rebuildLog(true);
    if (!data.exists && !lines.length) {
      state.logs = [{ line: t('dash.log.missingRun', { name }), className: 'dim' }];
      rebuildLog(true);
    }
  } catch (error) {
    if (!state.runLog.text) {
      state.logs = [{ line: t('dash.log.readFailed', { error: describeError(error) }), className: 'err' }];
      rebuildLog(true);
    }
  } finally {
    state.runLog.loading = false;
  }
}

export async function refreshRunLog() {
  if (currentTab() !== 'dashboard') return;
  if (state.job?.status === 'running') return;
  const run = currentLogRun();
  if (run) await loadRunLog(run);
}

export function clearLog() {
  state.logs = [];
  const box = $('#log');
  if (box) box.innerHTML = '';
  updateLogCount();
  setHidden('#log-jump', true);
}

export function copyLog() {
  copyText(state.logs.map((entry) => entry.line).join('\n'), t('job.msg2'));
}

export function logErrorMessage() {
  const errors = state.logs.filter((entry) => entry.className === 'err').slice(-3);
  const lines = errors.length ? errors : state.logs.slice(-3);
  return lines.map((entry) => entry.line).join('\n');
}

export function cacheStageLabel(part) {
  return part === 'text' ? t('cache.textEncode') : part === 'latent' ? t('cache.latentPart') : t('nav.cache');
}

export function cacheStagePercent(stage) {
  const total = Number(stage?.total) || 0;
  const done = Number(stage?.done) || 0;
  if (total > 0) return clamp((done / total) * 100, 0, 100);
  return stage?.complete ? 100 : 0;
}

function setCacheStage(part, patch) {
  const previous = state.cache.stages?.[part] || {};
  const stage = {
    done: 0,
    total: 0,
    rate: '',
    unit: '',
    pending: 0,
    line: '',
    started: false,
    complete: false,
    ...previous,
    ...patch,
  };
  state.cache.stages[part] = stage;
  state.cache.active = part;
  state.cache.progress = stage;
  renderCacheJob();
  renderActivity();
}

function resetCacheProgress(parts = []) {
  state.cache.parts = Array.isArray(parts) ? parts.filter((part) => part === 'latent' || part === 'text') : [];
  state.cache.stages = { latent: null, text: null };
  state.cache.active = state.cache.parts[0] || '';
  state.cache.progress = null;
}

export function beginCacheProgress(parts = []) {
  resetCacheProgress(parts);
  renderCacheJob();
}

export function completeCacheProgress() {
  const parts = state.cache.parts?.length
    ? state.cache.parts.slice()
    : ['latent', 'text'].filter((part) => state.cache.stages?.[part]?.started);
  state.cache.parts = parts;
  for (const part of parts) {
    const stage = state.cache.stages?.[part] || {};
    const total = Number(stage.total) || 0;
    state.cache.stages[part] = {
      ...stage,
      done: total,
      total,
      rate: '',
      pending: 0,
      complete: true,
    };
  }
  state.cache.active = '';
  state.cache.progress = null;
}

function parseCacheProgress(line) {
  if (!line) return;
  let match = line.match(/^latents (\d+)\/(\d+)\s+([\d.]+)\s+(img\/s)/);
  if (match) {
    const done = parseInt(match[1], 10);
    const total = parseInt(match[2], 10);
    setCacheStage('latent', {
      done, total, rate: match[3], unit: match[4], line,
      started: true, complete: total > 0 && done >= total,
    });
    return;
  }
  match = line.match(/^text (\d+)\/(\d+)\s+([\d.]+)\s+(cap\/s)/);
  if (match) {
    const done = parseInt(match[1], 10);
    const total = parseInt(match[2], 10);
    setCacheStage('text', {
      done, total, rate: match[3], unit: match[4], line,
      started: true, complete: total > 0 && done >= total,
    });
    return;
  }
  // "text: 32/624 captions" announces pending/total BEFORE work starts,
  // so this is a 0% progress marker, not done=32.
  match = line.match(/^text: (\d+)\/(\d+) captions/);
  if (match) {
    setCacheStage('text', {
      done: 0, total: parseInt(match[2], 10),
      pending: parseInt(match[1], 10), rate: '', unit: t('cache.msg5'), line,
      started: true, complete: false,
    });
    return;
  }
  // "latents: 1234 images" is the pending count; start from zero.
  match = line.match(/^latents: (\d+) images/);
  if (match) {
    setCacheStage('latent', {
      done: 0, total: parseInt(match[1], 10),
      pending: parseInt(match[1], 10), rate: '', unit: t('common.imagesUnit'), line,
      started: true, complete: false,
    });
    return;
  }
  if (/^latents: nothing to do/.test(line)) {
    setCacheStage('latent', {
      done: 0, total: 0, pending: 0, rate: '', unit: '', line,
      started: true, complete: true,
    });
    return;
  }
  if (/^index: .* items\)/.test(line)) {
    // encode-text also writes an index at the end; only treat it as latent
    // completion while latent is the active stage (text-only runs must leave
    // the latent bar at 0%).
    if (state.cache.active !== 'text' && (state.cache.parts.includes('latent') || state.cache.stages.latent)) {
      const stage = state.cache.stages.latent || {};
      setCacheStage('latent', {
        done: Number(stage.total) || 0,
        total: Number(stage.total) || 0,
        rate: '', unit: '', line, complete: true,
      });
    }
  }
}

function parseCaptionLog(lines) {
  const phases = [];
  let current = null;
  const startPhase = (total = 0) => {
    current = {
      total: Number(total) || 0,
      done: 0,
      ok: 0,
      skip: 0,
      fail: 0,
      failures: [],
      current: '',
      rate: '',
      role: '',
      workers: null,
      started: false,
    };
    phases.push(current);
  };
  for (const raw of lines || []) {
    const line = String(raw == null ? '' : raw).trim();
    if (!line) continue;
    const header = line.match(/(?:^|\s)images:\s*(\d+)(?:\s|$)/);
    if (header) {
      startPhase(Number(header[1]));
      const role = line.match(/(?:^|\s)role:\s*(train|reg)(?:\s|$)/);
      const workers = line.match(/(?:^|\s)workers:\s*(\d+)/);
      if (role) current.role = role[1];
      if (workers) current.workers = Number(workers[1]);
      continue;
    }
    const progress = line.match(/^(\d+)\/(\d+)\s+(ok|skip)\s+(.+?)\s+([\d.]+)\s+img\/s\s*$/i);
    if (progress) {
      const total = Number(progress[2]);
      if (!current) startPhase(total);
      else if (current.total && total !== current.total) {
        if (current.started || current.done > 0) startPhase(total);
        else current.total = total;
      } else if (!current.total) {
        current.total = total;
      }
      const done = Number(progress[1]);
      current.done = Math.max(current.done, done);
      if (progress[3].toLowerCase() === 'skip') current.skip += 1;
      else current.ok += 1;
      current.current = progress[4].trim();
      current.rate = progress[5];
      current.started = true;
      continue;
    }
    const fail = line.match(/^FAIL\s+(.+)$/);
    if (fail) {
      if (!current) startPhase(0);
      const body = fail[1];
      const split = body.indexOf(': ');
      const file = (split >= 0 ? body.slice(0, split) : body).trim();
      const reason = (split >= 0 ? body.slice(split + 2) : '').trim();
      current.fail += 1;
      current.current = file;
      current.failures.push({ file, reason });
      current.started = true;
    }
  }
  for (const phase of phases) {
    phase.done = Math.max(phase.done, phase.ok + phase.skip + phase.fail);
    if (phase.total > 0) phase.done = Math.min(phase.done, phase.total);
  }
  const used = phases.filter((phase) => phase.started || phase.done || phase.total);
  const aggregate = { total: 0, ok: 0, skip: 0, fail: 0, failures: [], current: '', rate: '' };
  for (const phase of used) {
    aggregate.total += phase.total;
    aggregate.ok += phase.ok;
    aggregate.skip += phase.skip;
    aggregate.fail += phase.fail;
    aggregate.failures.push(...phase.failures);
  }
  const last = used[used.length - 1] || null;
  if (last) {
    aggregate.current = last.current;
    aggregate.rate = last.rate;
    aggregate.role = last.role;
    aggregate.workers = last.workers;
  }
  return { phases: used, current: last, aggregate };
}

function captionPhaseLabel(index) {
  const phase = state.captionProgress?.phases?.[index] || state.captionProgress?.current;
  if (phase?.role === 'reg') return t('dataset.reg.title');
  if (phase?.role === 'train') return t('caption.datasets.train');
  const datasets = state.captionProgress?.datasets || [];
  if (datasets.length === 1) return datasets[0] === 'reg' ? t('dataset.reg.title') : t('caption.datasets.train');
  return index > 0 ? t('dataset.reg.title') : t('caption.datasets.train');
}

function syncCaptionProgressFromLogs() {
  const lines = (state.logs || []).map((entry) => (typeof entry === 'string' ? entry : entry.line));
  const parsed = parseCaptionLog(lines);
  const progress = state.captionProgress;
  progress.phases = parsed.phases;
  progress.current = parsed.current;
  progress.aggregate = parsed.aggregate;
  if (parsed.aggregate.role) progress.role = parsed.aggregate.role;
  if (parsed.aggregate.workers != null) progress.workers = parsed.aggregate.workers;
  renderCaptionJobPanel();
  renderActivity();
}

export function beginCaptionProgress(job, selection = null) {
  state.captionProgress = {
    jobId: job?.id || '',
    kind: job?.kind || '',
    status: 'running',
    phases: [],
    current: null,
    aggregate: { total: 0, ok: 0, skip: 0, fail: 0, failures: [], current: '', rate: '' },
    role: '',
    workers: null,
    datasets: Array.isArray(selection?.datasets) ? selection.datasets.slice() : [],
    styles: Array.isArray(selection?.styles) ? selection.styles.slice() : [],
  };
  renderCaptionJobPanel();
  renderActivity();
}

export function finalizeCaptionProgress(job, status) {
  const lines = (state.logs || []).map((entry) => (typeof entry === 'string' ? entry : entry.line));
  const parsed = parseCaptionLog(lines);
  const previous = state.captionProgress || {};
  state.captionProgress = {
    jobId: job?.id || '',
    kind: job?.kind || '',
    status,
    phases: parsed.phases,
    current: parsed.current,
    aggregate: parsed.aggregate,
    role: parsed.aggregate.role || previous.role || '',
    workers: parsed.aggregate.workers ?? previous.workers ?? null,
    datasets: previous.datasets || [],
    styles: previous.styles || [],
  };
  renderCaptionJobPanel();
}

export function renderCaptionJobPanel() {
  const panel = $('#caption-job-panel');
  if (!panel) return;
  const progress = state.captionProgress;
  const status = progress.status || 'idle';
  const running = status === 'running';
  const chip = $('#caption-job-chip');
  const chipMap = {
    idle: ['chip-muted', t('common.idle')],
    running: ['chip-run', t('caption.msg1')],
    done: ['chip-ok', t('caption.msg2')],
    error: ['chip-err', t('caption.msg3')],
    stopped: ['chip-muted', t('job.msg7')],
  };
  const [chipClass, chipText] = chipMap[status] || chipMap.idle;
  if (chip) {
    chip.className = `chip ${chipClass}`;
    chip.textContent = chipText;
  }
  setHidden('#btn-caption-stop', !running);
  setHidden('#caption-job-live', !running);
  setHidden('#caption-job-result', running || status === 'idle');

  const desc = $('#caption-job-desc');
  if (desc) {
    if (running) {
      const phaseIndex = Math.max(0, (progress.phases?.length || 1) - 1);
      const meta = progress.workers != null ? `${progress.workers} workers` : '';
      desc.textContent = progress.current
        ? `${captionPhaseLabel(phaseIndex)}${meta ? ` · ${meta}` : ''}`
        : t('caption.msg4');
    } else if (status === 'done') {
      desc.textContent = t('caption.msg5');
    } else if (status === 'error') {
      desc.textContent = t('caption.msg6');
    } else if (status === 'stopped') {
      desc.textContent = t('caption.msg7');
    } else {
      desc.textContent = t('caption.job.desc');
    }
  }

  if (running) {
    const current = progress.current;
    const total = Number(current?.total) || 0;
    const done = Number(current?.done) || 0;
    const percent = total ? clamp((done / total) * 100, 0, 100) : 0;
    const failed = Number(current?.fail) || 0;
    setText('#caption-job-count', `${fmtInt(done)} / ${fmtInt(total)}`);
    setText('#caption-job-status', t('caption.job.progressStatus', { done: fmtInt(done), total: fmtInt(total), rate: current?.rate ? t('caption.job.rate', { rate: current.rate }) : t('caption.job.noRate'), failed: fmtInt(failed) }));
    setText('#caption-job-current', t('caption.job.current', { current: current?.current || t('caption.job.waitingLog') }));
    const bar = $('#caption-job-bar');
    if (bar) bar.style.width = `${percent.toFixed(1)}%`;
    const barWrap = $('#caption-job-progress');
    if (barWrap) barWrap.setAttribute('aria-valuenow', String(Math.round(percent)));
    return;
  }

  const aggregate = progress.aggregate || { total: 0, ok: 0, skip: 0, fail: 0, failures: [] };
  const failures = aggregate.failures || [];
  const hasData = aggregate.total > 0 || aggregate.ok > 0 || aggregate.skip > 0 || aggregate.fail > 0;
  setText('#caption-job-summary', hasData
    ? t('caption.job.summary', { total: fmtInt(aggregate.total), ok: fmtInt(aggregate.ok), skip: fmtInt(aggregate.skip), fail: fmtInt(aggregate.fail) })
    : t('caption.msg8'));
  setHidden('#caption-failures', !failures.length);
  if (failures.length) {
    setText('#caption-failures-summary', t('caption.failuresWithCount', { count: fmtInt(failures.length) }));
    const list = $('#caption-failure-list');
    if (list) {
      list.innerHTML = failures.slice(0, 20).map((failure) => `<li><span class="caption-failure-file">${escapeHtml(failure.file || t('caption.unknownFile'))}</span><span class="caption-failure-reason">${escapeHtml(failure.reason || t('caption.unknownReason'))}</span></li>`).join('');
    }
    setHidden('#caption-failures-more', failures.length <= 20);
    if (failures.length > 20) setText('#caption-failures-more', t('caption.failuresMore', { count: fmtInt(failures.length) }));
  }
}
