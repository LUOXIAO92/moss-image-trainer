/* =========================================================================
 * js/pages/cache.js — Cache
 * split from web/static/app.js, source 6549-6858 (11 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { startJob } from '../components/jobsse.js';
import { cacheStageLabel, cacheStagePercent } from '../components/logview.js';
import { saveConfig } from '../config/file.js';
import { markDirty, refreshYamlFromConfig, showYamlError, updateYamlLineCount } from '../config/form.js';
import { defaultCacheDir } from '../config/model.js';
import { dumpYaml, parseYaml } from '../config/yaml.js';
import { api } from '../core/api.js';
import { $, getPath, setHidden, setPath, setText } from '../core/dom.js';
import { fmtDuration, fmtInt } from '../core/fmt.js';
import { jobLabel, t } from '../core/i18n.js';
import { showAlert, showSuccess } from '../core/notify.js';
import { CAPTION_SOURCES, state, workspaceParam } from '../core/state.js';
import { refreshDatasetAfterCaption } from './dataset.js';

function cachePath() {
  return ($('#cfg-cache-dir')?.value || defaultCacheDir() || getPath(state.config, 'cache.dir') || '').trim();
}

function cacheRegPath() {
  const input = $('#cfg-cache-reg-dir');
  return input ? String(input.value || '').trim() : String(getPath(state.config, 'cache.reg_dir') || '').trim();
}

function applyCaptionSource(value) {
  const source = CAPTION_SOURCES.includes(value) ? value : 'tag_caption';
  const select = $('#cfg-caption-source');
  if (select) select.value = source;
  if (!state.config || typeof state.config !== 'object') state.config = {};
  if (state.mode === 'yaml') {
    try {
      const parsed = parseYaml($('#yaml')?.value || '');
      setPath(parsed, 'dataset.moss_caption_source', source);
      state.config = parsed;
      const textarea = $('#yaml');
      if (textarea) textarea.value = dumpYaml(parsed);
      updateYamlLineCount();
    } catch (error) {
      showYamlError(error);
      return false;
    }
  } else {
    setPath(state.config, 'dataset.moss_caption_source', source);
    refreshYamlFromConfig();
  }
  markDirty();
  return true;
}

export async function saveCaptionSource(value) {
  if (!applyCaptionSource(value)) return false;
  const saved = await saveConfig({ silent: true });
  if (saved) {
    showSuccess(t('caption.msg12'), t('caption.msg13'), [], 'caption-source-saved');
    refreshDatasetAfterCaption();
  }
  return saved;
}

export function renderCacheFromData(data) {
  if (!data) return;
  const indexReady = Boolean(data.index_ready ?? data.ready);
  const legacy = Boolean(data.legacy_index);
  const ready = Boolean(data.ready) && indexReady && !legacy;
  const missingText = Math.max(0, Number(data.missing_text) || 0);
  const textMissing = !legacy && missingText > 0;
  const countsVisible = indexReady && !legacy;
  const previewReady = Boolean(data.preview_ready);
  const items = Number(data.items) || 0;
  const trainItems = Number.isFinite(Number(data.train_items))
    ? Number(data.train_items)
    : Math.max(0, items - (Number(data.reg_items) || 0));
  const regItems = Number(data.reg_items) || 0;
  const trainFiles = Number.isFinite(Number(data.train_files)) ? Number(data.train_files) : null;
  const regFiles = Number.isFinite(Number(data.reg_files)) ? Number(data.reg_files) : null;
  const regInput = $('#cfg-reg-path');
  const regConfigured = Boolean(String(regInput ? regInput.value : getPath(state.config, 'dataset.reg_path') || '').trim());
  const pill = $('#cache-ready-pill');
  if (pill) pill.className = `status-pill ${legacy || textMissing ? 'is-warn' : ready ? 'is-ok' : 'is-muted'}`;
  setText('#cache-ready-text', legacy
    ? t('cache.msg6')
    : textMissing
      ? t('cache.msg7')
      : ready
        ? (previewReady ? t('cache.msg8') : t('cache.msg9'))
        : t('cache.msg10'));
  setHidden('#cache-legacy-warning', !legacy);
  setText('#cache-train-path', data.dir_rel || data.dir || '--');
  setText('#cache-train-count', textMissing
    ? t('cache.missingTextCount', { count: fmtInt(missingText) })
    : countsVisible
      ? t('cache.itemCount', { count: fmtInt(trainItems), files: trainFiles == null ? '' : t('cache.fileCountSuffix', { count: fmtInt(trainFiles) }) })
      : '—');
  const trainRow = $('#cache-check-train');
  if (trainRow) trainRow.className = `check-row ${textMissing ? 'is-missing' : 'is-info'}`;
  setHidden('#cache-missing-text-hint', !textMissing);
  if (!regConfigured) {
    setText('#cache-reg-path', t('cache.msg11'));
    setText('#cache-reg-count', '—');
  } else {
    setText('#cache-reg-path', data.reg_dir_rel || data.reg_dir || '--');
    setText('#cache-reg-count', countsVisible
      ? t('cache.itemCount', { count: fmtInt(regItems), files: regFiles == null ? '' : t('cache.fileCountSuffix', { count: fmtInt(regFiles) }) })
      : '—');
  }
  const indexRow = $('#cache-check-index');
  if (indexRow) {
    indexRow.className = `check-row ${legacy ? 'is-info' : indexReady ? 'is-ok' : 'is-missing'}`;
    setText(indexRow.querySelector('.check-state'), legacy ? t('cache.msg12') : indexReady ? t('export.ready') : t('cache.msg13'));
  }
  const previewRow = $('#cache-check-preview');
  if (previewRow) {
    previewRow.className = `check-row ${previewReady ? 'is-ok' : 'is-info'}`;
    setText(previewRow.querySelector('.check-state'), previewReady ? t('cache.msg14') : t('cache.msg15'));
  }
  const next = $('#cache-next');
  if (next) {
    const paragraph = next.querySelector('p');
    const link = next.querySelector('a');
    if (link) link.className = ready ? 'btn btn-primary btn-sm' : 'btn btn-ghost btn-sm';
    if (ready && previewReady) {
      setText(paragraph, t('cache.next.ready', { count: fmtInt(items) }));
      if (link) { link.textContent = t('cache.msg16'); link.href = '#train'; }
    } else if (ready) {
      setText(paragraph, t('cache.next.previewMissing', { count: fmtInt(items) }));
      if (link) { link.textContent = t('cache.backToTrain'); link.href = '#train'; }
    } else if (legacy) {
      setText(paragraph, t('cache.msg17'));
      if (link) { link.textContent = t('cache.msg18'); link.href = '#dataset'; }
    } else if (textMissing) {
      setText(paragraph, t('cache.next.textMissing', { count: fmtInt(items), missing: fmtInt(missingText) }));
      if (link) { link.textContent = t('cache.backToTrain'); link.href = '#train'; }
    } else {
      setText(paragraph, t('cache.msg19'));
      if (link) { link.textContent = t('cache.backToTrain'); link.href = '#train'; }
    }
  }
  renderCacheJob();
}

export async function loadCache() {
  if (!state.profileName) {
    state.cache.data = null;
    state.cache.progress = null;
    applyCacheJob(null);
    setText('#cache-ready-text', t('workspace.emptyTitle'));
    setText('#cache-train-path', '--');
    setText('#cache-train-count', '—');
    setText('#cache-reg-path', '--');
    setText('#cache-reg-count', '—');
    setText('#cache-last-line', t('cache.msg20'));
    setHidden('#cache-legacy-warning', true);
    setHidden('#cache-missing-text-hint', true);
    const trainRow = $('#cache-check-train');
    if (trainRow) trainRow.className = 'check-row is-info';
    const pill = $('#cache-ready-pill');
    if (pill) pill.className = 'status-pill is-muted';
    renderCacheJob();
    return;
  }
  const path = cachePath();
  if (!path) {
    state.cache.data = null;
    state.cache.progress = null;
    applyCacheJob(null);
    setText('#cache-ready-text', t('cache.msg21'));
    const pill = $('#cache-ready-pill');
    if (pill) pill.className = 'status-pill is-warn';
    setText('#cache-train-path', '--');
    setText('#cache-train-count', '—');
    setText('#cache-reg-path', '--');
    setText('#cache-reg-count', '—');
    setText('#cache-last-line', t('cache.msg22'));
    setHidden('#cache-legacy-warning', true);
    setHidden('#cache-missing-text-hint', true);
    const trainRow = $('#cache-check-train');
    if (trainRow) trainRow.className = 'check-row is-info';
    for (const id of ['cache-check-index', 'cache-check-preview']) {
      const row = document.getElementById(id);
      if (row) {
        row.classList.remove('is-ok', 'is-missing');
        setText(row.querySelector('.check-state'), t('dataset.reg.notConfigured'));
      }
    }
    renderCacheJob();
    return;
  }
  try {
    const params = new URLSearchParams({ path, workspace: workspaceParam(), reg_path: cacheRegPath() });
    const data = await api('/api/cache?' + params.toString());
    state.cache.data = data;
    state.cache.error = null;
    applyCacheJob(data.job || null);
    renderCacheFromData(data);
    renderCacheJob();
  } catch (error) {
    state.cache.error = error;
    setText('#cache-ready-text', t('cache.msg23'));
    const pill = $('#cache-ready-pill');
    if (pill) pill.className = 'status-pill is-err';
    setText('#cache-last-line', error.detail || error.message);
    showAlert({ type: 'warn', title: t('cache.msg24'), text: error.detail || error.message, timeout: 6000 });
  }
}

let cacheTimer = null;

// Poll faster than the page's 6 s heartbeat while a build is live, and refresh
// the readiness panel once it settles.
function applyCacheJob(job) {
  const wasRunning = Boolean(state.cache.jobRunning);
  state.cache.job = job;
  state.cache.jobRunning = diskJobRunning(job);
  if (cacheTimer) {
    clearTimeout(cacheTimer);
    cacheTimer = null;
  }
  if (state.cache.jobRunning) {
    cacheTimer = setTimeout(() => { loadCache().catch(() => {}); }, 1500);
  } else if (wasRunning) {
    // the build just ended: the index/preview it wrote are only visible now
    cacheTimer = setTimeout(() => { loadCache().catch(() => {}); }, 800);
  }
  updateCacheStartState();
}

export function readCacheParts() {
  const parts = [];
  if ($('#cache-part-latent')?.checked) parts.push('latent');
  if ($('#cache-part-text')?.checked) parts.push('text');
  return parts;
}

export function updateCacheStartState() {
  const button = $('#btn-cache-start');
  if (!button) return;
  const running = Boolean(state.job && state.job.status === 'running')
    || Boolean(state.activeTraining)
    || diskJobRunning(state.cache.job);
  const busy = running || state.busy;
  for (const id of ['cache-part-latent', 'cache-part-text']) {
    const checkbox = $(`#${id}`);
    if (!checkbox) continue;
    checkbox.disabled = busy;
    const row = checkbox.closest('.cache-part');
    if (row) row.classList.toggle('is-disabled', busy);
  }
  const parts = readCacheParts();
  let hint = '';
  if (!parts.length) hint = t('cache.selectOne');
  else if (!state.profileName) hint = t('job.msg29');
  button.disabled = busy || Boolean(hint);
  button.title = hint || t('cache.msg25');
  setText('#cache-start-hint', hint);
  setHidden('#cache-start-hint', !hint);
}

function renderCacheStage(part, stage) {
  const percent = cacheStagePercent(stage);
  const done = Number(stage?.done) || 0;
  const total = Number(stage?.total) || 0;
  const count = $(`#cache-${part}-count`);
  const bar = $(`#cache-${part}-bar`);
  const progress = $(`#cache-${part}-progress`);
  const row = $(`#cache-stage-${part}`);
  if (count) {
    count.textContent = stage
      ? (total > 0 ? `${fmtInt(done)} / ${fmtInt(total)}` : (stage.complete ? t('caption.msg2') : '0 / 0'))
      : '0 / 0';
  }
  if (bar) bar.style.width = `${percent.toFixed(1)}%`;
  if (progress) progress.setAttribute('aria-valuenow', String(Math.round(percent)));
  if (row) {
    row.classList.toggle('is-active', Boolean(state.job?.status === 'running' && state.job.kind === 'cache' && state.cache.active === part));
    row.classList.toggle('is-complete', Boolean(stage?.complete));
  }
}

// A cache build publishes its progress to the workspace, so a run started from a
// terminal shows up here too. The local stages win while this tab owns the job.
function diskJobRunning(job) {
  return Boolean(job && (job.state === 'running' || job.state === 'starting'));
}

function diskCacheStages(job) {
  const key = String(job.started || 0);
  if (state.cache.diskKey !== key) {
    state.cache.diskKey = key;
    state.cache.diskStages = { latent: null, text: null };
  }
  const stages = state.cache.diskStages;
  const running = diskJobRunning(job);
  const history = job.history || {};
  for (const part of ['latent', 'text']) {
    const entry = history[part];
    if (!entry) continue;
    const total = Number(entry.total) || 0;
    const done = Number(entry.done) || 0;
    stages[part] = {
      done,
      total,
      rate: '',
      unit: entry.unit || (part === 'text' ? 'cap/s' : 'img/s'),
      started: true,
      complete: Boolean(entry.complete) || Boolean(entry.failed) || (total > 0 && done >= total),
    };
  }
  const phase = job.phase === 'text' ? 'text' : job.phase === 'latent' ? 'latent' : '';
  if (phase && running) {
    const total = Number(job.total) || 0;
    const done = Number(job.done) || 0;
    stages[phase] = {
      ...(stages[phase] || {}),
      done,
      total,
      rate: job.rate ? String(job.rate) : '',
      unit: job.unit || (phase === 'text' ? 'cap/s' : 'img/s'),
      started: true,
      complete: total > 0 && done >= total,
    };
  }
  return stages;
}

function effectiveCacheView() {
  const local = state.job && state.job.kind === 'cache' ? state.job : null;
  if (local && local.status === 'running') return null;      // this tab streams it live
  // A training run owns the GPU; the cache panel must say so instead of showing
  // the last build as if nothing else were happening.
  if (state.job && state.job.status === 'running' && state.job.kind !== 'cache') return null;
  const disk = state.cache.job;
  if (!disk) return null;
  const state_ = disk.state;
  const status = state_ === 'running' || state_ === 'starting'
    ? 'running'
    : state_ === 'done' ? 'done' : 'error';
  return {
    job: { kind: 'cache', status, created: disk.started || 0, disk },
    stages: diskCacheStages(disk),
    active: disk.phase === 'text' || disk.phase === 'latent' ? disk.phase : '',
  };
}

export function renderCacheJob() {
  const disk = effectiveCacheView();
  const job = disk ? disk.job : state.job;
  const isCache = Boolean(job && job.kind === 'cache');
  const stages = (disk ? disk.stages : state.cache.stages) || {};
  const active = disk ? disk.active : (state.cache.active || '');
  const activeStage = active ? stages[active] : null;
  const activeLabel = cacheStageLabel(active);
  const desc = $('#cache-job-desc');
  const label = $('#cache-job-label');
  const meta = $('#cache-job-meta');
  const lastLine = $('#cache-last-line');
  const stop = $('#cache-stop');
  renderCacheStage('latent', stages.latent);
  renderCacheStage('text', stages.text);
  if (stop) stop.classList.toggle('hidden', !(job && job.status === 'running' && isCache));

  if (!job) {
    renderCacheStage('latent', null);
    renderCacheStage('text', null);
    if (desc) desc.textContent = t('cache.job.idleDesc');
    if (label) label.textContent = t('common.idle');
    if (meta) meta.textContent = '--';
    if (lastLine) lastLine.textContent = t('cache.waiting');
    return;
  }
  if (job.status === 'running' && isCache) {
    const elapsed = job.created ? Math.max(0, Date.now() / 1000 - Number(job.created)) : null;
    const percent = activeStage ? cacheStagePercent(activeStage) : null;
    if (desc) desc.textContent = active ? t('cache.job.stageRunning', { stage: activeLabel }) : t('cache.msg26');
    if (label) label.textContent = active === 'text' ? t('cache.msg27') : active === 'latent' ? t('cache.msg28') : t('nav.cache');
    if (meta) meta.textContent = [
      activeStage && Number(activeStage.total) > 0 ? `${(percent || 0).toFixed(0)}%` : '',
      activeStage && Number(activeStage.total) > 0
        ? t('cache.job.progressCount', { done: fmtInt(activeStage.done), total: fmtInt(activeStage.total), rate: activeStage.rate ? t('cache.job.progressRate', { rate: activeStage.rate, unit: activeStage.unit }) : '' })
        : '',
      elapsed != null ? t('job.elapsed', { duration: fmtDuration(elapsed) }) : '',
    ].filter(Boolean).join(' · ') || t('job.msg11');
    if (lastLine) lastLine.textContent = state.logs[state.logs.length - 1]?.line
      || (job.disk ? t('cache.job.stageRunning', { stage: activeLabel }) : t('cache.waiting'));
  } else if (job.status === 'running') {
    renderCacheStage('latent', null);
    renderCacheStage('text', null);
    if (desc) desc.textContent = t('cache.job.otherRunning', { kind: jobLabel(job.kind) });
    if (label) label.textContent = t('cache.msg29');
    if (meta) meta.textContent = t('cache.job.occupied', { kind: jobLabel(job.kind) });
    if (lastLine) lastLine.textContent = t('cache.msg30');
  } else if (!isCache) {
    renderCacheStage('latent', null);
    renderCacheStage('text', null);
    if (desc) desc.textContent = t('cache.job.idleDesc');
    if (label) label.textContent = t('common.idle');
    if (meta) meta.textContent = '--';
    if (lastLine) lastLine.textContent = t('cache.waiting');
  } else {
    if (desc) desc.textContent = t('cache.job.lastStatus', { status: job.status === 'done' ? t('common.done') : job.status === 'error' ? t('common.failed') : t('common.stopped') });
    if (label) label.textContent = t('common.idle');
    if (meta) meta.textContent = job.status === 'done' ? t('caption.msg2') : job.status === 'error' ? t('caption.msg3') : t('job.msg7');
    if (lastLine) lastLine.textContent = state.logs[state.logs.length - 1]?.line
      || (job.disk ? t('cache.job.externalLog') : t('cache.waiting'));
  }
}

export function startCacheJob() {
  const parts = readCacheParts();
  if (!parts.length) {
    updateCacheStartState();
    return;
  }
  if (!cachePath()) {
    showAlert({ type: 'warn', title: t('cache.msg31'), text: t('cache.msg32'), timeout: 4000 });
    $('#cfg-cache-dir')?.focus();
    return;
  }
  startJob('cache', { parts }).catch(() => {});
}
