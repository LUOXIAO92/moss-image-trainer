/* =========================================================================
 * js/pages/dataset.js — Dataset
 * split from web/static/app.js, source 5940-6543 (26 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { saveConfig } from '../config/file.js';
import { markDirty, readFormIntoConfig, refreshYamlFromConfig, syncFormFromConfig, updateOutputPreview, updateRegCacheEffective, updateRegCachePlaceholder, updateYamlLineCount } from '../config/form.js';
import { dumpYaml, parseYaml } from '../config/yaml.js';
import { api } from '../core/api.js';
import { $, deletePath, escapeHtml, fileUrl, getPath, setHidden, setPath, setText, thumbUrl } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { captionSourceLabel, t } from '../core/i18n.js';
import { openModal } from '../core/modal.js';
import { showAlert, showError } from '../core/notify.js';
import { isCaptionJob, state } from '../core/state.js';
import { currentTab } from '../main.js';
import { loadCaptionSettings } from './caption.js';
import { runSetupCheck } from './settings.js';

function resetDatasetView() {
  state.dataset.data = null;
  state.dataset.error = null;
  setText('#ds-images', '--');
  setText('#ds-images-sub', t('dataset.notScanned'));
  const imagesSub = $('#ds-images-sub');
  if (imagesSub) imagesSub.title = '';
  setText('#ds-captions', '--');
  setText('#ds-captions-sub', '--');
  setText('#ds-caption-styles', t('dataset.captionStyles'));
  setHidden('#ds-sync', true);
  setHidden('#ds-error', true);
  const progress = $('#ds-caption-progress');
  if (progress) progress.setAttribute('aria-valuenow', '0');
  const bar = $('#ds-caption-bar');
  if (bar) bar.style.width = '0%';
  const thumbs = $('#ds-thumbs');
  if (thumbs) thumbs.innerHTML = '';
  setHidden('#ds-thumbs-empty', false);
  setText('#ds-thumb-note', t('dataset.thumbNote'));
}

function datasetCaptionStats(data) {
  const images = Number(data.images) || 0;
  const withCaption = Number(data.with_caption) || 0;
  const coverage = images ? (withCaption / images) * 100 : 0;
  const source = captionSourceLabel(data.source);
  const sourceLine = source ? t('dataset.captionSourceLine', { source }) : '';
  return {
    images,
    withCaption,
    coverage,
    value: images ? `${fmtInt(withCaption)} / ${fmtInt(images)}` : fmtInt(withCaption),
    sub: images
      ? [t('dataset.coverage', { percent: coverage.toFixed(1) }), sourceLine].filter(Boolean).join(' · ')
      : sourceLine || t('dataset.msg1'),
  };
}

function datasetStyleSummary(data) {
  const styles = data?.styles || {};
  const tag = Number(styles.tag_caption);
  const official = Number(styles.official);
  return t('dataset.styleSummary', { tag: Number.isFinite(tag) ? fmtInt(tag) : '--', official: Number.isFinite(official) ? fmtInt(official) : '--' });
}

function renderDataset(data) {
  const stats = datasetCaptionStats(data);
  const { images, coverage } = stats;
  setText('#ds-images', fmtInt(images));
  setText('#ds-images-sub', data.path ? t('dataset.scannedPath', { path: data.path }) : t('dataset.msg2'));
  const imagesSub = $('#ds-images-sub');
  if (imagesSub) imagesSub.title = data.path || '';
  setText('#ds-captions', stats.value);
  setText('#ds-captions-sub', stats.sub);
  setText('#ds-caption-styles', datasetStyleSummary(data));
  const progress = $('#ds-caption-progress');
  if (progress) progress.setAttribute('aria-valuenow', String(Math.round(coverage)));
  const bar = $('#ds-caption-bar');
  if (bar) bar.style.width = `${coverage.toFixed(1)}%`;
  const grid = $('#ds-thumbs');
  if (grid) {
    grid.innerHTML = '';
    const paths = data.thumbnails || [];
    let broken = 0;
    for (const path of paths) {
      grid.append(createThumbCard(path, () => { broken += 1; updateThumbNote(broken, paths.length, '#ds-thumb-note'); }));
    }
    setHidden('#ds-thumbs-empty', paths.length > 0);
    setText('#ds-thumb-note', paths.length
      ? t('dataset.previewNote', { count: paths.length })
      : t('dataset.msg3'));
  }
}

function renderDatasetSync(data) {
  const sync = $('#ds-sync');
  if (!sync) return;
  const images = Number(data.images) || 0;
  const withCaption = Number(data.with_caption) || 0;
  const missing = images - withCaption;
  setText('#ds-sync-title', t('dataset.synced'));
  setText('#ds-sync-text', t('dataset.syncText', { path: state.configPath, missing: missing > 0 ? t('dataset.syncMissing', { count: missing }) : '' }));
  sync.classList.remove('hidden');
}

function updateThumbNote(broken, total, selector = '#ds-thumb-note') {
  setText(selector, broken ? t('dataset.thumbBroken', { broken, total }) : t('dataset.previewNote', { count: total }));
}

function createThumbCard(path, onBroken) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'thumb-card';
  button.title = path;
  button.setAttribute('aria-label', t('dataset.previewAria', { path }));
  const image = document.createElement('img');
  image.loading = 'lazy';
  image.decoding = 'async';
  image.width = 384;
  image.height = 384;
  image.alt = path.split('/').pop() || t('dataset.msg4');
  image.src = thumbUrl(path, 384);
  image.addEventListener('error', () => {
    if (!image.dataset.fallback) {
      image.dataset.fallback = '1';
      image.src = fileUrl(path);
      return;
    }
    const broken = document.createElement('div');
    broken.className = 'thumb-broken';
    broken.textContent = t('dataset.msg5');
    if (button.contains(image)) button.replaceChild(broken, image);
    if (typeof onBroken === 'function') onBroken();
  });
  image.addEventListener('click', () => openModal(fileUrl(path), path));
  button.append(image);
  button.addEventListener('click', (event) => {
    if (event.target === image) return;
    openModal(fileUrl(path), path);
  });
  return button;
}

function applyDatasetToConfig(path, data) {
  if (!state.config) return;
  const datasetField = $('#cfg-dataset-path');
  if (datasetField) datasetField.value = path;

  if (state.mode === 'yaml') {
    try {
      const parsed = parseYaml($('#yaml')?.value || '');
      setPath(parsed, 'dataset.path', path);
      state.config = parsed;
      const textarea = $('#yaml');
      if (textarea) textarea.value = dumpYaml(parsed);
      updateYamlLineCount();
    } catch (_) {
      setPath(state.config, 'dataset.path', path);
    }
  } else {
    readFormIntoConfig();
    refreshYamlFromConfig();
  }
  syncFormFromConfig(state.config);
  markDirty();
  updateOutputPreview();
  runSetupCheck().catch(() => {});
}

function resolveWorkspacePath(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (/^(?:[A-Za-z]:[\\/]|\/)/.test(raw)) return raw;
  if (raw.startsWith('workspaces/')) return raw;
  return state.workspacePath ? `${state.workspacePath}/${raw}` : raw;
}

function datasetScanUrl(path) {
  const source = String(getPath(state.config, 'dataset.moss_caption_source') || 'tag_caption').trim() || 'tag_caption';
  return '/api/dataset?path=' + encodeURIComponent(path) + '&source=' + encodeURIComponent(source);
}

function resetRegDatasetView(message = t('dataset.reg.empty')) {
  state.dataset.reg = null;
  state.dataset.regError = null;
  setText('#ds-reg-state', t('dataset.reg.notConfigured'));
  const chip = $('#ds-reg-state');
  if (chip) chip.className = 'chip chip-muted';
  setText('#ds-reg-images', '--');
  setText('#ds-reg-images-sub', t('dataset.notScanned'));
  const regImagesSub = $('#ds-reg-images-sub');
  if (regImagesSub) regImagesSub.title = '';
  setText('#ds-reg-captions', '--');
  setText('#ds-reg-captions-sub', '--');
  setText('#ds-reg-caption-styles', t('dataset.captionStyles'));
  setText('#ds-reg-thumb-note', t('dataset.thumbNote'));
  setHidden('#ds-reg-error', true);
  const progress = $('#ds-reg-caption-progress');
  if (progress) progress.setAttribute('aria-valuenow', '0');
  const bar = $('#ds-reg-caption-bar');
  if (bar) bar.style.width = '0%';
  const thumbs = $('#ds-reg-thumbs');
  if (thumbs) thumbs.innerHTML = '';
  const empty = $('#ds-reg-empty');
  if (empty) {
    empty.classList.remove('hidden');
    const p = empty.querySelector('p');
    if (p) p.textContent = message;
  }
}

function renderRegDataset(data, rawPath) {
  const stats = datasetCaptionStats(data);
  const { images, coverage } = stats;
  state.dataset.reg = data;
  state.dataset.regError = null;
  setText('#ds-reg-state', t('dataset.reg.state', { count: fmtInt(images) }));
  const chip = $('#ds-reg-state');
  if (chip) chip.className = 'chip chip-ok';
  setText('#ds-reg-images', fmtInt(images));
  setText('#ds-reg-images-sub', rawPath || data.path || t('dataset.msg2'));
  const regImagesSub = $('#ds-reg-images-sub');
  if (regImagesSub) regImagesSub.title = rawPath || data.path || '';
  setText('#ds-reg-captions', stats.value);
  setText('#ds-reg-captions-sub', stats.sub);
  setText('#ds-reg-caption-styles', datasetStyleSummary(data));
  const progress = $('#ds-reg-caption-progress');
  if (progress) progress.setAttribute('aria-valuenow', String(Math.round(coverage)));
  const bar = $('#ds-reg-caption-bar');
  if (bar) bar.style.width = `${coverage.toFixed(1)}%`;
  setHidden('#ds-reg-error', true);

  const thumbs = $('#ds-reg-thumbs');
  const paths = Array.isArray(data.thumbnails) ? data.thumbnails : [];
  if (thumbs) {
    thumbs.innerHTML = '';
    let broken = 0;
    for (const path of paths) {
      thumbs.append(createThumbCard(path, () => { broken += 1; updateThumbNote(broken, paths.length, '#ds-reg-thumb-note'); }));
    }
    setText('#ds-reg-thumb-note', paths.length
      ? t('dataset.previewNote', { count: paths.length })
      : t('dataset.msg3'));
  }
  const empty = $('#ds-reg-empty');
  if (empty) {
    empty.classList.toggle('hidden', paths.length > 0);
    const p = empty.querySelector('p');
    if (p) p.textContent = paths.length ? '' : t('dataset.msg6');
  }
}

function applyRegDatasetToConfig(path) {
  if (!state.config) return;
  const regField = $('#cfg-reg-path');
  if (regField) regField.value = path || '';
  updateRegCachePlaceholder();
  updateRegCacheEffective();
  if (state.mode === 'yaml') {
    try {
      const parsed = parseYaml($('#yaml')?.value || '');
      if (path) setPath(parsed, 'dataset.reg_path', path);
      else deletePath(parsed, 'dataset.reg_path');
      state.config = parsed;
      const textarea = $('#yaml');
      if (textarea) textarea.value = dumpYaml(parsed);
      updateYamlLineCount();
    } catch (_) {
      if (path) setPath(state.config, 'dataset.reg_path', path);
      else deletePath(state.config, 'dataset.reg_path');
    }
  } else {
    readFormIntoConfig();
    if (path) setPath(state.config, 'dataset.reg_path', path);
    else deletePath(state.config, 'dataset.reg_path');
    refreshYamlFromConfig();
  }
  markDirty();
  runSetupCheck().catch(() => {});
}

function clearRegDataset() {
  resetRegDatasetView();
  applyRegDatasetToConfig('');
}

export async function scanRegDataset() {
  const input = $('#cfg-reg-path');
  const raw = (input?.value || '').trim();
  if (!raw) {
    clearRegDataset();
    await saveConfig({ silent: true });
    return;
  }
  state.dataset.regScanSeq = (state.dataset.regScanSeq || 0) + 1;
  state.dataset.regLoading = false;
  state.dataset.regAutoPath = '';
  const path = resolveWorkspacePath(raw);
  const button = $('#btn-ds-reg-scan');
  if (button) { button.disabled = true; button.textContent = t('dataset.msg7'); }
  setText('#ds-reg-state', t('dataset.msg7'));
  const chip = $('#ds-reg-state');
  if (chip) chip.className = 'chip chip-muted';
  setHidden('#ds-reg-error', true);
  try {
    const data = await api(datasetScanUrl(path));
    state.dataset.regPath = data.path || path;
    renderRegDataset(data, raw);
    applyRegDatasetToConfig(data.path || path);
    await saveConfig({ silent: true });
  } catch (error) {
    state.dataset.reg = null;
    state.dataset.regPath = raw;
    resetRegDatasetView(t('dataset.msg8'));
    state.dataset.regError = error;
    setText('#ds-reg-state', t('dataset.scanFailed'));
    const chip = $('#ds-reg-state');
    if (chip) chip.className = 'chip chip-err';
    setText('#ds-reg-error-text', t('dataset.readDirError', { error: error.detail || error.message }));
    setHidden('#ds-reg-error', false);
  } finally {
    if (button) { button.disabled = false; button.innerHTML = `<svg aria-hidden="true"><use href="#i-scan"></use></svg>${escapeHtml(t('dataset.reg.scan'))}`; }
  }
}

async function autoLoadRegDataset() {
  const input = $('#cfg-reg-path');
  const raw = String(input ? input.value : getPath(state.config, 'dataset.reg_path') || '').trim();
  if (!raw) {
    resetRegDatasetView();
    state.dataset.regPath = '';
    return;
  }
  const resolved = resolveWorkspacePath(raw);
  if (state.dataset.reg && state.dataset.regPath === resolved) {
    renderRegDataset(state.dataset.reg, raw);
    return;
  }
  if (state.dataset.regLoading && state.dataset.regAutoPath === raw) return;
  if (state.dataset.regAutoPath === raw && state.dataset.regError) return;
  const seq = (state.dataset.regScanSeq || 0) + 1;
  state.dataset.regScanSeq = seq;
  state.dataset.regLoading = true;
  state.dataset.regAutoPath = raw;
  setHidden('#ds-reg-error', true);
  setText('#ds-reg-state', t('dataset.msg7'));
  const chip = $('#ds-reg-state');
  if (chip) chip.className = 'chip chip-muted';
  try {
    const data = await api(datasetScanUrl(resolved));
    if (seq !== state.dataset.regScanSeq) return;
    state.dataset.regPath = data.path || resolved;
    renderRegDataset(data, raw);
  } catch (error) {
    if (seq !== state.dataset.regScanSeq) return;
    state.dataset.reg = null;
    state.dataset.regError = error;
    state.dataset.regPath = resolved;
    resetRegDatasetView(t('dataset.msg8'));
    state.dataset.regError = error;
    setText('#ds-reg-state', t('dataset.scanFailed'));
    const failedChip = $('#ds-reg-state');
    if (failedChip) failedChip.className = 'chip chip-err';
    setText('#ds-reg-error-text', t('dataset.readDirError', { error: error.detail || error.message }));
    setHidden('#ds-reg-error', false);
  } finally {
    if (seq === state.dataset.regScanSeq) {
      state.dataset.regLoading = false;
      state.dataset.regAutoPath = '';
    }
  }
}

export async function autoLoadDataset() {
  autoLoadRegDataset().catch(() => {});
  const input = $('#cfg-dataset-path');
  const path = String(input?.value || getPath(state.config, 'dataset.path') || '').trim();
  if (!path) {
    resetDatasetView();
    return;
  }
  const resolved = resolveWorkspacePath(path);
  if (state.dataset.loading && state.dataset.loadingPath === path) return;
  if (state.dataset.data && state.dataset.path === resolved) {
    renderDataset(state.dataset.data);
    return;
  }
  if (state.dataset.autoPath === path && state.dataset.error) return;
  const seq = (state.dataset.scanSeq || 0) + 1;
  state.dataset.scanSeq = seq;
  state.dataset.loading = true;
  state.dataset.loadingPath = path;
  state.dataset.autoPath = path;
  setHidden('#ds-error', true);
  try {
    const data = await api(datasetScanUrl(resolved));
    if (seq !== state.dataset.scanSeq) return;
    state.dataset.data = data;
    state.dataset.path = data.path || resolved;
    state.dataset.error = null;
    state.dataset.lastPath = path;
    renderDataset(data);
    setHidden('#ds-sync', true);
  } catch (error) {
    if (seq !== state.dataset.scanSeq) return;
    state.dataset.data = null;
    state.dataset.error = error;
    state.dataset.path = resolved;
    resetDatasetView();
    state.dataset.error = error;
    setText('#ds-error-text', t('dataset.readDirError', { error: error.detail || error.message }));
    setHidden('#ds-error', false);
  } finally {
    if (seq === state.dataset.scanSeq) {
      state.dataset.loading = false;
      state.dataset.loadingPath = '';
    }
  }
}

export function refreshDatasetAfterCaption() {
  state.dataset.data = null;
  state.dataset.reg = null;
  state.dataset.autoPath = '';
  state.dataset.regAutoPath = '';
  state.dataset.loading = false;
  state.dataset.loadingPath = '';
  state.dataset.scanSeq = (state.dataset.scanSeq || 0) + 1;
  state.dataset.regLoading = false;
  state.dataset.regScanSeq = (state.dataset.regScanSeq || 0) + 1;
  if (currentTab() === 'dataset') autoLoadDataset().catch(() => {});
  if (currentTab() === 'caption') loadCaptionDatasetInfo().catch(() => {});
}

export function readCaptionSelection() {
  const datasets = [];
  if ($('#caption-dataset-train')?.checked) datasets.push('train');
  if ($('#caption-dataset-reg')?.checked) datasets.push('reg');
  const styles = [];
  if ($('#caption-style-tag')?.checked) styles.push('tag_caption');
  if ($('#caption-style-official')?.checked) styles.push('official');
  return { datasets, styles, overwrite: Boolean($('#caption-overwrite')?.checked) };
}

export function updateCaptionStartState() {
  const button = $('#btn-caption-start');
  if (!button) return;
  const running = Boolean(state.job && state.job.status === 'running') || Boolean(state.activeTraining);
  const captionRunning = Boolean(state.job && state.job.status === 'running' && isCaptionJob(state.job.kind));
  const busy = running || state.busy;
  for (const [key, id] of [['train', 'caption-dataset-train'], ['reg', 'caption-dataset-reg']]) {
    const checkbox = $(`#${id}`);
    if (!checkbox) continue;
    const entry = state.captionRun.datasets[key] || {};
    const unavailable = state.captionRun.initialized && !entry.available;
    checkbox.disabled = busy || unavailable;
    const row = checkbox.closest('.caption-choice');
    if (row) row.classList.toggle('is-disabled', busy || unavailable);
  }
  for (const id of ['caption-style-tag', 'caption-style-official', 'caption-overwrite']) {
    const input = $(`#${id}`);
    if (input) input.disabled = busy;
  }
  const selection = readCaptionSelection();
  let hint = '';
  if (!state.profileName) hint = t('job.msg29');
  else if (!state.captionRun.initialized || state.captionRun.loading) hint = t('caption.msg9');
  else if (!selection.datasets.length) hint = t('caption.selectDataset');
  else if (!selection.styles.length) hint = t('job.msg33');
  button.disabled = busy || Boolean(hint);
  button.title = hint || t('caption.msg10');
  setText('#caption-start-hint', hint);
  setHidden('#caption-start-hint', !hint);
  setHidden('#btn-caption-stop', !captionRunning);
}

function captionDatasetRow(entry, key) {
  const isReg = key === 'reg';
  if (state.captionRun.loading && entry.path) return t('common.loadingShort');
  if (!entry.path) return isReg ? t('dataset.msg9') : t('dataset.msg10');
  if (entry.error) return isReg ? t('dataset.msg9') : t('dataset.msg11');
  return t('dataset.captionRow', { path: entry.displayPath || entry.path, count: fmtInt(entry.images || 0) });
}

function renderCaptionDatasetState() {
  for (const [key, id] of [['train', 'caption-dataset-train-meta'], ['reg', 'caption-dataset-reg-meta']]) {
    const entry = state.captionRun.datasets[key] || {};
    const meta = $(`#${id}`);
    if (meta) {
      meta.textContent = captionDatasetRow(entry, key);
      meta.title = entry.error || entry.path || '';
    }
    const checkbox = $(`#caption-dataset-${key}`);
    if (!checkbox) continue;
    if (!entry.available) checkbox.checked = false;
    const row = checkbox.closest('.caption-choice');
    if (row) row.classList.toggle('is-disabled', !entry.available);
  }
  updateCaptionStartState();
}

async function fetchCaptionDataset(key, path) {
  if (!path) {
    return { available: false, path: '', displayPath: '', images: null, error: '' };
  }
  try {
    const data = await api(datasetScanUrl(resolveWorkspacePath(path)));
    return {
      available: true,
      path: data.path || path,
      displayPath: path,
      images: Number(data.images) || 0,
      error: '',
    };
  } catch (error) {
    return {
      available: false,
      path,
      displayPath: path,
      images: null,
      error: error.detail || error.message || t('dataset.msg12'),
    };
  }
}

export async function loadCaptionDatasetInfo() {
  const seq = (state.captionRun.seq || 0) + 1;
  state.captionRun.seq = seq;
  state.captionRun.loading = true;
  const trainPath = String($('#cfg-dataset-path')?.value || getPath(state.config, 'dataset.path') || '').trim();
  const regPath = String($('#cfg-reg-path')?.value || getPath(state.config, 'dataset.reg_path') || '').trim();
  state.captionRun.datasets.train = { ...(state.captionRun.datasets.train || {}), path: trainPath, displayPath: trainPath };
  state.captionRun.datasets.reg = { ...(state.captionRun.datasets.reg || {}), path: regPath, displayPath: regPath };
  renderCaptionDatasetState();
  if (!state.profileName) {
    state.captionRun.loading = false;
    state.captionRun.initialized = true;
    state.captionRun.datasets = {
      train: { available: false, path: '', displayPath: '', images: null, error: '' },
      reg: { available: false, path: '', displayPath: '', images: null, error: '' },
    };
    renderCaptionDatasetState();
    return;
  }
  const [train, reg] = await Promise.all([
    fetchCaptionDataset('train', trainPath),
    fetchCaptionDataset('reg', regPath),
  ]);
  if (seq !== state.captionRun.seq) return;
  const wasInitialized = state.captionRun.initialized;
  const previousPaths = { ...(state.captionRun.paths || {}) };
  const previousAvailable = {
    train: Boolean(state.captionRun.datasets?.train?.available),
    reg: Boolean(state.captionRun.datasets?.reg?.available),
  };
  state.captionRun.datasets = { train, reg };
  state.captionRun.loading = false;
  state.captionRun.initialized = true;
  state.captionRun.paths = { train: trainPath, reg: regPath };
  for (const [key, id] of [['train', 'caption-dataset-train'], ['reg', 'caption-dataset-reg']]) {
    const checkbox = $(`#${id}`);
    const entry = state.captionRun.datasets[key];
    if (!checkbox) continue;
    if (entry.available && (!wasInitialized || (entry.path && !previousPaths[key]) || !previousAvailable[key])) checkbox.checked = true;
    if (!entry.available) checkbox.checked = false;
  }
  renderCaptionDatasetState();
}

export async function loadCaptionPage() {
  if (!state.captionSettingsDirty) {
    loadCaptionSettings().catch((error) => showError(t('caption.msg11'), error.detail || error.message));
  }
  await loadCaptionDatasetInfo();
}

export async function scanDataset() {
  const input = $('#cfg-dataset-path');
  const path = (input?.value || '').trim();
  if (!path) {
    showAlert({ type: 'warn', title: t('dataset.msg13'), text: t('dataset.msg14'), timeout: 4000 });
    input?.focus();
    return;
  }
  const button = $('#btn-scan');
  if (button) { button.disabled = true; button.textContent = t('dataset.msg7'); }
  state.dataset.scanSeq = (state.dataset.scanSeq || 0) + 1;
  state.dataset.loading = false;
  state.dataset.loadingPath = '';
  state.dataset.autoPath = path;
  setHidden('#ds-error', true);
  setHidden('#ds-sync', true);
  try {
    const data = await api(datasetScanUrl(path));
    state.dataset.data = data;
    state.dataset.path = data.path || path;
    state.dataset.error = null;
    state.dataset.lastPath = path;
    renderDataset(data);
    applyDatasetToConfig(data.path || path, data);
    const saved = await saveConfig({ silent: true });
    if (saved) renderDatasetSync(data);
  } catch (error) {
    state.dataset.data = null;
    state.dataset.error = error;
    resetDatasetView();
    const detail = error.detail || error.message;
    setText('#ds-error-text', t('dataset.readDirError', { error: detail }));
    setHidden('#ds-error', false);
    setText('#ds-images', '--');
    setText('#ds-images-sub', t('dataset.msg15'));
    showError(t('dataset.msg16'), t('dataset.readDirError', { error: error.detail || error.message }), [
      { label: t('common.retry'), onClick: () => scanDataset() },
    ], 'dataset-scan-error');
  } finally {
    if (button) { button.disabled = false; button.innerHTML = `<svg aria-hidden="true"><use href="#i-scan"></use></svg>${escapeHtml(t('dataset.scan'))}`; }
  }
}
