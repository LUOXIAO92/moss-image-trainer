/* =========================================================================
 * js/pages/export.js — Export / settings
 * split from web/static/app.js, source 7091-7738 (29 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { api, describeError } from '../core/api.js';
import { $, escapeHtml, fileUrl, setText } from '../core/dom.js';
import { fmtBytesFromMb, fmtInt, fmtNum } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { showError, showSuccess } from '../core/notify.js';
import { state, workspaceParam } from '../core/state.js';

function exportStepLabel(step) {
  return step === 'final' ? 'final' : `step ${fmtInt(step)}`;
}

function exportStepToken(step) {
  return step === 'final' ? '_final' : `_step${String(step).padStart(6, '0')}`;
}

function selectedExportSteps() {
  return [...state.exportSelected].filter((step) => step !== 'final').sort((a, b) => Number(a) - Number(b));
}

export function updateExportStatus(text, className = 'chip-muted') {
  const chip = $('#export-state');
  if (!chip) return;
  chip.className = `chip ${className}`;
  chip.textContent = text;
}

export function updateExportSelectionUi() {
  const numeric = state.exportCheckpoints.filter((item) => item.step !== 'final');
  const selected = selectedExportSteps();
  setText('#export-selected-count', t('export.selectedCount', { count: selected.length }));
  const all = $('#export-select-all');
  if (all) {
    all.checked = numeric.length > 0 && numeric.every((item) => state.exportSelected.has(String(item.step)));
    all.indeterminate = selected.length > 0 && !all.checked;
  }
  const batch = $('#btn-export-batch');
  if (batch) batch.disabled = state.exportRunning || !selected.length;
  const rangeGroup = $('#export-range-group');
  const rangeDisabled = selected.length < 2;
  if (rangeGroup) rangeGroup.classList.toggle('is-disabled', rangeDisabled);
  for (const id of ['#export-range-from', '#export-range-to', '#export-range-every']) {
    const input = $(id);
    if (input) input.disabled = rangeDisabled;
  }
}

function existingExportForStep(step) {
  const token = exportStepToken(step);
  const candidates = (state.loras || []).filter((item) => {
    if (item.state || item.name.endsWith('_state.safetensors')) return false;
    return item.name.includes(token);
  });
  return candidates.sort((a, b) => {
    const score = (item) => /_(comfy|diffusers|kohya)\.safetensors$/.test(item.name) ? 1 : 0;
    return score(a) - score(b) || a.name.localeCompare(b.name);
  })[0] || null;
}

export function renderExportStepRows() {
  const body = $('#loras-body');
  if (!body) return;
  const checkpoints = state.exportCheckpoints || [];
  if (!checkpoints.length) {
    body.innerHTML = `<tr><td colspan="5" class="table-empty">${escapeHtml(t('export.noCheckpoints'))}</td></tr>`;
    updateExportSelectionUi();
    return;
  }
  body.innerHTML = checkpoints.map((item) => {
    const step = item.step;
    const numeric = step !== 'final';
    const selected = numeric && state.exportSelected.has(String(step));
    const existing = existingExportForStep(step);
    const train = item.train == null || item.train === '' ? '—' : fmtNum(item.train, 4);
    const evalLoss = item.eval == null || item.eval === '' ? '—' : fmtNum(item.eval, 4);
    const action = state.exportDestination === 'server' ? t('common.save') : t('export.msg1');
    const title = existing
      ? (state.exportDestination === 'server' ? t('export.msg2') : t('export.msg3'))
      : (state.exportDestination === 'server' ? t('export.serverPath') : t('export.msg4'));
    return `<tr>
      <td class="export-check-col"><input type="checkbox" data-export-step="${escapeHtml(String(step))}" ${selected ? 'checked' : ''} ${numeric ? '' : 'disabled'} aria-label="${escapeHtml(t('export.selectStepAria', { step: exportStepLabel(step) }))}"></td>
      <td class="export-step-cell">${escapeHtml(exportStepLabel(step))}</td>
      <td class="export-loss">${train}</td>
      <td class="export-loss">${evalLoss}</td>
      <td class="actions-cell export-action-col"><button class="btn btn-xs" type="button" title="${escapeHtml(title)}" data-export-download="${escapeHtml(String(step))}">${action}</button></td>
    </tr>`;
  }).join('');
  updateExportSelectionUi();
}

function initExportRange() {
  const numeric = state.exportCheckpoints.filter((item) => item.step !== 'final').map((item) => Number(item.step)).sort((a, b) => a - b);
  if (!numeric.length) return;
  const every = numeric.length > 1 ? (numeric[1] - numeric[0]) : 1;
  const from = $('#export-range-from');
  const to = $('#export-range-to');
  const step = $('#export-range-every');
  if (from) from.value = String(numeric[0]);
  if (to) to.value = String(numeric[numeric.length - 1]);
  if (step) step.value = String(Math.max(1, every));
}

export function applyExportRange() {
  const numeric = state.exportCheckpoints.filter((item) => item.step !== 'final').map((item) => Number(item.step)).sort((a, b) => a - b);
  const from = Number($('#export-range-from')?.value);
  const to = Number($('#export-range-to')?.value);
  const every = Math.max(1, Number($('#export-range-every')?.value) || 1);
  if (!Number.isFinite(from) || !Number.isFinite(to) || to < from) return;
  state.exportSelected = new Set(numeric.filter((value) => value >= from && value <= to && (value - from) % every === 0).map((value) => String(value)));
  renderExportStepRows();
}

function triggerDownload(path) {
  if (!path) return;
  const anchor = document.createElement('a');
  anchor.href = fileUrl(path);
  anchor.download = '';
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}

function waitForExportJob(jobId) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 30 * 60 * 1000;
    const tick = async () => {
      try {
        const jobs = await api('/api/jobs');
        const job = (jobs || []).find((item) => item.id === jobId);
        if (!job) return reject(new Error(t('export.msg5')));
        if (job.status !== 'running') {
          return job.status === 'done' ? resolve(job) : reject(new Error(job.status === 'stopped' ? t('export.msg6') : t('job.msg28')));
        }
        if (Date.now() > deadline) return reject(new Error(t('export.msg7')));
        setTimeout(tick, 1000);
      } catch (error) {
        reject(error);
      }
    };
    tick();
  });
}

function exportFormats() {
  const formats = [];
  if ($('#export-format-kohya')?.checked) formats.push('kohya');
  if ($('#export-format-diffusers')?.checked) formats.push('diffusers');
  return formats;
}

function exportStepsSpec(steps) {
  // A single numeric step is ambiguous for the CLI range parser; a comma list
  // forces explicit enumeration.
  return steps.length === 1 ? `${steps[0]},${steps[0]}` : steps.join(',');
}

function exportPayload(steps) {
  return {
    run: $('#export-run')?.value || '',
    steps: exportStepsSpec(steps),
    formats: exportFormats(),
    state: Boolean($('#export-with-state')?.checked),
  };
}

async function exportBatch(steps, targetDir = '', zip = false) {
  if (!steps.length) {
    showError(t('export.msg8'), t('export.msg9'));
    return null;
  }
  const formats = exportFormats();
  if (!formats.length) {
    showError(t('export.msg10'), t('export.msg11'));
    return null;
  }
  updateExportStatus(t('export.msg12'), 'chip-run');
  state.exportRunning = true;
  const body = exportPayload(steps);
  if (targetDir) body.target_dir = targetDir;
  if (zip) body.zip = true;
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/export-batch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    await waitForExportJob(data.id);
    await loadLoras();
    return data;
  } catch (error) {
    updateExportStatus(t('job.msg28'), 'chip-err');
    showError(t('job.msg28'), describeError(error));
    return null;
  } finally {
    state.exportRunning = false;
    updateExportSelectionUi();
  }
}

export function closeExportDirectory(value) {
  const dialog = $('#export-dir-dialog');
  const resolve = state.exportDirResolve;
  state.exportDirResolve = null;
  if (dialog?.open && typeof dialog.close === 'function') dialog.close();
  if (resolve) resolve(value || null);
}

export function defaultExportDirectory() {
  return state.exportTargetDir || `${state.workspacePath || ('workspaces/' + state.profileName)}/exports`;
}

function renderExportDirectory(data, requested) {
  const list = $('#export-dir-list');
  const pathInput = $('#export-dir-path');
  const preview = $('#export-dir-preview');
  const notice = $('#export-dir-notice');
  if (!list || !pathInput || !preview) return;
  const current = data.path || requested;
  state.exportDirPath = current;
  pathInput.value = current;
  preview.textContent = current;
  if (notice) {
    notice.classList.toggle('hidden', data.exists !== false);
    notice.textContent = data.exists === false ? t('export.dir.notice') : '';
  }
  const dirs = (data.entries || []).filter((entry) => entry.is_dir);
  const rows = [];
  if (data.parent && data.parent !== current) {
    rows.push(`<button type="button" class="export-dir-entry" data-export-dir="${escapeHtml(data.parent)}"><strong>..</strong><span class="mono">${escapeHtml(t('common.parentDir'))}</span></button>`);
  }
  rows.push(...dirs.map((entry) => `<button type="button" class="export-dir-entry" data-export-dir="${escapeHtml(entry.path)}"><strong>${escapeHtml(entry.name)}</strong><span class="mono">${escapeHtml(t('common.directory'))}</span></button>`));
  list.innerHTML = rows.join('') || `<div class="export-dir-empty">${escapeHtml(t('export.noSubdirs'))}</div>`;
}

export async function loadExportDirectory(path) {
  const list = $('#export-dir-list');
  const pathInput = $('#export-dir-path');
  const preview = $('#export-dir-preview');
  if (!list || !pathInput || !preview) return;
  const target = String(path ?? pathInput.value ?? '').trim() || defaultExportDirectory();
  state.exportDirPath = target;
  pathInput.value = target;
  preview.textContent = target;
  const seq = ++state.exportDirRequestSeq;
  try {
    const data = await api('/api/fs/list?path=' + encodeURIComponent(target));
    if (seq !== state.exportDirRequestSeq) return;
    renderExportDirectory(data, target);
  } catch (error) {
    if (seq !== state.exportDirRequestSeq) return;
    const notice = $('#export-dir-notice');
    if (notice) notice.classList.add('hidden');
    list.innerHTML = `<div class="export-dir-empty">${escapeHtml(describeError(error))}</div>`;
  }
}

function chooseExportDirectory() {
  const dialog = $('#export-dir-dialog');
  if (!dialog) return Promise.resolve(null);
  return new Promise((resolve) => {
    state.exportDirResolve = resolve;
    const initial = state.exportTargetDir || `${state.workspacePath || ('workspaces/' + state.profileName)}/exports`;
    state.exportDirPath = initial;
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
    else dialog.setAttribute('open', '');
    loadExportDirectory(initial);
  });
}

export function closePathBrowser(value = null) {
  const dialog = $('#path-browser-dialog');
  state.pathBrowser.resolve = null;
  if (dialog?.open) dialog.close();
  if (state.pathBrowser.resolveValue) {
    state.pathBrowser.resolveValue(value || null);
    state.pathBrowser.resolveValue = null;
  }
}

export function choosePathFromBrowser(fieldId, mode = 'dir') {
  const dialog = $('#path-browser-dialog');
  const field = document.getElementById(fieldId);
  if (!dialog || !field) return Promise.resolve(null);
  return new Promise((resolve) => {
    state.pathBrowser = {
      fieldId, mode: mode === 'file' ? 'file' : mode === 'either' ? 'either' : 'dir',
      dir: '', selected: '', resolveValue: resolve,
    };
    const raw = String(field.value || '').trim();
    let initial = raw;
    if ((state.pathBrowser.mode === 'file' || state.pathBrowser.mode === 'either') && raw) {
      initial = raw.replace(/\/[^/]+$/, '') || '/';
      state.pathBrowser.selected = raw.endsWith('.safetensors') ? raw : '';
    }
    if (!initial || !initial.startsWith('/')) {
      initial = field.placeholder?.startsWith('/') ? field.placeholder : (state.workspaceAbsPath || state.workspacePath || '/');
    }
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
    else dialog.setAttribute('open', '');
    setText('#path-browser-title', state.pathBrowser.mode === 'file' ? t('common.msg45') : state.pathBrowser.mode === 'either' ? t('common.msg46') : t('path.title'));
    setText('#path-browser-confirm', state.pathBrowser.mode === 'file' ? t('common.msg47') : state.pathBrowser.mode === 'either' ? t('common.select') : t('common.msg48'));
    loadPathBrowser(initial).catch(() => {});
  });
}

export async function loadPathBrowser(path) {
  const list = $('#path-browser-list');
  const input = $('#path-browser-path');
  const notice = $('#path-browser-notice');
  if (!list || !input || !state.pathBrowser.mode) return;
  const target = String(path || '').trim() || state.workspaceAbsPath || '/';
  input.value = target;
  try {
    const data = await api('/api/fs/list?path=' + encodeURIComponent(target));
    state.pathBrowser.dir = data.path || target;
    input.value = state.pathBrowser.dir;
    if (notice) {
      notice.classList.toggle('hidden', data.exists !== false);
      notice.textContent = data.exists === false ? t('path.notice') : '';
    }
    const entries = (data.entries || []).filter((entry) => entry.is_dir
      || (state.pathBrowser.mode !== 'dir' && /\.safetensors$/i.test(entry.name)));
    const rows = [];
    if (data.parent && data.parent !== state.pathBrowser.dir) {
      rows.push(`<button type="button" class="export-dir-entry" data-path-dir="${escapeHtml(data.parent)}"><strong>..</strong><span class="path-browser-entry-kind">${escapeHtml(t('common.parentDir'))}</span></button>`);
    }
    for (const entry of entries) {
      const selected = state.pathBrowser.mode !== 'dir' && state.pathBrowser.selected === entry.path;
      rows.push(`<button type="button" class="export-dir-entry ${selected ? 'is-selected' : ''}" data-path-${entry.is_dir ? 'dir' : 'file'}="${escapeHtml(entry.path)}">
        <span class="path-browser-entry-name">${escapeHtml(entry.name)}</span>
        <span class="path-browser-entry-kind">${entry.is_dir ? escapeHtml(t('common.directory')) : `${fmtBytesFromMb(entry.size_mb)}`}</span>
      </button>`);
    }
    list.innerHTML = rows.join('') || `<div class="export-dir-empty">${escapeHtml(t('export.noEntries'))}</div>`;
    const selected = state.pathBrowser.mode !== 'dir' && state.pathBrowser.selected
      ? state.pathBrowser.selected
      : state.pathBrowser.dir;
    setText('#path-browser-selected', selected || '--');
  } catch (error) {
    list.innerHTML = `<div class="export-dir-empty">${escapeHtml(describeError(error))}</div>`;
  }
}

export function setExportDestination(mode) {
  state.exportDestination = mode === 'browser' ? 'browser' : 'server';
  const server = $('#export-dest-server');
  const browser = $('#export-dest-browser');
  if (server) server.checked = state.exportDestination === 'server';
  if (browser) browser.checked = state.exportDestination === 'browser';
  const batch = $('#btn-export-batch');
  if (batch) batch.textContent = state.exportDestination === 'server' ? t('common.save') : t('export.msg1');
  renderExportStepRows();
}

export async function exportSelected() {
  const steps = selectedExportSteps();
  if (!steps.length) {
    showError(t('export.msg8'), t('export.msg9'));
    return;
  }
  if (state.exportDestination === 'server') {
    const target = await chooseExportDirectory();
    if (!target) return;
    state.exportTargetDir = target;
    const data = await exportBatch(steps, target, false);
    if (!data) return;
    renderExportStepRows();
    updateExportStatus(t('common.saved'), 'chip-ok');
    showSuccess(t('export.msg13'), t('export.batchDone', { count: steps.length, dir: data.out_dir }), [], 'batch-path-done');
  } else {
    const data = await exportBatch(steps, '', true);
    if (!data) return;
    renderExportStepRows();
    updateExportStatus(t('export.msg14'), 'chip-ok');
    triggerDownload(data.zip);
    showSuccess(t('export.msg15'), data.zip, [], 'batch-zip-done');
  }
}

export async function downloadSingleStep(step) {
  if (state.exportRunning) return;
  if (state.exportDestination === 'browser') {
    const existing = existingExportForStep(step);
    if (existing) {
      triggerDownload(existing.path);
      renderExportStepRows();
      updateExportStatus(t('export.msg14'), 'chip-ok');
      return;
    }
  }
  state.exportRunning = true;
  updateExportStatus(state.exportDestination === 'server' ? t('common.msg51') : t('common.msg52'), 'chip-run');
  updateExportSelectionUi();
  try {
    if (state.exportDestination === 'server' && step !== 'final') {
      const target = await chooseExportDirectory();
      if (!target) { state.exportRunning = false; updateExportSelectionUi(); return; }
      state.exportTargetDir = target;
      const data = await exportBatch([String(step)], target, false);
      if (!data) return;
      updateExportStatus(t('common.saved'), 'chip-ok');
    } else {
      const formats = exportFormats();
      if (!formats.length) throw new Error(t('common.msg53'));
      const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/export', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run: $('#export-run')?.value, step, formats, state: Boolean($('#export-with-state')?.checked) }),
      });
      await waitForExportJob(data.id);
      await loadLoras();
      const file = (data.files || []).find((name) => !name.endsWith('_state.safetensors')) || data.files?.[0];
      const item = (state.loras || []).find((entry) => entry.name === file);
      if (item) triggerDownload(item.path);
      updateExportStatus(state.exportDestination === 'server' ? t('common.saved') : t('export.msg14'), 'chip-ok');
    }
    renderExportStepRows();
  } catch (error) {
    updateExportStatus(t('caption.msg3'), 'chip-err');
    showError(t('common.msg54'), describeError(error));
  } finally {
    state.exportRunning = false;
    updateExportSelectionUi();
  }
}

export async function loadExportPage() {
  if (!state.profileName) {
    state.exportRuns = [];
    state.exportCheckpoints = [];
    state.exportSelected.clear();
    const runSelect = $('#export-run');
    if (runSelect) runSelect.innerHTML = `<option value="">${escapeHtml(t('train.profile.noWorkspace'))}</option>`;
    renderExportStepRows();
    return;
  }
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/runs');
    state.exportRuns = Array.isArray(data.runs) ? data.runs : [];
    const select = $('#export-run');
    if (!select) return;
    const previous = select.value;
    select.innerHTML = state.exportRuns.length
      ? state.exportRuns.map((run) => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)} · ${fmtInt(run.step)}/${fmtInt(run.total)}</option>`).join('')
      : `<option value="">${escapeHtml(t('train.noRuns'))}</option>`;
    select.value = state.exportRuns.some((run) => run.name === previous) ? previous : (state.exportRuns[0]?.name || '');
    await onExportRunChange();
  } catch (error) {
    showError(t('export.msg16'), describeError(error));
  }
}

export async function onExportRunChange() {
  const run = state.exportRuns.find((item) => item.name === ($('#export-run')?.value || ''));
  state.exportCheckpoints = run?.checkpoints || [];
  const numeric = state.exportCheckpoints.filter((item) => item.step !== 'final').map((item) => Number(item.step));
  state.exportSelected = new Set(numeric.map((value) => String(value)));
  initExportRange();
  state.exportTargetDir = `${state.workspacePath || ('workspaces/' + state.profileName)}/exports`;
  updateExportStatus(t('export.ready'));
  setExportDestination(state.exportDestination);
  await loadLoras();
}

export async function loadLoras() {
  const run = $('#export-run')?.value || '';
  if (!state.profileName || !run) {
    state.loras = [];
    renderExportStepRows();
    return;
  }
  try {
    const query = new URLSearchParams({ workspace: workspaceParam(), run, include_state: '1' });
    const list = await api('/api/loras?' + query.toString());
    state.loras = Array.isArray(list) ? list : [];
    renderExportStepRows();
  } catch (error) {
    state.loras = [];
    renderExportStepRows();
  }
}
