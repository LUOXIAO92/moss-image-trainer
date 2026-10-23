/* =========================================================================
 * js/config/form.js — Config <-> form
 * split from web/static/app.js, source 913-1562 (32 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { FIELD_DEFS, LAYER_COUNT, LORA_MODULES, absoluteWorkspacePath, arrayFromPrompts, compactLayerRange, defaultCacheDir, defaultRegCacheDir, displayCacheValue, displayRegCacheValue, parseLayerRange, parseIntList, resolvedCacheBase } from './model.js';
import { dumpYaml, parseYaml } from './yaml.js';
import { $, $$, clamp, deletePath, escapeHtml, getPath, setPath, setText } from '../core/dom.js';
import { fmtLr } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { showAlert } from '../core/notify.js';
import { state } from '../core/state.js';
import { resetModelProbes } from '../pages/settings.js';

export function syncAttentionControls() {
  const attention = $('#cfg-attention');
  const binTokens = $('#cfg-bin-tokens');
  if (!attention || !binTokens) return;
  binTokens.disabled = attention.value !== 'sdpa';
}

function layerRowHtml(row) {
  const targets = row.targets && row.targets.length ? row.targets : LORA_MODULES;
  return `<div class="layer-row">
    <input class="layer-range mono" type="text" spellcheck="false" value="${escapeHtml(row.range)}" placeholder="0-7" aria-label="${escapeHtml(t('train.layer.rangeAria'))}">
    <div class="layer-modules" aria-label="${escapeHtml(t('train.layer.modulesAria'))}">
      ${LORA_MODULES.map((name) => `<label><input type="checkbox" value="${name}" ${targets.includes(name) ? 'checked' : ''}> ${name}</label>`).join('')}
    </div>
    <button class="btn btn-ghost btn-xs layer-remove" type="button" aria-label="${escapeHtml(t('train.layer.removeAria'))}">${escapeHtml(t('common.delete'))}</button>
  </div>`;
}

function renderLayerRows(rows) {
  const box = $('#layer-rows');
  if (!box) return;
  box.innerHTML = rows.map(layerRowHtml).join('');
}

function syncLayerControls(state) {
  const layers = Array.isArray(state.layers)
    ? state.layers.map((x) => parseInt(x, 10)).filter((n) => Number.isFinite(n) && n >= 0 && n < LAYER_COUNT).sort((a, b) => a - b)
    : [];
  const targets = Array.isArray(state.targets) && state.targets.length ? state.targets.map(String) : [...LORA_MODULES];
  const rows = [];
  if (!layers.length) {
    rows.push({ range: `0-${LAYER_COUNT - 1}`, targets });
  } else {
    let start = layers[0];
    let prev = layers[0];
    for (const n of layers.slice(1)) {
      if (n === prev + 1) { prev = n; continue; }
      rows.push({ range: start === prev ? `${start}` : `${start}-${prev}`, targets });
      start = n; prev = n;
    }
    rows.push({ range: start === prev ? `${start}` : `${start}-${prev}`, targets });
  }
  renderLayerRows(rows);
  readLayerControlsIntoConfig();
}

function collectLayerRows() {
  const rows = [];
  for (const row of $$('#layer-rows .layer-row')) {
    const range = row.querySelector('.layer-range')?.value || '';
    const targets = [...row.querySelectorAll('.layer-modules input:checked')].map((input) => input.value);
    rows.push({ range, targets });
  }
  return rows;
}

function updateLayerSummary(layers, targets, warning = '') {
  const uniqueLayers = [...new Set(layers || [])].sort((a, b) => a - b);
  const range = !uniqueLayers.length || uniqueLayers.length === LAYER_COUNT
    ? t('train.layer.all', { count: LAYER_COUNT })
    : t('train.layer.count', { range: compactLayerRange(uniqueLayers), count: uniqueLayers.length });
  setText('#layer-summary', `${range} · ${targets.length ? targets.join(', ') : t('train.layer.unselected')}`);
  const box = $('#layer-warning');
  if (box) {
    const message = warning || (targets.length ? '' : t('train.msg7'));
    box.textContent = message;
    box.classList.toggle('hidden', !message);
  }
}

export function readLayerControlsIntoConfig() {
  const rows = collectLayerRows();
  const errors = [];
  const layers = [];
  const perRowTargets = [];
  for (const row of rows) {
    try {
      layers.push(...parseLayerRange(row.range));
    } catch (error) {
      errors.push(error.message);
    }
    if (!row.targets.length) errors.push(t('train.layer.missingModules', { range: row.range || '?' }));
    perRowTargets.push(row.targets.join(','));
  }
  const uniqueLayers = [...new Set(layers)].sort((a, b) => a - b);
  const allTargets = [...new Set(rows.flatMap((row) => row.targets))];
  const mixedTargets = new Set(perRowTargets.filter(Boolean)).size > 1;
  state.layerErrors = errors;
  if (errors.length) {
    updateLayerSummary(uniqueLayers, allTargets, errors.join('；'));
    return;
  }
  if (state.config && typeof state.config === 'object') {
    state.config.lora = state.config.lora || {};
    // all 42 explicit indices are equivalent to "all"; keep the portable empty form
    state.config.lora.layers = uniqueLayers.length === LAYER_COUNT ? [] : uniqueLayers;
    state.config.lora.targets = allTargets;
  }
  updateLayerSummary(
    uniqueLayers, allTargets,
    mixedTargets ? t('train.msg8') : '',
  );
}

export function addLayerRow() {
  const rows = collectLayerRows();
  const used = [];
  for (const row of rows) {
    try { used.push(...parseLayerRange(row.range)); } catch (_) { /* keep adding anyway */ }
  }
  const start = used.length ? Math.max(...used) + 1 : 0;
  const range = start > LAYER_COUNT - 1 ? '' : (start + 7 > LAYER_COUNT - 1 ? `${start}-${LAYER_COUNT - 1}` : `${start}-${start + 7}`);
  const html = `<div class="layer-row">
    <input class="layer-range mono" type="text" spellcheck="false" value="${range}" placeholder="0-7" aria-label="${escapeHtml(t('train.layer.rangeAria'))}">
    <div class="layer-modules" aria-label="${escapeHtml(t('train.layer.modulesAria'))}">
      ${LORA_MODULES.map((name) => `<label><input type="checkbox" value="${name}" checked> ${name}</label>`).join('')}
    </div>
    <button class="btn btn-ghost btn-xs layer-remove" type="button" aria-label="${escapeHtml(t('train.layer.removeAria'))}">${escapeHtml(t('common.delete'))}</button>
  </div>`;
  $('#layer-rows')?.insertAdjacentHTML('beforeend', html);
  $$('#layer-rows .layer-row').at(-1)?.querySelector('.layer-range')?.focus();
  if (state.mode === 'form') { markDirty(); scheduleFormSync(); }
  readLayerControlsIntoConfig();
}

export function updateLrEffective() {
  const lrRaw = $('#cfg-lr')?.value || '';
  const steps = parseInt($('#cfg-steps')?.value || '0', 10) || 0;
  const lr = Number(lrRaw);
  const warmupRaw = $('#cfg-warmup')?.value || '';
  const warmupValue = Number(warmupRaw);
  const scheduler = $('#cfg-lr-scheduler')?.value || 'constant';
  let warmupSteps = null;
  if (Number.isFinite(warmupValue)) {
    warmupSteps = warmupValue > 0 && warmupValue < 1
      ? Math.max(1, Math.round(warmupValue * Math.max(steps, 1)))
      : Math.max(0, Math.round(warmupValue));
  }
  const lrText = Number.isFinite(lr) && lr > 0 ? fmtLr(lr) : t('common.msg9');
  const firstText = Number.isFinite(lr) && lr > 0 && warmupSteps
    ? fmtLr(lr / warmupSteps) : lrText;
  setText('#lr-effective', warmupSteps == null
    ? t('train.params.effectiveInvalidWarmup', { scheduler, lr: lrText, first: firstText })
    : t('train.params.effective', { scheduler, warmup: warmupSteps, lr: lrText, first: firstText }));
  const conversion = $('#warmup-conversion');
  if (conversion) {
    if (!Number.isFinite(warmupValue)) conversion.textContent = t('common.msg10');
    else if (warmupValue > 0 && warmupValue < 1) conversion.textContent = t('train.params.warmupRatio', { raw: warmupRaw, steps: warmupSteps, total: steps || '?', percent: (warmupValue * 100).toFixed(1) });
    else conversion.textContent = t('train.params.warmupAbsolute', { steps: warmupSteps });
  }
}

function syncRegCachePathControl(config = state.config) {
  const input = $('#cfg-cache-reg-dir');
  if (!input) return;
  input.placeholder = defaultRegCacheDir() ? t('cache.msg1') : t('cache.msg2');
  input.value = displayRegCacheValue(getPath(config, 'cache.reg_dir'));
  updateRegCacheEffective();
}

export function updateRegCachePlaceholder() {
  const input = $('#cfg-cache-reg-dir');
  if (input) input.placeholder = defaultRegCacheDir() ? t('cache.msg1') : t('cache.msg2');
}

export function updateRegCacheEffective() {
  const input = $('#cfg-cache-reg-dir');
  const raw = String(input?.value || '').trim();
  const value = raw ? absoluteWorkspacePath(raw) : defaultRegCacheDir();
  setText('#cfg-cache-reg-effective', t('dataset.effectivePath', { path: value || '--' }));
}

function syncCachePathControl(config = state.config) {
  const input = $('#cfg-cache-dir');
  if (!input) return;
  input.placeholder = defaultCacheDir() ? t('cache.msg3') : t('cache.msg4');
  input.value = displayCacheValue(getPath(config, 'cache.dir'));
  syncRegCachePathControl(config);
  updateCacheEffective();
}

export function updateCachePlaceholder() {
  const input = $('#cfg-cache-dir');
  if (input) input.placeholder = defaultCacheDir() ? t('cache.msg3') : t('cache.msg4');
}

export function updateCacheEffective() {
  const value = resolvedCacheBase();
  setText('#cfg-cache-effective', t('dataset.effectivePath', { path: value || '--' }));
}

export function syncSaveStateControls() {
  const saveState = $('#cfg-save-state');
  const select = $('#cfg-save-state-strategy');
  const row = $('#save-state-row');
  if (!saveState || !select) return;
  const enabled = Boolean(saveState.checked);
  select.disabled = !enabled;
  if (row) row.classList.toggle('is-disabled', !enabled);
  select.title = enabled
    ? t('common.msg11')
    : t('train.state.disabledTitle');
}

export function syncFormFromConfig(config) {
  for (const field of FIELD_DEFS) {
    const element = document.getElementById(field.id);
    if (!element) continue;
    const raw = getPath(config, field.path);
    if (raw === undefined || raw === null || raw === '') {
      element.value = field.blankDefault ? '' : String(field.default ?? '');
    } else if (field.type === 'intlist') {
      element.value = (Array.isArray(raw) ? raw : [raw]).join(', ');
    } else {
      element.value = String(raw);
    }
  }

  syncCachePathControl(config);
  syncAttentionControls();
  const checkpoint = getPath(config, 'vram.gradient_checkpointing');
  const checkpointField = $('#cfg-grad-ckpt');
  if (checkpointField) {
    checkpointField.value = checkpoint === true || checkpoint === 'true'
      ? 'true'
      : checkpoint === false || checkpoint === 'false' ? 'false' : 'auto';
  }

  const prompts = $('#cfg-prompts');
  if (prompts) prompts.value = arrayFromPrompts(getPath(config, 'preview.prompts')).join('\n');
  const saveState = $('#cfg-save-state');
  const saveStateRaw = getPath(config, 'train.save_state');
  if (saveState) {
    saveState.checked = saveStateRaw !== false && saveStateRaw !== 'off' && saveStateRaw !== 'false';
    const strategy = $('#cfg-save-state-strategy');
    if (strategy) strategy.value = saveStateRaw === 'all' ? 'all' : 'last';
    syncSaveStateControls();
  }
  const exportFormats = getPath(config, 'export.formats');
  const formats = Array.isArray(exportFormats) && exportFormats.length ? exportFormats : ['kohya'];
  if ($('#cfg-export-kohya')) $('#cfg-export-kohya').checked = formats.includes('kohya');
  if ($('#cfg-export-diffusers')) $('#cfg-export-diffusers').checked = formats.includes('diffusers');
  const layerState = { layers: getPath(config, 'lora.layers') || [], targets: getPath(config, 'lora.targets') || [] };
  syncLayerControls(layerState);
  updateLrEffective();
  resetModelProbes();
  updateOutputPreview();
}

export function readFormIntoConfig() {
  if (!state.config || typeof state.config !== 'object') state.config = {};
  for (const field of FIELD_DEFS) {
    const element = document.getElementById(field.id);
    if (!element) continue;
    const raw = String(element.value || '').trim();
    if (raw === '') {
      if (field.blankDefault) deletePath(state.config, field.path);
      else if (field.required) setPath(state.config, field.path, '');
      continue;
    }
    let value;
    if (field.type === 'number' || field.type === 'float') {
      value = field.integer ? parseInt(raw, 10) : parseFloat(raw);
      if (!Number.isFinite(value)) continue;
    } else if (field.type === 'intlist') {
      const parsed = parseIntList(raw);
      if (!parsed) continue;
      value = parsed;
    } else {
      value = raw;
    }
    setPath(state.config, field.path, value);
  }
  const promptsElement = $('#cfg-prompts');
  if (promptsElement) {
    setPath(state.config, 'preview.prompts',
      promptsElement.value.split('\n').map((line) => line.trim()).filter(Boolean));
  }
  const checkpointField = $('#cfg-grad-ckpt');
  if (checkpointField) {
    const value = checkpointField.value;
    setPath(state.config, 'vram.gradient_checkpointing',
      value === 'true' ? true : value === 'false' ? false : 'auto');
  }
  const saveState = $('#cfg-save-state');
  if (saveState) {
    const strategy = $('#cfg-save-state-strategy')?.value === 'all' ? 'all' : 'last';
    setPath(state.config, 'train.save_state', saveState.checked ? strategy : 'off');
  }
  const formats = [];
  if ($('#cfg-export-kohya')?.checked) formats.push('kohya');
  if ($('#cfg-export-diffusers')?.checked) formats.push('diffusers');
  setPath(state.config, 'export.formats', formats.length ? formats : ['kohya']);
  const negative = $('#cfg-preview-negative');
  if (negative) setPath(state.config, 'preview.negative_prompt', String(negative.value || ''));
  readLayerControlsIntoConfig();
  updateLrEffective();
  updateOutputPreview();
  return state.config;
}

export function updateOutputPreview() {
  const root = (state.settings?.paths?.workspaces || 'workspaces').trim().replace(/\/+$/, '') || 'workspaces';
  const wsPath = state.workspacePath || `${root}/${state.profileName || '<workspace>'}`;
  const hint = `${wsPath}/runs/<run-id>`;
  setText('#profile-output-hint', hint);
  if (state.trainTotal === 0 && !state.currentRun) {
    setText('#train-run-name', t('common.msg12'));
    setText('#train-run-dir', `${wsPath}/runs/<run-id>`);
  }
}

export function markDirty(dirty = true) {
  state.dirty = Boolean(dirty);
  const chip = $('#config-dirty');
  const stateChip = $('#config-state');
  if (chip) {
    chip.className = `chip ${state.dirty ? 'chip-warn' : 'chip-muted'}`;
    chip.textContent = state.dirty ? t('common.msg13') : t('common.unmodified');
  }
  if (stateChip) {
    stateChip.className = `chip ${state.dirty ? 'chip-warn' : 'chip-muted'}`;
    stateChip.textContent = state.dirty ? t('common.msg14') : t('common.saved');
  }
  const profileDirty = $('#profile-dirty');
  if (profileDirty) {
    profileDirty.className = `chip ${state.dirty ? 'chip-warn' : 'chip-muted'}`;
    profileDirty.textContent = state.dirty ? t('common.msg13') : t('common.saved');
  }
}

let formSyncTimer = null;

export function scheduleFormSync() {
  clearTimeout(formSyncTimer);
  formSyncTimer = setTimeout(() => {
    formSyncTimer = null;
    if (state.mode !== 'form') return;
    readFormIntoConfig();
    refreshYamlFromConfig();
  }, 160);
}

function flushFormSync() {
  if (formSyncTimer) {
    clearTimeout(formSyncTimer);
    formSyncTimer = null;
  }
  if (state.mode === 'form') {
    readFormIntoConfig();
    refreshYamlFromConfig();
  }
}

export function updateYamlLineCount() {
  const textarea = $('#yaml');
  if (!textarea) return;
  const count = textarea.value ? textarea.value.split('\n').length : 0;
  setText('#yaml-line-count', t('common.lines', { count }));
  renderYamlGutter(count, state.yamlError?.line || null);
}

function renderYamlGutter(count, errorLine = null) {
  const gutter = $('#yaml-gutter');
  if (!gutter) return;
  const lines = [];
  for (let index = 1; index <= Math.max(1, count); index += 1) {
    lines.push(errorLine === index ? `<span class="gutter-error">${index}</span>` : String(index));
  }
  gutter.innerHTML = lines.join('\n');
  syncYamlScroll();
}

export function syncYamlScroll() {
  const textarea = $('#yaml');
  const gutter = $('#yaml-gutter');
  if (textarea && gutter) gutter.scrollTop = textarea.scrollTop;
}

export function refreshYamlFromConfig() {
  if (state.mode !== 'form') return;
  const textarea = $('#yaml');
  if (!textarea) return;
  const scrollTop = textarea.scrollTop;
  const text = dumpYaml(state.config || {});
  textarea.value = text;
  textarea.scrollTop = scrollTop;
  clearYamlError();
  updateYamlLineCount();
}

export function getCurrentYaml() {
  if (state.mode === 'yaml') return $('#yaml')?.value || '';
  readFormIntoConfig();
  const text = dumpYaml(state.config || {});
  const textarea = $('#yaml');
  if (textarea) textarea.value = text;
  updateYamlLineCount();
  return text;
}

export function showYamlError(error) {
  const box = $('#yaml-error');
  if (!box) return;
  state.yamlError = {
    message: error.message || String(error),
    line: error.line || null,
    column: error.column || null,
  };
  const where = state.yamlError.line
    ? (state.yamlError.column
      ? t('train.yaml.locationWithColumn', { line: state.yamlError.line, column: state.yamlError.column })
      : t('train.yaml.location', { line: state.yamlError.line }))
    : '';
  box.textContent = t('train.yaml.parseError', {
    where,
    message: state.yamlError.message,
    hint: state.mode === 'yaml' ? t('train.yaml.clickHint') : '',
  });
  box.classList.remove('hidden');
  renderYamlGutter(($('#yaml')?.value || '').split('\n').length, state.yamlError.line);
  $('#yaml-details')?.setAttribute('open', '');
}

export function clearYamlError() {
  state.yamlError = null;
  const box = $('#yaml-error');
  if (box) {
    box.classList.add('hidden');
    box.textContent = '';
  }
  renderYamlGutter(($('#yaml')?.value || '').split('\n').length, null);
}

export function focusYamlLine(line) {
  const textarea = $('#yaml');
  if (!textarea || state.mode !== 'yaml') return;
  const lines = textarea.value.split('\n');
  const target = clamp(Number(line) - 1, 0, Math.max(0, lines.length - 1));
  let offset = 0;
  for (let index = 0; index < target; index += 1) offset += lines[index].length + 1;
  textarea.focus();
  textarea.setSelectionRange(offset, offset + (lines[target] || '').length);
  textarea.scrollTop = Math.max(0, target * 19.2 - 40);
  syncYamlScroll();
}

export function setMode(mode, options = {}) {
  if (mode === state.mode) return true;
  const textarea = $('#yaml');
  if (mode === 'yaml') {
    flushFormSync();
    let parsed;
    try {
      parsed = parseYaml(textarea?.value || '');
    } catch (error) {
      showYamlError(error);
      showAlert({
        type: 'error', title: t('common.msg15'),
        text: error.line
          ? t('train.yaml.fixBeforeAdvancedWithLine', { error: error.message, line: error.line })
          : t('train.yaml.fixBeforeAdvanced', { error: error.message }),
        persistent: true,
      });
      return false;
    }
    state.mode = 'yaml';
    if (state.profileName) parsed.name = state.profileName;
    state.config = parsed;
    syncFormFromConfig(parsed);
    textarea.readOnly = false;
    textarea.setAttribute('aria-label', t('common.msg16'));
    $('#yaml-editor')?.classList.add('is-editing');
    $('#yaml-actions')?.classList.remove('hidden');
    $('#btn-yaml-rebuild')?.classList.remove('hidden');
    setText('#yaml-mode-chip', t('common.msg17'));
    setText('#yaml-note', t('common.msg18'));
    setText('#mode-note', t('common.msg19'));
    setText('#btn-yaml-toggle', t('common.msg20'));
    setText('#btn-yaml-rebuild', t('common.msg21'));
    $('#config-form')?.classList.add('is-readonly');
    for (const element of $$('#config-form input, #config-form select, #config-form textarea, .dataset-config-fields input, .dataset-config-fields select, .dataset-config-fields textarea')) element.disabled = true;
    $('#mode-form')?.setAttribute('aria-pressed', 'false');
    $('#mode-yaml')?.setAttribute('aria-pressed', 'true');
    $('#mode-form')?.classList.remove('is-on');
    $('#mode-yaml')?.classList.add('is-on');
    const details = $('#yaml-details');
    if (details) details.open = true;
    markDirty();
    updateYamlLineCount();
    if (options.focus !== false) textarea.focus();
    return true;
  }

  let parsed;
  try {
    parsed = parseYaml(textarea?.value || '');
  } catch (error) {
    showYamlError(error);
    return false;
  }
  state.mode = 'form';
  state.config = parsed;
  syncFormFromConfig(parsed);
  if (textarea) {
    textarea.readOnly = true;
    textarea.setAttribute('aria-label', t('common.msg22'));
  }
  $('#yaml-editor')?.classList.remove('is-editing');
  $('#yaml-actions')?.classList.add('hidden');
  $('#btn-yaml-rebuild')?.classList.add('hidden');
  setText('#yaml-mode-chip', t('train.yaml.readonly'));
  setText('#yaml-note', t('common.msg23'));
  setText('#mode-note', t('common.msg24'));
  setText('#btn-yaml-toggle', t('common.msg25'));
  $('#config-form')?.classList.remove('is-readonly');
  for (const element of $$('#config-form input, #config-form select, #config-form textarea, .dataset-config-fields input, .dataset-config-fields select, .dataset-config-fields textarea')) element.disabled = false;
  syncAttentionControls();
  $('#mode-form')?.setAttribute('aria-pressed', 'true');
  $('#mode-yaml')?.setAttribute('aria-pressed', 'false');
  $('#mode-form')?.classList.add('is-on');
  $('#mode-yaml')?.classList.remove('is-on');
  const details = $('#yaml-details');
  if (details) details.open = false;
  clearYamlError();
  refreshYamlFromConfig();
  markDirty();
  if (options.focus !== false) $('#cfg-model-dit')?.focus();
  return true;
}
