/* =========================================================================
 * js/config/form.js — Config <-> form
 * split from web/static/app.js, source 913-1562 (32 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { FIELD_DEFS, LAYER_COUNT, LORA_MODULES, absoluteWorkspacePath, arrayFromPrompts, compactLayerRange, defaultCacheDir, defaultRegCacheDir, displayCacheValue, displayRegCacheValue, parseLayerRange, parseIntList, residentToSwap, resolvedCacheBase, swapToResident } from './model.js';
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

export function syncQuantizeControls() {
  const dit = $('#cfg-quantize-dit');
  const dtype = $('#cfg-quantize-dtype');
  const conditioning = $('#cfg-quantize-conditioning');
  if (!dit || !dtype || !conditioning) return;
  const enabled = String(dit.value || 'none').toLowerCase() !== 'none';
  for (const select of [dtype, conditioning]) {
    select.disabled = !enabled;
    if (enabled) {
      select.removeAttribute('title');
      select.removeAttribute('aria-disabled');
    } else {
      select.title = t('train.params.quantizeDisabledTitle');
      select.setAttribute('aria-disabled', 'true');
    }
  }
}

// The config stores offloaded blocks; the form asks for resident ones. A blank
// field keeps vram.blocks_to_swap out of the YAML so the preset still decides.
function syncResidentBlocksControl(config = state.config) {
  const input = $('#cfg-resident-blocks');
  if (!input) return;
  const resident = swapToResident(getPath(config, 'vram.blocks_to_swap'));
  input.value = resident == null ? '' : String(resident);
}

// A value the select does not list (float32, a shifted sigma mode) stays
// visible as a one-off option instead of blanking the control and losing the
// key on the next save.
function syncSelectValue(select, value, fallback = '') {
  if (!select) return;
  for (const option of select.querySelectorAll('option[data-config-value]')) option.remove();
  const raw = value == null ? '' : String(value);
  if (raw && !Array.from(select.options).some((option) => option.value === raw)) {
    const option = document.createElement('option');
    option.value = raw;
    option.textContent = raw;
    option.dataset.configValue = 'true';
    select.append(option);
  }
  select.value = raw || fallback;
}

const PRESET_TIERS = ['32', '24', '16', '12', '8', '6'];

// The preset line is built in JS, so it cannot carry inline data-i18n text.
// These Chinese strings are its fallback until the language packs have the
// keys; the translations win as soon as they exist.
const PRESET_FALLBACK = {
  'train.preset.fromPreset': '预设',
  'train.preset.overridden': '你改过',
  'train.preset.on': '开',
  'train.preset.off': '关',
  'train.preset.quantize': '量化 {value}（{source}）',
  'train.preset.conditioning': 'conditioning {value}（{source}）',
  'train.preset.resident': '常驻 {count} 层（{source}）',
  'train.preset.checkpointing': '梯度检查点 {state}（{source}）',
  'train.preset.compile': 'compile {state}（{source}）',
  'train.preset.optimizer8bit': '8bit 优化器 {state}（{source}）',
  'train.preset.autoLine': '按显卡自动选：{tier}G · {summary}',
  'train.preset.explicit': '预设 {tier}G 生效：{summary}',
};

function presetT(key, vars = {}) {
  const entry = state.i18n.entries?.[key];
  const template = typeof entry === 'string' ? entry : PRESET_FALLBACK[key];
  if (typeof template !== 'string') return key;
  return template.replace(/\{(\w+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match
  ));
}

// Mirrors moss_lora/config.py::apply_preset: the first tier that is at most
// card_gb + 1 wins, and cards below 6 GB fall back to the 6 GB tier.
function presetTierForCard(totalGb) {
  const card = Number(totalGb);
  if (!Number.isFinite(card) || card <= 0) return '';
  for (const tier of PRESET_TIERS) {
    if (card >= Number(tier) - 1) return tier;
  }
  return '6';
}

function presetSource(fromPreset) {
  return fromPreset ? presetT('train.preset.fromPreset') : presetT('train.preset.overridden');
}

function presetOnOff(value) {
  return value ? presetT('train.preset.on') : presetT('train.preset.off');
}

function residentValue(value) {
  return Math.max(0, Math.min(LAYER_COUNT, Math.round(Number(value) || 0)));
}

function presetBool(value) {
  return value === true || String(value).toLowerCase() === 'true';
}

// What the resolved preset sets, and which of those knobs the config
// overrides. The tokens report the preset values: a knob is 预设 when the
// config sits at its "follow the preset" sentinel/default (auto checkpointing,
// -1/blank blocks, false optimizer, none quantization) or already equals the
// preset, and 你改过 when it carries a different deliberate value.
export function updatePresetSummary() {
  const line = $('#vram-preset-summary');
  if (!line) return;
  const config = state.config && typeof state.config === 'object' ? state.config : {};
  const vram = config.vram || {};
  const selected = String($('#cfg-vram-preset')?.value || vram.preset || 'auto').trim();
  const automatic = selected === '' || selected === 'auto';
  const tier = automatic ? presetTierForCard(state.system?.gpu?.total_gb) : selected.replace(/GB$/i, '');
  const preset = tier && state.presets ? state.presets[tier] : null;
  if (!preset) {
    line.textContent = '';
    return;
  }

  const presetResident = LAYER_COUNT - residentValue(preset.blocks_to_swap);
  const residentRaw = String($('#cfg-resident-blocks')?.value ?? '').trim();
  const residentCount = residentRaw === '' ? null : swapToResident(residentToSwap(residentRaw));
  const residentFromPreset = residentCount == null || residentCount === presetResident;

  const checkpointRaw = String($('#cfg-grad-ckpt')?.value || vram.gradient_checkpointing || 'auto');
  const presetCheckpoint = presetBool(preset.gradient_checkpointing);
  const checkpointValue = checkpointRaw === 'true' ? true : checkpointRaw === 'false' ? false : null;
  const checkpointFromPreset = checkpointRaw === 'auto' || checkpointValue === presetCheckpoint;

  const quantizeRaw = String($('#cfg-quantize-dit')?.value || vram.quantize_dit || 'none');
  const presetQuantize = String(preset.quantize_dit || 'none');
  const quantizeFromPreset = quantizeRaw === 'none' || quantizeRaw === presetQuantize;
  const quantizeDtype = String($('#cfg-quantize-dtype')?.value || vram.quantize_dtype || 'nf4');
  const quantizeText = presetQuantize === 'none'
    ? presetT('train.preset.off')
    : `${quantizeDtype}·${presetQuantize}`;
  const quantizeSourceFromPreset = quantizeFromPreset
    && (presetQuantize === 'none' || quantizeDtype === 'nf4');

  const optimizerRaw = Boolean(vram.optimizer_in_8bit);
  const presetOptimizer = presetBool(preset.optimizer_in_8bit);
  const optimizerFromPreset = !optimizerRaw || optimizerRaw === presetOptimizer;

  const compileRaw = vram.compile;
  const compileValue = compileRaw == null
    ? true
    : !(compileRaw === false || String(compileRaw).toLowerCase() === 'false');
  const presetCompile = presetBool(preset.compile);
  // compile has no "follow the preset" sentinel: true is the backend default,
  // so only an explicit false against a compiling preset counts as an override.
  const compileFromPreset = compileValue || !presetCompile;

  const conditioning = String($('#cfg-quantize-conditioning')?.value || vram.quantize_conditioning || 'none');
  const parts = [
    presetT('train.preset.quantize', { value: quantizeText, source: presetSource(quantizeSourceFromPreset) }),
  ];
  if (conditioning !== 'none') {
    parts.push(presetT('train.preset.conditioning', { value: conditioning, source: presetSource(false) }));
  }
  parts.push(presetT('train.preset.resident', { count: presetResident, source: presetSource(residentFromPreset) }));
  parts.push(presetT('train.preset.checkpointing', { state: presetOnOff(presetCheckpoint), source: presetSource(checkpointFromPreset) }));
  parts.push(presetT('train.preset.compile', { state: presetOnOff(presetCompile), source: presetSource(compileFromPreset) }));
  parts.push(presetT('train.preset.optimizer8bit', { state: presetOnOff(presetOptimizer), source: presetSource(optimizerFromPreset) }));

  const summary = parts.join(' · ');
  line.textContent = automatic
    ? presetT('train.preset.autoLine', { tier, summary })
    : presetT('train.preset.explicit', { tier, summary });
}

export function syncFormFromConfig(config) {
  for (const field of FIELD_DEFS) {
    if (field.type === 'resident') continue;
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
  syncQuantizeControls();
  syncResidentBlocksControl(config);
  syncSelectValue($('#cfg-dtype'), getPath(config, 'model.dtype'), 'bfloat16');
  syncSelectValue($('#cfg-sigma-mode'), getPath(config, 'train.sigma.mode'), '');
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
  updatePresetSummary();
}

export function readFormIntoConfig() {
  if (!state.config || typeof state.config !== 'object') state.config = {};
  for (const field of FIELD_DEFS) {
    if (field.type === 'resident') continue;
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
  const residentElement = $('#cfg-resident-blocks');
  if (residentElement) {
    const raw = String(residentElement.value || '').trim();
    const swap = raw === '' ? null : residentToSwap(raw);
    if (swap == null) deletePath(state.config, 'vram.blocks_to_swap');
    else setPath(state.config, 'vram.blocks_to_swap', swap);
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
  updatePresetSummary();
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
  // The dashboard estimate watches the same edits; this is the one signal every
  // form/YAML change and config load already goes through, so it needs no poll.
  document.dispatchEvent(new CustomEvent('moss:config-dirty'));
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
