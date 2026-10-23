/* =========================================================================
 * js/pages/settings.js — Export / settings
 * split from web/static/app.js, source 7740-8501 (31 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { getCurrentYaml, updateOutputPreview } from '../config/form.js';
import { api, describeError } from '../core/api.js';
import { $, escapeHtml, getPath, setHidden, setText } from '../core/dom.js';
import { fmtBytesFromMb, fmtInt } from '../core/fmt.js';
import { fetchI18nState, languageName, refreshLocalizedUi, renderLanguageSelectors, t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { state } from '../core/state.js';
import { loadSystem } from '../layout/topbar.js';
import { markCaptionSettingsDirty, renderCaptionSettings } from './caption.js';

const SETTINGS_FIELD_MAP = {
  'paths.workspaces': 'set-workspaces-path',
  'runtime.allocator': 'set-allocator',
  'runtime.gc_threshold': 'set-gc-threshold',
};

const SETTINGS_ALLOCATORS = ['expandable', 'async', 'off'];

const DEFAULT_GC_THRESHOLD = 0.9;

export function settingsField(id) { return document.getElementById(id); }

function runtimeAllocator(runtime) {
  const value = String(runtime?.allocator || '').trim().toLowerCase();
  if (SETTINGS_ALLOCATORS.includes(value)) return value;
  // Old settings used runtime.expandable_segments; both boolean states now
  // fall back to the new default because expandable is the default allocator.
  return 'expandable';
}

function runtimeGcThreshold(runtime) {
  const value = Number(runtime?.gc_threshold);
  if (!Number.isFinite(value)) return DEFAULT_GC_THRESHOLD;
  return Math.min(0.99, Math.max(0, value));
}

export function syncAllocatorControls() {
  const allocator = settingsField('set-allocator');
  const threshold = settingsField('set-gc-threshold');
  if (!threshold) return;
  threshold.disabled = allocator?.value !== 'expandable';
}

export function markSettingsDirty(dirty = true) {
  state.settingsDirty = Boolean(dirty);
  const chip = $('#settings-state');
  if (chip) {
    chip.className = `chip ${state.settingsDirty ? 'chip-warn' : 'chip-muted'}`;
    chip.textContent = state.settingsDirty ? t('common.msg13') : t('common.saved');
  }
}

function renderSettingsStatus() {
  const status = state.settingsStatus || {};
  const workspacesStatus = settingsField('set-workspaces-status');
  if (workspacesStatus) {
    const ok = Boolean(status.paths?.workspaces_exists);
    const abs = status.paths?.workspaces || state.settingsResolved?.paths?.workspaces || '';
    workspacesStatus.className = `field-help ${ok ? 'ok-mark' : 'warn-mark'}`;
    workspacesStatus.textContent = ok
      ? t('settings.paths.exists', { path: abs })
      : t('settings.paths.notCreated', { path: abs });
  }
}

function i18nBadge(language) {
  if (language.custom) return { label: t('settings.language.custom'), kind: 'chip-muted' };
  if (language.overridden) return { label: t('settings.language.overridden'), kind: 'chip-warn' };
  return { label: t('settings.language.builtin'), kind: 'chip-muted' };
}

export function renderI18nSettings() {
  const list = $('#i18n-language-list');
  if (!list) return;
  renderLanguageSelectors();
  renderI18nImportFileName();
  const languages = state.i18n.languages || [];
  setText('#i18n-language-summary', languages.length
    ? t('settings.language.summary', { count: languages.length, current: languageName() })
    : '');
  if (!languages.length) {
    list.innerHTML = `<div class="i18n-language-empty">${escapeHtml(t('settings.language.none'))}</div>`;
    return;
  }
  list.innerHTML = languages.map((language) => {
    const badge = i18nBadge(language);
    const current = language.code === state.i18n.current;
    const totalKeys = Object.keys(state.i18n.entries || {}).length;
    const missing = Math.max(Number(language.missing) || 0, totalKeys - (Number(language.keys) || 0));
    const name = escapeHtml(language.name || language.code);
    const code = escapeHtml(language.code);
    return `<div class="i18n-language-item ${current ? 'is-current' : ''}" data-i18n-language="${code}">
      <div class="i18n-language-ident">
        <span class="i18n-language-name">${name}</span>
        <span class="i18n-language-code mono">${code}</span>
        ${current ? `<span class="chip chip-ok">${escapeHtml(t('settings.language.inUse'))}</span>` : ''}
      </div>
      <div class="i18n-language-badges">
        <span class="chip ${badge.kind}">${escapeHtml(badge.label)}</span>
        <span class="i18n-language-missing">${escapeHtml(t('settings.language.missing', { count: missing }))}</span>
      </div>
      <div class="i18n-language-actions">
        ${current ? '' : `<button class="btn btn-ghost btn-xs" type="button" data-i18n-use="${code}">${escapeHtml(t('common.use'))}</button>`}
        <button class="btn btn-danger-ghost btn-xs" type="button" data-i18n-delete="${code}" data-i18n-builtin="${language.builtin ? '1' : ''}" data-i18n-name="${name}">${escapeHtml(language.builtin ? t('settings.language.restore') : t('common.delete'))}</button>
      </div>
    </div>`;
  }).join('');
}

function setI18nImportHint(text = '', kind = '') {
  const hint = $('#i18n-import-hint');
  if (!hint) return;
  hint.className = `field-help${kind ? ` ${kind}` : ''}`;
  hint.textContent = text;
}

export function renderI18nImportFileName() {
  const label = $('#i18n-import-name');
  if (!label) return;
  const name = String(state.i18n.importFileName || '').trim();
  label.textContent = name || t('settings.language.noFile');
  label.title = name;
}

export async function readI18nImportFile(event) {
  const file = event.target?.files?.[0];
  if (!file) return;
  state.i18n.importFileName = file.name;
  renderI18nImportFileName();
  try {
    const text = await file.text();
    const textarea = $('#i18n-import-text');
    if (textarea) textarea.value = text;
    let pack = null;
    try { pack = JSON.parse(text); } catch (_) { /* parsed on import */ }
    const code = $('#i18n-import-code');
    if (code && !code.value.trim()) {
      if (pack && typeof pack.$code === 'string') code.value = pack.$code;
      else if (/\.json$/i.test(file.name)) code.value = file.name.replace(/\.json$/i, '');
    }
    setI18nImportHint(t('settings.language.fileLoaded', { name: file.name }), 'ok-mark');
  } catch (error) {
    setI18nImportHint(t('settings.language.fileFailed', { error: error.message }), 'warn-mark');
  }
}

export async function importI18nPack() {
  const button = $('#i18n-import-btn');
  const textarea = $('#i18n-import-text');
  const codeInput = $('#i18n-import-code');
  const raw = String(textarea?.value || '').trim();
  if (!raw) {
    setI18nImportHint(t('settings.language.needJson'), 'warn-mark');
    textarea?.focus();
    return;
  }
  let pack;
  try {
    pack = JSON.parse(raw);
  } catch (error) {
    setI18nImportHint(t('settings.language.invalidJson', { error: error.message }), 'warn-mark');
    return;
  }
  if (!pack || typeof pack !== 'object' || Array.isArray(pack)) {
    setI18nImportHint(t('settings.language.invalidPack'), 'warn-mark');
    return;
  }
  const code = String(codeInput?.value || pack.$code || '').trim().toLowerCase();
  if (!/^[a-z][a-z0-9-]{1,7}$/.test(code)) {
    setI18nImportHint(t('settings.language.invalidCode'), 'warn-mark');
    codeInput?.focus();
    return;
  }
  const values = Object.entries(pack).filter(([key, value]) => !key.startsWith('$') && typeof value === 'string');
  if (!values.length) {
    setI18nImportHint(t('settings.language.emptyPack'), 'warn-mark');
    return;
  }
  pack.$code = code;
  if (!pack.$name) pack.$name = code;
  if (button) button.disabled = true;
  setI18nImportHint(t('settings.language.importing'), '');
  try {
    await api('/api/i18n/' + encodeURIComponent(code), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pack),
    });
    const fileInput = $('#i18n-import-file');
    if (fileInput) fileInput.value = '';
    state.i18n.importFileName = '';
    await fetchI18nState(state.i18n.current);
    if (code === state.i18n.current) await refreshLocalizedUi();
    else renderI18nSettings();
    setI18nImportHint(t('settings.language.importedHint', { code }), 'ok-mark');
    showSuccess(t('settings.language.importedTitle'), t('settings.language.importedText', { code, count: values.length }), [], `i18n-import-${code}`);
  } catch (error) {
    setI18nImportHint(describeError(error), 'warn-mark');
    showError(t('settings.language.importFailed'), describeError(error), [], 'i18n-import-error');
  } finally {
    if (button) button.disabled = false;
  }
}

export async function deleteI18nLanguage(code, builtin, name) {
  if (!code) return;
  const label = name || code;
  const prompt = builtin
    ? t('settings.language.confirmRestore', { name: label })
    : t('settings.language.confirmRemove', { name: label });
  if (!window.confirm(prompt)) return;
  try {
    const result = await api('/api/i18n/' + encodeURIComponent(code), { method: 'DELETE' });
    await fetchI18nState(state.i18n.current);
    await refreshLocalizedUi();
    if (!result.removed) {
      showAlert({ type: 'info', title: t('settings.language.noUserPackTitle'), text: t('settings.language.noUserPackText', { name: label }), timeout: 5000, key: 'i18n-delete-none' });
    } else if (builtin) {
      showSuccess(t('settings.language.restoredTitle'), t('settings.language.restoredText', { name: label }), [], `i18n-restore-${code}`);
    } else {
      showSuccess(t('settings.language.removedTitle'), t('settings.language.removedText', { name: label }), [], `i18n-remove-${code}`);
    }
  } catch (error) {
    showError(t('settings.language.deleteFailed'), describeError(error), [], 'i18n-delete-error');
  }
}

function renderSettingsForm() {
  const settings = state.settings;
  if (!settings) return;
  const set = (id, value) => { const el = settingsField(id); if (el) el.value = value == null ? '' : String(value); };
  const workspacesField = settingsField('set-workspaces-path');
  const workspacesAbs = state.settingsResolved?.paths?.workspaces || state.settingsStatus?.paths?.workspaces || '';
  const rawWorkspaces = String(settings.paths?.workspaces || '').trim();
  if (workspacesField) {
    workspacesField.placeholder = workspacesAbs || '/absolute/path/to/workspaces';
    workspacesField.value = rawWorkspaces.startsWith('/') ? rawWorkspaces : '';
  }
  const runtime = settings.runtime || {};
  const allocator = settingsField('set-allocator');
  if (allocator) allocator.value = runtimeAllocator(runtime);
  const threshold = settingsField('set-gc-threshold');
  if (threshold) threshold.value = String(runtimeGcThreshold(runtime));
  syncAllocatorControls();
  const file = $('#settings-file-path');
  if (file) file.textContent = state.settingsPath || 'settings.yaml';
  renderSettingsStatus();
  renderI18nSettings();
  markSettingsDirty(false);
  updateOutputPreview();
}

function readSettingsForm() {
  const allocatorValue = String(settingsField('set-allocator')?.value || '').trim().toLowerCase();
  const allocator = SETTINGS_ALLOCATORS.includes(allocatorValue) ? allocatorValue : 'expandable';
  const rawThreshold = parseFloat(settingsField('set-gc-threshold')?.value);
  const threshold = Number.isFinite(rawThreshold)
    ? Math.round(Math.min(0.99, Math.max(0, rawThreshold)) * 100) / 100
    : DEFAULT_GC_THRESHOLD;
  return {
    paths: {
      workspaces: settingsField('set-workspaces-path')?.value.trim()
        || state.settingsResolved?.paths?.workspaces
        || state.settingsStatus?.paths?.workspaces
        || 'workspaces',
    },
    runtime: {
      allocator,
      gc_threshold: threshold,
    },
  };
}

export function clearSettingsErrors() {
  setHidden('#settings-alert', true);
  setText('#settings-alert-text', '');
  const list = $('#settings-error-list');
  if (list) list.innerHTML = '';
  for (const id of Object.values(SETTINGS_FIELD_MAP)) {
    const input = settingsField(id);
    input?.removeAttribute('aria-invalid');
    input?.removeAttribute('title');
  }
}

function showSettingsErrors(errors = {}, warnings = []) {
  const box = $('#settings-alert');
  if (!box) return;
  box.classList.remove('hidden');
  setText('#settings-alert-title', t('settings.unsaved'));
  setText('#settings-alert-text', warnings.length ? warnings.join('；') : t('settings.msg1'));
  const list = $('#settings-error-list');
  if (list) {
    list.innerHTML = Object.entries(errors).map(([path, message]) => {
      const id = SETTINGS_FIELD_MAP[path];
      return `<li>${id ? `<button type="button" class="form-error-link" data-settings-field="${escapeHtml(id)}">` : ''}${escapeHtml(path)}：${escapeHtml(message)}${id ? '</button>' : ''}</li>`;
    }).join('');
  }
  for (const [path, message] of Object.entries(errors)) {
    const id = SETTINGS_FIELD_MAP[path];
    if (!id) continue;
    const input = settingsField(id);
    if (!input) continue;
    input.setAttribute('aria-invalid', 'true');
    input.title = message;
  }
  box.focus?.();
}

const MODEL_PROBE_FIELDS = {
  dit: 'cfg-model-dit',
  text_encoder: 'cfg-model-te',
  vae: 'cfg-model-vae',
};

export const MODEL_FIELD_KINDS = Object.fromEntries(Object.entries(MODEL_PROBE_FIELDS).map(([kind, id]) => [id, kind]));

export function setModelStatus(kind, result) {
  const el = document.getElementById(`${MODEL_PROBE_FIELDS[kind]}-status`);
  if (!el) return;
  el.className = 'model-status';
  if (!result) {
    el.textContent = t('train.model.notProbed');
    return;
  }
  if (result.ok) {
    el.classList.add('is-ok');
    el.textContent = result.size_mb != null
      ? t('train.model.passedSize', { size: fmtBytesFromMb(result.size_mb) })
      : t('train.msg78');
    return;
  }
  el.classList.add('is-err');
  el.textContent = result.reason || t('train.msg79');
}

export function resetModelProbes() {
  for (const kind of Object.keys(MODEL_PROBE_FIELDS)) setModelStatus(kind, null);
}

export async function probeModel(kind) {
  const id = MODEL_PROBE_FIELDS[kind];
  const input = id ? document.getElementById(id) : null;
  const button = document.querySelector(`[data-model-probe="${kind}"]`);
  if (!input) return;
  const value = input.value.trim();
  if (!value) {
    setModelStatus(kind, { ok: false, reason: t('train.msg80') });
    return;
  }
  if (button) { button.disabled = true; button.textContent = t('train.msg81'); }
  try {
    const result = await api('/api/models/probe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, path: value }),
    });
    setModelStatus(kind, result);
  } catch (error) {
    setModelStatus(kind, { ok: false, reason: describeError(error) });
  } finally {
    if (button) { button.disabled = false; button.textContent = t('train.model.probe'); }
  }
}

export function focusField(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (!el) return;
  el.focus({ preventScroll: true });
  el.scrollIntoView({ block: 'center' });
}

export function openOnboarding() {
  const dialog = $('#onboarding-dialog');
  if (!dialog || state.onboardingShown) return;
  state.onboardingShown = true;
  if (typeof dialog.showModal === 'function') {
    if (!dialog.open) dialog.showModal();
  } else {
    dialog.setAttribute('open', '');
  }
}

export function closeOnboarding() {
  const dialog = $('#onboarding-dialog');
  if (!dialog) return;
  if (typeof dialog.close === 'function') dialog.close();
  else dialog.removeAttribute('open');
}

export async function loadSettings() {
  const data = await api('/api/settings');
  state.settings = data.settings;
  state.settingsResolved = data.resolved;
  state.settingsStatus = data.status;
  state.settingsPath = data.path || 'settings.yaml';
  state.firstRun = Boolean(data.first_run);
  state.captionPrompts = data.caption_prompts || state.captionPrompts;
  renderSettingsForm();
  if (!state.captionSettingsDirty) {
    state.captionService = data.caption_service || state.captionService;
    state.captionClearKey = false;
    renderCaptionSettings(data.settings?.caption || {}, state.captionPrompts, state.captionService);
    markCaptionSettingsDirty(false);
  }
  clearSettingsErrors();
  return data;
}

export async function saveSettings({ silent = false } = {}) {
  const button = $('#btn-save-settings');
  if (button) button.disabled = true;
  clearSettingsErrors();
  try {
    const data = await api('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: readSettingsForm() }),
    });
    state.settings = data.settings;
    state.settingsResolved = data.resolved;
    state.settingsStatus = data.status;
    state.settingsPath = data.path || state.settingsPath;
    renderSettingsForm();
    clearSettingsErrors();
    markSettingsDirty(false);
    const warnings = data.warnings || [];
    if (warnings.length && !silent) {
      showAlert({ type: 'warn', title: t('settings.msg2'), text: warnings.join('；'), persistent: false, timeout: 8000, key: 'settings-warn' });
    } else if (!silent) {
      showSuccess(t('settings.msg3'), t('settings.msg4'), [], 'settings-saved');
    }
    runSetupCheck().catch(() => {});
    loadSystem().catch(() => {});
    return true;
  } catch (error) {
    if (error.status === 400 && error.body) {
      if (error.body.status) {
        state.settingsStatus = error.body.status;
        renderSettingsStatus();
      }
      if (error.body.errors) showSettingsErrors(error.body.errors, error.body.warnings || []);
      else showSettingsErrors({ settings: describeError(error) || t('settings.msg5') });
    } else if (error.status === 400 && describeError(error)) {
      showSettingsErrors({ settings: describeError(error) });
    } else {
      showError(t('settings.msg6'), describeError(error), [
        { label: t('common.retry'), onClick: () => saveSettings() },
      ]);
    }
    return false;
  } finally {
    if (button) button.disabled = false;
  }
}

function setupCheckItem({ key, ok, title, detail, tab, tabLabel, focus }) {
  const action = focus
    ? `<a class="btn btn-ghost btn-sm" href="#${tab}" data-focus-field="${escapeHtml(focus)}">${escapeHtml(tabLabel || t('setup.goFix'))}</a>`
    : `<a class="btn btn-ghost btn-sm" href="#${tab}">${escapeHtml(tabLabel || t('setup.goFix'))}</a>`;
  return `<article class="setup-item ${ok ? 'is-ok' : 'is-missing'}" data-setup-key="${escapeHtml(key)}">
    <span class="setup-icon" aria-hidden="true"><svg><use href="${ok ? '#i-check' : '#i-alert'}"></use></svg></span>
    <div class="setup-copy">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(detail)}</p>
    </div>
    ${ok ? `<span class="chip chip-ok">${escapeHtml(t('export.ready'))}</span>` : action}
  </article>`;
}

function renderSetupChecks(checks, error = '') {
  const panel = $('#setup-panel');
  if (!panel) return;
  const pending = checks.filter((item) => !item.ok).length;
  setText('#setup-summary', error ? t('train.msg82') : pending ? t('setup.pending', { count: pending }) : t('train.msg83'));
  const summary = $('#setup-summary');
  if (summary) summary.className = `chip ${pending || error ? 'chip-warn' : 'chip-ok'}`;
  const box = $('#setup-items');
  if (box) {
    box.innerHTML = checks.map((item) => setupCheckItem(item)).join('')
      + (error ? `<div class="setup-error">${escapeHtml(error)}</div>` : '');
  }
  const shouldShow = Boolean(error) || pending > 0;
  panel.classList.toggle('hidden', !shouldShow);
}

export async function runSetupCheck() {
  if (!state.profileName) {
    state.setup.data = { checks: [], error: '' };
    const panel = $('#setup-panel');
    if (panel) panel.classList.add('hidden');
    return;
  }
  if (state.setup.loading) return;
  state.setup.loading = true;
  try {
    if (!state.settings) await loadSettings().catch(() => {});
    const yamlText = getCurrentYaml();
    let data = null;
    let requestError = '';
    try {
      data = await api('/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml: yamlText, workspace: state.profileName }),
      });
    } catch (error) {
      requestError = describeError(error);
    }
    const problems = data?.problems || [];
    const yamlError = data?.yaml_error;
    const modelProblems = problems.filter((problem) => /^model\./.test(problem));
    const ditPath = String(getPath(state.config, 'model.dit') || '').trim();
    const vaePath = String(getPath(state.config, 'model.vae') || '').trim();
    const tePath = String(getPath(state.config, 'model.text_encoder') || '').trim();
    const modelsOk = Boolean(ditPath) && Boolean(vaePath) && modelProblems.length === 0;
    const missingModels = [];
    if (!ditPath) missingModels.push('DiT');
    if (!vaePath) missingModels.push('VAE');
    if (!tePath) missingModels.push(t('train.msg84'));
    const fileProblems = modelProblems.filter((problem) => !/is empty/.test(problem));
    const modelsDetail = fileProblems.length
      ? fileProblems.join('；')
      : missingModels.length
        ? t('setup.modelsMissing', { models: missingModels.join(t('common.listSeparator')) })
        : t('train.msg85');
    const modelFocus = !ditPath ? 'cfg-model-dit' : !vaePath ? 'cfg-model-vae' : 'cfg-model-te';
    const datasetPath = String(getPath(state.config, 'dataset.path') || '').trim();
    const datasetProblems = problems.filter((problem) => /^dataset\./.test(problem));
    const datasetOk = Boolean(datasetPath) && datasetProblems.length === 0;
    const datasetDetail = !datasetPath
      ? t('train.msg86')
      : datasetProblems.length ? datasetProblems.join(t('common.sentenceSeparator')) : t('setup.datasetReadable', { path: datasetPath });
    const cache = data?.cache;
    const cacheOk = Boolean(cache?.ready);
    const cacheDetail = cacheOk
      ? t('setup.cacheReady', { count: fmtInt(cache.items), preview: cache?.preview_ready ? '' : t('setup.previewMissing') })
      : cache?.legacy_index
        ? t('setup.cacheLegacy', { dir: cache?.dir_rel || getPath(state.config, 'cache.dir') || cache?.dir || t('settings.notConfigured') })
        : t('setup.cacheMissing', { dir: cache?.dir_rel || getPath(state.config, 'cache.dir') || cache?.dir || t('settings.notConfigured') });
    const checks = [
      { key: 'models', ok: modelsOk, title: t('train.model.legend'), detail: modelsDetail, tab: 'train',
        tabLabel: t('train.msg87'), focus: modelsOk ? '' : modelFocus },
      { key: 'dataset', ok: datasetOk, title: t('nav.dataset'), detail: datasetDetail, tab: 'dataset', tabLabel: t('train.msg88') },
      { key: 'cache', ok: cacheOk, title: t('nav.cache'), detail: cacheDetail, tab: 'cache', tabLabel: cache?.legacy_index ? t('train.msg89') : t('train.msg90') },
    ];
    const error = yamlError
      ? t('setup.yamlParseError', { line: yamlError.line || '?', error: yamlError.message })
      : requestError ? t('setup.requestFailed', { error: requestError }) : '';
    state.setup.data = { checks, error };
    renderSetupChecks(checks, error);
  } finally {
    state.setup.loading = false;
  }
}
