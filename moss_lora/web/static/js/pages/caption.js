/* =========================================================================
 * js/pages/caption.js — Export / settings
 * split from web/static/app.js, source 7781-8353 (12 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { api } from '../core/api.js';
import { $, clamp, setHidden } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { state } from '../core/state.js';
import { settingsField } from './settings.js';

export function markCaptionSettingsDirty(dirty = true) {
  state.captionSettingsDirty = Boolean(dirty);
  const chip = $('#caption-settings-state');
  if (chip) {
    chip.className = `chip ${state.captionSettingsDirty ? 'chip-warn' : 'chip-muted'}`;
    chip.textContent = state.captionSettingsDirty ? t('common.msg13') : t('common.saved');
  }
}

const DEFAULT_CAPTION_SETTINGS = {
  trigger: '',
  workers: 6,
  retries: 3,
  temperature: 1.0,
  top_p: 0.95,
  max_tokens: 16384,
  thinking: true,
  base_url: '',
  api_key: '',
  model: '',
  prompt_tags: '',
  prompt_natural: '',
};

function captionNumber(id, fallback, min, max, integer = false) {
  const raw = String(settingsField(id)?.value ?? '').trim();
  const parsed = Number(raw);
  let value = raw !== '' && Number.isFinite(parsed) ? parsed : fallback;
  if (integer) value = Math.trunc(value);
  return clamp(value, min, max);
}

function readCaptionSettings() {
  return {
    trigger: String(settingsField('set-caption-trigger')?.value || '').trim(),
    workers: captionNumber('set-caption-workers', DEFAULT_CAPTION_SETTINGS.workers, 1, 32, true),
    retries: captionNumber('set-caption-retries', DEFAULT_CAPTION_SETTINGS.retries, 1, 10, true),
    temperature: captionNumber('set-caption-temperature', DEFAULT_CAPTION_SETTINGS.temperature, 0, 2),
    top_p: captionNumber('set-caption-top-p', DEFAULT_CAPTION_SETTINGS.top_p, 0.01, 1),
    max_tokens: captionNumber('set-caption-max-tokens', DEFAULT_CAPTION_SETTINGS.max_tokens, 256, 32768, true),
    thinking: Boolean(settingsField('set-caption-thinking')?.checked),
    base_url: String(settingsField('set-caption-base-url')?.value || '').trim(),
    api_key: String(settingsField('set-caption-api-key')?.value || '').trim(),
    model: String(settingsField('set-caption-model')?.value || '').trim(),
    prompt_tags: String(settingsField('set-caption-prompt-tags')?.value || ''),
    prompt_natural: String(settingsField('set-caption-prompt-natural')?.value || ''),
  };
}

export function updateCaptionTriggerWarning(showWarning = true) {
  const trigger = String(settingsField('set-caption-trigger')?.value || '').trim();
  setHidden('#set-caption-trigger-warning', !(showWarning && !trigger));
}

export function renderCaptionApiKeyStatus(service = state.captionService) {
  const source = String(service?.key_source || 'none');
  const status = $('#caption-api-key-status');
  if (!status) return;
  let text = t('dataset.reg.notConfigured');
  let kind = 'warn-mark';
  if (state.captionClearKey) {
    text = t('caption.msg14');
    kind = 'warn-mark';
  } else if (source === 'literal') {
    text = t('caption.msg15');
    kind = 'warn-mark';
  } else if (source === 'missing') {
    text = t('caption.msg16');
    kind = 'warn-mark';
  } else if (source === 'env' || source === 'legacy') {
    text = t('common.saved');
    kind = 'ok-mark';
  }
  status.className = `field-help ${kind}`;
  status.textContent = text;
}

export function updateCaptionApiKeyButton(service = state.captionService) {
  const button = $('#btn-clear-caption-key');
  const input = settingsField('set-caption-api-key');
  if (!button) return;
  const source = String(service?.key_source || 'none');
  button.disabled = state.captionClearKey || (!String(input?.value || '').trim() && source === 'none');
}

export function renderCaptionSettings(caption = {}, prompts = state.captionPrompts, service = state.captionService) {
  const values = { ...DEFAULT_CAPTION_SETTINGS, ...(caption || {}) };
  const promptDefaults = prompts || {};
  const serviceKey = service && Object.prototype.hasOwnProperty.call(service, 'api_key')
    ? service.api_key
    : values.api_key;
  const promptTags = values.prompt_tags == null ? '' : String(values.prompt_tags);
  const promptNatural = values.prompt_natural == null ? '' : String(values.prompt_natural);
  const fallbackPrompt = t('caption.prompts.placeholder');
  const set = (id, value) => {
    const element = settingsField(id);
    if (element) element.value = value == null ? '' : String(value);
  };
  const check = (id, value) => {
    const element = settingsField(id);
    if (element) element.checked = Boolean(value);
  };
  set('set-caption-trigger', values.trigger);
  set('set-caption-workers', values.workers);
  set('set-caption-retries', values.retries);
  set('set-caption-temperature', values.temperature);
  set('set-caption-top-p', values.top_p);
  set('set-caption-max-tokens', values.max_tokens);
  set('set-caption-base-url', values.base_url);
  set('set-caption-api-key', serviceKey);
  set('set-caption-model', values.model);
  renderCaptionApiKeyStatus(service);
  updateCaptionApiKeyButton(service);
  check('set-caption-thinking', values.thinking);
  const tagsPrompt = settingsField('set-caption-prompt-tags');
  if (tagsPrompt) {
    tagsPrompt.value = promptTags;
    tagsPrompt.placeholder = promptDefaults.tags || fallbackPrompt;
  }
  const naturalPrompt = settingsField('set-caption-prompt-natural');
  if (naturalPrompt) {
    naturalPrompt.value = promptNatural;
    naturalPrompt.placeholder = promptDefaults.natural || fallbackPrompt;
  }
  const resetButton = $('#btn-reset-caption-prompts');
  if (resetButton) resetButton.disabled = !promptDefaults.tags && !promptDefaults.natural;
  updateCaptionTriggerWarning(false);
}

export function resetCaptionPrompts() {
  const prompts = state.captionPrompts || {};
  if (!prompts.tags && !prompts.natural) {
    showAlert({ type: 'warn', title: t('caption.msg17'), text: t('caption.msg18'), timeout: 5000 });
    return;
  }
  const tags = settingsField('set-caption-prompt-tags');
  const natural = settingsField('set-caption-prompt-natural');
  const changed = Boolean(tags?.value || natural?.value);
  if (tags) {
    tags.value = '';
    tags.placeholder = prompts.tags || t('caption.prompts.placeholder');
  }
  if (natural) {
    natural.value = '';
    natural.placeholder = prompts.natural || t('caption.prompts.placeholder');
  }
  if (changed) markCaptionSettingsDirty();
  showAlert({ type: 'info', title: t('caption.msg19'), text: t('caption.msg20'), timeout: 5000 });
}

export function clearCaptionApiKey() {
  state.captionClearKey = true;
  const input = settingsField('set-caption-api-key');
  if (input) input.value = '';
  renderCaptionApiKeyStatus();
  updateCaptionApiKeyButton();
  markCaptionSettingsDirty();
}

export async function loadCaptionSettings() {
  const dirtyAtStart = state.captionSettingsDirty;
  const data = await api('/api/settings');
  state.settings = data.settings;
  state.settingsResolved = data.resolved;
  state.settingsStatus = data.status;
  state.settingsPath = data.path || state.settingsPath;
  state.captionPrompts = data.caption_prompts || state.captionPrompts;
  state.captionService = data.caption_service || state.captionService;
  state.captionClearKey = false;
  if (!state.captionSettingsDirty || dirtyAtStart) {
    renderCaptionSettings(data.settings?.caption || {}, state.captionPrompts, state.captionService);
    markCaptionSettingsDirty(false);
  }
  return data;
}

export async function saveCaptionSettings({ silent = false } = {}) {
  const button = $('#btn-save-caption');
  if (button) button.disabled = true;
  try {
    const caption = readCaptionSettings();
    const keySource = String(state.captionService?.key_source || 'none');
    if (state.captionClearKey) {
      caption.api_key = '';
    } else if (!caption.api_key && (keySource === 'literal' || keySource === 'legacy')) {
      delete caption.api_key;
    }
    const data = await api('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: { caption } }),
    });
    state.settings = data.settings;
    state.settingsResolved = data.resolved;
    state.settingsStatus = data.status;
    state.settingsPath = data.path || state.settingsPath;
    state.captionService = data.caption_service || state.captionService;
    state.captionClearKey = false;
    renderCaptionSettings(data.settings?.caption || caption, state.captionPrompts, state.captionService);
    markCaptionSettingsDirty(false);
    const warnings = data.warnings || [];
    if (warnings.length && !silent) {
      showAlert({ type: 'info', title: t('caption.msg21'), text: warnings.join('；'), timeout: 12000, key: 'caption-settings-warn' });
    } else if (!silent) {
      showSuccess(t('caption.msg21'), t('caption.msg22'), [], 'caption-settings-saved');
    }
    return true;
  } catch (error) {
    showError(t('caption.msg23'), error.detail || error.message, [
      { label: t('common.retry'), onClick: () => saveCaptionSettings() },
    ], 'caption-settings-error');
    return false;
  } finally {
    if (button) button.disabled = false;
  }
}
