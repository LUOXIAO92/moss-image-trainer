/* =========================================================================
 * js/core/i18n.js — Small utilities
 * split from web/static/app.js, source 298-462 (14 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { renderRunStrip, updateJobUi } from '../components/jobs.js';
import { updateLogCount } from '../components/logview.js';
import { updateCacheEffective, updateCachePlaceholder, updateRegCacheEffective, updateRegCachePlaceholder, updateYamlLineCount } from '../config/form.js';
import { renderConfigList } from '../config/list.js';
import { renderProfileOptions } from '../config/profiles.js';
import { api } from './api.js';
import { $, escapeHtml, setText } from './dom.js';
import { showError, showSuccess } from './notify.js';
import { CAPTION_SOURCE_KEYS, JOB_LABEL_KEYS, state } from './state.js';
import { renderSystemChips } from '../layout/topbar.js';
import { route } from '../main.js';
import { renderDashboard } from '../pages/dashboard.js';
import { renderI18nImportFileName, renderI18nSettings } from '../pages/settings.js';

function formatI18n(template, vars = {}) {
  return String(template).replace(/\{(\w+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match
  ));
}

export function t(key, vars = {}) {
  const value = state.i18n.entries?.[key];
  return typeof value === 'string' ? formatI18n(value, vars) : key;
}

function hasI18nEntry(key) {
  return typeof state.i18n.entries?.[key] === 'string';
}

export function applyI18n(root = document) {
  const scope = root || document;
  const elements = [];
  if (scope.nodeType === 1) elements.push(scope);
  if (typeof scope.querySelectorAll === 'function') elements.push(...scope.querySelectorAll('*'));
  for (const element of elements) {
    if (element.tagName === 'TITLE') continue;
    const key = element.getAttribute?.('data-i18n');
    if (key && hasI18nEntry(key) && !element.hasAttribute('data-i18n-skip')) element.textContent = t(key);
    for (const [dataAttr, property] of [
      ['data-i18n-title', 'title'],
      ['data-i18n-placeholder', 'placeholder'],
      ['data-i18n-aria', 'aria-label'],
    ]) {
      const attrKey = element.getAttribute?.(dataAttr);
      if (attrKey && hasI18nEntry(attrKey)) element.setAttribute(property, t(attrKey));
    }
  }
}

export function jobLabel(kind) { return JOB_LABEL_KEYS[kind] ? t(JOB_LABEL_KEYS[kind]) : (kind || t('job.generic')); }

export function captionSourceLabel(source) {
  return CAPTION_SOURCE_KEYS[source] ? t(CAPTION_SOURCE_KEYS[source]) : '';
}

export function localeCode(code = state.i18n.current) {
  return ({ zh: 'zh-CN', en: 'en-US', ja: 'ja-JP', ko: 'ko-KR' })[code] || code || 'zh-CN';
}

export function languageName(code = state.i18n.current) {
  const found = (state.i18n.languages || []).find((item) => item.code === code);
  return found?.name || code || '—';
}

export function renderLanguageSelectors() {
  document.documentElement.lang = state.i18n.current === 'zh' ? 'zh-CN' : (state.i18n.current || 'zh');
  const languages = state.i18n.languages || [];
  const options = languages.map((language) => (
    `<option value="${escapeHtml(language.code)}">${escapeHtml(language.name)}</option>`
  )).join('');
  for (const selector of ['#ui-language-select', '#i18n-current-select']) {
    const select = $(selector);
    if (!select) continue;
    select.innerHTML = options;
    select.value = state.i18n.current || 'zh';
    select.disabled = !languages.length;
  }
  setText('#i18n-current-chip', languageName());
  const link = $('#i18n-template-link');
  if (link) link.href = `/api/i18n/template?base=${encodeURIComponent(state.i18n.current || 'zh')}`;
}

function applyI18nPayload(meta, pack) {
  state.i18n.languages = Array.isArray(meta?.languages) ? meta.languages : state.i18n.languages;
  state.i18n.current = meta?.current || state.i18n.current || 'zh';
  state.i18n.entries = pack && pack.code === state.i18n.current && pack.entries && typeof pack.entries === 'object'
    ? pack.entries
    : {};
  renderLanguageSelectors();
  applyI18n(document);
}

export async function fetchI18nState(preferred = state.i18n.current || 'zh') {
  const meta = await api('/api/i18n');
  const current = meta?.current || preferred || 'zh';
  let pack = null;
  try {
    pack = await api('/api/i18n/' + encodeURIComponent(current));
  } catch (error) {
    if (current !== 'zh') pack = await api('/api/i18n/zh').catch(() => null);
  }
  if (pack && pack.code !== current) {
    meta.current = pack.code;
  }
  applyI18nPayload(meta, pack);
  return { meta, pack };
}

export async function loadI18n() {
  const preferred = state.i18n.current || 'zh';
  const [metaResult, packResult] = await Promise.allSettled([
    api('/api/i18n'),
    api('/api/i18n/' + encodeURIComponent(preferred)),
  ]);
  const meta = metaResult.status === 'fulfilled' ? metaResult.value : null;
  let current = meta?.current || preferred || 'zh';
  let pack = packResult.status === 'fulfilled' ? packResult.value : null;
  if (!pack || pack.code !== current) {
    pack = await api('/api/i18n/' + encodeURIComponent(current)).catch(() => null);
  }
  if (!pack && current !== 'zh') {
    pack = await api('/api/i18n/zh').catch(() => null);
  }
  if (pack && pack.code !== current) current = pack.code;
  if (!pack) current = 'zh';
  if (meta) {
    applyI18nPayload({ ...meta, current }, pack);
  } else {
    state.i18n.current = pack?.code || 'zh';
    state.i18n.entries = pack?.entries || {};
    state.i18n.languages = [{ code: state.i18n.current, name: pack?.name || state.i18n.current || 'zh' }];
    renderLanguageSelectors();
    applyI18n(document);
  }
}

export async function setUiLanguage(code) {
  if (!code || code === state.i18n.current) return;
  const selects = ['#ui-language-select', '#i18n-current-select'].map((selector) => $(selector)).filter(Boolean);
  for (const select of selects) select.disabled = true;
  try {
    await api('/api/i18n/current', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    await fetchI18nState(code);
    await refreshLocalizedUi();
    showSuccess(t('settings.language.changedTitle'), t('settings.language.changedText', { name: languageName(code) }), [], 'language-changed');
  } catch (error) {
    renderLanguageSelectors();
    showError(t('settings.language.changeFailed'), error.detail || error.message, [], 'language-change-error');
  } finally {
    for (const select of selects) select.disabled = !state.i18n.languages.length;
  }
}

export async function refreshLocalizedUi() {
  const scrollY = window.scrollY;
  applyI18n(document);
  renderLanguageSelectors();
  renderI18nSettings();
  renderI18nImportFileName();
  if (state.system) renderSystemChips();
  updateCachePlaceholder();
  updateRegCachePlaceholder();
  updateCacheEffective();
  updateRegCacheEffective();
  updateLogCount();
  updateYamlLineCount();
  renderProfileOptions();
  renderConfigList();
  updateJobUi();
  renderRunStrip();
  if (state.dashboard.data) renderDashboard(state.dashboard.data);
  await route();
  applyI18n(document);
  window.scrollTo({ top: scrollY, behavior: 'auto' });
}
