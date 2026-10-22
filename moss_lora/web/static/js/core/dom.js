/* =========================================================================
 * js/core/dom.js — header
 * split from web/static/app.js, source 12-296 (13 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { workspaceParam } from './state.js';

export const $ = (selector, root = document) => root.querySelector(selector);

export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

export function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

export function setText(selector, text) {
  const el = typeof selector === 'string' ? $(selector) : selector;
  if (el) el.textContent = text == null ? '' : String(text);
}

export function setHidden(selector, hidden) {
  const el = typeof selector === 'string' ? $(selector) : selector;
  if (el) el.classList.toggle('hidden', Boolean(hidden));
}

function setChip(selector, variant, text) {
  const el = typeof selector === 'string' ? $(selector) : selector;
  if (!el) return;
  el.className = `chip ${variant || 'chip-muted'}`;
  const textEl = el.querySelector('span:last-child');
  if (textEl) textEl.textContent = text;
  else el.textContent = text;
}

function debounce(fn, delay) {
  let timer = null;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
  wrapped.cancel = () => clearTimeout(timer);
  return wrapped;
}

export function getPath(object, path) {
  return String(path).split('.').reduce((acc, key) => (acc == null ? undefined : acc[key]), object);
}

export function setPath(object, path, value) {
  const keys = String(path).split('.');
  let cursor = object;
  for (let i = 0; i < keys.length - 1; i += 1) {
    if (cursor[keys[i]] == null || typeof cursor[keys[i]] !== 'object') cursor[keys[i]] = {};
    cursor = cursor[keys[i]];
  }
  cursor[keys[keys.length - 1]] = value;
  return object;
}

export function deletePath(object, path) {
  const keys = String(path).split('.');
  let cursor = object;
  for (let i = 0; i < keys.length - 1; i += 1) {
    if (cursor == null || typeof cursor !== 'object') return object;
    cursor = cursor[keys[i]];
  }
  if (cursor && typeof cursor === 'object') delete cursor[keys[keys.length - 1]];
  return object;
}

export function fileUrl(path) {
  return '/api/files?path=' + encodeURIComponent(path);
}

const THUMB_STAMP = Date.now();   // bust the browser cache after a re-render

export function thumbUrl(path, width = 384) {
  const workspace = workspaceParam();
  return '/api/thumb?path=' + encodeURIComponent(path) + '&w=' + encodeURIComponent(width)
    + (workspace ? '&workspace=' + encodeURIComponent(workspace) : '')
    + '&v=' + THUMB_STAMP;
}
