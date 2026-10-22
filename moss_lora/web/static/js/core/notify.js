/* =========================================================================
 * js/core/notify.js — Alerts
 * split from web/static/app.js, source 517-604 (5 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { $ } from './dom.js';
import { t } from './i18n.js';
import { state } from './state.js';

function alertIcon(type) {
  return type === 'success' ? '#i-check' : type === 'error' ? '#i-alert' : type === 'warn' ? '#i-alert' : '#i-info';
}

function dismissAlert(element) {
  if (!element) return;
  const key = element.dataset.key || '';
  if (key) state.alertKeys.delete(key);
  element.remove();
}

export function showAlert(options) {
  const {
    type = 'info', title = '', text = '', actions = [],
    key = '', persistent = false, timeout = 4500,
  } = options || {};
  const region = $('#alert-region');
  if (!region) return null;
  if (key && state.alertKeys.has(key)) {
    const existing = region.querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (existing) return existing;
  }

  const alert = document.createElement('div');
  alert.className = `alert alert-${type}`;
  alert.setAttribute('role', type === 'error' ? 'alert' : 'status');
  if (key) {
    alert.dataset.key = key;
    state.alertKeys.add(key);
  }

  const icon = document.createElement('span');
  icon.className = 'alert-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.innerHTML = `<svg><use href="${alertIcon(type)}"></use></svg>`;

  const copy = document.createElement('div');
  copy.className = 'alert-copy';
  const titleEl = document.createElement('div');
  titleEl.className = 'alert-title';
  titleEl.textContent = title;
  const textEl = document.createElement('div');
  textEl.className = 'alert-text';
  textEl.textContent = text;
  copy.append(titleEl);
  if (text) copy.append(textEl);

  if (actions.length) {
    const actionWrap = document.createElement('div');
    actionWrap.className = 'alert-actions';
    for (const action of actions) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `btn ${action.primary ? 'btn-primary' : 'btn-ghost'} btn-xs`;
      button.textContent = action.label;
      button.addEventListener('click', () => {
        if (typeof action.onClick === 'function') action.onClick();
        if (action.dismiss !== false) dismissAlert(alert);
      });
      actionWrap.append(button);
    }
    copy.append(actionWrap);
  }

  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'btn btn-ghost btn-xs alert-close';
  close.setAttribute('aria-label', t('common.msg2'));
  close.textContent = t('common.off');
  close.addEventListener('click', () => dismissAlert(alert));

  alert.append(icon, copy, close);
  region.append(alert);
  while (region.children.length > 3) dismissAlert(region.firstElementChild);

  if (!persistent && timeout > 0) {
    setTimeout(() => dismissAlert(alert), timeout);
  }
  return alert;
}

export function showError(title, text, actions = [], key = '') {
  return showAlert({ type: 'error', title, text, actions, key, persistent: true });
}

export function showSuccess(title, text, actions = [], key = '') {
  return showAlert({ type: 'success', title, text, actions, key, timeout: 5000 });
}
