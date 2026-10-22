/* =========================================================================
 * js/core/modal.js — Modal
 * split from web/static/app.js, source 8536-8570 (3 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { $, $$, setText } from './dom.js';
import { t } from './i18n.js';
import { state } from './state.js';

export function focusableInModal() {
  const modal = $('#modal');
  if (!modal) return [];
  return $$('a[href], area[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])', modal)
    .filter((element) => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length > 0
      && getComputedStyle(element).visibility !== 'hidden');
}

export function openModal(src, caption) {
  const modal = $('#modal');
  if (!modal) return;
  state.modal.lastFocus = document.activeElement;
  const image = $('#modal-img');
  if (image) {
    image.src = src;
    image.alt = caption || t('samplePreview.title');
  }
  setText('#modal-cap', caption || '');
  modal.classList.remove('hidden');
  document.body.classList.add('modal-open');
  $('#modal-close')?.focus();
}

export function closeModal() {
  const modal = $('#modal');
  if (!modal || modal.classList.contains('hidden')) return;
  modal.classList.add('hidden');
  document.body.classList.remove('modal-open');
  const image = $('#modal-img');
  if (image) image.removeAttribute('src');
  if (state.modal.lastFocus && document.contains(state.modal.lastFocus)) {
    state.modal.lastFocus.focus();
  }
  state.modal.lastFocus = null;
}
