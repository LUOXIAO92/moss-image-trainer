/* =========================================================================
 * js/config/validate.js — Form validation
 * split from web/static/app.js, source 2363-2518 (7 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { FIELD_DEFS, parseIntList } from './model.js';
import { $, escapeHtml, getPath } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';

function fieldErrorElement(field) {
  const input = document.getElementById(field.id);
  if (!input) return null;
  const wrapper = input.closest('.field');
  if (!wrapper) return null;
  let error = wrapper.querySelector('.field-error');
  if (!error) {
    error = document.createElement('span');
    error.className = 'field-error';
    error.id = `error-${field.id}`;
    wrapper.append(error);
  }
  return error;
}

export function setFieldError(field, message) {
  const input = document.getElementById(field.id);
  if (!input) return;
  if (!message) {
    input.removeAttribute('aria-invalid');
    const error = input.closest('.field')?.querySelector('.field-error');
    if (error) {
      error.textContent = '';
      error.classList.add('hidden');
    }
    // Nothing to restore when the field never carried an error: the markup's
    // own aria-describedby is still in place.
    const described = input.dataset.originalDescribedby;
    if (described === undefined) return;
    if (described) input.setAttribute('aria-describedby', described);
    else input.removeAttribute('aria-describedby');
    return;
  }
  input.setAttribute('aria-invalid', 'true');
  const error = fieldErrorElement(field);
  if (error) {
    error.textContent = message;
    error.classList.remove('hidden');
  }
  if (!input.dataset.originalDescribedby) {
    input.dataset.originalDescribedby = input.getAttribute('aria-describedby') || '';
  }
  const ids = new Set((input.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
  ids.add(`error-${field.id}`);
  input.setAttribute('aria-describedby', Array.from(ids).join(' '));
}

export function validateField(field) {
  const input = document.getElementById(field.id);
  if (!input) return null;
  const raw = String(input.value || '').trim();
  if (field.required && raw === '') return t('common.required');
  if ((field.type === 'number' || field.type === 'float') && raw !== '') {
    const value = field.integer ? parseInt(raw, 10) : parseFloat(raw);
    if (!Number.isFinite(value)) return t('common.msg26');
    if (field.min != null && value < field.min) {
      if (field.min === 0) return t('common.msg27');
      return t('train.validation.min', { min: field.min });
    }
    if (field.integer && !Number.isInteger(value)) return t('common.msg28');
    if (field.multipleOf && value % field.multipleOf !== 0) return t('train.validation.multiple', { value: field.multipleOf });
  }
  if (field.type === 'resident' && raw !== '') {
    const value = Number(raw);
    if (!Number.isInteger(value) || value < 0 || value > field.max) {
      return t('train.validation.residentRange', { max: field.max });
    }
  }
  if (field.type === 'previewResident' && raw !== '') {
    const value = Number(raw);
    if (!Number.isInteger(value) || value < -1 || value > field.max) {
      return t('train.validation.previewResidentRange', { max: field.max });
    }
  }
  if (field.type === 'intlist' && raw !== '') {
    const list = parseIntList(raw);
    if (!list) return t('common.msg29');
    if (list.some((n) => n < (field.min ?? 0) || n > (field.max ?? 41))) {
      return t('train.validation.layerRange', { min: field.min ?? 0, max: field.max ?? 41 });
    }
  }
  if (field.id === 'cfg-alpha' && Number(input.value) <= 0) return t('common.msg30');
  if (field.id === 'cfg-lr' && raw !== '' && !(Number(raw) > 0)) return t('common.msg31');
  if (field.id === 'cfg-warmup' && raw !== '' && !(Number(raw) >= 0)) return t('common.msg32');
  return null;
}

export function validateConfigValues(config) {
  const errors = [];
  if (!String(getPath(config, 'dataset.path') || '').trim()) errors.push(t('train.msg37'));
  const numeric = (path, fallback) => {
    const value = getPath(config, path);
    return value == null || value === '' ? fallback : Number(value);
  };
  const intCheck = (path, fallback, label, min) => {
    const value = numeric(path, fallback);
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < min) {
      errors.push(t('train.validation.integerAtLeast', { label, min }));
    }
  };
  intCheck('train.steps', 1000, t('train.params.steps'), 1);
  const lr = numeric('train.lr', 0.0001);
  if (!Number.isFinite(lr) || lr <= 0) errors.push(t('common.msg31'));
  const warmup = numeric('train.warmup', 50);
  if (!Number.isFinite(warmup) || warmup < 0) errors.push(t('train.msg38'));
  intCheck('train.batch_size', 1, t('train.params.batch'), 1);
  intCheck('train.grad_accum', 1, t('train.params.accum'), 1);
  intCheck('lora.rank', 32, t('train.params.rank'), 1);
  intCheck('lora.alpha', 32, t('train.params.alpha'), 1);
  intCheck('train.save_every', 250, t('train.intervals.save'), 0);
  intCheck('train.sample_every', 250, t('train.intervals.sample'), 0);
  intCheck('train.eval_every', 100, t('train.intervals.eval'), 0);
  intCheck('preview.width', 1024, t('train.preview.width'), 64);
  intCheck('preview.height', 1024, t('train.preview.height'), 64);
  intCheck('preview.steps', 50, t('train.preview.steps'), 1);
  const width = numeric('preview.width', 1024);
  const height = numeric('preview.height', 1024);
  if (Number.isFinite(width) && width % 16 !== 0) errors.push(t('train.msg39'));
  if (Number.isFinite(height) && height % 16 !== 0) errors.push(t('train.msg40'));
  const guidance = numeric('preview.guidance', 3.0);
  if (!Number.isFinite(guidance) || guidance < 0) errors.push(t('train.msg41'));
  const shift = numeric('preview.shift', 5.0);
  if (!Number.isFinite(shift) || shift < 0) errors.push(t('train.msg42'));
  const seed = numeric('preview.seed', 42);
  if (!Number.isInteger(seed) || seed < -1) errors.push(t('train.msg43'));
  const skip = getPath(config, 'preview.uncond_skip_layers');
  const skipList = Array.isArray(skip) ? skip : (skip == null || skip === '' ? [22] : [skip]);
  if (!skipList.length || skipList.some((n) => !Number.isInteger(Number(n)) || Number(n) < 0 || Number(n) > 41)) {
    errors.push(t('train.msg44'));
  }
  const attention = String(getPath(config, 'train.attention') || 'auto');
  if (!['auto', 'flash', 'sdpa'].includes(attention)) errors.push(t('train.msg45'));
  const binTokens = numeric('train.bin_tokens', 0);
  if (!Number.isInteger(binTokens) || binTokens < 0) errors.push(t('train.msg46'));
  return errors;
}

export function validateForm(showInline = true) {
  const errors = [];
  for (const field of FIELD_DEFS) {
    const message = validateField(field);
    if (showInline) setFieldError(field, message);
    if (message) errors.push({ field, message });
  }
  if (state.layerErrors?.length) {
    errors.push({ field: { id: 'btn-add-layer-row', label: t('common.msg33') }, message: state.layerErrors.join('；') });
  }
  return errors;
}

export function showFormErrors(errors) {
  const box = $('#form-errors');
  const list = $('#form-errors-list');
  if (!box || !list) return;
  if (!errors.length) {
    box.classList.add('hidden');
    list.innerHTML = '';
    return;
  }
  list.innerHTML = errors.map(({ field, message }) => (
    `<li><button type="button" class="form-error-link" data-field="${escapeHtml(field.id)}">${escapeHtml(t(field.label))}：${escapeHtml(message)}</button></li>`
  )).join('');
  box.classList.remove('hidden');
  box.focus({ preventScroll: false });
  box.scrollIntoView({ block: 'nearest' });
}

export function clearFormErrors() {
  showFormErrors([]);
  for (const field of FIELD_DEFS) setFieldError(field, null);
}
