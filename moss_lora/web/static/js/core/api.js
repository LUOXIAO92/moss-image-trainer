/* =========================================================================
 * js/core/api.js — API
 * split from web/static/app.js, source 468-511 (2 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { t } from './i18n.js';

class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status || 0;
    this.detail = options.detail || message;
    // The server sends either a plain string or {code, params, message}.
    this.code = options.code || '';
    this.params = options.params || null;
    this.line = options.line || null;
    this.column = options.column || null;
    this.body = options.body || null;
  }
}

/**
 * Human text for one server error payload: `{code, params, message}` or a
 * plain string.
 *
 * The i18n pack holds `err.<code>`; a pack that has not caught up falls back to
 * the server's Chinese `message`. Returns '' when there is nothing to say.
 */
export function describeDetail(detail) {
  if (detail && typeof detail === 'object' && detail.code) {
    // Codes already carry the "err." prefix; tolerate a bare one too.
    const key = String(detail.code).startsWith('err.') ? detail.code : `err.${detail.code}`;
    const text = t(key, detail.params || {});
    if (text !== key) return text;
    if (detail.message) return detail.message;
    return '';
  }
  return typeof detail === 'string' ? detail : '';
}

/** Same, for a thrown ApiError. Plain strings stay as they are. */
export function describeError(error) {
  const detail = error?.body && typeof error.body === 'object' ? error.body.detail : null;
  const text = describeDetail(detail);
  if (text) return text;
  if (typeof error?.detail === 'string' && error.detail) return error.detail;
  return error?.message || String(error ?? '');
}

export async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new ApiError(t('alert.connectFailed', { error: error.message }), { status: 0 });
  }
  const contentType = response.headers.get('content-type') || '';
  let body = null;
  try {
    body = contentType.includes('application/json') ? await response.json() : await response.text();
  } catch (_) {
    body = null;
  }
  if (!response.ok) {
    let detail = '';
    let code = '';
    let params = null;
    if (body && typeof body === 'object') {
      const raw = body.detail || body.message || body;
      if (raw && typeof raw === 'object') {
        code = raw.code || '';
        params = raw.params || null;
        // Kept as text so every caller that prints `error.detail` still has a
        // sentence; describeError() upgrades it to the current language.
        detail = raw.message || JSON.stringify(raw);
      } else {
        detail = String(raw || JSON.stringify(body));
      }
    } else {
      detail = String(body || '');
    }
    throw new ApiError(`${path} → HTTP ${response.status}${detail ? '：' + detail : ''}`, {
      status: response.status,
      detail,
      code,
      params,
      line: body && typeof body === 'object' ? body.line : null,
      column: body && typeof body === 'object' ? body.column : null,
      body: body && typeof body === 'object' ? body : null,
    });
  }
  return body;
}
