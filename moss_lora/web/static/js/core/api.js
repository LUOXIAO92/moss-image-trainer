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
    this.line = options.line || null;
    this.column = options.column || null;
    this.body = options.body || null;
  }
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
    if (body && typeof body === 'object') {
      detail = body.detail || body.message || JSON.stringify(body);
      if (typeof detail !== 'string') detail = JSON.stringify(detail);
    } else {
      detail = String(body || '');
    }
    throw new ApiError(`${path} → HTTP ${response.status}${detail ? '：' + detail : ''}`, {
      status: response.status,
      detail,
      line: body && typeof body === 'object' ? body.line : null,
      column: body && typeof body === 'object' ? body.column : null,
      body: body && typeof body === 'object' ? body : null,
    });
  }
  return body;
}
