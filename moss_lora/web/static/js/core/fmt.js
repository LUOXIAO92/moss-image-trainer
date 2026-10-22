/* =========================================================================
 * js/core/fmt.js — Small utilities
 * split from web/static/app.js, source 182-226 (6 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { t } from './i18n.js';

export function fmtNum(value, digits = 4) {
  if (value == null || !Number.isFinite(Number(value))) return '--';
  const n = Number(value);
  if (n !== 0 && Math.abs(n) < 1e-3) return n.toExponential(2);
  return n.toFixed(digits);
}

export function fmtInt(value) {
  if (value == null || !Number.isFinite(Number(value))) return '--';
  return Math.round(Number(value)).toLocaleString('en-US');
}

export function fmtLr(value) {
  if (value == null || !Number.isFinite(Number(value))) return '--';
  return Number(value).toExponential(2);
}

export function fmtDuration(seconds) {
  if (!Number.isFinite(Number(seconds)) || Number(seconds) < 0) return '--';
  const total = Math.round(Number(seconds));
  if (total < 60) return t('common.seconds', { value: total });
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 60) return t('common.minutesSeconds', { minutes, seconds: String(rest).padStart(2, '0') });
  const hours = Math.floor(minutes / 60);
  return t('common.hoursMinutes', { hours, minutes: String(minutes % 60).padStart(2, '0') });
}

export function fmtTime(timestamp) {
  if (!timestamp) return '--';
  const date = new Date(Number(timestamp) * 1000);
  if (Number.isNaN(date.getTime())) return '--';
  const pad = (n) => String(n).padStart(2, '0');
  const now = new Date();
  const sameYear = date.getFullYear() === now.getFullYear();
  const prefix = sameYear ? '' : `${date.getFullYear()}-`;
  return `${prefix}${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function fmtBytesFromMb(mb) {
  if (!Number.isFinite(Number(mb))) return '--';
  const value = Number(mb);
  if (value >= 1024) return `${(value / 1024).toFixed(2)} GB`;
  return `${value.toFixed(1)} MB`;
}
