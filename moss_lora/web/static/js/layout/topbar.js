/* =========================================================================
 * js/layout/topbar.js — Export / settings
 * split from web/static/app.js, source 8503-8530 (2 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { api } from '../core/api.js';
import { $, setText } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';

export async function loadSystem() {
  const data = await api('/api/system');
  state.system = data;
  state.lastSystemAt = Date.now();
  renderSystemChips();
}

export function renderSystemChips() {
  const gpu = state.system?.gpu || {};
  const parts = [];
  if (gpu.utilization != null && Number.isFinite(Number(gpu.utilization))) {
    parts.push(t('gpu.usage', { value: Math.round(Number(gpu.utilization)) }));
  }
  if (gpu.temperature != null && Number.isFinite(Number(gpu.temperature))) {
    parts.push(t('gpu.temperature', { value: Math.round(Number(gpu.temperature)) }));
  }
  if (gpu.used_gb != null && gpu.total_gb != null) {
    parts.push(t('gpu.memoryCompact', { used: Number(gpu.used_gb).toFixed(1), total: Number(gpu.total_gb).toFixed(1) }));
  }
  if (gpu.power != null && Number.isFinite(Number(gpu.power))) {
    parts.push(t('gpu.power', { value: Math.round(Number(gpu.power)) }));
  }
  setText('#gpu-chip-text', gpu.name ? (parts.length ? parts.join(' · ') : 'GPU --') : t('gpu.unavailable'));
  const chip = $('#gpu-chip');
  if (chip) chip.title = gpu.name
    ? t('gpu.titleWithName', { name: gpu.name })
    : t('gpu.unavailable');
}
