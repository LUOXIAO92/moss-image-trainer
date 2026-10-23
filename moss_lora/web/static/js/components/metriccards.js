/* =========================================================================
 * js/components/metriccards.js — Dashboard
 * split from web/static/app.js, source 5069-5683 (30 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { drawChart, nearestPoint } from './chart.js';
import { api, describeError } from '../core/api.js';
import { $, $$, clamp, escapeHtml, setText } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { showAlert } from '../core/notify.js';
import { state } from '../core/state.js';
import { initSampleCompareSelection } from '../pages/dashboard.js';

const WIDGET_KEYS = ['loss', 'lr', 'sec', 'vram'];

const WIDGET_COLUMNS = 8;

const WIDGET_GAP = 10;

const WIDGET_ROW = 140;

const METRIC_CARD_COLORS = { lr: '#a78bfa', sec: '#fb923c', vram: '#f87171' };

const METRIC_CARD_LABELS = { lr: 'learning rate', sec: 's/step', vram: 'gpu' };

const METRIC_HOVER_LABELS = { lr: 'learning rate', sec: 's/step', vram: 'train.msg77' };

const metricHovers = {};

export function defaultDashboardLayout() {
  return {
    version: 2,
    checked: ['loss', 'lr', 'sec', 'vram'],
    cards: {
      loss: { x: 0, y: 0, w: 2, h: 4 },
      lr: { x: 2, y: 0, w: 2, h: 4 },
      sec: { x: 4, y: 0, w: 2, h: 4 },
      vram: { x: 6, y: 0, w: 2, h: 4 },
    },
    ranges: { loss: { min: null, max: null }, lr: { min: null, max: null }, sec: { min: null, max: null }, vram: { min: null, max: null } },
    sampleRuns: null,
    sampleCollapsed: false,
  };
}

function normalizeDashboardLayout(raw) {
  const layout = raw && typeof raw === 'object' ? JSON.parse(JSON.stringify(raw)) : defaultDashboardLayout();
  layout.checked = Array.isArray(layout.checked) ? layout.checked.filter((key) => WIDGET_KEYS.includes(key)) : ['loss'];
  layout.cards = layout.cards && typeof layout.cards === 'object' ? layout.cards : {};
  for (const key of layout.checked) {
    if (!layout.cards[key]) layout.cards[key] = { x: 0, y: 0, w: 2, h: 4 };
  }
  layout.ranges = layout.ranges && typeof layout.ranges === 'object' ? layout.ranges : {};
  layout.sampleRuns = Array.isArray(layout.sampleRuns) ? layout.sampleRuns : null;
  layout.sampleCollapsed = layout.sampleCollapsed === true;
  for (const key of WIDGET_KEYS) {
    const range = layout.ranges[key] || {};
    layout.ranges[key] = {
      min: range.min == null || range.min === '' ? null : (Number.isFinite(Number(range.min)) ? Number(range.min) : null),
      max: range.max == null || range.max === '' ? null : (Number.isFinite(Number(range.max)) ? Number(range.max) : null),
    };
    if (key === 'lr' && layout.ranges[key].min === 0 && layout.ranges[key].max === 1) {
      layout.ranges[key] = { min: null, max: null };
    }
  }
  return layout;
}

function widgetGridMetrics() {
  const grid = $('#widget-grid');
  const width = grid?.clientWidth || 900;
  const col = Math.max(36, (width - WIDGET_GAP * (WIDGET_COLUMNS - 1)) / WIDGET_COLUMNS);
  return { grid, width, col, row: WIDGET_ROW, gap: WIDGET_GAP };
}

function cardSlots(layout) {
  return layout.checked.filter((key) => key !== 'loss').map((key) => layout.cards[key]).filter(Boolean);
}

function findWidgetSlot(layout, w = 4, h = 4) {
  const occupied = Object.entries(layout.cards)
    .filter(([key]) => layout.checked.includes(key))
    .map(([, card]) => card);
  for (let y = 0; y < 40; y += 1) {
    for (let x = 0; x + w <= WIDGET_COLUMNS; x += 1) {
      const candidate = { x, y, w, h };
      const overlap = occupied.some((card) => (
        x < card.x + card.w && x + w > card.x && y < card.y + card.h && y + h > card.y
      ));
      if (!overlap) return candidate;
    }
  }
  return { x: 0, y: 0, w, h };
}

export function positionWidgets() {
  const layout = state.layout;
  const grid = $('#widget-grid');
  if (!layout || !grid) return;
  const { col, row, gap } = widgetGridMetrics();
  let maxBottom = 2;
  for (const [key, card] of Object.entries(layout.cards)) {
    if (!layout.checked.includes(key)) continue;
    const element = document.querySelector(`[data-widget="${key}"]`);
    if (!element) continue;
    const x = Math.max(0, Math.min(WIDGET_COLUMNS - 1, Number(card.x) || 0));
    const y = Math.max(0, Number(card.y) || 0);
    const w = Math.max(2, Math.min(WIDGET_COLUMNS - x, Number(card.w) || 4));
    const h = Math.max(2, Number(card.h) || 4);
    element.style.left = `${x * (col + gap)}px`;
    element.style.top = `${y * (row + gap)}px`;
    element.style.width = `${w * col + (w - 1) * gap}px`;
    element.style.height = `${h * row + (h - 1) * gap}px`;
    maxBottom = Math.max(maxBottom, y + h);
  }
  grid.style.height = `${maxBottom * (row + gap)}px`;
}

function metricCanvas(key) {
  return document.querySelector(`#metric-canvas-${key}`);
}

function formatMetricNumber(key, value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '--';
  if (key === 'lr') return number.toExponential(1);
  if (key === 'vram') return number.toFixed(1);
  return number.toFixed(2);
}

function metricGpuVisible() {
  return {
    mem: state.gpuVisible?.mem !== false,
    util: state.gpuVisible?.util !== false,
  };
}

function syncGpuLegend() {
  const hasUtil = (state.charts.dash?.metricSeries?.gpu || []).length > 0;
  for (const button of $$('[data-gpu-series]')) {
    const series = button.dataset.gpuSeries;
    const on = state.gpuVisible?.[series] !== false;
    button.classList.toggle('is-on', on);
    button.setAttribute('aria-pressed', String(on));
    if (series === 'util') {
      button.disabled = !hasUtil;
      button.title = hasUtil ? t('common.msg41') : t('common.msg42');
    }
  }
}

function drawMetricCard(key) {
  const canvas = metricCanvas(key);
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const width = Math.max(20, canvas.clientWidth || wrap?.clientWidth || 260);
  const height = Math.max(20, canvas.clientHeight || wrap?.clientHeight || 140);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const isGpu = key === 'vram';
  const series = state.charts.dash?.metricSeries || {};
  const memPoints = isGpu ? (series.vram || []) : (series[key] || []);
  const utilPoints = isGpu ? (series.gpu || []) : [];
  const gpuVisible = metricGpuVisible();
  const drawMem = isGpu && !gpuVisible.mem ? [] : memPoints;
  const drawUtil = isGpu && gpuVisible.util ? utilPoints : [];
  const allPoints = [...drawMem, ...drawUtil];
  if (!allPoints.length) {
    canvas._metricPlot = null;
    ctx.fillStyle = '#7c8ba1'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText(t('common.msg43'), width / 2, height / 2);
    hideMetricTooltip(key);
    setText(`#metric-last-${key}`, '--');
    if (isGpu) syncGpuLegend();
    return;
  }

  const scalePoints = memPoints.length ? memPoints : allPoints;
  const values = scalePoints.map((p) => p.y);
  const dataMin = Math.min(...values);
  let min = dataMin;
  let max = Math.max(...values);
  const range = metricRange(key);
  const rangeMin = range.min == null ? null : Number(range.min);
  const rangeMax = range.max == null ? null : Number(range.max);
  const manual = Number.isFinite(rangeMin) && Number.isFinite(rangeMax) && rangeMin < rangeMax;
  if (manual) {
    min = rangeMin;
    max = rangeMax;
  } else if (max - min < 1e-12) {
    min = min - Math.max(Math.abs(min) * 0.05, 1e-6);
    max = max + Math.max(Math.abs(max) * 0.05, 1e-6);
  }
  if (!manual && dataMin >= 0) min = 0;
  canvas.dataset.yMin = String(min);
  canvas.dataset.yMax = String(max);
  canvas.dataset.yMode = manual ? 'manual' : 'auto';
  const pad = { l: 48, r: isGpu ? 36 : 12, t: 12, b: 24 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const xMax = Math.max(...allPoints.map((p) => p.x), 1);
  const xOf = (v) => pad.l + (v / xMax) * plotW;
  const yOf = (v) => pad.t + (1 - (v - min) / (max - min)) * plotH;
  const yUtil = (v) => pad.t + (1 - clamp(Number(v) || 0, 0, 100) / 100) * plotH;

  ctx.strokeStyle = 'rgba(148, 163, 184, .07)';
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 3; i += 1) {
    const y = pad.t + plotH * i / 3;
    ctx.beginPath(); ctx.moveTo(pad.l, Math.round(y) + .5); ctx.lineTo(pad.l + plotW, Math.round(y) + .5); ctx.stroke();
    const value = max - (max - min) * i / 3;
    ctx.fillText(formatMetricNumber(key, value), pad.l - 5, y);
  }
  if (isGpu && drawUtil.length) {
    ctx.textAlign = 'left';
    ctx.fillStyle = '#fbbf24';
    for (let i = 0; i <= 3; i += 1) {
      const y = pad.t + plotH * i / 3;
      const value = 100 - (100 * i) / 3;
      ctx.fillText(`${Math.round(value)}%`, pad.l + plotW + 5, y);
    }
    ctx.fillStyle = '#94a3b8';
  }

  const strokeSeries = (points, color, dashed, alpha, lineWidth) => {
    if (!points.length) return;
    ctx.save();
    ctx.beginPath();
    points.forEach((p, i) => { const x = xOf(p.x), y = yOf(p.y); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(dashed ? [5, 3] : []);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  };
  strokeSeries(drawMem, METRIC_CARD_COLORS[key], false, .95, 1.8);
  if (isGpu && drawUtil.length) {
    ctx.save();
    ctx.beginPath();
    drawUtil.forEach((p, i) => { const x = xOf(p.x), y = yUtil(p.y); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.strokeStyle = '#fbbf24';
    ctx.globalAlpha = .95;
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }
  if (drawMem.length) {
    const last = drawMem[drawMem.length - 1];
    ctx.beginPath(); ctx.arc(xOf(last.x), yOf(last.y), 3, 0, Math.PI * 2); ctx.fillStyle = METRIC_CARD_COLORS[key]; ctx.fill();
  }
  if (drawUtil.length) {
    const last = drawUtil[drawUtil.length - 1];
    ctx.beginPath(); ctx.arc(xOf(last.x), yUtil(last.y), 3, 0, Math.PI * 2); ctx.fillStyle = '#fbbf24'; ctx.fill();
  }

  if (metricHovers[key] != null) {
    const hoverX = xOf(metricHovers[key]);
    if (hoverX >= pad.l - 1 && hoverX <= pad.l + plotW + 1) {
      ctx.save();
      ctx.strokeStyle = 'rgba(226, 232, 240, .38)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(hoverX, pad.t);
      ctx.lineTo(hoverX, pad.t + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      const memPoint = nearestPoint(drawMem, metricHovers[key]);
      if (memPoint) {
        ctx.beginPath(); ctx.arc(xOf(memPoint.x), yOf(memPoint.y), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = METRIC_CARD_COLORS[key]; ctx.fill();
        ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.5; ctx.stroke();
      }
      const utilPoint = nearestPoint(drawUtil, metricHovers[key]);
      if (utilPoint) {
        ctx.beginPath(); ctx.arc(xOf(utilPoint.x), yUtil(utilPoint.y), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24'; ctx.fill();
        ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.5; ctx.stroke();
      }
      ctx.restore();
    }
  }

  canvas._metricPlot = { pad, plotW, plotH, xMax, xOf, yOf, yUtil, memPoints: drawMem, utilPoints: drawUtil, isGpu };
  if (drawMem.length) setText(`#metric-last-${key}`, formatMetricNumber(key, drawMem[drawMem.length - 1].y));
  if (isGpu) syncGpuLegend();
}

function metricTooltipRows(key, step) {
  const canvas = metricCanvas(key);
  const plot = canvas?._metricPlot;
  if (!plot) return [];
  const rows = [];
  if (plot.isGpu) {
    const memPoint = nearestPoint(plot.memPoints, step);
    if (memPoint) rows.push({ label: t('train.msg77'), value: `${formatMetricNumber('vram', memPoint.y)} GiB` });
    const utilPoint = nearestPoint(plot.utilPoints, step);
    if (utilPoint) rows.push({ label: t('common.msg44'), value: `${Math.round(clamp(utilPoint.y, 0, 100))}%` });
  } else {
    const point = nearestPoint(plot.memPoints, step);
    if (point) rows.push({ label: t(METRIC_HOVER_LABELS[key] || key), value: formatMetricNumber(key, point.y) });
  }
  return rows;
}

function updateMetricTooltip(key, step) {
  const tooltip = document.getElementById(`metric-tooltip-${key}`);
  const canvas = metricCanvas(key);
  const plot = canvas?._metricPlot;
  if (!tooltip || !plot) return;
  const rows = metricTooltipRows(key, step);
  if (!rows.length) { tooltip.classList.add('hidden'); return; }
  tooltip.innerHTML = `<div class="tooltip-title">${escapeHtml(t('dash.chart.tooltipStep', { step: fmtInt(step) }))}</div>${rows.map((row) => `<div class="tooltip-row"><span>${escapeHtml(row.label)}</span><span class="mono">${escapeHtml(row.value)}</span></div>`).join('')}`;
  tooltip.classList.remove('hidden');
  const wrap = canvas.parentElement;
  const x = plot.xOf(step);
  const tooltipWidth = tooltip.offsetWidth || 150;
  const wrapWidth = wrap.clientWidth || 280;
  let left = x + 12;
  if (left + tooltipWidth > wrapWidth - 4) left = Math.max(4, x - tooltipWidth - 12);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = '4px';
}

function hideMetricTooltip(key) {
  document.getElementById(`metric-tooltip-${key}`)?.classList.add('hidden');
}

function bindMetricHover(card, key) {
  const canvas = metricCanvas(key);
  const wrap = card.querySelector('.metric-chart-wrap');
  if (!canvas || !wrap) return;
  const move = (event) => {
    const plot = canvas._metricPlot;
    if (!plot) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    if (x < plot.pad.l - 6 || x > plot.pad.l + plot.plotW + 6) {
      metricHovers[key] = null;
      drawMetricCard(key);
      hideMetricTooltip(key);
      return;
    }
    const rawStep = ((x - plot.pad.l) / plot.plotW) * plot.xMax;
    const nearest = nearestPoint([...plot.memPoints, ...plot.utilPoints], rawStep);
    if (!nearest) return;
    metricHovers[key] = nearest.x;
    drawMetricCard(key);
    updateMetricTooltip(key, nearest.x);
  };
  const leave = () => {
    metricHovers[key] = null;
    drawMetricCard(key);
    hideMetricTooltip(key);
  };
  wrap.addEventListener('pointermove', move);
  wrap.addEventListener('pointerdown', move);
  wrap.addEventListener('pointerleave', leave);
}

export function drawMetricCards() {
  for (const key of state.layout?.checked || []) {
    if (key !== 'loss') drawMetricCard(key);
  }
}

export function renderWidgetCards() {
  const layout = state.layout;
  const grid = $('#widget-grid');
  if (!layout || !grid) return;
  const lossCard = $('#widget-loss');
  if (lossCard) {
    lossCard.style.display = layout.checked.includes('loss') ? '' : 'none';
    if (layout.checked.includes('loss')) bindWidgetCard(lossCard, 'loss');
  }
  const wanted = layout.checked.filter((key) => key !== 'loss');
  for (const card of $$('#widget-grid .widget-card[data-widget]')) {
    const key = card.dataset.widget;
    if (key !== 'loss' && !wanted.includes(key)) card.remove();
  }
  for (const key of wanted) {
    let card = document.querySelector(`[data-widget="${key}"]`);
    if (!card) {
      const label = METRIC_CARD_LABELS[key] || key;
      const gpuLegend = key === 'vram' ? `
        <div class="metric-gpu-legend" aria-label="${escapeHtml(t('dash.metric.gpuLegendAria'))}">
          <button class="legend-toggle is-on" type="button" data-gpu-series="mem" aria-pressed="true"><span class="legend-line" style="color:${METRIC_CARD_COLORS.vram}"></span>${escapeHtml(t('dash.metric.memory'))}</button>
          <button class="legend-toggle is-on" type="button" data-gpu-series="util" aria-pressed="true"><span class="legend-line" style="color:#fbbf24;border-top-style:dashed"></span>${escapeHtml(t('dash.metric.utilization'))}</button>
        </div>
        <p class="metric-gpu-usage mono" id="metric-gpu-usage">--</p>` : '';
      grid.insertAdjacentHTML('beforeend', `<article class="widget-card" data-widget="${key}">
        <header class="widget-head"><span class="widget-title"><span class="widget-label">${escapeHtml(label)}</span> <span class="mono muted" id="metric-last-${key}">--</span></span>
          <div class="widget-settings" aria-label="${escapeHtml(t('dash.metric.yRangeAria', { label }))}"><span>Y</span><input class="input-compact" id="metric-y-min-${key}" type="text" inputmode="decimal" placeholder="${escapeHtml(t('common.auto'))}" aria-label="${escapeHtml(t('dash.metric.yMinAria', { label }))}"><span>–</span><input class="input-compact" id="metric-y-max-${key}" type="text" inputmode="decimal" placeholder="${escapeHtml(t('common.auto'))}" aria-label="${escapeHtml(t('dash.metric.yMaxAria', { label }))}"></div>
          <button class="widget-close" type="button" data-close-widget="${key}" aria-label="${escapeHtml(t('dash.metric.closeAria', { label }))}">×</button>
          <span class="widget-drag" title="${escapeHtml(t('dash.widget.drag'))}">⠿</span>
        </header>
        ${gpuLegend}
        <div class="metric-chart-wrap"><canvas id="metric-canvas-${key}" role="img" aria-label="${escapeHtml(t('dash.metric.chartAria', { label }))}"></canvas><div class="chart-tooltip metric-tooltip hidden" id="metric-tooltip-${key}" role="tooltip"></div></div>
        <span class="widget-resize" title="${escapeHtml(t('dash.widget.resize'))}"></span>
      </article>`);
      card = document.querySelector(`[data-widget="${key}"]`);
      bindWidgetCard(card, key);
    }
  }
  syncLayoutControls();
  syncRangeInputs();
  positionWidgets();
  drawMetricCards();
}

function syncRangeInputs() {
  if (!state.layout) return;
  for (const key of WIDGET_KEYS) {
    const range = metricRange(key);
    const minId = key === 'loss' ? 'loss-y-min' : `metric-y-min-${key}`;
    const maxId = key === 'loss' ? 'loss-y-max' : `metric-y-max-${key}`;
    const minInput = document.getElementById(minId);
    const maxInput = document.getElementById(maxId);
    if (minInput && document.activeElement !== minInput) minInput.value = range.min == null ? '' : formatMetricNumber(key, range.min);
    if (maxInput && document.activeElement !== maxInput) maxInput.value = range.max == null ? '' : formatMetricNumber(key, range.max);
  }
}

function syncLayoutControls() {
  if (!state.layout) return;
  state.chartMetrics = { loss: false, lr: false, sec: false, vram: false };
  for (const key of state.layout.checked) state.chartMetrics[key] = true;
  $('#btn-reset-layout')?.addEventListener('click', () => { resetDashboardLayout().catch(() => {}); });
  $('#widget-grid')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-close-widget]');
    if (button) closeWidgetCard(button.dataset.closeWidget).catch(() => {});
  });
  for (const input of $$('#chart-controls [data-metric]')) {
    input.checked = Boolean(state.chartMetrics[input.dataset.metric]);
  }
}

function widgetRect(card) {
  const metrics = widgetGridMetrics();
  return {
    left: card.x * (metrics.col + metrics.gap),
    top: card.y * (metrics.row + metrics.gap),
    width: card.w * metrics.col + (card.w - 1) * metrics.gap,
    height: card.h * metrics.row + (card.h - 1) * metrics.gap,
  };
}

function positionPlaceholder(placeholder, card) {
  if (!placeholder || !card) return;
  const front = $('#widget-grid');
  if (!front) return;
  const metrics = widgetGridMetrics();
  placeholder.style.left = `${card.x * (metrics.col + metrics.gap)}px`;
  placeholder.style.top = `${card.y * (metrics.row + metrics.gap)}px`;
  placeholder.style.width = `${card.w * metrics.col + (card.w - 1) * metrics.gap}px`;
  placeholder.style.height = `${card.h * metrics.row + (card.h - 1) * metrics.gap}px`;
}

function bindWidgetCard(card, key) {
  if (!card || card.dataset.bound === '1') return;
  card.dataset.bound = '1';
  const startPointer = (event, mode) => {
    if (event.button != null && event.button !== 0) return;
    event.preventDefault();
    const layout = state.layout;
    const value = layout?.cards?.[key];
    if (!value) return;
    const metrics = widgetGridMetrics();
    const startX = event.clientX, startY = event.clientY;
    const initial = { ...value };
    const rect = widgetRect(initial);
    const target = { ...initial };
    const placeholder = document.createElement('div');
    placeholder.className = 'widget-placeholder';
    $('#widget-grid')?.appendChild(placeholder);
    positionPlaceholder(placeholder, target);
    card.classList.add('is-dragging');
    card.style.zIndex = '5';
    card.setPointerCapture?.(event.pointerId);
    const move = (moveEvent) => {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      if (mode === 'move') {
        card.style.left = `${rect.left + dx}px`;
        card.style.top = `${rect.top + dy}px`;
        target.x = Math.max(0, Math.min(WIDGET_COLUMNS - initial.w, Math.round((rect.left + dx) / (metrics.col + metrics.gap))));
        target.y = Math.max(0, Math.round((rect.top + dy) / (metrics.row + metrics.gap)));
      } else {
        const w = Math.max(metrics.col, rect.width + dx);
        const h = Math.max(metrics.row, rect.height + dy);
        card.style.width = `${w}px`;
        card.style.height = `${h}px`;
        target.w = Math.max(2, Math.min(WIDGET_COLUMNS - initial.x, Math.round((w + metrics.gap) / (metrics.col + metrics.gap))));
        target.h = Math.max(2, Math.round((h + metrics.gap) / (metrics.row + metrics.gap)));
      }
      positionPlaceholder(placeholder, target);
    };
    const up = () => {
      card.removeEventListener('pointermove', move);
      card.removeEventListener('pointerup', up);
      card.removeEventListener('pointercancel', up);
      card.classList.remove('is-dragging');
      card.style.zIndex = '';
      placeholder.remove();
      Object.assign(value, target);
      positionWidgets();
      if (key !== 'loss') drawMetricCard(key); else drawChart('dash');
      saveDashboardLayout();
    };
    card.addEventListener('pointermove', move);
    card.addEventListener('pointerup', up);
    card.addEventListener('pointercancel', up);
  };
  const bindRange = (id, which) => {
    const input = card.querySelector('#' + id);
    if (!input) return;
    input.addEventListener('input', () => setMetricRange(key, which, input.value));
    input.addEventListener('change', () => setMetricRange(key, which, input.value));
    input.addEventListener('blur', () => syncRangeInputs());
  };
  bindRange(key === 'loss' ? 'loss-y-min' : `metric-y-min-${key}`, 'min');
  bindRange(key === 'loss' ? 'loss-y-max' : `metric-y-max-${key}`, 'max');
  if (key !== 'loss') bindMetricHover(card, key);
  for (const button of card.querySelectorAll('[data-gpu-series]')) {
    button.addEventListener('click', () => {
      const series = button.dataset.gpuSeries;
      if (!['mem', 'util'].includes(series)) return;
      state.gpuVisible = state.gpuVisible || { mem: true, util: true };
      state.gpuVisible[series] = state.gpuVisible[series] === false;
      syncGpuLegend();
      drawMetricCard('vram');
    });
  }
  card.querySelector('.widget-drag')?.addEventListener('pointerdown', (event) => startPointer(event, 'move'));
  card.querySelector('.widget-resize')?.addEventListener('pointerdown', (event) => startPointer(event, 'resize'));
}

export function metricRange(key) {
  return state.layout?.ranges?.[key] || { min: null, max: null };
}

function setMetricRange(key, which, value) {
  if (!state.layout) state.layout = defaultDashboardLayout();
  state.layout.ranges = state.layout.ranges || {};
  state.layout.ranges[key] = state.layout.ranges[key] || { min: null, max: null };
  const raw = String(value || '').trim();
  const number = raw === '' ? null : Number(raw);
  state.layout.ranges[key][which] = Number.isFinite(number) ? number : null;
  if (key === 'loss') drawChart('dash'); else drawMetricCard(key);
  saveDashboardLayout();
}

async function resetDashboardLayout() {
  state.layout = defaultDashboardLayout();
  state.layoutLoaded = true;
  state.sampleCompareInitialized = false;
  state.sampleCompareSelection = new Set();
  state.sampleCompareData = [];
  state.layout.sampleCollapsed = false;
  if (state.layoutSaving) await new Promise((resolve) => setTimeout(resolve, 120));
  renderWidgetCards();
  drawChart('dash');
  await saveDashboardLayout();
  initSampleCompareSelection();
}

async function closeWidgetCard(key) {
  const input = document.querySelector(`#chart-controls [data-metric="${key}"]`);
  if (input) input.checked = false;
  await toggleMetricCard(key, false);
}

export async function loadDashboardLayout() {
  if (!state.profileName) {
    state.layout = defaultDashboardLayout();
    state.layoutLoaded = true;
    renderWidgetCards();
    return;
  }
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/layout');
    state.layout = normalizeDashboardLayout(data.layout);
    state.layoutLoaded = true;
    renderWidgetCards();
    drawChart('dash');
    if (state.dashboard.data) initSampleCompareSelection();
  } catch (error) {
    state.layout = defaultDashboardLayout();
    state.layoutLoaded = true;
    renderWidgetCards();
  }
}

export async function saveDashboardLayout() {
  if (!state.layout || !state.profileName || state.layoutSaving) return;
  state.layoutSaving = true;
  try {
    await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/layout', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layout: state.layout }),
    });
  } catch (error) {
    showAlert({ type: 'warn', title: t('dash.msg11'), text: describeError(error), timeout: 5000, key: 'layout-save' });
  } finally {
    state.layoutSaving = false;
  }
}

export async function toggleMetricCard(key, checked) {
  if (!state.layout) state.layout = defaultDashboardLayout();
  const set = new Set(state.layout.checked);
  if (checked) set.add(key); else set.delete(key);
  if (key === 'loss' && !checked) set.delete('loss');
  state.layout.checked = WIDGET_KEYS.filter((item) => set.has(item));
  if (checked && !state.layout.cards[key]) state.layout.cards[key] = findWidgetSlot(state.layout, 2, 4);
  renderWidgetCards();
  drawChart('dash');
  await saveDashboardLayout();
}
