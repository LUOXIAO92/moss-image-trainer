/* =========================================================================
 * js/components/chart.js — Charts
 * split from web/static/app.js, source 2524-3082 (20 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { drawMetricCards, metricRange } from './metriccards.js';
import { $, clamp, escapeHtml } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';

const CHART_COLORS = {
  train: '#38bdf8',
  eval: '#fbbf24',
  avg: '#4ade80',
};

const CHART_LABELS = { train: 'dash.chart.rawLoss', eval: 'dash.chart.eval', avg: 'dash.chart.average' };

function rgba(hex, alpha) {
  const value = String(hex || '#38bdf8').replace('#', '');
  const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function chartConfig(name) {
  return {
    canvas: '#dash-chart', wrap: '#dash-chart-wrap', tooltip: '#dash-tooltip',
    empty: '#dash-empty', readout: '#dash-readout', legend: '#dash-legend',
  };
}

function emptySeries() {
  return { train: [], eval: [], avg: [] };
}

export function initCharts() {
  for (const name of ['dash']) {
    const config = chartConfig(name);
    const canvas = $(config.canvas);
    const wrap = $(config.wrap);
    if (!canvas || !wrap) continue;
    const legendElement = $(config.legend);
    const chart = {
      name, canvas, wrap,
      tooltip: $(config.tooltip),
      empty: $(config.empty),
      readout: $(config.readout),
      legend: legendElement,
      ctx: canvas.getContext('2d'),
      series: emptySeries(),
      visible: { train: true, eval: true, avg: true },
      hoverStep: null,
      hoverSource: '',
      lastPlot: null,
      ariaLabel: '',
    };
    state.charts[name] = chart;

    const pointerMove = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      chart.hoverSource = 'pointer';
      moveChartHover(chart, x, y);
    };
    wrap.addEventListener('pointermove', pointerMove);
    wrap.addEventListener('pointerdown', pointerMove);
    wrap.addEventListener('pointerleave', () => {
      if (chart.hoverSource === 'pointer') {
        chart.hoverSource = '';
        chart.hoverStep = null;
        drawChart(name);
        hideChartTooltip(chart);
      }
    });
    wrap.addEventListener('focus', () => {
      if (chart.hoverStep == null) {
        const points = visiblePoints(chart);
        if (points.length) {
          chart.hoverStep = points[points.length - 1].x;
          chart.hoverSource = 'keyboard';
          drawChart(name);
        }
      }
    });
    wrap.addEventListener('blur', () => {
      if (chart.hoverSource === 'keyboard') {
        chart.hoverSource = '';
        chart.hoverStep = null;
        drawChart(name);
        hideChartTooltip(chart);
      }
    });
    wrap.addEventListener('keydown', (event) => chartKeydown(event, chart));
    if (legendElement) {
      legendElement.addEventListener('click', (event) => {
        const button = event.target.closest('.legend-toggle');
        if (!button) return;
        const series = button.dataset.series;
        if (!series || !(series in chart.visible)) return;
        chart.visible[series] = !chart.visible[series];
        button.setAttribute('aria-pressed', String(chart.visible[series]));
        drawChart(name);
      });
    }
    if (typeof ResizeObserver === 'function') {
      const observer = new ResizeObserver(() => requestAnimationFrame(() => drawChart(name)));
      observer.observe(wrap);
    } else {
      window.addEventListener('resize', () => drawChart(name));
    }
    drawChart(name);
  }
}

function visiblePoints(chart) {
  const points = [];
  if (state.chartMetrics.loss !== false) {
    for (const key of ['train', 'eval', 'avg']) {
      if (!chart.visible[key]) continue;
      for (const point of chart.series[key] || []) points.push(point);
    }
  }
  for (const run of chart.compare || []) {
    for (const key of ['train', 'eval']) {
      if (run.visible?.[key] === false) continue;
      for (const point of run.series?.[key] || []) points.push(point);
    }
  }
  points.sort((a, b) => a.x - b.x);
  return points;
}

export function nearestPoint(points, step) {
  if (!points || !points.length) return null;
  let best = points[0];
  let distance = Math.abs(best.x - step);
  for (const point of points) {
    const next = Math.abs(point.x - step);
    if (next < distance) { best = point; distance = next; }
  }
  return best;
}

export function buildLine(points, key = 'y') {
  return (points || []).map((point) => ({
    x: Number(point.x ?? point.step),
    y: Number(point.y ?? point[key]),
  }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
    .sort((a, b) => a.x - b.x);
}

export function drawChart(name) {
  const chart = state.charts[name];
  if (!chart) return;
  const { canvas, ctx } = chart;
  const width = canvas.clientWidth || chart.wrap.clientWidth || 600;
  const height = canvas.clientHeight || chart.wrap.clientHeight || 280;
  if (width < 20 || height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const allVisible = visiblePoints(chart);
  const hasData = Object.values(chart.series).some((points) => points && points.length)
    || (chart.compare || []).some((run) => ['train', 'eval'].some((key) => (
      run.visible?.[key] !== false && (run.series?.[key] || []).length
    )));
  if (chart.empty) chart.empty.classList.toggle('hidden', hasData);
  if (!hasData) {
    chart.lastPlot = null;
    hideChartTooltip(chart);
    return;
  }
  if (!allVisible.length) {
    chart.lastPlot = null;
    hideChartTooltip(chart);
    return;
  }

  const compact = width < 520;
  const pad = {
    l: compact ? 42 : 50,
    r: compact ? 12 : 18,
    t: 16,
    b: compact ? 30 : 34,
  };
  const plotW = Math.max(20, width - pad.l - pad.r);
  const plotH = Math.max(20, height - pad.t - pad.b);

  let xMax = 0;
  for (const point of allVisible) xMax = Math.max(xMax, point.x);
  if (chart.total && Number.isFinite(chart.total)) xMax = Math.max(xMax, chart.total);
  if (xMax <= 0) xMax = 1;
  const xMin = 0;

  let yMin = Infinity;
  let yMax = -Infinity;
  for (const point of allVisible) {
    yMin = Math.min(yMin, point.y);
    yMax = Math.max(yMax, point.y);
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) return;
  const lossRange = metricRange('loss');
  const manual = Number.isFinite(lossRange.min) && Number.isFinite(lossRange.max) && lossRange.min < lossRange.max && allVisible.length > 0;
  if (manual) {
    yMin = lossRange.min;
    yMax = lossRange.max;
  } else {
    if (yMax - yMin < 1e-6) {
      const padValue = Math.max(Math.abs(yMax) * 0.01, 0.005);
      yMin -= padValue;
      yMax += padValue;
    }
    const span = yMax - yMin;
    yMin -= span * 0.08;
    yMax += span * 0.08;
  }
  const originalMin = yMin;
  const originalMax = yMax;
  const span = yMax - yMin;

  const xOf = (value) => pad.l + ((value - xMin) / (xMax - xMin || 1)) * plotW;
  const yOf = (value) => pad.t + (1 - (value - yMin) / (yMax - yMin || 1)) * plotH;
  const plot = { pad, plotW, plotH, xMin, xMax, yMin, yMax, xOf, yOf };
  chart.lastPlot = plot;

  const mono = 'ui-monospace, SFMono-Regular, Consolas, monospace';
  ctx.font = `11px ${mono}`;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  // grid + y ticks
  ctx.strokeStyle = 'rgba(148, 163, 184, .055)';
  ctx.lineWidth = 1;
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i += 1) {
    const y = pad.t + (plotH * i) / yTicks;
    ctx.beginPath();
    ctx.moveTo(pad.l, Math.round(y) + 0.5);
    ctx.lineTo(pad.l + plotW, Math.round(y) + 0.5);
    ctx.stroke();
    const value = yMax - ((yMax - yMin) * i) / yTicks;
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(value.toFixed(span < 0.02 ? 4 : 3), pad.l - 7, y);
  }

  // x ticks
  const xTicks = Math.min(compact ? 4 : 5, Math.max(2, Math.ceil(xMax)));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= xTicks; i += 1) {
    const x = pad.l + (plotW * i) / xTicks;
    ctx.strokeStyle = 'rgba(148, 163, 184, .025)';
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, pad.t);
    ctx.lineTo(Math.round(x) + 0.5, pad.t + plotH);
    ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    const value = Math.round(xMin + ((xMax - xMin) * i) / xTicks);
    ctx.fillText(fmtInt(value), x, pad.t + plotH + 8);
  }

  // axis captions
  ctx.fillStyle = '#7c8ba1';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(t('dash.msg3'), pad.l, 0);
  ctx.textAlign = 'right';
  ctx.fillText(t('dash.msg4'), pad.l + plotW, pad.t + plotH + 20);

  const drawCompare = () => {
    for (const run of chart.compare || []) {
      for (const key of ['train', 'eval']) {
        if (run.visible?.[key] === false) continue;
        const points = countSeries(run.series?.[key] || [], compact ? 500 : 1200);
        if (!points.length) continue;
        ctx.save();
        ctx.beginPath();
        points.forEach((point, index) => {
          const x = xOf(point.x);
          const y = yOf(point.y);
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = run.color;
        ctx.globalAlpha = key === 'eval' ? .92 : .78;
        ctx.lineWidth = key === 'eval' ? 1.45 : 1.25;
        ctx.setLineDash(key === 'eval' ? [5, 3] : []);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }
    }
  };

  const drawSeries = (key) => {
    if (!chart.visible[key]) return;
    const points = countSeries(chart.series[key], compact ? 500 : 1200);
    if (!points.length) return;
    const color = CHART_COLORS[key];
    ctx.save();
    ctx.beginPath();
    let started = false;
    for (const point of points) {
      const x = xOf(point.x);
      const y = yOf(point.y);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.globalAlpha = key === 'train' ? .34 : key === 'eval' ? .78 : 1;
    ctx.lineWidth = key === 'train' ? 1.05 : key === 'eval' ? 1.55 : 2.15;
    if (key === 'train') ctx.setLineDash([4, 3]);
    else if (key === 'eval') ctx.setLineDash([5, 4]);
    else ctx.setLineDash([]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    if (key === 'eval') {
      const markerStride = Math.max(1, Math.ceil(points.length / 40));
      ctx.save();
      ctx.globalAlpha = .82;
      ctx.fillStyle = CHART_COLORS.eval;
      for (let index = 0; index < points.length; index += markerStride) {
        const point = points[index];
        drawDiamond(ctx, xOf(point.x), yOf(point.y), 2.6);
      }
      ctx.restore();
    }
    if (key === 'avg') {
      const last = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(xOf(last.x), yOf(last.y), 3, 0, Math.PI * 2);
      ctx.fillStyle = '#bbf7d0';
      ctx.fill();
    }
  };
  drawSeries('train');
  drawSeries('eval');
  drawSeries('avg');
  drawCompare();

  // crosshair + readout
  if (chart.hoverStep != null) {
    const hoverX = xOf(chart.hoverStep);
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
      for (const key of ['train', 'eval', 'avg']) {
        if (!chart.visible[key]) continue;
        const point = nearestPoint(chart.series[key], chart.hoverStep);
        if (!point) continue;
        ctx.beginPath();
        ctx.arc(xOf(point.x), yOf(point.y), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = CHART_COLORS[key];
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.restore();
      updateChartTooltip(chart);
    }
  } else {
    hideChartTooltip(chart);
  }

  // accessible summary
  const trainPoints = chart.series.train || [];
  const evalPoints = chart.series.eval || [];
  const latest = trainPoints[trainPoints.length - 1];
  const summary = latest
    ? t('dash.chart.ariaWithData', { train: trainPoints.length, eval: evalPoints.length, loss: latest.y.toFixed(4) })
    : t('dash.chart.ariaNoData', { count: allVisible.length });
  if (summary !== chart.ariaLabel) {
    chart.ariaLabel = summary;
    chart.wrap.setAttribute('aria-label', summary + t('dash.msg5'));
  }
  chart.wrap.setAttribute('data-y-min', originalMin.toFixed(5));
  chart.wrap.setAttribute('data-y-max', originalMax.toFixed(5));
}

function countSeries(points, limit) {
  if (!points || points.length <= limit) return points || [];
  const stride = Math.ceil(points.length / limit);
  const sampled = [];
  for (let index = 0; index < points.length; index += stride) sampled.push(points[index]);
  if (sampled[sampled.length - 1] !== points[points.length - 1]) sampled.push(points[points.length - 1]);
  return sampled;
}

function drawDiamond(ctx, x, y, radius) {
  ctx.beginPath();
  ctx.moveTo(x, y - radius);
  ctx.lineTo(x + radius, y);
  ctx.lineTo(x, y + radius);
  ctx.lineTo(x - radius, y);
  ctx.closePath();
  ctx.fill();
}

function moveChartHover(chart, x, y) {
  const plot = chart.lastPlot;
  if (!plot) return;
  const { pad, plotW, xMin, xMax } = plot;
  if (x < pad.l - 8 || x > pad.l + plotW + 8) {
    chart.hoverStep = null;
    hideChartTooltip(chart);
    drawChart(chart.name);
    return;
  }
  const rawStep = xMin + ((x - pad.l) / plotW) * (xMax - xMin);
  const points = visiblePoints(chart).filter((point) => point.y != null);
  const nearest = nearestPoint(points, rawStep);
  if (!nearest) return;
  chart.hoverStep = nearest.x;
  drawChart(chart.name);
  updateChartTooltip(chart);
}

function chartKeydown(event, chart) {
  const points = visiblePoints(chart);
  if (!points.length) return;
  const steps = Array.from(new Set(points.map((point) => point.x))).sort((a, b) => a - b);
  let index = chart.hoverStep == null ? steps.length - 1 : steps.indexOf(chart.hoverStep);
  if (index < 0) index = steps.length - 1;
  let handled = true;
  if (event.key === 'ArrowRight') index = clamp(index + 1, 0, steps.length - 1);
  else if (event.key === 'ArrowLeft') index = clamp(index - 1, 0, steps.length - 1);
  else if (event.key === 'Home') index = 0;
  else if (event.key === 'End') index = steps.length - 1;
  else if (event.key === 'Escape') {
    chart.hoverStep = null;
    chart.hoverSource = '';
    hideChartTooltip(chart);
    drawChart(chart.name);
    return;
  } else handled = false;
  if (!handled) return;
  event.preventDefault();
  chart.hoverStep = steps[index];
  chart.hoverSource = 'keyboard';
  drawChart(chart.name);
  updateChartTooltip(chart);
}

function tooltipRowsFor(chart, step) {
  const rows = [];
  for (const key of ['train', 'avg', 'eval']) {
    if (!chart.visible[key]) continue;
    const point = nearestPoint(chart.series[key], step);
    if (!point) continue;
    rows.push({
      key,
      step: point.x,
      value: point.y,
      label: t(CHART_LABELS[key]),
    });
  }
  return rows;
}

function updateChartTooltip(chart) {
  if (!chart.tooltip || !chart.lastPlot || chart.hoverStep == null) return;
  const rows = tooltipRowsFor(chart, chart.hoverStep);
  if (!rows.length) { hideChartTooltip(chart); return; }
  const titleStep = rows.find((row) => row.key === 'train')?.step ?? chart.hoverStep;
  chart.tooltip.innerHTML = `
    <div class="tooltip-title">${escapeHtml(t('dash.chart.tooltipStep', { step: fmtInt(titleStep) }))}</div>
    ${rows.map((row) => `<div class="tooltip-row"><span>${escapeHtml(row.label)}${row.step !== titleStep ? ` <span class="mono">${escapeHtml(t('dash.chart.tooltipStepSuffix', { step: fmtInt(row.step) }))}</span>` : ''}</span><span class="is-${row.key} mono">${row.value.toFixed(4)}</span></div>`).join('')}
  `;
  chart.tooltip.classList.remove('hidden');
  const plot = chart.lastPlot;
  const x = plot.xOf(chart.hoverStep);
  const tooltipWidth = chart.tooltip.offsetWidth || 180;
  const tooltipHeight = chart.tooltip.offsetHeight || 70;
  const wrapWidth = chart.wrap.clientWidth || 600;
  let left = x + 14;
  if (left + tooltipWidth > wrapWidth - 4) left = Math.max(4, x - tooltipWidth - 14);
  const top = clamp(plot.pad.t + 2, 4, Math.max(4, (chart.wrap.clientHeight || 280) - tooltipHeight - 4));
  chart.tooltip.style.left = `${left}px`;
  chart.tooltip.style.top = `${top}px`;

  if (chart.readout && chart.hoverSource === 'keyboard') {
    chart.readout.textContent = t('dash.chart.readoutStep', { step: fmtInt(titleStep) }) + rows.map((row) => (
      t('dash.chart.readoutRow', { label: row.label, value: row.value.toFixed(4), recent: row.step !== titleStep ? t('dash.chart.readoutRecent', { step: fmtInt(row.step) }) : '' })
    )).join(t('dash.chart.readoutJoin'));
  }
}

function hideChartTooltip(chart) {
  if (chart.tooltip) chart.tooltip.classList.add('hidden');
}

export function movingAverage(points, window) {
  const width = Math.max(1, Math.floor(Number(window) || 1));
  const sorted = (points || []).slice().sort((a, b) => a.x - b.x);
  const out = [];
  const queue = [];
  let sum = 0;
  for (const point of sorted) {
    queue.push(point.y);
    sum += point.y;
    while (queue.length > width) sum -= queue.shift();
    out.push({ x: point.x, y: sum / queue.length });
  }
  return out;
}

function metricSeriesFromRows(rows) {
  const out = { lr: [], sec: [], vram: [], gpu: [] };
  for (const row of rows || []) {
    const x = Number(row.step ?? row.x);
    if (!Number.isFinite(x)) continue;
    const lr = Number(row.lr);
    const sec = Number(row.sec);
    const peak = Number(row.peak ?? row.vram);
    const gpu = Number(row.gpu);
    if (Number.isFinite(lr)) out.lr.push({ x, y: lr });
    if (Number.isFinite(sec) && sec > 0) out.sec.push({ x, y: sec });
    if (Number.isFinite(peak) && peak > 0) out.vram.push({ x, y: peak });
    if (Number.isFinite(gpu) && gpu > 0) out.gpu.push({ x, y: clamp(gpu, 0, 100) });
  }
  return out;
}

export function setChartSeries(name, series, total = 0, rows = []) {
  const chart = state.charts[name];
  if (!chart) return;
  chart.total = Number(total) || 0;
  chart.metricSeries = series.metrics || metricSeriesFromRows(rows);
  chart.source = {
    train: buildLine(series.train || []),
    eval: buildLine(series.eval || []),
  };
  chart.series = {
    train: chart.source.train,
    eval: chart.source.eval,
    avg: movingAverage(chart.source.train, state.smoothWindow),
  };
  drawMetricCards();
  drawChart(name);
}

export function applySmoothWindow() {
  const input = $('#dash-smooth');
  const value = Math.max(1, Math.floor(Number(input?.value) || 6));
  state.smoothWindow = value;
  if (input && Number(input.value) !== value) input.value = String(value);
  const chart = state.charts.dash;
  if (chart?.source) {
    chart.series.avg = movingAverage(chart.source.train, value);
    drawChart('dash');
  }
}
