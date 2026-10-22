/* =========================================================================
 * js/pages/samples.js — Samples
 * split from web/static/app.js, source 6864-7037 (6 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { api } from '../core/api.js';
import { $, escapeHtml, fileUrl, setHidden, setText, thumbUrl } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { openModal } from '../core/modal.js';
import { showAlert } from '../core/notify.js';
import { state, workspaceParam } from '../core/state.js';

export function sampleMeta(path, item = null) {
  const filename = String(path || '').split('/').pop() || '';
  const stepFrom = (value) => {
    const match = String(value || '').match(/step[_-]?(\d+)/i);
    return match ? parseInt(match[1], 10) : null;
  };
  if (item && item.run) {
    return { run: item.run, step: item.step ?? stepFrom(filename), filename };
  }
  const normalized = String(path || '').replace(/\\/g, '/');
  const parts = normalized.split('/');
  let runParts = [];
  const wsIndex = parts.lastIndexOf('workspaces');
  if (wsIndex >= 0 && parts[wsIndex + 2] === 'runs') {
    // workspaces/<ws>/runs/<run>/samples/...
    runParts = [`${parts[wsIndex + 1]}/${parts[wsIndex + 3]}`];
  } else {
    const sampleIndex = parts.lastIndexOf('samples');
    if (sampleIndex > 0) {
      const head = parts.slice(0, sampleIndex);
      const outputIndex = head.lastIndexOf('output');
      runParts = outputIndex >= 0 ? head.slice(outputIndex + 1) : head.slice(-2);
    } else {
      const outputIndex = parts.lastIndexOf('output');
      if (outputIndex >= 0) runParts = parts.slice(outputIndex + 1, -1);
      else if (parts.length >= 3) runParts = parts.slice(-2, -1);
    }
  }
  return {
    run: runParts.filter(Boolean).join('/') || 'output',
    step: stepFrom(filename) ?? stepFrom(normalized),
    filename,
  };
}

export function createSampleCard(path, options = {}) {
  const meta = options.run ? options : sampleMeta(path);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'sample-card';
  button.title = path;
  const image = document.createElement('img');
  image.loading = 'lazy';
  image.decoding = 'async';
  image.width = options.small ? 384 : 512;
  image.height = options.small ? 384 : 512;
  const altParts = [];
  if (meta.run) altParts.push(meta.run);
  if (meta.step != null) altParts.push(t('common.stepNumber', { step: meta.step }));
  altParts.push(t('nav.samples'));
  image.alt = altParts.join(' ');
  image.src = thumbUrl(path, options.small ? 384 : 512);
  image.addEventListener('error', () => {
    if (!image.dataset.fallback) {
      image.dataset.fallback = '1';
      image.src = fileUrl(path);
      return;
    }
    button.classList.add('is-broken');
  });
  const badge = document.createElement('span');
  badge.className = 'sample-badge';
  badge.textContent = meta.step != null ? t('common.stepNumber', { step: fmtInt(meta.step) }) : (meta.filename || String(path).split('/').pop() || t('nav.samples'));
  button.append(image, badge);
  button.addEventListener('click', () => openModal(fileUrl(path), `${meta.run} · ${meta.step != null ? t('common.stepNumber', { step: fmtInt(meta.step) }) : meta.filename}`));
  return button;
}

export async function loadSamples() {
  try {
    const list = await api('/api/samples?workspace=' + encodeURIComponent(workspaceParam()));
    state.samples.items = (Array.isArray(list) ? list : []).filter((item) => !/\/trajectory\.png$/i.test(item.path));
    renderSamples();
  } catch (error) {
    showAlert({ type: 'warn', title: t('dash.msg17'), text: error.detail || error.message, timeout: 6000 });
  }
}

function sampleRuns() {
  const map = new Map();
  for (const item of state.samples.items) {
    const meta = sampleMeta(item.path, item);
    if (!map.has(meta.run)) map.set(meta.run, { run: meta.run, count: 0, latest: 0, items: [] });
    const group = map.get(meta.run);
    group.count += 1;
    group.latest = Math.max(group.latest, item.mtime || 0);
    group.items.push(item);
  }
  return Array.from(map.values()).sort((a, b) => b.latest - a.latest);
}

function sampleStepGroups(items) {
  const map = new Map();
  for (const item of items || []) {
    const meta = sampleMeta(item.path, item);
    if (!map.has(meta.step)) map.set(meta.step, []);
    map.get(meta.step).push({ item, meta });
  }
  return Array.from(map.entries())
    .sort((a, b) => (b[0] ?? -1) - (a[0] ?? -1))
    .map(([step, entries]) => {
      entries.sort((a, b) => {
        const ai = (a.meta.filename.match(/step[_-]?\d+[_-](\d+)/i) || [])[1];
        const bi = (b.meta.filename.match(/step[_-]?\d+[_-](\d+)/i) || [])[1];
        return (Number(ai) || 0) - (Number(bi) || 0);
      });
      return { step, entries };
    });
}

export function renderSamples() {
  const select = $('#samples-filter');
  const groups = sampleRuns();
  if (select) {
    const previous = state.samples.filter;
    select.innerHTML = `<option value="all">${escapeHtml(t('samples.allRuns'))}</option>${groups.map((group) => `<option value="${escapeHtml(group.run)}">${escapeHtml(group.run)} · ${escapeHtml(t('common.imagesCount', { count: group.count }))}</option>`).join('')}`;
    select.value = groups.some((group) => group.run === previous) ? previous : 'all';
    state.samples.filter = select.value;
  }
  const filter = state.samples.filter;
  const visible = filter === 'all' ? groups : groups.filter((group) => group.run === filter);
  const container = $('#samples-groups');
  const empty = $('#samples-empty');
  setText('#samples-count', t('samples.count', { images: state.samples.items.length, runs: groups.length }));
  if (!container) return;
  container.innerHTML = '';
  if (!visible.length) {
    setHidden(empty, false);
    return;
  }
  setHidden(empty, true);
  for (const group of visible) {
    const stepGroups = sampleStepGroups(group.items);
    const latestStep = stepGroups.find((entry) => entry.step != null)?.step;
    const section = document.createElement('section');
    section.className = 'sample-group';
    const header = document.createElement('header');
    header.className = 'sample-group-head';
    const title = document.createElement('h2');
    title.textContent = group.run;
    const count = document.createElement('span');
    count.className = 'mono';
    count.textContent = t('samples.groupCount', { images: group.count, steps: stepGroups.length, latest: latestStep != null ? t('samples.latestStep', { step: fmtInt(latestStep) }) : '' });
    const link = document.createElement('a');
    link.className = 'btn btn-ghost btn-xs';
    link.href = '#dashboard';
    link.textContent = t('dash.msg18');
    link.addEventListener('click', () => {
      state.dashboard.run = group.run;
    });
    header.append(title, count, link);
    section.append(header);
    for (const entry of stepGroups) {
      const stepBox = document.createElement('div');
      stepBox.className = 'sample-step-group';
      const stepHead = document.createElement('div');
      stepHead.className = 'sample-step-head';
      const stepTitle = document.createElement('h3');
      stepTitle.textContent = entry.step == null ? t('dash.msg19') : `step ${fmtInt(entry.step)}`;
      const stepCount = document.createElement('span');
      stepCount.className = 'mono';
      stepCount.textContent = t('common.imagesCount', { count: entry.entries.length });
      stepHead.append(stepTitle, stepCount);
      const grid = document.createElement('div');
      grid.className = 'sample-grid';
      for (const { item, meta } of entry.entries) {
        grid.append(createSampleCard(item.path, { run: meta.run, step: meta.step }));
      }
      stepBox.append(stepHead, grid);
      section.append(stepBox);
    }
    container.append(section);
  }
}
