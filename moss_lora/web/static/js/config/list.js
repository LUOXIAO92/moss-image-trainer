/* =========================================================================
 * js/config/list.js — run selector
 *
 * A run is the config: the picker lists runs, the dashboard lists the same
 * runs, and there is no pseudo-option for history. Selecting a run loads its
 * own runs/<run>/config.yaml.
 * ========================================================================= */

import { updateJobUi } from '../components/jobs.js';
import { startJob } from '../components/jobsse.js';
import { applyFrozenState, switchRun } from './file.js';
import { $, escapeHtml, setHidden, setText } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';

function runEntryLabel(entry) {
  const bits = [entry.name || ''];
  if (entry.total) bits.push(t('train.config.steps', { count: fmtInt(entry.total) }));
  if (entry.step) bits.push(`${fmtInt(entry.step)}`);
  return bits.join(' · ');
}

function runSummaryText(entry) {
  if (!entry) return '--';
  const config = entry.config || {};
  const steps = (config.train || {}).steps ?? entry.total;
  const rank = (config.lora || {}).rank;
  const lr = (config.train || {}).lr;
  return [
    steps ? t('train.config.steps', { count: fmtInt(steps) }) : '',
    rank ? `rank ${rank}` : '',
    lr ? `lr ${Number(lr).toExponential(2)}` : '',
  ].filter(Boolean).join(' · ') || '--';
}

export function renderConfigIdentity() {
  const active = (state.runs || []).find((entry) => entry.name === state.activeRun);
  setText('#config-form-title', active?.name || t('train.config'));
  setText('#config-path', active ? `runs/${active.name}/config.yaml` : '--');
}

export function renderConfigList() {
  const select = $('#config-select');
  const summary = $('#config-summary');
  if (!select) return;
  const runs = state.runs || [];
  const live = applyFrozenState();
  select.innerHTML = runs.length
    ? runs.map((entry) => `<option value="${escapeHtml(entry.name)}">${escapeHtml(runEntryLabel(entry))}</option>`).join('')
    : `<option value="">${escapeHtml(t('runEmpty.option'))}</option>`;
  if (state.activeRun) select.value = state.activeRun;
  select.disabled = !runs.length || live;
  const active = runs.find((entry) => entry.name === state.activeRun);
  if (summary) summary.textContent = runSummaryText(active);
  renderConfigIdentity();
  const hasRun = Boolean(state.activeRun);
  for (const id of ['#btn-config-save', '#btn-config-delete']) {
    const button = $(id);
    if (button) button.disabled = !hasRun || Boolean(active?.live);
  }
  const newButton = $('#btn-config-new');
  if (newButton) newButton.disabled = !state.profileName;
  renderDashboardConfigOptions();
}

function dashboardConfigOptions() {
  // Same list, same order as the train tab: a run is the config.
  return (state.runs || []).map((entry) => ({ value: entry.name, label: entry.name }));
}

export function hasDashboardConfigSelection() {
  const select = $('#dash-config-select');
  if (!select || select.disabled) return false;
  return Boolean(select.value);
}

export function renderDashboardConfigOptions() {
  const select = $('#dash-config-select');
  if (!select) return;
  const options = dashboardConfigOptions();
  if (!options.length) {
    select.innerHTML = `<option value="">${escapeHtml(t('dash.train.noConfigOption'))}</option>`;
    select.disabled = true;
    setHidden('#dash-config-line', true);
    setHidden('#dash-config-empty', false);
  } else {
    select.innerHTML = options
      .map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`)
      .join('');
    const active = options.some((option) => option.value === state.activeRun)
      ? state.activeRun
      : options[0].value;
    select.value = active;
    select.disabled = !state.profileName;
    setHidden('#dash-config-line', false);
    setHidden('#dash-config-empty', true);
    renderDashboardConfigSummary(active);
  }
  updateJobUi();
}

function renderDashboardConfigSummary(activeName = state.activeRun) {
  const entry = (state.runs || []).find((item) => item.name === activeName) || null;
  setText('#dash-config-path', entry ? `runs/${entry.name}/config.yaml` : '--');
  setText('#dash-config-summary', runSummaryText(entry));
}

export async function selectDashboardConfig(target) {
  if (!state.profileName || !target) return false;
  if (target === state.activeRun) {
    renderDashboardConfigOptions();
    return true;
  }
  await switchRun(target);
  renderDashboardConfigOptions();
  return state.activeRun === target;
}

export async function ensureDashboardConfigSelection() {
  if (!state.profileName) {
    renderDashboardConfigOptions();
    return;
  }
  const options = dashboardConfigOptions();
  const allowed = new Set(options.map((option) => option.value));
  if (allowed.has(state.activeRun)) {
    renderDashboardConfigOptions();
    return;
  }
  const target = options[0]?.value || '';
  if (target && !state.dirty) await selectDashboardConfig(target);
  else renderDashboardConfigOptions();
}

export async function startDashboardTraining() {
  const target = $('#dash-config-select')?.value || '';
  if (!target) {
    renderDashboardConfigOptions();
    return;
  }
  if (target !== state.activeRun) {
    const ok = await selectDashboardConfig(target);
    if (!ok) return;
  }
  startJob('train');
}
