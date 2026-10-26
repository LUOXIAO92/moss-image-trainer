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
import { showAlert } from '../core/notify.js';
import { state } from '../core/state.js';

// The picker is async; keep the user's target selected while the config loads
// instead of letting a dashboard poll snap it back to the previous run.
let pendingDashboardRun = '';

function shortRunName(value) {
  return String(value || '').split('/').pop();
}

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
  // Same list, same order as the train tab: a run is the config. The full
  // workspace/run label is the stable identity shown to the user.
  return (state.runs || []).map((entry) => ({
    value: entry.name,
    label: entry.label || (state.profileName ? `${state.profileName}/${entry.name}` : entry.name),
  }));
}

function dashboardConfigEntry(target) {
  const wanted = String(target || '');
  const short = shortRunName(wanted);
  return (state.runs || []).find((entry) => entry.name === wanted || entry.name === short
    || entry.label === wanted) || null;
}

export function hasDashboardConfigSelection() {
  const select = $('#dash-config-select');
  if (!select || select.disabled) return false;
  return Boolean(select.value);
}

function latestCheckpoint(entry) {
  const checkpoints = Array.isArray(entry?.checkpoints)
    ? entry.checkpoints.filter((item) => item && item.file && item.step !== 'final')
    : [];
  return checkpoints.sort((a, b) => Number(a.step) - Number(b.step)).pop() || null;
}

function resumeInfo(entry) {
  if (!entry) return { resumable: false, file: '', checkpointStep: 0, step: 0, total: 0, done: false };
  const latest = latestCheckpoint(entry);
  const reported = String(entry.checkpoint || '').split(/[\\/]/).pop();
  const file = (reported && !/^final\./i.test(reported) ? reported : latest?.file) || '';
  const parsed = file.match(/step(\d+)/i);
  const checkpointStep = Number(entry.checkpoint_step ?? latest?.step ?? (parsed ? parsed[1] : 0));
  const step = Number(entry.step || 0);
  const total = Number(entry.total || 0);
  // The backend owns this verdict. Only fall back to checkpoint presence when
  // talking to an older API, never to `step < total` alone.
  const resumable = entry.resumable === false ? false : (entry.resumable === true || Boolean(file));
  return {
    resumable,
    file: file || (resumable ? 'latest' : ''),
    checkpointStep: Number.isFinite(checkpointStep) ? checkpointStep : 0,
    step: Number.isFinite(step) ? step : 0,
    total: Number.isFinite(total) ? total : 0,
    done: total > 0 && step >= total && !resumable,
  };
}

export function updateDashboardResumeState() {
  const button = $('#btn-resume');
  if (!button) return;
  const entry = dashboardConfigEntry(state.activeRun);
  const info = resumeInfo(entry);
  const running = state.job?.status === 'running' || Boolean(state.activeTraining);
  button.dataset.resume = info.file;
  button.dataset.resumable = info.resumable ? '1' : '0';
  button.disabled = !info.resumable || running || state.busy || !state.profileName;
  button.setAttribute('aria-disabled', String(button.disabled));
  button.title = info.resumable ? t('dash.train.resumeHint') : t('dash.train.resumeUnavailableText');
  setHidden('#dash-resume-line', !entry);
  if (!entry) {
    setText('#dash-resume-status', '--');
    return;
  }
  if (info.resumable) {
    if (info.checkpointStep > 0 && info.step > 0 && info.total > 0) {
      setText('#dash-resume-status', t('dash.train.resumeStatus', {
        step: fmtInt(info.step),
        total: fmtInt(info.total),
        checkpoint: fmtInt(info.checkpointStep),
        next: fmtInt(info.checkpointStep + 1),
      }));
    } else if (info.checkpointStep > 0) {
      setText('#dash-resume-status', t('dash.train.resumeCheckpointOnly', {
        checkpoint: fmtInt(info.checkpointStep),
        next: fmtInt(info.checkpointStep + 1),
      }));
    } else {
      setText('#dash-resume-status', t('dash.train.resumeLatest'));
    }
  } else if (info.done) {
    setText('#dash-resume-status', t('dash.train.resumeDone'));
  } else {
    setText('#dash-resume-status', t('dash.train.resumeUnavailable'));
  }
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
    const activeShort = shortRunName(state.activeRun);
    const activeOption = options.find((option) => option.value === state.activeRun
      || shortRunName(option.value) === activeShort);
    const desired = pendingDashboardRun && options.some((option) => option.value === pendingDashboardRun)
      ? pendingDashboardRun
      : (activeOption?.value || options[0].value);
    select.value = desired;
    select.disabled = !state.profileName;
    setHidden('#dash-config-line', false);
    setHidden('#dash-config-empty', true);
    renderDashboardConfigSummary(desired);
  }
  updateDashboardResumeState();
  updateJobUi();
}

function renderDashboardConfigSummary(activeName = state.activeRun) {
  const entry = (state.runs || []).find((item) => item.name === activeName) || null;
  setText('#dash-config-path', entry ? `runs/${entry.name}/config.yaml` : '--');
  setText('#dash-config-summary', runSummaryText(entry));
}

export async function selectDashboardConfig(target) {
  if (!state.profileName || !target) return false;
  const entry = dashboardConfigEntry(target);
  const name = entry?.name || shortRunName(target);
  pendingDashboardRun = name;
  renderDashboardConfigOptions();
  try {
    if (name !== state.activeRun) await switchRun(name);
    const current = dashboardConfigEntry(state.activeRun) || entry;
    if (current) state.dashboard.run = current.label || (state.profileName ? `${state.profileName}/${current.name}` : current.name);
    return state.activeRun === name;
  } finally {
    pendingDashboardRun = '';
    renderDashboardConfigOptions();
  }
}

export async function ensureDashboardConfigSelection() {
  if (!state.profileName) {
    renderDashboardConfigOptions();
    return;
  }
  const options = dashboardConfigOptions();
  const allowed = new Set(options.map((option) => option.value));
  if (allowed.has(state.activeRun) || allowed.has(shortRunName(state.activeRun))) {
    renderDashboardConfigOptions();
    return;
  }
  const target = options[0]?.value || '';
  if (target && !state.dirty) await selectDashboardConfig(target);
  else renderDashboardConfigOptions();
}

export async function startDashboardTraining(options = {}) {
  const target = $('#dash-config-select')?.value || '';
  if (!target) {
    renderDashboardConfigOptions();
    return false;
  }
  if (target !== state.activeRun) {
    const ok = await selectDashboardConfig(target);
    if (!ok) return false;
  }
  if (!options.resume) return startJob('train');
  const entry = dashboardConfigEntry(state.activeRun);
  const info = resumeInfo(entry);
  if (!info.resumable) {
    updateDashboardResumeState();
    showAlert({
      type: 'warn',
      title: t('dash.train.resumeUnavailable'),
      text: t('dash.train.resumeUnavailableText'),
      timeout: 5000,
    });
    return false;
  }
  // Resume never reruns the no-LoRA baseline: that is a fresh-run step and
  // would overwrite the baseline the user already has.
  return startJob('train', { resume: info.file || 'latest', baseline: false });
}
