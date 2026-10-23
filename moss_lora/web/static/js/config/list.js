/* =========================================================================
 * js/config/list.js — Config <-> form
 * split from web/static/app.js, source 1746-1982 (16 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { updateJobUi } from '../components/jobs.js';
import { startJob } from '../components/jobsse.js';
import { switchConfig } from './file.js';
import { clearYamlError, markDirty, syncFormFromConfig, updateYamlLineCount } from './form.js';
import { renderProfileOptions } from './profiles.js';
import { clearFormErrors } from './validate.js';
import { dumpYaml, parseYaml } from './yaml.js';
import { api } from '../core/api.js';
import { $, escapeHtml, getPath, setHidden, setText } from '../core/dom.js';
import { fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';
import { autoLoadDataset } from '../pages/dataset.js';
import { loadWorkspaceRuns } from '../pages/workspace.js';

function configEntryLabel(entry) {
  if (entry?.virtual && entry.name === 'default') return t('train.msg14');
  const bits = [entry.name || ''];
  if (entry.steps) bits.push(t('train.config.steps', { count: fmtInt(entry.steps) }));
  if (entry.rank) bits.push(`rank ${entry.rank}`);
  const precision = String(entry.name || '').match(/(bf16|fp16|nf4|int8)/i);
  if (precision) bits.push(precision[1].toLowerCase());
  return bits.join(' · ');
}

function configDisplayPath(name = state.activeConfigName, virtual = false) {
  if (!name || virtual || name === 'default' && !(state.configs || []).some((item) => item.name === name && !item.virtual)) {
    return 'workspace.yaml';
  }
  return `configs/${name}.yaml`;
}

export function renderConfigIdentity() {
  const active = (state.configs || []).find((entry) => entry.name === state.activeConfigName);
  setText('#config-form-title', active?.name || t('train.config'));
  setText('#config-path', configDisplayPath(active?.name, Boolean(active?.virtual)));
}

function configSummaryText(entry) {
  if (!entry) return '--';
  return [entry.steps ? t('train.config.steps', { count: fmtInt(entry.steps) }) : '', entry.rank ? `rank ${entry.rank}` : '', (String(entry.name || '').match(/(bf16|fp16|nf4|int8)/i) || [])[1]?.toLowerCase(), entry.lr ? `lr ${Number(entry.lr).toExponential(2)}` : ''].filter(Boolean).join(' · ') || '--';
}

export function renderConfigList() {
  const select = $('#config-select');
  const summary = $('#config-summary');
  if (!select) return;
  const realConfigs = state.configs || [];
  const configs = state.activeConfigName === 'default' && !realConfigs.some((entry) => entry.name === 'default')
    ? [{
      name: 'default',
      virtual: true,
      path: state.configPath,
      steps: getPath(state.config, 'train.steps'),
      rank: getPath(state.config, 'lora.rank'),
      lr: getPath(state.config, 'train.lr'),
    }, ...realConfigs]
    : realConfigs;
  select.innerHTML = configs.length
    ? configs.map((entry) => `<option value="${escapeHtml(entry.name)}">${escapeHtml(configEntryLabel(entry))}</option>`).join('')
    : `<option value="">${escapeHtml(t('train.noConfigs'))}</option>`;
  if (state.activeConfigName) select.value = state.activeConfigName;
  select.disabled = !configs.length;
  const active = configs.find((entry) => entry.name === state.activeConfigName);
  if (summary) summary.textContent = configSummaryText(active);
  renderConfigIdentity();
  const virtual = Boolean(active?.virtual);
  for (const id of ['#btn-config-save', '#btn-config-saveas']) {
    const button = $(id);
    if (button) button.disabled = !state.profileName;
  }
  for (const id of ['#btn-config-rename', '#btn-config-delete']) {
    const button = $(id);
    if (button) button.disabled = !active || virtual;
  }
  renderDashboardConfigOptions();
}

function dashboardConfigEntries() {
  // Same entries, same order as the train tab: one workspace must not present
  // two different config lists depending on which page you are looking at.
  return (state.configs || []).filter((entry) => {
    const name = String(entry?.name || '');
    return name && name !== 'default';
  });
}

function workspaceDefaultConfigReady() {
  try {
    const base = parseYaml(state.workspaceDefaultYaml || '{}');
    return Boolean(base && base.train && typeof base.train === 'object'
      && base.lora && typeof base.lora === 'object');
  } catch (error) {
    return false;
  }
}

function dashboardConfigOptions() {
  const options = dashboardConfigEntries().map((entry) => ({
    value: entry.name,
    label: entry.name,
  }));
  const snapshot = runSnapshotOption();
  if (snapshot) options.unshift(snapshot);
  if (workspaceDefaultConfigReady()) {
    options.push({ value: 'default', label: t('dash.msg1') });
  }
  return options;
}

export function hasDashboardConfigSelection() {
  const select = $('#dash-config-select');
  if (!select || select.disabled) return false;
  // The snapshot is read-only history: it enables no launch button.
  return Boolean(select.value) && select.value !== RUN_SNAPSHOT;
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
    const snapshot = runSnapshotOption();
    const active = snapshot
      ? snapshot.value
      : (options.some((option) => option.value === state.activeConfigName)
        ? state.activeConfigName
        : options[0].value);
    select.value = active;
    select.disabled = !state.profileName;
    // While a historic run is selected, its own snapshot is the only config
    // shown; the "next run" line would describe something else.
    setHidden('#dash-config-line', Boolean(snapshot));
    setHidden('#dash-config-empty', true);
    if (!snapshot) renderDashboardConfigSummary(active);
  }
  updateJobUi();
}

function renderDashboardConfigSummary(activeName = state.activeConfigName || 'default') {
  const entry = (state.configs || []).find((item) => item.name === activeName && !item.virtual) || null;
  const virtual = !entry;
  const summaryEntry = entry || {
    name: activeName,
    steps: getPath(state.config, 'train.steps'),
    rank: getPath(state.config, 'lora.rank'),
    lr: getPath(state.config, 'train.lr'),
  };
  setText('#dash-config-path', configDisplayPath(activeName, virtual));
  setText('#dash-config-summary', configSummaryText(summaryEntry));
}

async function loadWorkspaceDefaultConfig() {
  const baseData = await api('/api/config?workspace=' + encodeURIComponent(state.profileName));
  state.workspaceDefaultYaml = baseData.yaml || '';
  const base = parseYaml(baseData.yaml || '{}');
  state.activeConfigName = 'default';
  state.config = base;
  state.savedYaml = dumpYaml(base);
  state.savedAt = Date.now();
  state.configPath = baseData.path || state.configPath;
  const textarea = $('#yaml');
  if (textarea) textarea.value = state.savedYaml;
  syncFormFromConfig(base);
  clearFormErrors();
  clearYamlError();
  updateYamlLineCount();
  markDirty(false);
  renderConfigList();
  renderProfileOptions();
  renderConfigIdentity();
  if ($('#cfg-run-baseline')) $('#cfg-run-baseline').checked = false;
  loadWorkspaceRuns().catch(() => {});
  autoLoadDataset().catch(() => {});
  return state.config;
}

export async function selectDashboardConfig(target, { confirmChange = true } = {}) {
  if (!state.profileName) return false;
  if (target === 'default' && !workspaceDefaultConfigReady()) {
    renderDashboardConfigOptions();
    return false;
  }
  if (target === 'default') {
    const alreadyWorkspace = state.activeConfigName === 'default' && /workspace\.ya?ml$/i.test(state.configPath || '');
    if (alreadyWorkspace) {
      renderDashboardConfigOptions();
      return true;
    }
    if (state.dirty && confirmChange && !window.confirm(t('dash.msg2'))) {
      renderDashboardConfigOptions();
      return false;
    }
    await loadWorkspaceDefaultConfig();
  } else {
    if (target === state.activeConfigName) {
      renderDashboardConfigOptions();
      return true;
    }
    await switchConfig(target);
    if (state.activeConfigName !== target) {
      renderDashboardConfigOptions();
      return false;
    }
  }
  renderDashboardConfigOptions();
  return true;
}

export async function ensureDashboardConfigSelection() {
  if (!state.profileName) {
    renderDashboardConfigOptions();
    return;
  }
  const options = dashboardConfigOptions();
  const allowed = new Set(options.map((option) => option.value));
  if (allowed.has(state.activeConfigName)) {
    renderDashboardConfigOptions();
    return;
  }
  const target = options[0]?.value || '';
  if (target && !state.dirty) await selectDashboardConfig(target, { confirmChange: false });
  else renderDashboardConfigOptions();
}

// A run's config lives in its own directory and nowhere else. While a historic
// run is selected the picker shows that snapshot instead of a config name, so
// the panel never describes a run with a file that is not its own.
export const RUN_SNAPSHOT = '__run_snapshot__';

export function runSnapshotOption() {
  const data = state.dashboard.data;
  if (!data || !data.run || !data.run_config_path) return null;
  // Once the user picks a real config, the picker belongs to the next run
  // again; inspecting a run puts it back on that run's own snapshot.
  if (state.dashboard.configPicked) return null;
  const name = String(data.run).split('/').pop();
  return { value: RUN_SNAPSHOT, label: `${t('train.configSource.run')} · ${name}` };
}

export function armRunSnapshot() {
  state.dashboard.configPicked = false;
}

export function isRunSnapshotSelected() {
  return $('#dash-config-select')?.value === RUN_SNAPSHOT;
}

export async function startDashboardTraining() {
  const target = $('#dash-config-select')?.value || '';
  if (!target) return;
  if (target !== state.activeConfigName) {
    const ok = await selectDashboardConfig(target, { confirmChange: true });
    if (!ok) return;
  }
  startJob('train');
}

export function mergeConfigIntoWorkspace(base, selected, name) {
  const merged = JSON.parse(JSON.stringify(base || {}));
  for (const key of ['lora', 'train', 'vram', 'preview', 'sigma', 'export']) {
    if (selected && selected[key] !== undefined) merged[key] = JSON.parse(JSON.stringify(selected[key]));
  }
  merged.name = name;
  merged.model = JSON.parse(JSON.stringify(base.model || {}));
  if (selected && selected.model && typeof selected.model === 'object') {
    for (const key of ['text_encoder', 'text_encoder_config']) {
      if (selected.model[key] !== undefined) merged.model[key] = selected.model[key];
    }
  }
  merged.dataset = base.dataset;
  merged.cache = base.cache;
  merged.output_dir = base.output_dir;
  return merged;
}
