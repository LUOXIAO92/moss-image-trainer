/* =========================================================================
 * js/config/file.js — the selected run's config
 *
 * One config, one home: runs/<run>/config.yaml. This module loads it into the
 * form, writes it back, and creates new runs. There is no second file and no
 * overlay: what you save is what the run uses.
 * ========================================================================= */

import { loadDashboardLayout } from '../components/metriccards.js';
import { clearYamlError, getCurrentYaml, markDirty, readFormIntoConfig, refreshYamlFromConfig, showYamlError, syncFormFromConfig, updateYamlLineCount } from './form.js';
import { renderConfigIdentity, renderConfigList } from './list.js';
import { renderProfileOptions } from './profiles.js';
import { clearFormErrors } from './validate.js';
import { dumpYaml, parseYaml } from './yaml.js';
import { api, describeError } from '../core/api.js';
import { $, setHidden, setText } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { state } from '../core/state.js';
import { autoLoadDataset } from '../pages/dataset.js';
import { runSetupCheck } from '../pages/settings.js';
import { loadWorkspaceRuns } from '../pages/workspace.js';

export function activeRunEntry() {
  return (state.runs || []).find((entry) => entry.name === state.activeRun) || null;
}

export function activeRunIsLive() {
  return Boolean(activeRunEntry()?.live);
}

function runConfigPath(run = state.activeRun) {
  return run ? `runs/${run}/config.yaml` : '';
}

async function refreshRunList() {
  const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/runs');
  state.runs = Array.isArray(data.runs) ? data.runs : [];
  return state.runs;
}

function showEmptyState() {
  state.config = {};
  state.savedYaml = '';
  state.savedAt = Date.now();
  state.configPath = '';
  const textarea = $('#yaml');
  if (textarea) textarea.value = '';
  syncFormFromConfig({});
  clearFormErrors();
  clearYamlError();
  updateYamlLineCount();
  markDirty(false);
  renderConfigList();
}

export async function loadConfig(run = null) {
  if (!state.profileName) {
    state.runs = [];
    state.activeRun = '';
    state.dashboard.run = null;
    showEmptyState();
    renderProfileOptions();
    setHidden('#setup-panel', true);
    return state.config;
  }

  state.workspacePath = (state.profiles.find((item) => item.name === state.profileName) || {}).path || state.workspacePath;
  if (state.workspacePath) {
    try {
      const abs = await api('/api/fs/list?path=' + encodeURIComponent(state.workspacePath));
      state.workspaceAbsPath = abs.path || '';
    } catch (error) {
      state.workspaceAbsPath = '';
    }
  }

  await refreshRunList();
  let name = String(run || '') || localStorage.getItem(`moss.run.${state.profileName}`) || state.activeRun || '';
  if (!state.runs.some((entry) => entry.name === name)) name = state.runs[0]?.name || '';
  state.activeRun = name;
  // One config owns the whole dashboard: switching the run here must also
  // switch the chart, resume state, and sample picker to that same run.
  const selected = state.runs.find((entry) => entry.name === name);
  state.dashboard.run = name
    ? (selected?.label || (state.profileName ? `${state.profileName}/${name}` : name))
    : null;
  if (!name) {
    // No run yet: nothing to edit until one is created. The run browser still
    // has to drop the previous workspace's list.
    showEmptyState();
    renderProfileOptions();
    renderConfigIdentity();
    loadWorkspaceRuns().catch(() => {});
    return state.config;
  }

  const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName)
    + '/runs/' + encodeURIComponent(name) + '/config');
  state.config = parseYaml(data.yaml || '{}');
  state.savedYaml = data.yaml || '';
  state.savedAt = Date.now();
  state.configPath = data.path || runConfigPath(name);
  setText('#config-path', runConfigPath(name));
  localStorage.setItem(`moss.run.${state.profileName}`, name);
  const textarea = $('#yaml');
  if (textarea) textarea.value = state.savedYaml;
  syncFormFromConfig(state.config);
  clearFormErrors();
  clearYamlError();
  updateYamlLineCount();
  markDirty(false);
  renderConfigList();
  renderProfileOptions();
  renderConfigIdentity();
  if ($('#cfg-run-baseline')) $('#cfg-run-baseline').checked = false;
  loadDashboardLayout().catch(() => {});
  loadWorkspaceRuns().catch(() => {});
  autoLoadDataset().catch(() => {});
  return state.config;
}

export async function switchRun(name) {
  if (!name || name === state.activeRun) { renderConfigList(); return; }
  if (state.dirty && !window.confirm(t('dash.msg2'))) {
    renderConfigList();
    return;
  }
  await loadConfig(name);
}

export function openNewRunDialog() {
  if (!state.profileName) {
    showAlert({ type: 'warn', title: t('workspace.emptyTitle'), text: t('train.msg15'), timeout: 4000 });
    return;
  }
  const dialog = $('#config-dialog');
  setText('#config-dialog-title', t('runNew.title'));
  setText('#config-dialog-desc', t('runNew.desc'));
  setHidden('#config-source-row', false);
  const input = $('#config-name-input');
  if (input) {
    input.value = newRunName();
    input.removeAttribute('aria-invalid');
  }
  const source = $('#config-source-select');
  if (source) {
    source.innerHTML = [
      `<option value="">${t('runNew.fromWorkspace')}</option>`,
      ...(state.runs || []).map((entry) => `<option value="${entry.name}">${t('runNew.fromRun', { name: entry.name })}</option>`),
    ].join('');
    source.value = '';
  }
  setHidden('#config-dialog-error', true);
  if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
  else dialog.setAttribute('open', '');
  setTimeout(() => { input?.focus(); input?.select(); }, 30);
}

function newRunName() {
  const stamp = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  return `run-${stamp.getFullYear()}${pad(stamp.getMonth() + 1)}${pad(stamp.getDate())}`
    + `-${pad(stamp.getHours())}${pad(stamp.getMinutes())}`;
}

export function closeConfigDialog() {
  const dialog = $('#config-dialog');
  if (!dialog) return;
  if (typeof dialog.close === 'function' && dialog.open) dialog.close();
  else dialog.removeAttribute('open');
}

export async function submitConfigDialog(event) {
  event.preventDefault();
  const input = $('#config-name-input');
  const errorBox = $('#config-dialog-error');
  const name = String(input?.value || '').trim().replace(/\.ya?ml$/i, '');
  const show = (message) => {
    if (errorBox) { errorBox.textContent = message; errorBox.classList.remove('hidden'); }
    input?.setAttribute('aria-invalid', 'true');
    input?.focus();
  };
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) {
    show(t('train.msg19'));
    return;
  }
  const from = $('#config-source-select')?.value || '';
  const confirm = $('#config-dialog-confirm');
  if (confirm) confirm.disabled = true;
  try {
    await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, from }),
    });
    closeConfigDialog();
    localStorage.setItem(`moss.run.${state.profileName}`, name);
    state.activeRun = name;
    await loadConfig(name);
    showSuccess(t('runNew.created'), `runs/${name}/config.yaml`, [], 'run-created');
  } catch (error) {
    show(error.status === 409
      ? `${t('runNew.exists')}：${name}`
      : (describeError(error)));
  } finally {
    if (confirm) confirm.disabled = false;
  }
}

export async function deleteCurrentRun() {
  if (!state.profileName || !state.activeRun) return;
  if (activeRunIsLive()) {
    showAlert({ type: 'warn', title: t('runFrozen.title'), text: t('runFrozen.text'), timeout: 5000 });
    return;
  }
  const run = state.activeRun;
  if (!window.confirm(t('runDelete.confirm', { name: run }))) return;
  try {
    await api('/api/workspaces/' + encodeURIComponent(state.profileName)
      + '/runs/' + encodeURIComponent(run), { method: 'DELETE' });
    localStorage.removeItem(`moss.run.${state.profileName}`);
    state.activeRun = '';
    await loadConfig();
    showSuccess(t('runDelete.done'), `runs/${run}`, [], 'run-deleted');
  } catch (error) {
    showError(t('runDelete.failed'), describeError(error));
  }
}

export async function saveConfig(options = {}) {
  if (!state.profileName) {
    showAlert({ type: 'warn', title: t('workspace.emptyTitle'), text: t('train.msg33'), timeout: 5000 });
    return false;
  }
  if (!state.activeRun) {
    showAlert({ type: 'warn', title: t('runEmpty.title'), text: t('runEmpty.desc'), timeout: 5000 });
    return false;
  }
  if (activeRunIsLive()) {
    showAlert({ type: 'warn', title: t('runFrozen.title'), text: t('runFrozen.text'), timeout: 5000 });
    return false;
  }
  readFormIntoConfig();
  const yamlText = getCurrentYaml();
  try {
    parseYaml(yamlText);
  } catch (error) {
    showYamlError(error);
    showError(t('train.msg34'), error.line
      ? t('train.yaml.errorWithLine', { error: error.message, line: error.line })
      : t('train.yaml.errorNoLine', { error: error.message }));
    return false;
  }
  const button = $('#btn-config-save');
  if (button) button.disabled = true;
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName)
      + '/runs/' + encodeURIComponent(state.activeRun) + '/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml: yamlText }),
    });
    state.savedYaml = yamlText;
    state.savedAt = Date.now();
    state.configPath = data.path || runConfigPath();
    markDirty(false);
    clearYamlError();
    await refreshRunList();
    renderConfigList();
    setText('#config-path', runConfigPath());
    refreshYamlFromConfig();
    if (!options.silent) {
      showSuccess(t('train.msg35'), t('train.config.savedPath', { path: state.configPath }), [], 'config-saved');
    }
    runSetupCheck().catch(() => {});
    return true;
  } catch (error) {
    if (error.code === 'err.run.frozen' || error.status === 409) {
      showAlert({ type: 'warn', title: t('runFrozen.title'), text: t('runFrozen.text'), timeout: 5000 });
    } else if (error.line) {
      showYamlError({ message: describeError(error), line: error.line, column: error.column });
    }
    showError(t('train.msg36'), describeError(error), [
      { label: t('common.retry'), onClick: () => saveConfig() },
    ]);
    return false;
  } finally {
    if (button) button.disabled = false;
    applyFrozenState();
  }
}

export function applyFrozenState() {
  const live = activeRunIsLive();
  const note = $('#config-frozen-note');
  if (note) {
    note.textContent = live ? t('runFrozen.text') : '';
    setHidden('#config-frozen-note', !live);
  }
  const controls = document.querySelectorAll('#config-form input, #config-form select, #config-form textarea');
  for (const control of controls) {
    if (control.dataset.alwaysEnabled === 'true') continue;
    control.disabled = live;
  }
  const yaml = $('#yaml');
  if (yaml) yaml.readOnly = live;
  for (const id of ['#btn-config-save', '#btn-config-delete']) {
    const button = $(id);
    if (button) button.disabled = live;
  }
  return live;
}
