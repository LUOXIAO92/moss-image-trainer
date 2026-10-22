/* =========================================================================
 * js/config/file.js — Config <-> form
 * split from web/static/app.js, source 1984-2357 (7 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { loadDashboardLayout } from '../components/metriccards.js';
import { clearYamlError, getCurrentYaml, markDirty, readFormIntoConfig, refreshYamlFromConfig, showYamlError, syncFormFromConfig, updateYamlLineCount } from './form.js';
import { mergeConfigIntoWorkspace, renderConfigIdentity, renderConfigList } from './list.js';
import { renderProfileOptions } from './profiles.js';
import { clearFormErrors } from './validate.js';
import { dumpYaml, parseYaml } from './yaml.js';
import { api } from '../core/api.js';
import { $, setHidden, setText } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { state } from '../core/state.js';
import { autoLoadDataset } from '../pages/dataset.js';
import { runSetupCheck } from '../pages/settings.js';
import { loadWorkspaceRuns } from '../pages/workspace.js';

export async function loadConfig(path = null) {
  if (!state.profileName) {
    state.config = {};
    state.savedYaml = '';
    state.savedAt = Date.now();
    state.configPath = '';
    state.configs = [];
    state.activeConfigName = '';
    const textarea = $('#yaml');
    if (textarea) textarea.value = '';
    renderConfigList();
    syncFormFromConfig({});
    clearFormErrors();
    clearYamlError();
    updateYamlLineCount();
    markDirty(false);
    renderProfileOptions();
    setHidden('#setup-panel', true);
      return state.config;
  }

  const baseData = await api('/api/config?workspace=' + encodeURIComponent(state.profileName));
  state.workspacePath = (state.profiles.find((item) => item.name === state.profileName) || {}).path || state.workspacePath;
  if (state.workspacePath) {
    try {
      const abs = await api('/api/fs/list?path=' + encodeURIComponent(state.workspacePath));
      state.workspaceAbsPath = abs.path || '';
    } catch (error) {
      state.workspaceAbsPath = '';
    }
  }
  state.workspaceDefaultYaml = baseData.yaml || '';
  const list = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs');
  state.configs = Array.isArray(list.configs) ? list.configs : [];
  if (!state.configs.length) {
    state.configs = [{ name: 'default', virtual: true, path: baseData.path }];
  }
  let name = localStorage.getItem(`moss.config.${state.profileName}`) || '';
  if (path) name = String(path);
  if (!state.configs.some((entry) => entry.name === name)) name = state.configs[0].name;
  state.activeConfigName = name;
  const active = state.configs.find((entry) => entry.name === name);
  let selectedYaml = baseData.yaml || '';
  let configPath = active?.path || baseData.path;
  if (active && !active.virtual) {
    const selected = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs/' + encodeURIComponent(active.name));
    selectedYaml = selected.yaml || '';
    configPath = selected.entry?.path || configPath;
  }
  const base = parseYaml(baseData.yaml || '{}');
  const selected = parseYaml(selectedYaml || '{}');
  state.config = mergeConfigIntoWorkspace(base, selected, name);
  state.savedYaml = dumpYaml(state.config);
  state.savedAt = Date.now();
  state.configPath = configPath;
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

export async function switchConfig(name) {
  if (!name || name === state.activeConfigName) { renderConfigList(); return; }
  if (state.dirty && !window.confirm(t('dash.msg2'))) {
    renderConfigList();
    return;
  }
  localStorage.setItem(`moss.config.${state.profileName}`, name);
  await loadConfig(name);
}

export function openConfigDialog(mode) {
  if (!state.profileName) {
    showAlert({ type: 'warn', title: t('workspace.emptyTitle'), text: t('train.msg15'), timeout: 4000 });
    return;
  }
  const dialog = $('#config-dialog');
  const active = state.activeConfigName || 'default';
  state.configDialog.mode = mode;
  const title = mode === 'new' ? t('configDialog.title') : mode === 'saveas' ? t('train.msg16') : t('train.msg17');
  setText('#config-dialog-title', title);
  setText('#config-dialog-desc', mode === 'new'
    ? t('configDialog.desc')
    : t('train.msg18'));
  setHidden('#config-source-row', mode !== 'new');
  const input = $('#config-name-input');
  if (input) {
    input.value = mode === 'new' ? `config-${Date.now().toString().slice(-6)}` : `${active}-copy`;
    input.removeAttribute('aria-invalid');
  }
  const source = $('#config-source-select');
  if (source) source.value = 'current';
  setHidden('#config-dialog-error', true);
  if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
  else dialog.setAttribute('open', '');
  setTimeout(() => { input?.focus(); input?.select(); }, 30);
}

export function closeConfigDialog() {
  const dialog = $('#config-dialog');
  if (!dialog) return;
  if (typeof dialog.close === 'function' && dialog.open) dialog.close();
  else dialog.removeAttribute('open');
}

export async function submitConfigDialog(event) {
  event.preventDefault();
  const mode = state.configDialog.mode;
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
  const oldName = state.activeConfigName;
  const oldVirtual = state.configs.find((entry) => entry.name === oldName)?.virtual;
  readFormIntoConfig();
  let yamlText;
  if (mode === 'new' && $('#config-source-select')?.value === 'workspace') {
    yamlText = state.workspaceDefaultYaml;
  } else {
    yamlText = getCurrentYaml();
  }
  const confirm = $('#config-dialog-confirm');
  if (confirm) confirm.disabled = true;
  try {
    const parsed = parseYaml(yamlText);
    parsed.name = name;
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, yaml: dumpYaml(parsed), overwrite: false }),
    });
    if (mode === 'rename' && oldName && !oldVirtual && oldName !== name) {
      await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs/' + encodeURIComponent(oldName), { method: 'DELETE' });
    }
    closeConfigDialog();
    localStorage.setItem(`moss.config.${state.profileName}`, name);
    await loadConfig(name);
    showSuccess(mode === 'rename' ? t('train.msg20') : mode === 'saveas' ? t('train.msg21') : t('train.msg22'), `${name} · ${data.entry?.path || ''}`, [], 'config-created');
  } catch (error) {
    show(error.detail || error.message);
  } finally {
    if (confirm) confirm.disabled = false;
  }
}

export async function deleteCurrentConfig() {
  if (!state.profileName) return;
  const active = state.configs.find((entry) => entry.name === state.activeConfigName);
  if (!active || active.virtual) return;
  if (!window.confirm(t('train.config.confirmDelete', { name: active.name }))) return;
  try {
    await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs/' + encodeURIComponent(active.name), { method: 'DELETE' });
    localStorage.removeItem(`moss.config.${state.profileName}`);
    await loadConfig();
    showSuccess(t('train.msg23'), active.name, [], 'config-deleted');
  } catch (error) {
    showError(t('train.msg24'), error.detail || error.message);
  }
}

export async function saveConfig(options = {}) {
  if (!state.profileName) {
    showAlert({ type: 'warn', title: t('workspace.emptyTitle'), text: t('train.msg33'), timeout: 5000 });
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
  const target = state.activeConfigName || 'default';
  const button = $('#btn-config-save');
  if (button) button.disabled = true;
  try {
    const parsed = parseYaml(yamlText);
    parsed.name = target;
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: target, yaml: dumpYaml(parsed), overwrite: true }),
    });
    state.savedYaml = dumpYaml(parsed);
    state.savedAt = Date.now();
    state.config.name = target;
    state.configPath = data.entry?.path || state.configPath;
    localStorage.setItem(`moss.config.${state.profileName}`, target);
    localStorage.setItem('moss.workspace', state.profileName);
    markDirty(false);
    clearYamlError();
    const list = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/configs');
    state.configs = Array.isArray(list.configs) ? list.configs : [];
    if (!state.configs.some((entry) => entry.name === target)) {
      state.configs.unshift({ name: target, path: data.entry?.path || '' });
    }
    state.activeConfigName = target;
    renderConfigList();
    setText('#config-path', state.configPath);
      refreshYamlFromConfig();
    if (!options.silent) {
      showSuccess(t('train.msg35'), t('train.config.savedPath', { path: data.entry?.path || state.configPath }), [], 'config-saved');
    }
    runSetupCheck().catch(() => {});
    return true;
  } catch (error) {
    if (error.line) showYamlError({ message: error.detail || error.message, line: error.line, column: error.column });
    showError(t('train.msg36'), error.detail || error.message, [
      { label: t('common.retry'), onClick: () => saveConfig() },
    ]);
    return false;
  } finally {
    if (button) button.disabled = false;
  }
}
