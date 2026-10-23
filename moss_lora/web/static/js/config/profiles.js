/* =========================================================================
 * js/config/profiles.js — Config <-> form
 * split from web/static/app.js, source 1564-2297 (7 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { updateJobUi } from '../components/jobs.js';
import { currentLogRun, loadRunLog } from '../components/logview.js';
import { loadConfig } from './file.js';
import { getCurrentYaml } from './form.js';
import { ensureDashboardConfigSelection, renderConfigIdentity } from './list.js';
import { api, describeError } from '../core/api.js';
import { $, escapeHtml, setHidden, setText } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { state } from '../core/state.js';
import { currentTab } from '../main.js';
import { loadCache } from '../pages/cache.js';
import { clearCompareState, loadActiveTraining, loadDashboard } from '../pages/dashboard.js';
import { loadLoras } from '../pages/export.js';
import { loadSamples } from '../pages/samples.js';
import { runSetupCheck } from '../pages/settings.js';
import { loadTrainHistory } from '../pages/train.js';

function profileNameFromPath(path) {
  const base = String(path || '').split('/').pop() || '';
  return base.replace(/\.ya?ml$/i, '') || 'default';
}

export function renderProfileOptions() {
  const select = $('#profile-select');
  if (!select) return;
  const current = state.profileName || '';
  const profiles = state.profiles || [];
  select.innerHTML = profiles.length
    ? profiles.map((profile) => `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name)}</option>`).join('')
    : `<option value="">${escapeHtml(t('train.profile.noWorkspaceNew'))}</option>`;
  if (profiles.some((profile) => profile.name === current)) select.value = current;
  else if (profiles.length) select.value = profiles[0].name;
  select.disabled = !profiles.length;

  const dashWorkspace = $('#dash-workspace-select');
  if (dashWorkspace) {
    dashWorkspace.innerHTML = profiles.length
      ? profiles.map((profile) => `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name)}</option>`).join('')
      : `<option value="">${escapeHtml(t('train.profile.noWorkspace'))}</option>`;
    if (profiles.some((profile) => profile.name === current)) dashWorkspace.value = current;
    else if (profiles.length) dashWorkspace.value = profiles[0].name;
    dashWorkspace.disabled = !profiles.length;
  }

  setText('#profile-path', state.workspacePath || '—');
  renderConfigIdentity();
  setText('#profile-output-hint', state.workspacePath ? `${state.workspacePath}/runs/<run-id>` : 'workspaces/<workspace>/runs/<run-id>');
  const emptyNotice = $('#workspace-empty-notice');
  if (emptyNotice) emptyNotice.classList.toggle('hidden', Boolean(state.profileName));
  const dirty = $('#profile-dirty');
  if (dirty) {
    dirty.className = `chip ${state.dirty ? 'chip-warn' : 'chip-muted'}`;
    dirty.textContent = state.dirty ? t('common.msg13') : t('common.saved');
  }
  updateJobUi();
}

export async function loadProfiles(options = {}) {
  const data = await api('/api/workspaces');
  state.workspacesRoot = data.root || 'workspaces';
  const profiles = Array.isArray(data.workspaces) ? [...data.workspaces] : [];
  state.profiles = profiles;
  const saved = localStorage.getItem('moss.workspace') || localStorage.getItem('moss.profile') || '';
  const wanted = options.select || saved || data.default || '';
  const chosen = profiles.find((profile) => profile.name === wanted) || profiles[0];
  if (chosen) {
    state.profileName = chosen.name;
    state.workspacePath = chosen.path || '';
    localStorage.setItem('moss.workspace', chosen.name);
  } else {
    state.profileName = '';
    state.workspacePath = '';
    state.configPath = '';
  }
  renderProfileOptions();
  return chosen || null;
}

export async function switchProfile(name) {
  if (!name || name === state.profileName) {
    renderProfileOptions();
    return;
  }
  const profile = (state.profiles || []).find((item) => item.name === name);
  if (!profile) return;
  if (state.dirty && !window.confirm(t('train.msg25'))) {
    renderProfileOptions();
    return;
  }
  const previous = {
    profileName: state.profileName, workspacePath: state.workspacePath,
    configPath: state.configPath, activeRun: state.activeRun,
  };
  try {
    state.profileName = profile.name;
    state.workspacePath = profile.path || '';
    state.configPath = '';
    state.activeRun = '';
    localStorage.setItem('moss.workspace', profile.name);
    await loadConfig();
    state.dashboard.run = null;
    state.currentRun = null;
    state.activeTraining = null;
    clearCompareState();
    state.series = { train: new Map(), avg: new Map(), eval: new Map(), meta: new Map() };
    state.trainTotal = 0;
    renderProfileOptions();
    showAlert({
      type: 'info',
      title: t('train.profile.switchedTitle', { name: profile.name }),
      text: profile.path || '',
      timeout: 3500,
    });
    if (currentTab() === 'dashboard') {
      await ensureDashboardConfigSelection();
      await loadDashboard(null, { silent: true });
      await loadActiveTraining();
      loadRunLog(currentLogRun()).catch(() => {});
    }
    await loadTrainHistory();
    loadSamples().catch(() => {});
    loadLoras().catch(() => {});
    loadCache().catch(() => {});
    runSetupCheck().catch(() => {});
  } catch (error) {
    state.profileName = previous.profileName;
    state.workspacePath = previous.workspacePath;
    state.configPath = previous.configPath;
    state.activeRun = previous.activeRun;
    renderProfileOptions();
    showError(t('train.msg26'), describeError(error));
  }
}

export function openProfileDialog(mode) {
  const dialog = $('#profile-dialog');
  if (!dialog) return;
  if (mode === 'saveas' && !state.profileName) {
    showAlert({ type: 'warn', title: t('train.msg27'), text: t('train.msg15'), timeout: 4500 });
    return;
  }
  if (mode === 'new' && state.dirty
      && !window.confirm(t('train.msg28'))) return;
  state.profileDialog.mode = mode;
  setText('#profile-dialog-title', mode === 'saveas' ? t('train.msg29') : t('train.profile.new'));
  setText('#profile-dialog-desc', mode === 'saveas'
    ? t('train.profile.saveAsDesc', { name: state.profileName || t('common.workspace') })
    : t('train.msg30'));
  setText('#profile-dialog-confirm', mode === 'saveas' ? t('train.msg31') : t('common.create'));
  const input = $('#profile-name-input');
  if (input) {
    const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    input.value = mode === 'saveas' ? `${state.profileName || 'workspace'}-copy` : `workspace-${stamp}`;
    input.removeAttribute('aria-invalid');
  }
  setHidden('#profile-dialog-error', true);
  if (typeof dialog.showModal === 'function') {
    if (!dialog.open) dialog.showModal();
  } else {
    dialog.setAttribute('open', '');
  }
  setTimeout(() => { input?.focus(); input?.select(); }, 30);
}

export function closeProfileDialog() {
  const dialog = $('#profile-dialog');
  if (!dialog) return;
  if (typeof dialog.close === 'function' && dialog.open) dialog.close();
  else dialog.removeAttribute('open');
}

export async function submitProfileDialog(event) {
  event.preventDefault();
  const mode = state.profileDialog.mode;
  const input = $('#profile-name-input');
  let name = String(input?.value || '').trim().replace(/\.ya?ml$/i, '');
  const errorBox = $('#profile-dialog-error');
  const show = (message) => {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.classList.remove('hidden');
    input?.setAttribute('aria-invalid', 'true');
    input?.focus();
  };
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) {
    show(t('train.msg19'));
    return;
  }
  const confirm = $('#profile-dialog-confirm');
  if (confirm) confirm.disabled = true;
  try {
    const payload = { name };
    const data = await api('/api/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    closeProfileDialog();
    clearCompareState();
    state.layoutLoaded = false; state.layout = null;
    await loadProfiles({ select: data.workspace.name });
    await loadConfig();
    state.currentRun = null;
    state.series = { train: new Map(), avg: new Map(), eval: new Map(), meta: new Map() };
    state.trainTotal = 0;
    await loadTrainHistory();
    showSuccess(t('train.msg32'), data.workspace.path, [], 'workspace-created');
    runSetupCheck().catch(() => {});
  } catch (error) {
    show(describeError(error));
  } finally {
    if (confirm) confirm.disabled = false;
  }
}
