/* =========================================================================
 * js/pages/workspace.js — Config <-> form
 * split from web/static/app.js, source 1604-7600 (17 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { startJob } from '../components/jobsse.js';
import { clearLog, pushLog } from '../components/logview.js';
import { loadConfig, switchRun } from '../config/file.js';
import { syncFormFromConfig } from '../config/form.js';
import { renderConfigList } from '../config/list.js';
import { loadProfiles } from '../config/profiles.js';
import { api, describeError } from '../core/api.js';
import { $, escapeHtml, setText } from '../core/dom.js';
import { fmtBytesFromMb, fmtInt } from '../core/fmt.js';
import { t } from '../core/i18n.js';
import { showAlert, showError, showSuccess } from '../core/notify.js';
import { state } from '../core/state.js';
import { currentTab, focusLogPanel } from '../main.js';

// A run's config comes from one place: its own directory, read-only. The API
// labels that source "run"; anything else is shown as-is.
function configSourceLabel(source) {
  if (!source) return t('train.noSnapshot');
  return source === 'run' ? t('workspace.runsTitle') : source;
}


export async function loadWorkspaceRuns() {
  if (!state.profileName) {
    state.runs = [];
    return;
  }
  try {
    const data = await api('/api/workspaces/' + encodeURIComponent(state.profileName) + '/runs');
    state.runs = Array.isArray(data.runs) ? data.runs : [];
  } catch (error) {
    state.runs = [];
    showAlert({ type: 'warn', title: t('workspace.msg1'), text: describeError(error), timeout: 5000 });
  }
}

const WORKSPACE_PART_LABELS = {
  train_cache: 'cache.trainCache',
  reg_cache: 'cache.regCache',
  runs: 'workspace.runsTitle',
  logs: 'common.msg49',
  baseline: 'common.msg50',
  exports: 'nav.export',
};

function workspaceTotal(usage) {
  return usage && Number.isFinite(Number(usage.total_mb)) ? fmtBytesFromMb(Number(usage.total_mb)) : '--';
}

function renderWorkspaceList() {
  const box = $('#workspace-list');
  if (!box) return;
  const profiles = state.workspacePage.list || [];
  box.innerHTML = profiles.length
    ? profiles.map((profile) => {
        const usage = state.workspacePage.usages[profile.name];
        return `<button type="button" class="workspace-list-item ${profile.name === state.profileName ? 'is-active' : ''}" data-workspace-select="${escapeHtml(profile.name)}">
          <span class="workspace-list-name"><span>${escapeHtml(profile.name)}</span><span class="workspace-list-size">${workspaceTotal(usage)}</span></span>
          <span class="workspace-list-path">${escapeHtml(profile.path || '')}</span>
        </button>`;
      }).join('')
    : `<p class="muted small">${escapeHtml(t('workspace.noWorkspace'))}</p>`;
}

function renderWorkspaceUsage(usage) {
  const body = $('#workspace-usage-body');
  const runsBody = $('#workspace-runs-body');
  if (!body || !runsBody) return;
  const available = Boolean(usage);
  for (const id of ['#btn-workspace-open', '#btn-workspace-delete-cache', '#btn-workspace-delete-baseline', '#btn-workspace-delete']) {
    const button = $(id);
    if (button) button.disabled = !available;
  }
  setText('#workspace-usage-name', usage?.workspace || '');
  setText('#workspace-usage-path', usage?.absolute || '--');
  setText('#workspace-usage-total', usage ? workspaceTotal(usage) : '--');
  if (!usage) {
    body.innerHTML = `<tr><td colspan="3" class="table-empty">${escapeHtml(t('workspace.selectPrompt'))}</td></tr>`;
    runsBody.innerHTML = `<tr><td colspan="4" class="table-empty">${escapeHtml(t('workspace.selectPrompt'))}</td></tr>`;
    return;
  }
  const parts = usage.parts || {};
  const rows = Object.keys(WORKSPACE_PART_LABELS).map((key) => {
    const part = parts[key] || {};
    return `<tr><td>${escapeHtml(t(WORKSPACE_PART_LABELS[key]))}</td><td class="num mono">${fmtBytesFromMb(Number(part.mb) || 0)}</td><td class="num mono">${fmtInt(Number(part.files) || 0)}</td></tr>`;
  });
  rows.push(`<tr class="workspace-total-row"><td>${escapeHtml(t('common.total'))}</td><td class="num mono">${workspaceTotal(usage)}</td><td class="num mono">${fmtInt(Object.values(parts).reduce((sum, part) => sum + (Number(part.files) || 0), 0))}</td></tr>`);
  body.innerHTML = rows.join('');
  const runs = Array.isArray(usage.runs) ? usage.runs : [];
  runsBody.innerHTML = runs.length
    ? runs.map((run) => `<tr>
        <td class="mono">${escapeHtml(run.run)}</td>
        <td class="num mono">${fmtBytesFromMb(Number(run.mb) || 0)}</td>
        <td class="num mono">${fmtInt(Number(run.files) || 0)}</td>
        <td class="actions-cell"><button class="btn btn-danger-ghost btn-xs" type="button" data-workspace-run="${escapeHtml(run.run)}">${escapeHtml(t('common.delete'))}</button></td>
      </tr>`).join('')
    : `<tr><td colspan="4" class="table-empty">${escapeHtml(t('workspace.noRuns'))}</td></tr>`;
}

function renderWorkspacePage() {
  const layout = $('#workspace-layout');
  const empty = $('#workspace-page-empty');
  const has = Boolean((state.workspacePage.list || []).length);
  if (layout) layout.classList.toggle('hidden', !has);
  if (empty) empty.classList.toggle('hidden', has);
  renderWorkspaceList();
  renderWorkspaceUsage(state.workspacePage.usage);
}

export async function loadWorkspacePage() {
  if (state.workspacePage.loading) return;
  state.workspacePage.loading = true;
  try {
    const data = await api('/api/workspaces');
    let profiles = Array.isArray(data.workspaces) ? data.workspaces : [];
    state.workspacePage.list = profiles;
    if (!state.profileName || !profiles.some((profile) => profile.name === state.profileName)) {
      await loadProfiles();
      profiles = Array.isArray(state.profiles) ? state.profiles : profiles;
      state.workspacePage.list = profiles;
      if (state.profileName) await loadConfig();
    }
    const current = state.workspacePage.list.find((profile) => profile.name === state.profileName);
    const usages = {};
    await Promise.all(state.workspacePage.list.map(async (profile) => {
      try {
        usages[profile.name] = await api('/api/workspaces/' + encodeURIComponent(profile.name) + '/usage');
      } catch (error) {
        usages[profile.name] = null;
      }
    }));
    state.workspacePage.usages = usages;
    state.workspacePage.usage = current ? usages[current.name] : null;
    renderWorkspacePage();
  } catch (error) {
    state.workspacePage.list = [];
    state.workspacePage.usage = null;
    renderWorkspacePage();
    showError(t('workspace.msg2'), describeError(error));
  } finally {
    state.workspacePage.loading = false;
  }
}

export function closeWorkspaceConfirm() {
  const dialog = $('#workspace-confirm-dialog');
  state.workspacePage.confirm = null;
  if (dialog?.open) dialog.close();
}

export function openWorkspaceConfirm({ title, text, requireName = '', confirmText = t('common.delete'), action = '', run = '' }) {
  const dialog = $('#workspace-confirm-dialog');
  if (!dialog) return;
  state.workspacePage.confirm = { requireName, confirmText, action, run };
  setText('#workspace-confirm-title', title);
  setText('#workspace-confirm-text', text);
  const nameRow = $('#workspace-confirm-name-row');
  const nameInput = $('#workspace-confirm-name');
  const button = $('#workspace-confirm-delete');
  if (nameRow) nameRow.classList.toggle('hidden', !requireName);
  if (nameInput) {
    nameInput.value = '';
    nameInput.placeholder = requireName || '';
  }
  if (button) {
    button.textContent = confirmText;
    button.disabled = Boolean(requireName);
  }
  if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
  else dialog.setAttribute('open', '');
  setTimeout(() => (requireName ? nameInput : button)?.focus(), 40);
}

async function reloadWorkspaceAfterDelete({ workspaceDeleted = false } = {}) {
  if (workspaceDeleted) {
    await loadProfiles();
    if (state.profileName) await loadConfig();
    else {
      state.config = {};
      state.runs = [];
      state.activeRun = '';
      renderConfigList();
      syncFormFromConfig({});
    }
  }
  await loadWorkspacePage();
}

export async function performWorkspaceDelete(action, run = '') {
  if (!state.profileName) return;
  const name = state.profileName;
  try {
    if (action === 'run') {
      await api('/api/workspaces/' + encodeURIComponent(name) + '/runs/' + encodeURIComponent(run), { method: 'DELETE' });
      showSuccess(t('workspace.msg3'), `${name}/${run}`, [], 'workspace-run-deleted');
      await loadWorkspacePage();
    } else if (action === 'cache') {
      await api('/api/workspaces/' + encodeURIComponent(name) + '/cache', { method: 'DELETE' });
      showSuccess(t('workspace.msg4'), name, [], 'workspace-cache-deleted');
      await loadWorkspacePage();
    } else if (action === 'baseline') {
      await api('/api/workspaces/' + encodeURIComponent(name) + '/baseline', { method: 'DELETE' });
      showSuccess(t('workspace.msg5'), name, [], 'workspace-baseline-deleted');
      await loadWorkspacePage();
    } else if (action === 'workspace') {
      await api('/api/workspaces/' + encodeURIComponent(name), { method: 'DELETE' });
      showSuccess(t('workspace.msg6'), name, [], 'workspace-deleted');
      await reloadWorkspaceAfterDelete({ workspaceDeleted: true });
    }
  } catch (error) {
    showError(t('workspace.msg7'), describeError(error));
  }
}
