'use strict';

/* =========================================================================
 * MOSS-Image LoRA trainer — front end, no build step, no dependencies
 *
 * State model:
 *   form mode  → form edits the in-memory Config, YAML is a read-only projection
 *   YAML mode  → the textarea is the source, the form is read-only projection
 * There is no third path; the two editors never write each other implicitly.
 * ========================================================================= */

import { initCharts } from './components/chart.js';
import { applyJobRunInfo, loadJobs, pollJobs, renderActivity, renderRunStrip, updateJobUi } from './components/jobs.js';
import { startJob } from './components/jobsse.js';
import { clearLog, currentLogRun, loadRunLog, pushLog, refreshRunLog, renderCaptionJobPanel } from './components/logview.js';
import { loadDashboardLayout } from './components/metriccards.js';
import { runPreflight } from './components/preflight.js';
import { loadConfig } from './config/file.js';
import { getCurrentYaml, refreshYamlFromConfig, setMode } from './config/form.js';
import { ensureDashboardConfigSelection } from './config/list.js';
import { loadProfiles } from './config/profiles.js';
import { dumpYaml, parseYaml } from './config/yaml.js';
import { api, describeError } from './core/api.js';
import { $, $$ } from './core/dom.js';
import { mountFragments } from './core/fragments.js';
import { applyI18n, loadI18n, t } from './core/i18n.js';
import { showAlert, showError, showSuccess } from './core/notify.js';
import { TABS, TAB_LABEL_KEYS, state } from './core/state.js';
import { bindEvents } from './events.js';
import { loadSystem } from './layout/topbar.js';
import { loadCache, renderCacheJob } from './pages/cache.js';
import { loadActiveTraining, loadDashboard } from './pages/dashboard.js';
import { autoLoadDataset, loadCaptionPage, scanDataset, scanRegDataset } from './pages/dataset.js';
import { loadExportPage, loadLoras } from './pages/export.js';
import { loadSamples } from './pages/samples.js';
import { loadSettings, openOnboarding, renderI18nSettings, runSetupCheck } from './pages/settings.js';
import { loadTrainHistory, refreshTrainSamples } from './pages/train.js';
import { loadWorkspacePage, loadWorkspaceRuns } from './pages/workspace.js';

export function currentTab() {
  const hash = (location.hash || '').replace(/^#/, '');
  return TABS.includes(hash) ? hash : 'train';
}

function setTabState(tab) {
  for (const link of $$('#tabs .tab')) {
    const active = link.dataset.tab === tab;
    link.classList.toggle('active', active);
    link.setAttribute('aria-selected', String(active));
    link.setAttribute('tabindex', active ? '0' : '-1');
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  for (const section of $$('.tabpane')) {
    section.classList.toggle('hidden', section.id !== `tab-${tab}`);
  }
  const activeLink = $(`#tab-link-${tab}`);
  if (activeLink) {
    const shell = activeLink.closest('.tabs');
    if (shell) {
      const linkRect = activeLink.getBoundingClientRect();
      const shellRect = shell.getBoundingClientRect();
      if (linkRect.left < shellRect.left || linkRect.right > shellRect.right) {
        activeLink.scrollIntoView({ block: 'nearest', inline: 'center' });
      }
    }
  }
  document.title = t('app.documentTitle', { tab: t(TAB_LABEL_KEYS[tab] || 'nav.train') });
}

export async function route() {
  const tab = currentTab();
  setTabState(tab);
  if (tab === 'dashboard') {
    await ensureDashboardConfigSelection();
    if (!state.layoutLoaded) await loadDashboardLayout();
    await loadDashboard(state.dashboard.run, {});
    await loadActiveTraining();
    loadRunLog(currentLogRun()).catch(() => {});
    loadDashboardPolling();
    loadActivePolling(true);
  } else {
    loadDashboardPolling(false);
  }
  if (tab !== 'train' && tab !== 'dashboard') loadActivePolling(false);
  if (tab === 'train') {
    loadActiveTraining().catch(() => {});
    loadActivePolling(true);
    loadTrainHistory().catch((error) => showAlert({ type: 'warn', title: t('common.msg57'), text: error.message, timeout: 5000 }));
    refreshTrainSamples().catch(() => {});
    runSetupCheck().catch(() => {});
  }
  if (tab === 'dataset') autoLoadDataset().catch(() => {});
  if (tab === 'cache') loadCache().catch(() => {});
  if (tab === 'caption') loadCaptionPage().catch((error) => showError(t('common.msg58'), describeError(error)));
  if (tab === 'samples') loadSamples().catch(() => {});
  if (tab === 'export') loadExportPage().catch(() => {});
  if (tab === 'workspace') loadWorkspacePage().catch(() => {});
  if (tab === 'settings') {
      if (!state.settingsDirty) loadSettings().catch((error) => showError(t('common.msg59'), describeError(error)));
    loadSystem().catch(() => {});
  }
  window.scrollTo({ top: 0, behavior: 'auto' });
}

let activePollingTimer = null;

function loadActivePolling(enabled = true) {
  if (activePollingTimer) {
    clearInterval(activePollingTimer);
    activePollingTimer = null;
  }
  if (!enabled) return;
  activePollingTimer = setInterval(() => {
    if (document.hidden) return;
    loadActiveTraining().catch(() => {});
  }, 5000);
}

let dashboardTimer = null;

function loadDashboardPolling(enabled = true) {
  if (dashboardTimer) {
    clearInterval(dashboardTimer);
    dashboardTimer = null;
  }
  if (!enabled) return;
  dashboardTimer = setInterval(() => {
    if (document.hidden || currentTab() !== 'dashboard' || state.busy) return;
    refreshRunLog().catch(() => {});
    loadDashboard(state.dashboard.run, { silent: true }).catch(() => {});
  }, 5000);
}

export async function readJobLogFile() {
  if (!state.job?.id) {
    const run = currentLogRun();
    if (run) { await loadRunLog(run, { force: true }); showSuccess(t('job.msg57'), run, [], 'run-log-refresh'); return; }
    showAlert({ type: 'info', title: t('job.msg58'), text: t('job.msg59'), timeout: 4000 });
    return;
  }
  try {
    const text = await api(`/api/jobs/${state.job.id}/log`);
    const lines = String(text || '').split('\n');
    state.logs = [];
    clearLog();
    for (const line of lines) {
      if (!line) continue;
      const match = line.match(/step (\d+)\/(\d+) loss ([0-9.]+) avg ([0-9.]+) lr ([0-9.eE+-]+) ([0-9.]+)s peak ([0-9.]+) GiB/);
      if (match) {
        const [_, step, total, loss, avg, lr, sec, peak] = match;
        pushLog(line, 'info');
        state.series.meta.set(Number(step), { lr: Number(lr), sec: Number(sec), vram: Number(peak), total: Number(total), loss: Number(loss), avg: Number(avg) });
      } else {
        pushLog(line, line.startsWith('$ ') ? 'cmd' : 'info');
      }
    }
    showSuccess(t('job.msg60'), state.job.logPath || '', [], 'log-file');
  } catch (error) {
    showError(t('job.msg61'), describeError(error));
  }
}

export function focusLogPanel() {
  if (currentTab() !== 'dashboard') location.hash = '#dashboard';
  setTimeout(() => {
    const panel = $('#log');
    if (panel) {
      panel.focus();
      panel.scrollTop = panel.scrollHeight;
    }
  }, 60);
}

export function activateTabFromKeyboard(link, direction) {
  const tab = link.dataset.tab;
  let index = TABS.indexOf(tab);
  if (direction === 'home') index = 0;
  else if (direction === 'end') index = TABS.length - 1;
  else index = (index + direction + TABS.length) % TABS.length;
  const next = $(`#tab-link-${TABS[index]}`);
  if (next) {
    location.hash = `#${TABS[index]}`;
    setTimeout(() => next.focus(), 0);
  }
}

export function downloadYaml() {
  const text = getCurrentYaml();
  const name = String(state.profileName || state.config?.name || 'workspace').replace(/[^\w.-]+/g, '-');
  const blob = new Blob([text], { type: 'text/yaml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${name}.yaml`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  showSuccess(t('train.msg91'), `${name}.yaml`, [], 'yaml-download');
}

function arrangeDashboardOrder() {
  const monitor = document.getElementById('monitor-panel');
  const kpis = document.getElementById('dash-kpis');
  const samples = document.getElementById('sample-compare-panel');
  if (!monitor || !kpis || !samples) return;
  const parent = monitor.parentNode;
  parent.insertBefore(kpis, monitor);
  parent.appendChild(samples);
}

async function boot() {
  // 外壳里没有页面 markup：先把 9 个片段挂上去，后面所有 $() 才有东西可查。
  try {
    await mountFragments(TABS);
  } catch (error) {
    document.getElementById('main')?.insertAdjacentHTML('afterbegin',
      `<p class="inline-error" role="alert">页面片段加载失败：${error.message}</p>`);
    throw error;
  }
  arrangeDashboardOrder();
  bindEvents();
  initCharts();
  if (!location.hash) {
    history.replaceState(null, '', '#train');
  }

  // 先单独把语言包等到位：loadSystem()/loadJobs() 会立刻把 t() 的文本写进顶栏，
  // 和 loadI18n() 并发跑的话，顶栏会先闪一串裸 key（gpu.unavailable 之类）。
  try {
    await loadI18n();
  } catch (error) {
    showError(t('common.msg89'), error?.message || String(error), [
      { label: t('common.reload'), onClick: () => location.reload() },
    ], 'init-error');
  }

  const results = await Promise.allSettled([
    loadSettings(),
    loadProfiles(),
    loadSystem(),
    loadJobs(),
    api('/api/presets').then((data) => { state.presets = data?.presets || null; }),
  ]);
  const failures = results.filter((result) => result.status === 'rejected');
  for (const failure of failures) {
    showError(t('common.msg89'), failure.reason?.message || String(failure.reason), [
      { label: t('common.reload'), onClick: () => location.reload() },
    ], 'init-error');
  }
  const configResult = await Promise.allSettled([loadConfig()]);
  for (const failure of configResult.filter((result) => result.status === 'rejected')) {
    showError(t('common.msg90'), failure.reason?.message || String(failure.reason), [
      { label: t('common.reload'), onClick: () => location.reload() },
    ], 'config-init-error');
  }
  applyI18n(document);
  renderI18nSettings();
  if (state.mode === 'form') refreshYamlFromConfig();
  renderRunStrip();
  updateJobUi();
  renderCaptionJobPanel();
  runSetupCheck().catch(() => {});
  await route();
  if (state.firstRun) openOnboarding();
  refreshTrainSamples().catch(() => {});
  loadLoras().catch(() => {});

  setInterval(() => { pollJobs().catch(() => {}); }, 4000);
  // The run browser is a snapshot taken when the tab opened. A run that starts
  // outside this page would never appear in it while it runs, so follow the job
  // poll while something is live.
  setInterval(() => {
    if (document.hidden || currentTab() !== 'train') return;
    if (state.job?.status !== 'running' && !state.activeTraining) return;
    loadWorkspaceRuns().catch(() => {});
  }, 4000);
  setInterval(() => {
    if (document.hidden) return;
    loadSystem().catch(() => {});
  }, 2000);
  setInterval(() => {
    if (currentTab() === 'cache' && !state.busy) loadCache().catch(() => {});
  }, 6000);

  // keep relative elapsed times fresh while a job runs
  setInterval(() => {
    if (state.job?.status === 'running') {
      renderActivity();
      renderCacheJob();
    }
  }, 1000);

  if (state.job?.status === 'running') {
    applyJobRunInfo(state.job).catch(() => {});
  }
}


// Small internal hooks for the headless verification script.
window.__mossUI = {
  state,
  parseYaml,
  dumpYaml,
  scanDataset,
  scanRegDataset,
  startJob,
  runPreflight,
  setMode,
  loadCache,
  loadDashboard,
  loadSamples,
  currentTab,
};

document.addEventListener('DOMContentLoaded', () => { boot(); });
