/* =========================================================================
 * js/events.js — Routing + events
 * split from web/static/app.js, source 8750-9421; behaviour unchanged
 * ========================================================================= */

import { applySmoothWindow, drawChart } from './components/chart.js'; import { startJob, stopJob } from './components/jobsse.js';
import { clearLog, copyLog, isLogNearBottom } from './components/logview.js'; import { drawMetricCards, positionWidgets, toggleMetricCard } from './components/metriccards.js';
import { runPreflight } from './components/preflight.js'; import { closeConfigDialog, deleteCurrentRun, openNewRunDialog, saveConfig, submitConfigDialog, switchRun } from './config/file.js';
import { addLayerRow, clearYamlError, focusYamlLine, markDirty, readLayerControlsIntoConfig, scheduleFormSync, setMode, showYamlError, syncAttentionControls, syncFormFromConfig, syncQuantizeControls, syncSaveStateControls, syncYamlScroll, updateCacheEffective, updateLrEffective, updateOutputPreview, applyPresetToForm, updateRegCacheEffective, updateRegCachePlaceholder, updateYamlLineCount } from './config/form.js'; import { selectDashboardConfig, startDashboardTraining } from './config/list.js';
import { FIELD_DEFS, LAYER_COUNT } from './config/model.js'; import { closeProfileDialog, openProfileDialog, submitProfileDialog, switchProfile } from './config/profiles.js';
import { setFieldError, validateField } from './config/validate.js'; import { dumpYaml, parseYaml } from './config/yaml.js';
import { copyText } from './core/clipboard.js'; import { $, $$, setHidden, setText } from './core/dom.js';
import { setUiLanguage, t } from './core/i18n.js'; import { closeModal, focusableInModal } from './core/modal.js';
import { showAlert, showError, showSuccess } from './core/notify.js'; import { state } from './core/state.js';
import { loadSystem } from './layout/topbar.js'; import { activateTabFromKeyboard, downloadYaml, focusLogPanel, readJobLogFile, route } from './main.js';
import { loadCache, saveCaptionSource, startCacheJob, updateCacheStartState } from './pages/cache.js'; import { clearCaptionApiKey, loadCaptionSettings, markCaptionSettingsDirty, renderCaptionApiKeyStatus, resetCaptionPrompts, saveCaptionSettings, updateCaptionApiKeyButton, updateCaptionTriggerWarning } from './pages/caption.js';
import { closeSampleCompare, initSampleCompareSelection, loadActiveTraining, loadDashboard, loadDashboardCompare, openCompareMenu, openSampleCompare, renderCompareLegend, toggleSampleCollapsed, toggleSampleCompareRun } from './pages/dashboard.js'; import { loadCaptionDatasetInfo, readCaptionSelection, scanDataset, scanRegDataset, updateCaptionStartState } from './pages/dataset.js';
import { applyExportRange, choosePathFromBrowser, closeExportDirectory, closePathBrowser, defaultExportDirectory, downloadSingleStep, exportSelected, loadExportDirectory, loadPathBrowser, onExportRunChange, renderExportStepRows, setExportDestination, updateExportSelectionUi } from './pages/export.js'; import { loadSamples, renderSamples } from './pages/samples.js';
import { MODEL_FIELD_KINDS, clearSettingsErrors, closeOnboarding, deleteI18nLanguage, focusField, importI18nPack, loadSettings, markSettingsDirty, probeModel, readI18nImportFile, runSetupCheck, saveSettings, setModelStatus, syncAllocatorControls } from './pages/settings.js'; import { refreshTrainSamples } from './pages/train.js';
import { closeWorkspaceConfirm, loadWorkspacePage, openWorkspaceConfirm, performWorkspaceDelete } from './pages/workspace.js';
import { describeError } from './core/api.js';

export function bindEvents() {
  window.addEventListener('hashchange', () => { route().catch((error) => showError(t('common.msg60'), error.message)); });

  const tablist = $('#tabs');
  if (tablist) {
    tablist.addEventListener('keydown', (event) => {
      const link = event.target.closest('.tab');
      if (!link) return;
      if (event.key === 'ArrowRight') { event.preventDefault(); activateTabFromKeyboard(link, 1); }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); activateTabFromKeyboard(link, -1); }
      else if (event.key === 'Home') { event.preventDefault(); activateTabFromKeyboard(link, 'home'); }
      else if (event.key === 'End') { event.preventDefault(); activateTabFromKeyboard(link, 'end'); }
      else if (event.key === ' ') { event.preventDefault(); location.hash = `#${link.dataset.tab}`; }
    });
  }

  $('#activity-open')?.addEventListener('click', focusLogPanel);
  $('#btn-read-log')?.addEventListener('click', readJobLogFile);
  $('#activity-stop')?.addEventListener('click', () => stopJob());

  for (const selector of ['#ui-language-select', '#i18n-current-select']) {
    $(selector)?.addEventListener('change', (event) => setUiLanguage(event.target.value));
  }
  $('#i18n-import-pick')?.addEventListener('click', () => $('#i18n-import-file')?.click());
  $('#i18n-import-file')?.addEventListener('change', (event) => readI18nImportFile(event));
  $('#i18n-import-btn')?.addEventListener('click', () => importI18nPack());
  $('#i18n-language-list')?.addEventListener('click', (event) => {
    const use = event.target.closest('[data-i18n-use]');
    if (use) { setUiLanguage(use.dataset.i18nUse); return; }
    const remove = event.target.closest('[data-i18n-delete]');
    if (remove) {
      deleteI18nLanguage(remove.dataset.i18nDelete, Boolean(remove.dataset.i18nBuiltin), remove.dataset.i18nName);
    }
  });

  $('#profile-select')?.addEventListener('change', (event) => switchProfile(event.target.value));
  $('#dash-workspace-select')?.addEventListener('change', (event) => switchProfile(event.target.value));
  $('#btn-profile-new')?.addEventListener('click', () => openProfileDialog('new'));
  $('#btn-profile-saveas')?.addEventListener('click', () => openProfileDialog('saveas'));
  $('#btn-empty-new')?.addEventListener('click', () => openProfileDialog('new'));
  $('#profile-form')?.addEventListener('submit', submitProfileDialog);
  $('#profile-dialog-close')?.addEventListener('click', closeProfileDialog);
  $('#profile-dialog-cancel')?.addEventListener('click', closeProfileDialog);
  $('#settings-error-list')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-settings-field]');
    if (button) document.getElementById(button.dataset.settingsField)?.focus();
  });
  $('#btn-save-settings')?.addEventListener('click', () => saveSettings());
  $('#btn-refresh-settings')?.addEventListener('click', () => {
    if (state.settingsDirty && !window.confirm(t('common.msg61'))) return;
    loadSettings().catch((error) => showError(t('common.msg59'), describeError(error)));
    loadSystem().catch(() => {});
  });
  $('#btn-save-caption')?.addEventListener('click', () => saveCaptionSettings());
  $('#btn-reset-caption-prompts')?.addEventListener('click', resetCaptionPrompts);
  $('#btn-clear-caption-key')?.addEventListener('click', clearCaptionApiKey);
  $('#btn-refresh-caption')?.addEventListener('click', () => {
    if (state.captionSettingsDirty && !window.confirm(t('common.msg62'))) return;
    loadCaptionSettings().catch((error) => showError(t('caption.msg11'), describeError(error)));
    loadCaptionDatasetInfo().catch(() => {});
  });
  for (const id of ['set-workspaces-path', 'set-allocator', 'set-gc-threshold']) {
    const input = document.getElementById(id);
    if (!input) continue;
    const handler = () => {
      if (id === 'set-allocator') syncAllocatorControls();
      markSettingsDirty();
      clearSettingsErrors();
    };
    input.addEventListener('input', handler);
    input.addEventListener('change', handler);
  }
  const captionPanel = $('#caption-settings-card');
  if (captionPanel) {
    const handler = (event) => {
      if (event?.target?.id === 'set-caption-trigger') updateCaptionTriggerWarning(true);
      if (event?.target?.id === 'set-caption-api-key') {
        if (String(event.target.value || '').trim()) state.captionClearKey = false;
        renderCaptionApiKeyStatus();
        updateCaptionApiKeyButton();
      }
      markCaptionSettingsDirty();
    };
    captionPanel.addEventListener('input', handler);
    captionPanel.addEventListener('change', handler);
  }
  for (const button of $$('[data-model-probe]')) {
    button.addEventListener('click', () => probeModel(button.dataset.modelProbe));
  }
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-focus-field]');
    if (!target) return;
    const id = target.dataset.focusField;
    setTimeout(() => focusField(id), 120);
  });
  $('#onboarding-close')?.addEventListener('click', closeOnboarding);
  $('#onboarding-later')?.addEventListener('click', closeOnboarding);
  $('#onboarding-start')?.addEventListener('click', () => {
    closeOnboarding();
    const hasWorkspace = (state.profiles || []).length > 0;
    if (!hasWorkspace) {
      openProfileDialog('new');
      return;
    }
    location.hash = '#train';
    setTimeout(() => focusField('cfg-model-dit'), 120);
  });
  $('#btn-setup-refresh')?.addEventListener('click', () => runSetupCheck());
  $('#btn-start')?.addEventListener('click', () => startDashboardTraining().catch((error) => showError(t('common.msg63'), describeError(error))));
  $('#btn-resume')?.addEventListener('click', () => startDashboardTraining({ resume: true }).catch((error) => showError(t('common.msg63'), describeError(error))));
  $('#btn-stop')?.addEventListener('click', () => stopJob());
  $('#btn-preflight')?.addEventListener('click', async () => {
    const button = $('#btn-preflight');
    if (button) button.disabled = true;
    try {
      await runPreflight({ forStart: false });
    } finally {
      if (button) button.disabled = false;
    }
  });

  $('#mode-form')?.addEventListener('click', () => setMode('form'));
  $('#mode-yaml')?.addEventListener('click', () => setMode('yaml'));
  $('#btn-yaml-toggle')?.addEventListener('click', () => setMode(state.mode === 'form' ? 'yaml' : 'form'));
  $('#btn-yaml-apply')?.addEventListener('click', () => setMode('form'));
  $('#btn-yaml-rebuild')?.addEventListener('click', () => {
    try {
      const parsed = parseYaml($('#yaml')?.value || '');
      state.config = parsed;
      const textarea = $('#yaml');
      if (textarea) textarea.value = dumpYaml(parsed);
      clearYamlError();
      updateYamlLineCount();
      syncFormFromConfig(parsed);
      markDirty();
      showSuccess(t('common.msg64'), t('common.msg65'), [], 'yaml-format');
    } catch (error) {
      showYamlError(error);
    }
  });
  $('#btn-yaml-check')?.addEventListener('click', async () => {
    const result = await runPreflight({ forStart: false });
    if (result.ok) showSuccess(t('common.msg66'), t('common.msg67'), [], 'yaml-check');
  });
  $('#btn-yaml-copy')?.addEventListener('click', () => copyText($('#yaml')?.value || '', t('common.msg68')));
  $('#btn-yaml-download')?.addEventListener('click', downloadYaml);
  $('#yaml')?.addEventListener('input', () => {
    if (state.mode !== 'yaml') return;
    markDirty();
    updateYamlLineCount();
    clearTimeout(state.yamlParseTimer);
    state.yamlParseTimer = setTimeout(() => {
      try {
        const parsed = parseYaml($('#yaml')?.value || '');
        state.config = parsed;
        syncFormFromConfig(parsed);
        clearYamlError();
        updateOutputPreview();
      } catch (error) {
        showYamlError(error);
      }
    }, 320);
  });
  $('#yaml')?.addEventListener('scroll', syncYamlScroll);
  $('#yaml-error')?.addEventListener('click', () => {
    if (!state.yamlError?.line) return;
    if (state.mode !== 'yaml') {
      showAlert({ type: 'info', title: t('common.msg69'), text: t('common.msg70'), timeout: 3000 });
      return;
    }
    focusYamlLine(state.yamlError.line);
  });

  for (const field of FIELD_DEFS) {
    const input = document.getElementById(field.id);
    if (!input) continue;
    input.addEventListener('blur', () => {
      if (state.mode !== 'form') return;
      setFieldError(field, validateField(field));
    });
    input.addEventListener('input', () => {
      if (state.mode !== 'form') return;
      if (field.id === 'cfg-attention') syncAttentionControls();
      if (field.id === 'cfg-quantize-dit') syncQuantizeControls();
      if (field.id === 'cfg-reg-path') { updateRegCachePlaceholder(); updateRegCacheEffective(); }
      if (field.id === 'cfg-cache-reg-dir') updateRegCacheEffective();
      if (MODEL_FIELD_KINDS[field.id]) setModelStatus(MODEL_FIELD_KINDS[field.id], null);
      setFieldError(field, null);
      markDirty();
      scheduleFormSync();
      updateOutputPreview();
    });
    input.addEventListener('change', () => {
      if (state.mode !== 'form') return;
      if (field.id === 'cfg-attention') syncAttentionControls();
      if (field.id === 'cfg-quantize-dit') syncQuantizeControls();
      if (field.id === 'cfg-cache-dir') updateCacheEffective();
      if (field.id === 'cfg-reg-path') { updateRegCachePlaceholder(); updateRegCacheEffective(); }
      if (field.id === 'cfg-cache-reg-dir') updateRegCacheEffective();
      markDirty();
      scheduleFormSync();
      updateOutputPreview();
    });
  }
  $('#cfg-vram-preset')?.addEventListener('change', (event) => {
    if (state.mode !== 'form') return;
    if (!event.target.value) return;
    applyPresetToForm(event.target.value);
    scheduleFormSync();
    updateOutputPreview();
  });
  for (const id of ['cfg-quantize-dtype', 'cfg-quantize-conditioning']) {
    document.getElementById(id)?.addEventListener('change', syncQuantizeControls);
  }
  for (const id of ['cfg-lr', 'cfg-warmup', 'cfg-lr-scheduler', 'cfg-steps']) {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener('input', updateLrEffective);
      element.addEventListener('change', updateLrEffective);
    }
  }
  $('#cfg-save-state')?.addEventListener('change', () => {
    syncSaveStateControls();
    if (state.mode !== 'form') return;
    markDirty();
    scheduleFormSync();
  });
  $('#cfg-save-state-strategy')?.addEventListener('change', () => {
    if (state.mode !== 'form') return;
    markDirty();
    scheduleFormSync();
  });
  for (const id of ['cfg-export-kohya', 'cfg-export-diffusers']) {
    document.getElementById(id)?.addEventListener('change', () => {
      if (state.mode !== 'form') return;
      markDirty();
      scheduleFormSync();
    });
  }
  const layerChanged = () => {
    if (state.mode !== 'form') return;
    readLayerControlsIntoConfig();
    markDirty();
    scheduleFormSync();
  };
  $('#layer-rows')?.addEventListener('input', layerChanged);
  $('#layer-rows')?.addEventListener('change', layerChanged);
  $('#layer-rows')?.addEventListener('click', (event) => {
    const remove = event.target.closest('.layer-remove');
    if (!remove) return;
    const rows = $$('#layer-rows .layer-row');
    if (rows.length <= 1) {
      const range = rows[0]?.querySelector('.layer-range');
      if (range) range.value = `0-${LAYER_COUNT - 1}`;
    } else {
      remove.closest('.layer-row')?.remove();
    }
    layerChanged();
  });
  $('#btn-add-layer-row')?.addEventListener('click', addLayerRow);

  $('#cfg-prompts')?.addEventListener('input', () => {
    if (state.mode !== 'form') return;
    markDirty();
    scheduleFormSync();
  });
  $('#config-form')?.addEventListener('submit', (event) => event.preventDefault());
  $('#form-errors-list')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-field]');
    if (!button) return;
    const input = document.getElementById(button.dataset.field);
    input?.focus();
    input?.scrollIntoView({ block: 'center' });
  });

  $('#config-select')?.addEventListener('change', (event) => switchRun(event.target.value));
  $('#btn-config-new')?.addEventListener('click', () => openNewRunDialog());
  $('#btn-config-save')?.addEventListener('click', () => saveConfig());
  $('#btn-config-delete')?.addEventListener('click', deleteCurrentRun);
  $('#config-dialog-close')?.addEventListener('click', closeConfigDialog);
  $('#config-dialog-cancel')?.addEventListener('click', closeConfigDialog);
  $('#config-dialog-form')?.addEventListener('submit', submitConfigDialog);


  $('#btn-scan')?.addEventListener('click', scanDataset);
  $('#cfg-dataset-path')?.addEventListener('keydown', (event) => { if (event.key === 'Enter') scanDataset(); });
  $('#btn-ds-retry')?.addEventListener('click', scanDataset);
  $('#btn-ds-reg-scan')?.addEventListener('click', () => scanRegDataset());
  $('#cfg-reg-path')?.addEventListener('keydown', (event) => { if (event.key === 'Enter') scanRegDataset(); });
  $('#btn-caption-start')?.addEventListener('click', () => {
    const selection = readCaptionSelection();
    if (!selection.datasets.length || !selection.styles.length) {
      updateCaptionStartState();
      showAlert({
        type: 'warn',
        title: !selection.datasets.length ? t('caption.selectDataset') : t('job.msg33'),
        text: !selection.datasets.length ? t('job.msg34') : t('job.msg35'),
        timeout: 5000,
      });
      return;
    }
    startJob('caption').catch(() => {});
  });
  for (const id of ['caption-dataset-train', 'caption-dataset-reg', 'caption-style-tag', 'caption-style-official', 'caption-overwrite']) {
    const input = document.getElementById(id);
    if (input) input.addEventListener('change', updateCaptionStartState);
  }
  $('#btn-caption-stop')?.addEventListener('click', () => stopJob());
  $('#cfg-caption-source')?.addEventListener('change', (event) => {
    saveCaptionSource(event.target.value).catch((error) => showError(t('common.msg71'), describeError(error)));
  });

  $('#btn-cache-refresh')?.addEventListener('click', () => loadCache());
  $('#btn-cache-start')?.addEventListener('click', startCacheJob);
  for (const id of ['cache-part-latent', 'cache-part-text', 'cache-overwrite']) {
    const input = document.getElementById(id);
    if (input) input.addEventListener('change', updateCacheStartState);
  }
  $('#cache-stop')?.addEventListener('click', () => stopJob());

  $('#btn-refresh-samples')?.addEventListener('click', () => loadSamples());
  $('#samples-filter')?.addEventListener('change', (event) => {
    state.samples.filter = event.target.value;
    renderSamples();
  });
  $('#export-run')?.addEventListener('change', () => { onExportRunChange().catch(() => {}); });
  for (const id of ['#export-range-from', '#export-range-to', '#export-range-every']) {
    $(id)?.addEventListener('change', applyExportRange);
  }
  $('#export-select-all')?.addEventListener('change', (event) => {
    const numeric = state.exportCheckpoints.filter((item) => item.step !== 'final').map((item) => item.step);
    state.exportSelected = event.target.checked ? new Set(numeric.map((value) => String(value))) : new Set();
    renderExportStepRows();
  });
  $('#btn-export-batch')?.addEventListener('click', () => { exportSelected().catch(() => {}); });
  $('#btn-workspace-refresh')?.addEventListener('click', () => loadWorkspacePage());
  $('#btn-workspace-open')?.addEventListener('click', () => {
    const path = state.workspacePage.usage?.absolute || state.workspacePage.usage?.path || state.workspacePath;
    if (path) copyText(path, t('common.msg72'));
  });
  $('#btn-workspace-delete-cache')?.addEventListener('click', () => {
    if (!state.profileName) return;
    openWorkspaceConfirm({
      title: t('common.msg73'), text: t('common.msg74'), confirmText: t('common.msg75'),
      action: 'cache',
    });
  });
  $('#btn-workspace-delete-baseline')?.addEventListener('click', () => {
    if (!state.profileName) return;
    openWorkspaceConfirm({
      title: t('common.msg76'), text: t('common.msg77'), confirmText: t('workspace.deleteBaseline'),
      action: 'baseline',
    });
  });
  $('#btn-workspace-delete')?.addEventListener('click', () => {
    if (!state.profileName) return;
    openWorkspaceConfirm({
      title: t('common.msg78'),
      text: t('workspace.confirm.deleteText', { name: state.profileName }),
      requireName: state.profileName, confirmText: t('common.msg79'), action: 'workspace',
    });
  });
  $('#workspace-list')?.addEventListener('click', (event) => {
    const item = event.target.closest('[data-workspace-select]');
    if (!item) return;
    const name = item.dataset.workspaceSelect;
    if (name === state.profileName) return;
    switchProfile(name).then(() => loadWorkspacePage()).catch(() => {});
  });
  $('#workspace-runs-body')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-workspace-run]');
    if (!button) return;
    const run = button.dataset.workspaceRun;
    openWorkspaceConfirm({
      title: t('workspace.confirm.deleteRun', { run }),
      text: t('common.msg80'),
      confirmText: t('common.msg81'), action: 'run', run,
    });
  });
  $('#workspace-confirm-name')?.addEventListener('input', (event) => {
    const confirm = state.workspacePage.confirm;
    const button = $('#workspace-confirm-delete');
    if (!confirm || !button) return;
    button.disabled = confirm.requireName ? event.target.value.trim() !== confirm.requireName : false;
  });
  $('#workspace-confirm-delete')?.addEventListener('click', async () => {
    const confirm = state.workspacePage.confirm;
    if (!confirm) return;
    if (confirm.requireName && String($('#workspace-confirm-name')?.value || '').trim() !== confirm.requireName) return;
    const action = confirm.action;
    const run = confirm.run;
    closeWorkspaceConfirm();
    await performWorkspaceDelete(action, run);
  });
  for (const id of ['#workspace-confirm-cancel', '#workspace-confirm-cancel-top']) {
    $(id)?.addEventListener('click', closeWorkspaceConfirm);
  }
  $('#workspace-confirm-dialog')?.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeWorkspaceConfirm();
  });
  for (const button of document.querySelectorAll('[data-path-browse]')) {
    button.addEventListener('click', () => {
      choosePathFromBrowser(button.dataset.pathBrowse, button.dataset.pathMode || 'dir').catch(() => {});
    });
  }
  $('#path-browser-path')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); loadPathBrowser(event.target.value).catch(() => {}); }
  });
  $('#path-browser-path')?.addEventListener('change', (event) => {
    loadPathBrowser(event.target.value).catch(() => {});
  });
  $('#path-browser-list')?.addEventListener('click', (event) => {
    const dir = event.target.closest('[data-path-dir]');
    if (dir) { loadPathBrowser(dir.dataset.pathDir).catch(() => {}); return; }
    const file = event.target.closest('[data-path-file]');
    if (file) {
      state.pathBrowser.selected = file.dataset.pathFile;
      for (const item of document.querySelectorAll('#path-browser-list .export-dir-entry')) item.classList.remove('is-selected');
      file.classList.add('is-selected');
      setText('#path-browser-selected', file.dataset.pathFile);
    }
  });
  $('#path-browser-confirm')?.addEventListener('click', () => {
    const mode = state.pathBrowser.mode;
    const value = mode === 'dir' ? state.pathBrowser.dir : (state.pathBrowser.selected || (mode === 'either' ? state.pathBrowser.dir : ''));
    if (!value) {
      showAlert({ type: 'warn', title: t('common.msg82'), text: t('common.msg83'), timeout: 3500 });
      return;
    }
    const field = document.getElementById(state.pathBrowser.fieldId);
    if (field) {
      field.value = value;
      field.dispatchEvent(new Event('input', { bubbles: true }));
      field.dispatchEvent(new Event('change', { bubbles: true }));
      if (field.id === 'cfg-cache-dir') updateCacheEffective();
    }
    closePathBrowser(value);
  });
  for (const id of ['#path-browser-cancel', '#path-browser-cancel-top']) {
    $(id)?.addEventListener('click', () => closePathBrowser(null));
  }
  $('#path-browser-dialog')?.addEventListener('cancel', (event) => {
    event.preventDefault();
    closePathBrowser(null);
  });
  const exportPathInput = $('#export-dir-path');
  exportPathInput?.addEventListener('input', () => {
    clearTimeout(state.exportDirTimer);
    state.exportDirPath = exportPathInput.value;
    setText('#export-dir-preview', exportPathInput.value || defaultExportDirectory());
    const notice = $('#export-dir-notice');
    if (notice) notice.classList.add('hidden');
    state.exportDirTimer = setTimeout(() => {
      loadExportDirectory(exportPathInput.value || defaultExportDirectory()).catch(() => {});
    }, 300);
  });
  exportPathInput?.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    clearTimeout(state.exportDirTimer);
    loadExportDirectory(exportPathInput.value || defaultExportDirectory()).catch(() => {});
  });
  exportPathInput?.addEventListener('change', () => {
    clearTimeout(state.exportDirTimer);
    loadExportDirectory(exportPathInput.value || defaultExportDirectory()).catch(() => {});
  });
  $('#export-dir-list')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-export-dir]');
    if (button) loadExportDirectory(button.dataset.exportDir).catch(() => {});
  });
  $('#btn-export-dir-new')?.addEventListener('click', () => {
    const name = String($('#export-dir-new-name')?.value || '').trim();
    if (!name) { $('#export-dir-new-name')?.focus(); return; }
    const base = String(state.exportDirPath || $('#export-dir-path')?.value || '').replace(/\/+$/, '');
    loadExportDirectory(`${base}/${name}`).catch(() => {});
  });
  $('#export-dir-confirm')?.addEventListener('click', () => {
    const path = String($('#export-dir-path')?.value || state.exportDirPath || '').trim() || defaultExportDirectory();
    state.exportTargetDir = path;
    closeExportDirectory(path);
  });
  for (const id of ['#export-dir-cancel', '#export-dir-cancel-top']) {
    $(id)?.addEventListener('click', () => closeExportDirectory(null));
  }
  $('#export-dir-dialog')?.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeExportDirectory(null);
  });
  $('#export-dest-server')?.addEventListener('change', () => setExportDestination('server'));
  $('#export-dest-browser')?.addEventListener('change', () => setExportDestination('browser'));
  $('#loras-body')?.addEventListener('change', (event) => {
    const checkbox = event.target.closest('[data-export-step]');
    if (!checkbox) return;
    const step = checkbox.dataset.exportStep;
    if (checkbox.checked) state.exportSelected.add(step);
    else state.exportSelected.delete(step);
    updateExportSelectionUi();
  });
  $('#loras-body')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-export-download]');
    if (button) downloadSingleStep(button.dataset.exportDownload).catch(() => {});
  });
  $('#btn-refresh-train-samples')?.addEventListener('click', () => refreshTrainSamples());
  $('#dash-refresh')?.addEventListener('click', () => {
    loadDashboard(state.dashboard.run, {}).then(() => loadDashboardCompare()).catch(() => {});
    loadActiveTraining().catch(() => {});
    loadSystem().catch(() => {});
  });
  $('#dash-config-select')?.addEventListener('change', (event) => {
    const target = event.target.value;
    if (!target) return;
    selectDashboardConfig(target)
      .then(async (ok) => {
        if (!ok) return;
        const entry = (state.runs || []).find((run) => run.name === target) || null;
        const runLabel = state.dashboard.run || entry?.label || target;
        await loadDashboard(runLabel, {});
        await loadDashboardCompare();
        initSampleCompareSelection();
      })
      .catch((error) => showError(t('common.msg84'), describeError(error)));
  });
  $('#dash-smooth')?.addEventListener('input', applySmoothWindow);
  $('#dash-smooth')?.addEventListener('change', applySmoothWindow);
  for (const input of $$('#chart-controls [data-metric]')) {
    input.checked = Boolean(state.chartMetrics[input.dataset.metric]);
    input.addEventListener('change', () => {
      toggleMetricCard(input.dataset.metric, input.checked);
    });
  }
  $('#btn-compare-toggle')?.addEventListener('click', (event) => {
    event.stopPropagation();
    openCompareMenu();
  });
  $('#compare-menu')?.addEventListener('change', (event) => {
    const input = event.target.closest('[data-compare-run]');
    if (!input) return;
    if (input.checked) state.compareRuns.add(input.dataset.compareRun);
    else state.compareRuns.delete(input.dataset.compareRun);
    setText('#compare-count', String(state.compareRuns.size));
    loadDashboardCompare().catch(() => {});
  });
  $('#btn-sample-collapse')?.addEventListener('click', toggleSampleCollapsed);
  $('#btn-sample-runs')?.addEventListener('click', () => {
    const menu = $('#sample-run-menu');
    if (!menu) return;
    menu.classList.toggle('hidden');
    $('#btn-sample-runs')?.setAttribute('aria-expanded', menu.classList.contains('hidden') ? 'false' : 'true');
  });
  $('#sample-run-menu')?.addEventListener('change', (event) => {
    const input = event.target.closest('[data-sample-run]');
    if (input) toggleSampleCompareRun(input.dataset.sampleRun, input.checked);
  });
  $('#sample-compare-grid')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-compare-step]');
    if (button) {
      state.sampleCompareTrigger = button;
      openSampleCompare(Number(button.dataset.compareStep));
    }
  });
  $('#sample-compare-close')?.addEventListener('click', closeSampleCompare);
  const sampleDialog = $('#sample-compare-dialog');
  sampleDialog?.addEventListener('click', (event) => {
    if (event.target !== sampleDialog) return;
    const rect = sampleDialog.getBoundingClientRect();
    const outside = event.clientX < rect.left || event.clientX > rect.right
      || event.clientY < rect.top || event.clientY > rect.bottom;
    if (outside) closeSampleCompare();
  });
  sampleDialog?.addEventListener('close', () => {
    const trigger = state.sampleCompareTrigger;
    state.sampleCompareTrigger = null;
    if (trigger && trigger.isConnected) requestAnimationFrame(() => trigger.focus());
  });
  $('#dash-compare-legend')?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-compare-run][data-compare-series]');
    if (!button) return;
    const item = state.compareData.find((run) => run.name === button.dataset.compareRun);
    if (!item) return;
    const series = button.dataset.compareSeries;
    if (!['train', 'eval'].includes(series)) return;
    item.visible = item.visible || { train: true, eval: true };
    item.visible[series] = item.visible[series] === false;
    renderCompareLegend();
    drawChart('dash');
  });
  document.addEventListener('click', (event) => {
    const dropdown = $('#compare-dropdown');
    if (dropdown && !dropdown.contains(event.target)) {
      $('#compare-menu')?.classList.add('hidden');
      $('#btn-compare-toggle')?.setAttribute('aria-expanded', 'false');
    }
  });

  $('#btn-pause-log')?.addEventListener('click', () => {
    state.logPaused = !state.logPaused;
    const button = $('#btn-pause-log');
    if (button) {
      button.textContent = state.logPaused ? t('common.msg85') : t('dash.log.pause');
      button.setAttribute('aria-pressed', String(state.logPaused));
    }
    if (!state.logPaused) {
      const box = $('#log');
      if (box) box.scrollTop = box.scrollHeight;
      setHidden('#log-jump', true);
    }
  });
  $('#btn-clear-log')?.addEventListener('click', clearLog);
  $('#btn-copy-log')?.addEventListener('click', copyLog);
  $('#log')?.addEventListener('scroll', () => {
    const box = $('#log');
    if (!box) return;
    if (state.logPaused) return;
    setHidden('#log-jump', isLogNearBottom(box));
  });
  $('#log-jump')?.addEventListener('click', () => {
    const box = $('#log');
    if (box) box.scrollTop = box.scrollHeight;
    setHidden('#log-jump', true);
  });

  $('#modal')?.addEventListener('click', (event) => {
    if (event.target === event.currentTarget) closeModal();
  });
  $('#modal-close')?.addEventListener('click', closeModal);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (!$('#modal')?.classList.contains('hidden')) closeModal();
    }
    if (event.key === 'Tab' && !$('#modal')?.classList.contains('hidden')) {
      const focusable = focusableInModal();
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });

  window.addEventListener('resize', () => {
    if (!state.layout) return;
    positionWidgets();
    drawMetricCards();
    drawChart('dash');
  });

  document.addEventListener('click', (event) => {
    if (!event.target.closest('.sample-run-picker')) $('#sample-run-menu')?.classList.add('hidden');
    const copyButton = event.target.closest('[data-copy]');
    if (copyButton) copyText(copyButton.dataset.copy, t('common.msg86'));
  });

  window.addEventListener('error', (event) => {
    if (String(event.message || '').includes('ResizeObserver loop')) {
      event.preventDefault();
      return;
    }
    if (event.message) {
      showAlert({ type: 'error', title: t('common.msg87'), text: event.message, persistent: true, key: 'window-error' });
    }
  });
  window.addEventListener('unhandledrejection', (event) => {
    const message = event.reason?.message || String(event.reason || t('common.msg88'));
    showAlert({ type: 'error', title: t('common.msg54'), text: message, persistent: true, key: 'unhandled-' + message.slice(0, 40) });
  });
}
