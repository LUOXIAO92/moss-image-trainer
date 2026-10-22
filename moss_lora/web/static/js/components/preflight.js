/* =========================================================================
 * js/components/preflight.js — Preflight
 * split from web/static/app.js, source 4424-4616 (3 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { clearYamlError, getCurrentYaml, showYamlError } from '../config/form.js';
import { FIELD_BY_ID } from '../config/model.js';
import { openProfileDialog } from '../config/profiles.js';
import { clearFormErrors, setFieldError, showFormErrors, validateConfigValues, validateForm } from '../config/validate.js';
import { parseYaml } from '../config/yaml.js';
import { api } from '../core/api.js';
import { $, getPath, setText } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';
import { renderCacheFromData } from '../pages/cache.js';

export function setPreflight({ level = 'warn', title = '', text = '', actions = [] }) {
  const box = $('#train-preflight');
  if (!box) return;
  box.className = `callout callout-${level}`;
  box.setAttribute('role', level === 'error' ? 'alert' : 'status');
  const iconUse = box.querySelector('.callout-icon use');
  if (iconUse) iconUse.setAttribute('href', level === 'success' ? '#i-check' : level === 'error' ? '#i-alert' : '#i-alert');
  setText('#train-preflight-title', title);
  setText('#train-preflight-text', text);
  const actionBox = $('#train-preflight-actions');
  if (actionBox) {
    actionBox.innerHTML = '';
    for (const action of actions) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `btn ${action.primary ? 'btn-primary' : 'btn-ghost'} btn-sm`;
      button.textContent = action.label;
      button.addEventListener('click', action.onClick);
      actionBox.append(button);
    }
  }
  box.classList.remove('hidden');
}

export function hidePreflight() {
  const box = $('#train-preflight');
  if (box) box.classList.add('hidden');
}

export async function runPreflight(options = {}) {
  if (!state.profileName) {
    setPreflight({
      level: 'warn', title: t('workspace.emptyTitle'),
      text: t('train.msg57'),
      actions: [{ label: t('train.profile.new'), primary: true, onClick: () => openProfileDialog('new') }],
    });
    return { ok: false, data: { next_tab: 'train', focus_field: '' } };
  }
  const yamlText = options.yamlText || getCurrentYaml();
  let parsed;
  try {
    parsed = parseYaml(yamlText);
    clearYamlError();
  } catch (error) {
    showYamlError(error);
    setPreflight({
      level: 'error',
      title: t('train.msg58'),
      text: t('train.lineError', { line: error.line || '?', error: error.message }),
      actions: [
        { label: t('train.msg59'), onClick: () => { const d = $('#yaml-details'); if (d) { d.open = true; d.scrollIntoView({ block: 'start' }); } } },
      ],
    });
    return { ok: false };
  }

  if (state.mode === 'form') {
    const localErrors = validateForm(true);
    if (localErrors.length) {
      showFormErrors(localErrors);
      const firstId = localErrors[0]?.field?.id || '';
      const nextTab = firstId === 'cfg-cache-dir' ? 'cache'
        : (firstId === 'cfg-dataset-path' || firstId.startsWith('cfg-reg-') || firstId === 'cfg-cache-reg-dir') ? 'dataset'
          : 'train';
      setPreflight({
        level: 'error',
        title: t('train.msg60'),
        text: localErrors.map((item) => `${t(item.field.label)}：${item.message}`).join(t('common.sentenceSeparator')),
        actions: [{ label: t('train.msg61'), onClick: () => showFormErrors(localErrors) }],
      });
      return { ok: false, data: { next_tab: nextTab, focus_field: firstId.startsWith('cfg-model-') ? firstId : '' } };
    }
    clearFormErrors();
  } else {
    clearFormErrors();
    const valueErrors = validateConfigValues(parsed);
    if (valueErrors.length) {
      setPreflight({
        level: 'error',
        title: t('train.msg62'),
        text: valueErrors.join('；'),
        actions: [{ label: t('train.msg59'), onClick: () => { const d = $('#yaml-details'); if (d) { d.open = true; d.scrollIntoView({ block: 'start' }); } } }],
      });
      return { ok: false, data: { next_tab: 'train', focus_field: '' } };
    }
  }

  let data;
  try {
    data = await api('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml: yamlText, workspace: state.profileName }),
    });
  } catch (error) {
    // Old server fallback: local config is already parseable, check cache only.
    const cachePath = getPath(parsed, 'cache.dir') || '';
    let cache = null;
    try {
      cache = cachePath ? await api('/api/cache?path=' + encodeURIComponent(cachePath) + '&workspace=' + encodeURIComponent(state.profileName)) : null;
    } catch (_) { cache = null; }
    const cacheOk = Boolean(cache?.ready);
    const legacy = Boolean(cache?.legacy_index);
    setPreflight({
      level: cacheOk ? 'warn' : 'error',
      title: cacheOk ? t('train.msg63') : legacy ? t('train.msg64') : t('train.msg65'),
      text: cacheOk ? t('train.msg66')
        : legacy ? t('train.msg67')
          : t('train.msg68'),
      actions: cacheOk ? [] : [{ label: t('train.msg69'), onClick: () => { location.hash = '#cache'; } }],
    });
    return { ok: cacheOk, data: cacheOk ? null : { next_tab: 'cache', focus_field: '' } };
  }

  if (data.yaml_error) {
    showYamlError({
      message: data.yaml_error.message,
      line: data.yaml_error.line,
      column: data.yaml_error.column,
    });
    setPreflight({
      level: 'error',
      title: t('train.msg58'),
      text: t('train.lineError', { line: data.yaml_error.line || '?', error: data.yaml_error.message }),
      actions: [
        { label: t('train.msg59'), onClick: () => { const d = $('#yaml-details'); if (d) { d.open = true; d.scrollIntoView({ block: 'start' }); } } },
      ],
    });
    return { ok: false, data };
  }

  const cache = data.cache || {};
  const missing = [];
  if (!cache.ready && !cache.legacy_index) missing.push(t('train.msg70'));
  const problems = [...(data.problems || [])];
  for (const problem of problems) {
    if (problem.startsWith('dataset.reg_path')) setFieldError(FIELD_BY_ID.get('cfg-reg-path'), problem);
  }
  if (cache.dir && cache.legacy_index) {
    problems.push(t('train.cacheLegacyProblem', { dir: cache.dir }));
  } else if (cache.dir && missing.length) {
    problems.push(t('train.cacheMissingProblem', { dir: cache.dir, missing: missing.join(t('common.listSeparator')) }));
  }
  if (data.run) {
    state.runPreview = data.run;
    state.currentRun = state.currentRun || {
      name: data.run.name,
      label: data.run.workspace ? `${data.run.workspace}/${data.run.name}` : data.run.name,
      profile: data.run.workspace || data.run.profile || state.profileName,
      workspace: data.run.workspace || state.profileName,
      outputDir: data.run.output_dir || '',
    };
    if (!state.series.train.size) {
      setText('#train-run-name', data.run.name || '--');
      setText('#train-run-dir', data.run.output_dir || '--');
    }
  }
  if (cache.dir) {
    state.cache.data = { ...cache };
    renderCacheFromData(state.cache.data);
  }

  const actions = [];
  if (missing.length) actions.push({ label: t('job.msg46'), primary: true, onClick: () => { location.hash = '#cache'; } });
  if (problems.some((problem) => /model\.|not found/.test(problem))) {
    actions.push({ label: t('train.msg71'), onClick: () => { location.hash = '#settings'; } });
  }
  actions.push({ label: t('train.msg72'), onClick: () => runPreflight() });

  if (problems.length) {
    setPreflight({
      level: 'error',
      title: t('train.preflight.title'),
      text: problems.join('；'),
      actions,
    });
    return { ok: false, data };
  }

  const warnings = (data.warnings || []).map((warning) => warning.replace(/。$/, ''));
  let text = t('train.preflight.writeTo', { path: data.run?.output_dir || t('common.outputDir') });
  if (data.run?.will_rename) text += t('train.msg73');
  if (warnings.length) text += t('train.preflight.withWarnings', { warnings: warnings.join(t('common.sentenceSeparator')) });
  if (state.dirty) text += t('train.msg74');
  if (cache.items) text += t('train.preflight.cacheReady', { count: cache.items });
  setPreflight({
    level: 'success',
    title: t('train.msg75'),
    text,
    actions: warnings.length ? [{ label: t('train.msg76'), onClick: hidePreflight }] : [],
  });
  return { ok: true, data, yamlText };
}
