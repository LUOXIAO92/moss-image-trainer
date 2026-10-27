/* =========================================================================
 * js/config/model.js — Config <-> form
 * split from web/static/app.js, source 872-1183 (14 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { $ } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';

export const LAYER_COUNT = 42;

export const FIELD_DEFS = [
  { id: 'cfg-model-dit', path: 'model.dit', type: 'text', label: 'train.model.dit', required: true, default: '' },
  { id: 'cfg-model-te', path: 'model.text_encoder', type: 'text', label: 'train.model.textEncoder', default: '' },
  { id: 'cfg-model-te-config', path: 'model.text_encoder_config', type: 'text', label: 'train.model.teConfig', default: '' },
  { id: 'cfg-model-vae', path: 'model.vae', type: 'text', label: 'train.model.vae', required: true, default: '' },
  { id: 'cfg-dtype', path: 'model.dtype', type: 'text', label: 'train.model.dtype', default: 'bfloat16' },
  { id: 'cfg-dataset-path', path: 'dataset.path', type: 'text', label: 'dataset.path', required: true, default: '' },
  { id: 'cfg-max-area', path: 'dataset.max_area', type: 'number', label: 'dataset.maxArea', min: 1, integer: true, default: 1048576 },
  { id: 'cfg-cache-dir', path: 'cache.dir', type: 'text', label: 'dataset.cacheDir', default: '' },
  { id: 'cfg-cache-reg-dir', path: 'cache.reg_dir', type: 'text', label: 'dataset.reg.cacheDir', default: '' },
  { id: 'cfg-cache-text-batch', path: 'cache.text_batch', type: 'number', label: 'cache.textBatch', min: 0, integer: true, default: 0 },
  { id: 'cfg-cache-vae-batch', path: 'cache.vae_batch', type: 'number', label: 'cache.vaeBatch', min: 0, integer: true, default: 0 },
  { id: 'cfg-reg-path', path: 'dataset.reg_path', type: 'text', label: 'dataset.reg.path', default: '' },
  { id: 'cfg-reg-caption', path: 'dataset.reg_caption', type: 'text', label: 'dataset.reg.caption', default: '' },
  { id: 'cfg-caption-source', path: 'dataset.moss_caption_source', type: 'text', label: 'cache.captionSource', default: 'tag_caption' },
  { id: 'cfg-steps', path: 'train.steps', type: 'number', label: 'train.params.steps', min: 1, integer: true, default: 1000 },
  { id: 'cfg-lr', path: 'train.lr', type: 'float', label: 'train.params.lr', min: 0.0000001, default: 0.0001, required: true },
  { id: 'cfg-lr-scheduler', path: 'train.lr_scheduler', type: 'text', label: 'train.params.scheduler', default: 'constant' },
  { id: 'cfg-warmup', path: 'train.warmup', type: 'float', label: 'warmup', min: 0, default: 50, required: true },
  { id: 'cfg-batch', path: 'train.batch_size', type: 'number', label: 'train.params.batch', min: 1, integer: true, default: 1 },
  { id: 'cfg-grad-accum', path: 'train.grad_accum', type: 'number', label: 'train.params.accum', min: 1, integer: true, default: 1 },
  { id: 'cfg-rank', path: 'lora.rank', type: 'number', label: 'train.params.rank', min: 1, integer: true, default: 32 },
  { id: 'cfg-alpha', path: 'lora.alpha', type: 'number', label: 'train.params.alpha', min: 1, integer: true, default: 32 },
  { id: 'cfg-lora-init', path: 'lora.init_from', type: 'text', label: 'train.params.initLora', default: '' },
  { id: 'cfg-seed-dit', path: 'train.seed', type: 'number', label: 'train.params.seed', min: 0, integer: true, default: 42, required: true },
  { id: 'cfg-save-every', path: 'train.save_every', type: 'number', label: 'train.intervals.save', min: 0, integer: true, default: 250 },
  { id: 'cfg-sample-every', path: 'train.sample_every', type: 'number', label: 'train.intervals.sample', min: 0, integer: true, default: 250 },
  { id: 'cfg-eval-every', path: 'train.eval_every', type: 'number', label: 'train.intervals.eval', min: 0, integer: true, default: 100 },
  { id: 'cfg-optimizer', path: 'train.optimizer', type: 'text', label: 'train.params.optimizer', default: 'adamw' },
  { id: 'cfg-attention', path: 'train.attention', type: 'text', label: 'Attention backend', default: 'auto' },
  { id: 'cfg-bin-tokens', path: 'train.bin_tokens', type: 'number', label: 'train.advanced.binSize', min: 0, integer: true, default: 0 },
  { id: 'cfg-resident-blocks', path: 'vram.blocks_to_swap', type: 'resident', label: 'train.vram.residentBlocks', min: 0, max: LAYER_COUNT, integer: true, default: 0 },
  { id: 'cfg-grad-ckpt', path: 'vram.gradient_checkpointing', type: 'bool', label: 'train.vram.checkpointing', default: true },
  { id: 'cfg-offload-activations', path: 'vram.offload_activations', type: 'bool', label: 'train.vram.offloadActivations', default: false },
  { id: 'cfg-compile', path: 'vram.compile', type: 'bool', label: 'train.vram.compile', default: true },
  { id: 'cfg-optimizer-8bit', path: 'vram.optimizer_in_8bit', type: 'bool', label: 'train.vram.optimizer8bit', default: false },
  { id: 'cfg-quantize-dit', path: 'vram.quantize_dit', type: 'text', label: 'train.vram.quantize', default: 'none' },
  { id: 'cfg-quantize-dtype', path: 'vram.quantize_dtype', type: 'text', label: 'train.vram.quantizeDtype', default: 'nf4' },
  { id: 'cfg-quantize-conditioning', path: 'vram.quantize_conditioning', type: 'text', label: 'train.vram.quantizeConditioning', default: 'none' },
  { id: 'cfg-sigma-mode', path: 'train.sigma.mode', type: 'text', label: 'train.defaults.sigmaMode', default: 'logit_normal' },
  { id: 'cfg-sigma-mu', path: 'train.sigma.mu', type: 'float', label: 'train.defaults.sigmaMu', default: -1.09 },
  { id: 'cfg-sigma-sigma', path: 'train.sigma.sigma', type: 'float', label: 'train.defaults.sigmaSigma', default: 0.95 },
  { id: 'cfg-sigma-cutoff', path: 'train.sigma.cutoff', type: 'float', label: 'train.defaults.sigmaCutoff', default: 0.95 },
  { id: 'cfg-sigma-uniform-mix', path: 'train.sigma.uniform_mix', type: 'float', label: 'train.defaults.sigmaUniformMix', default: 0.01 },
  { id: 'cfg-weight-decay', path: 'train.weight_decay', type: 'float', label: 'train.defaults.weightDecay', min: 0, default: 0.01 },
  { id: 'cfg-max-grad-norm', path: 'train.max_grad_norm', type: 'float', label: 'train.defaults.maxGradNorm', min: 0, default: 1.0 },
  { id: 'cfg-caption-dropout', path: 'train.caption_dropout', type: 'float', label: 'train.defaults.captionDropout', min: 0, max: 1, default: 0.0 },
  { id: 'cfg-ema', path: 'train.ema', type: 'float', label: 'train.defaults.ema', min: 0, default: 0 },
  { id: 'cfg-eval-batches', path: 'train.eval_batches', type: 'number', label: 'train.defaults.evalBatches', min: 0, integer: true, default: 8 },
  { id: 'cfg-eval-holdout', path: 'train.eval_holdout', type: 'float', label: 'train.defaults.evalHoldout', min: 0, default: 0.05 },
  { id: 'cfg-log-every', path: 'train.log_every', type: 'number', label: 'train.defaults.logEvery', min: 1, integer: true, default: 1 },
  { id: 'cfg-epochs', path: 'train.epochs', type: 'number', label: 'train.defaults.epochs', min: 0, integer: true, default: 0 },
  { id: 'cfg-preview-steps', path: 'preview.steps', type: 'number', label: 'train.preview.steps', min: 1, integer: true, default: 50 },
  { id: 'cfg-preview-guidance', path: 'preview.guidance', type: 'float', label: 'train.msg5', min: 0, default: 3.0 },
  { id: 'cfg-preview-shift', path: 'preview.shift', type: 'float', label: 'Shift', min: 0, default: 5.0 },
  { id: 'cfg-preview-seed', path: 'preview.seed', type: 'number', label: 'train.preview.seed', min: -1, integer: true, default: 42 },
  { id: 'cfg-preview-width', path: 'preview.width', type: 'number', label: 'train.preview.width', min: 64, integer: true, default: 1024, multipleOf: 16 },
  { id: 'cfg-preview-height', path: 'preview.height', type: 'number', label: 'train.preview.height', min: 64, integer: true, default: 1024, multipleOf: 16 },
  { id: 'cfg-preview-uncond', path: 'preview.uncond_skip_layers', type: 'intlist', label: 'Uncond skip layers', min: 0, max: 41, default: [22] },
  { id: 'cfg-preview-negative', path: 'preview.negative_prompt', type: 'text', label: 'train.preview.negative', default: '' },
  { id: 'cfg-preview-lora', path: 'preview.lora', type: 'text', label: 'train.preview.lora', default: '' },
  { id: 'cfg-preview-lora-scale', path: 'preview.lora_scale', type: 'float', label: 'train.preview.loraScale', min: 0, default: 1.0 },
  { id: 'cfg-preview-batch', path: 'preview.batch', type: 'number', label: 'train.preview.batch', min: 0, integer: true, default: 1 },
  { id: 'cfg-preview-vae-device', path: 'preview.vae_device', type: 'text', label: 'train.preview.vaeDevice', default: 'gpu' },
  { id: 'cfg-preview-vae-dtype', path: 'preview.vae_dtype', type: 'text', label: 'train.preview.vaeDtype', default: 'fp32' },
  { id: 'cfg-preview-resident', path: 'preview.blocks_to_swap', type: 'previewResident', label: 'train.preview.resident', min: -1, max: LAYER_COUNT, integer: true, default: -1 },
];

export const FIELD_BY_ID = new Map(FIELD_DEFS.map((field) => [field.id, field]));

const FIELD_BY_PATH = new Map(FIELD_DEFS.map((field) => [field.path, field]));

// The UI asks how many blocks stay on the card, while the config stores how
// many stream from host memory. Blank/null means "let the preset decide".
function blockCount(value) {
  if (value === '' || value == null) return null;
  const count = Number(value);
  if (!Number.isFinite(count)) return null;
  return Math.min(LAYER_COUNT, Math.max(0, Math.round(count)));
}

export function residentToSwap(resident) {
  const count = blockCount(resident);
  return count == null ? null : LAYER_COUNT - count;
}

export function swapToResident(swap) {
  if (swap === '' || swap == null) return null;
  const count = Number(swap);
  if (!Number.isFinite(count) || count < 0) return null;
  return LAYER_COUNT - Math.min(LAYER_COUNT, Math.round(count));
}

export function arrayFromPrompts(value) {
  return Array.isArray(value) ? value.map((x) => String(x)) : [];
}

export function parseIntList(text) {
  const parts = String(text || '').replace(/[\[\]]/g, '').split(/[,，\s]+/).filter(Boolean);
  const values = [];
  for (const part of parts) {
    if (!/^-?\d+$/.test(part)) return null;
    values.push(parseInt(part, 10));
  }
  return values.length ? values : null;
}

export const LORA_MODULES = ['q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down'];

export function compactLayerRange(nums) {
  if (!nums.length) return '';
  const sorted = [...new Set(nums)].sort((a, b) => a - b);
  const chunks = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (const n of sorted.slice(1)) {
    if (n === prev + 1) { prev = n; continue; }
    chunks.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = n; prev = n;
  }
  chunks.push(start === prev ? `${start}` : `${start}-${prev}`);
  return chunks.join(', ');
}

export function parseLayerRange(text) {
  const out = [];
  for (const rawPart of String(text || '').split(',')) {
    const part = rawPart.trim();
    if (!part) continue;
    let match = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (match) {
      let a = parseInt(match[1], 10);
      let b = parseInt(match[2], 10);
      if (a > b) [a, b] = [b, a];
      if (a < 0 || b > LAYER_COUNT - 1) throw new Error(t('train.layer.outOfRange', { max: LAYER_COUNT - 1, part }));
      for (let i = a; i <= b; i += 1) out.push(i);
      continue;
    }
    match = part.match(/^(\d+)$/);
    if (match) {
      const n = parseInt(match[1], 10);
      if (n < 0 || n > LAYER_COUNT - 1) throw new Error(t('train.layer.outOfRange', { max: LAYER_COUNT - 1, part }));
      out.push(n);
      continue;
    }
    throw new Error(t('train.layer.parseFailed', { part }));
  }
  if (!out.length) throw new Error(t('train.msg6'));
  return [...new Set(out)].sort((a, b) => a - b);
}

export function defaultCacheDir() {
  return state.workspaceAbsPath ? `${state.workspaceAbsPath}/train_cache` : '';
}

function cacheIsDefault(value) {
  const raw = String(value || '').trim().replace(/\/+$/, '');
  if (!raw) return true;
  if (raw === 'train_cache') return true;
  if (state.workspacePath && raw === `${state.workspacePath}/train_cache`) return true;
  if (state.profileName && raw === `workspaces/${state.profileName}/train_cache`) return true;
  if (defaultCacheDir() && raw === defaultCacheDir()) return true;
  return false;
}

export function displayCacheValue(value) {
  const raw = String(value || '').trim();
  if (!raw || cacheIsDefault(raw)) return '';
  if (raw.startsWith('/')) return raw;
  return state.workspaceAbsPath ? `${state.workspaceAbsPath}/${raw.replace(/^\.\//, '')}` : '';
}

export function resolvedCacheBase() {
  let value = String($('#cfg-cache-dir')?.value || '').trim() || defaultCacheDir();
  if (value && !value.startsWith('/') && state.workspaceAbsPath) {
    value = `${state.workspaceAbsPath}/${value.replace(/^\.\//, '')}`;
  }
  return value.replace(/\/+$/, '');
}

export function absoluteWorkspacePath(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.startsWith('/')) return raw.replace(/\/+$/, '');
  if (raw.startsWith('workspaces/') && state.workspaceAbsPath && state.workspacePath) {
    const workspaceRel = state.workspacePath.replace(/\/+$/, '');
    if (state.workspaceAbsPath.endsWith(workspaceRel)) {
      return `${state.workspaceAbsPath.slice(0, -workspaceRel.length)}${raw}`.replace(/\/+$/, '');
    }
  }
  return state.workspaceAbsPath
    ? `${state.workspaceAbsPath}/${raw.replace(/^\.\//, '')}`.replace(/\/+$/, '')
    : raw.replace(/\/+$/, '');
}

export function defaultRegCacheDir() {
  return state.workspaceAbsPath ? `${state.workspaceAbsPath}/reg_cache` : '';
}

function regCacheIsDefault(value) {
  const raw = String(value || '').trim().replace(/\/+$/, '');
  if (!raw) return true;
  if (defaultRegCacheDir() && raw === defaultRegCacheDir()) return true;
  return false;
}

export function displayRegCacheValue(value) {
  const raw = String(value || '').trim();
  if (!raw || regCacheIsDefault(raw)) return '';
  return absoluteWorkspacePath(raw);
}
