/* =========================================================================
 * js/core/state.js — header
 * split from web/static/app.js, source 15-8602 (4 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

export const TABS = ['dashboard', 'train', 'dataset', 'cache', 'samples', 'export', 'workspace', 'settings', 'caption'];

export const TAB_LABEL_KEYS = {
  dashboard: 'nav.dashboard', train: 'nav.train', dataset: 'nav.dataset', cache: 'nav.cache',
  samples: 'nav.samples', export: 'nav.export', workspace: 'nav.workspace', settings: 'nav.settings',
  caption: 'nav.caption',
};

export const MAX_LOG = 2000;

export const JOB_LABEL_KEYS = { train: 'job.train', cache: 'job.cache', caption: 'job.caption', export: 'job.export' };

const CAPTION_JOB_KINDS = new Set(['caption']);

export const CAPTION_SOURCE_KEYS = {
  tag_caption: 'caption.source.tag',
  official_long: 'caption.source.long',
  official_medium: 'caption.source.medium',
  official_short: 'caption.source.short',
};

export const CAPTION_SOURCES = Object.keys(CAPTION_SOURCE_KEYS);

export function isCaptionJob(kind) { return CAPTION_JOB_KINDS.has(kind); }

export const state = {
  i18n: { current: 'zh', entries: {}, languages: [], importFileName: '' },
  i18nLoading: false,
  system: null,
  presets: null,
  profiles: [],
  profileName: '',
  configPath: '',
  workspacePath: '',
  workspacesRoot: '',
  runs: [],
  activeRun: '',
  workspaceAbsPath: '',
  pathBrowser: { fieldId: '', mode: 'dir', dir: '', selected: '' },
  workspacePage: { list: [], usages: {}, usage: null, loading: false, confirm: null },
  configDialog: { mode: 'new' },
  exportRuns: [],
  exportExpected: [],
  exportRunName: '',
  exportStepValue: '',
  exportFormats: [],
  exportSelected: new Set(),
  exportCheckpoints: [],
  exportRunning: false,
  exportDestination: 'server',
  exportTargetDir: '',
  exportDirResolve: null,
  exportDirPath: '',
  exportDirTimer: null,
  exportDirRequestSeq: 0,
  settings: null,
  settingsResolved: null,
  settingsStatus: null,
  settingsDirty: false,
  captionSettingsDirty: false,
  captionPrompts: { tags: '', natural: '' },
  captionService: { base_url: '', model: '', api_key: '', has_key: false, key_source: 'none', key_saved: false },
  captionClearKey: false,
  captionRun: {
    initialized: false,
    loading: false,
    seq: 0,
    paths: { train: '', reg: '' },
    datasets: {
      train: { available: false, path: '', images: null, error: '' },
      reg: { available: false, path: '', images: null, error: '' },
    },
  },
  captionProgress: {
    jobId: '',
    kind: '',
    status: 'idle',
    phases: [],
    current: null,
    aggregate: { total: 0, ok: 0, skip: 0, fail: 0, failures: [], current: '', rate: '' },
    role: '',
    workers: null,
    datasets: [],
    styles: [],
  },
  settingsPath: 'settings.yaml',
  firstRun: false,
  onboardingShown: false,
  setup: { loading: false, data: null },
  profileDialog: { mode: 'new' },
  config: {},
  mode: 'form',                 // form | yaml
  dirty: false,
  savedAt: null,
  savedYaml: '',
  yamlError: null,
  yamlParseTimer: null,
  job: null,
  jobs: [],
  es: null,
  streamedJobId: null,
  eventCount: 0,
  logs: [],
  logPaused: false,
  sampleTimer: null,
  busy: false,
  currentRun: null,             // { name, outputDir } — runtime truth, not editor truth
  trainTotal: 0,
  series: { train: new Map(), avg: new Map(), eval: new Map(), meta: new Map() },
  liveTrain: false,
  lastEvalSync: 0,
  dashboard: { data: null, run: null, loading: false, lastError: '' },
  dataset: { data: null, path: '', error: null, lastPath: '', reg: null, regPath: '', regError: null, loading: false, loadingPath: '', autoPath: '', scanSeq: 0, regLoading: false, regAutoPath: '', regScanSeq: 0 },
  cache: {
    data: null, progress: null, stages: { latent: null, text: null }, active: '', parts: [], error: null,
    // progress published to the workspace, whoever started the build
    job: null, jobRunning: false, diskKey: '', diskStages: { latent: null, text: null },
  },
  samples: { items: [], filter: 'all' },
  loras: [],
  charts: {},
  modal: { lastFocus: null },
  alertKeys: new Set(),
  jobInfoFor: null,
  activityAnnounce: '',
  lastSystemAt: 0,
  runs: [],
  smoothWindow: 6,
  yMin: null,
  yMax: null,
  chartMetrics: { loss: true, lr: false, sec: false, vram: false },
  gpuVisible: { mem: true, util: true },
  layout: null,
  layoutSaving: false,
  layoutLoaded: false,
  activeRunName: '',
  activeTraining: null,
  runLog: { run: '', text: '', loading: false },
  compareRuns: new Set(),
  compareData: [],
  compareSampleRuns: [],
  sampleCompareSelection: new Set(),
  sampleCompareData: [],
  sampleCompareStep: null,
  sampleCompareTrigger: null,
  sampleCompareInitialized: false,
  sampleCompareLoading: false,
  baselineSamples: [],
  baselineLoading: false,
  baselineSource: '',
  compareVisible: {},
  activeRunConfig: null,
  compareLoading: false,
  comparePending: false,
  lastSseAt: 0,
  lastProgressAt: 0,
  fallbackTimer: null,
  fallbackBusy: false,
  logFileText: '',
  lastSseLogAt: 0,
  lastProgressAt: 0,
};

export function workspaceParam() {
  return state.profileName || '';
}
