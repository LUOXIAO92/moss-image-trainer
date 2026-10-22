#!/usr/bin/env node
/**
 * WebUI 冒烟 / 基线快照（零依赖，Node ESM，直连 Chromium DevTools Protocol）。
 *
 * 用法：
 *   node tests/ui/smoke.mjs --base http://127.0.0.1:7891 --cdp http://127.0.0.1:9401 \
 *     [--langs zh,en,ja,ko] [--out .tmp/ui-baseline.json] [--shots .tmp/ui-baseline-shots]
 *
 * 流程：固定 1440×1000 → 逐个语言 POST /api/i18n/current 并整页 reload
 *      → 逐个 tab 改 location.hash → 等页面稳定 → 截图 + 采集 DOM 度量与运行时事件。
 *
 * 只做观测，不修改产品代码；结束时把服务端语言恢复成 zh。
 */

import fs from 'node:fs';
import path from 'node:path';

const TABS = ['dashboard', 'train', 'dataset', 'cache', 'samples', 'export', 'workspace', 'settings', 'caption'];
// dashboard / cache 首屏有图表、缓存扫描等异步渲染，多给一点沉降时间。
const SLOW_TABS = new Set(['dashboard', 'cache']);
const VIEWPORT = { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false };

const HELP = `用法: node tests/ui/smoke.mjs [选项]

  --base <url>     WebUI 地址（默认 http://127.0.0.1:7891）
  --cdp <url>      Chromium CDP 地址（默认 http://127.0.0.1:9401）
  --langs <list>   逗号分隔的语言代码（默认 zh,en,ja,ko）
  --out <file>     JSON 输出路径；不传则写 stdout
  --shots <dir>    每个 tab 的整页截图画布输出目录；不传则不截图
  -h, --help       显示本帮助
`;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function parseArgs(argv) {
  const options = {
    base: 'http://127.0.0.1:7891',
    cdp: 'http://127.0.0.1:9401',
    langs: ['zh', 'en', 'ja', 'ko'],
    out: null,
    shots: null,
    help: false,
  };
  const takesValue = new Set(['--base', '--cdp', '--langs', '--out', '--shots']);
  for (let index = 0; index < argv.length; index += 1) {
    const raw = argv[index];
    const equal = raw.indexOf('=');
    const flag = equal >= 0 ? raw.slice(0, equal) : raw;
    let value = equal >= 0 ? raw.slice(equal + 1) : null;
    if (flag === '--help' || flag === '-h') {
      options.help = true;
      continue;
    }
    if (!takesValue.has(flag)) throw new Error(`未知参数: ${raw}`);
    if (value === null) {
      value = argv[index + 1];
      index += 1;
    }
    if (value === undefined || value === '' || value.startsWith('--')) {
      throw new Error(`参数 ${flag} 缺少值`);
    }
    if (flag === '--base') options.base = value.replace(/\/+$/, '');
    else if (flag === '--cdp') options.cdp = value.replace(/\/+$/, '');
    else if (flag === '--langs') {
      const langs = value.split(',').map((item) => item.trim().toLowerCase()).filter(Boolean);
      if (!langs.length) throw new Error('--langs 不能为空');
      options.langs = [...new Set(langs)];
    } else if (flag === '--out') options.out = path.resolve(process.cwd(), value);
    else if (flag === '--shots') options.shots = path.resolve(process.cwd(), value);
  }
  return options;
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!response.ok) {
    const detail = typeof body === 'string' ? body : JSON.stringify(body);
    throw new Error(`${init?.method || 'GET'} ${url} -> HTTP ${response.status}: ${detail.slice(0, 400)}`);
  }
  return body;
}

async function setLanguage(base, code) {
  await fetchJson(`${base}/api/i18n/current`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
}

/** 极简 CDP 客户端：一个页面 target 一条 WebSocket，事件与命令用 id 配对。 */
class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 0;
    this.pending = new Map();
    this.listeners = new Set();
    this.closed = false;
    this.requestMeta = new Map();
    this.buffer = [];
    this.capture = false;
    ws.addEventListener('message', (event) => this.onMessage(event));
    ws.addEventListener('close', () => {
      this.closed = true;
      for (const entry of this.pending.values()) {
        clearTimeout(entry.timer);
        entry.reject(new Error('CDP 连接已关闭'));
      }
      this.pending.clear();
    });
  }

  static async connect(cdpBase, baseUrl) {
    let targets = await fetchJson(`${cdpBase}/json/list`);
    let target = targets.find((item) => item.type === 'page' && String(item.url).startsWith(baseUrl))
      || targets.find((item) => item.type === 'page');
    if (!target) {
      target = await fetchJson(`${cdpBase}/json/new?${encodeURIComponent('about:blank')}`, { method: 'PUT' });
    }
    if (!target?.webSocketDebuggerUrl) throw new Error(`CDP 没有可用的 page target: ${cdpBase}`);
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', () => reject(new Error(`无法连接 CDP: ${target.webSocketDebuggerUrl}`)), { once: true });
    });
    return new Cdp(ws);
  }

  onMessage(event) {
    let message;
    try {
      message = JSON.parse(typeof event.data === 'string' ? event.data : String(event.data));
    } catch {
      return;
    }
    if (message.id) {
      const entry = this.pending.get(message.id);
      if (!entry) return;
      this.pending.delete(message.id);
      clearTimeout(entry.timer);
      if (message.error) entry.reject(new Error(`${entry.method}: ${JSON.stringify(message.error)}`));
      else entry.resolve(message.result);
      return;
    }
    if (message.method) {
      this.record(message);
      for (const listener of this.listeners) listener(message);
    }
  }

  send(method, params = {}, timeout = 30000) {
    if (this.closed) return Promise.reject(new Error(`CDP 已关闭，不能发送 ${method}`));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP 超时: ${method}`));
      }, timeout);
      this.pending.set(id, {
        method,
        timer,
        resolve: (value) => { clearTimeout(timer); resolve(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  waitEvent(method, timeout = 30000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.listeners.delete(listener);
        reject(new Error(`等待 CDP 事件超时: ${method}`));
      }, timeout);
      const listener = (message) => {
        if (message.method !== method) return;
        clearTimeout(timer);
        this.listeners.delete(listener);
        resolve(message.params);
      };
      this.listeners.add(listener);
    });
  }

  async evaluate(expression, { awaitPromise = true, timeout = 20000 } = {}) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise,
      returnByValue: true,
      userGesture: false,
    }, timeout);
    if (result.exceptionDetails) {
      const detail = result.exceptionDetails.exception?.description || result.exceptionDetails.text;
      throw new Error(`页面脚本异常: ${detail}`);
    }
    return result.result?.value;
  }

  /** 采集一条事件；capture 关闭时只维护网络请求表，不污染下一组样本。 */
  record(message) {
    const method = message.method;
    const params = message.params || {};
    if (method === 'Network.requestWillBeSent') {
      this.requestMeta.set(params.requestId, {
        url: params.request?.url || '',
        method: params.request?.method || 'GET',
        type: params.type || '',
      });
      return;
    }
    if (method === 'Network.loadingFinished') {
      this.requestMeta.delete(params.requestId);
      return;
    }
    if (!this.capture) return;

    if (method === 'Runtime.consoleAPICalled') {
      if (!['error', 'warning', 'assert'].includes(params.type)) return;
      const text = (params.args || []).map(formatRemoteObject).join(' ');
      const frame = topCallFrame(params.stackTrace);
      this.push({
        kind: 'console',
        level: params.type,
        text: truncate(text, 2000),
        url: frame.url,
        line: frame.line,
        column: frame.column,
      });
      return;
    }
    if (method === 'Runtime.exceptionThrown') {
      const details = params.exceptionDetails || {};
      this.push({
        kind: 'exception',
        level: 'error',
        text: truncate(details.exception?.description || details.text || '未知异常', 2000),
        url: details.url || topCallFrame(details.stackTrace).url,
        line: details.lineNumber != null ? details.lineNumber + 1 : topCallFrame(details.stackTrace).line,
        column: details.columnNumber != null ? details.columnNumber + 1 : topCallFrame(details.stackTrace).column,
      });
      return;
    }
    if (method === 'Log.entryAdded') {
      const entry = params.entry || {};
      if (!['error', 'warning'].includes(entry.level)) return;
      this.push({
        kind: 'log',
        level: entry.level,
        source: entry.source || '',
        text: truncate(entry.text || '', 2000),
        url: entry.url || '',
        line: entry.lineNumber != null ? entry.lineNumber + 1 : undefined,
      });
      return;
    }
    if (method === 'Network.loadingFailed') {
      const meta = this.requestMeta.get(params.requestId) || {};
      this.requestMeta.delete(params.requestId);
      const url = meta.url || '';
      // favicon 缺失是本地 WebUI 的常态，不纳入基线噪音。
      if (/favicon\.ico/i.test(url)) return;
      this.push({
        kind: 'request-failed',
        level: 'error',
        url,
        method: meta.method || '',
        resourceType: meta.type || params.type || '',
        errorText: params.errorText || '',
        canceled: Boolean(params.canceled),
        blockedReason: params.blockedReason || '',
      });
    }
  }

  push(event) {
    this.buffer.push(event);
  }

  takeEvents() {
    const events = this.buffer;
    this.buffer = [];
    return events;
  }
}

function formatRemoteObject(arg) {
  if (!arg || typeof arg !== 'object') return String(arg);
  if ('value' in arg) {
    const value = arg.value;
    if (typeof value === 'string') return value;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return arg.description || arg.unserializableValue || arg.type || '';
}

function topCallFrame(stackTrace) {
  const frame = stackTrace?.callFrames?.[0];
  if (!frame) return { url: '', line: undefined, column: undefined };
  return {
    url: frame.url || '',
    line: frame.lineNumber != null ? frame.lineNumber + 1 : undefined,
    column: frame.columnNumber != null ? frame.columnNumber + 1 : undefined,
  };
}

function truncate(text, limit) {
  const value = String(text ?? '');
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

/**
 * 在页面里执行的度量函数。保持自包含：会被序列化后通过 Runtime.evaluate 注入。
 * “可见”= CSS 可见（checkVisibility），不要求落在当前视口内。
 */
function browserMetrics(tab) {
  const pane = document.getElementById(`tab-${tab}`);
  if (!pane) throw new Error(`找不到面板 #tab-${tab}`);

  const visible = (element) => {
    if (!element || !element.isConnected) return false;
    if (typeof element.checkVisibility === 'function') {
      return element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });
    }
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  };

  const snippet = (text) => String(text || '').replace(/\s+/g, ' ').trim().slice(0, 160);

  // 生成一个够用、稳定的短选择器：优先 id，其次 tag + class，最后 nth-of-type。
  const selectorOf = (element) => {
    const parts = [];
    let node = element;
    for (let depth = 0; node && node.nodeType === 1 && depth < 3; depth += 1) {
      if (node.id) {
        parts.unshift(`#${node.id}`);
        break;
      }
      let part = node.tagName.toLowerCase();
      const classes = [...node.classList].slice(0, 2);
      if (classes.length) part += `.${classes.join('.')}`;
      else if (node.parentElement) {
        const siblings = [...node.parentElement.children].filter((item) => item.tagName === node.tagName);
        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
      }
      parts.unshift(part);
      node = node.parentElement;
    }
    return parts.join(' > ');
  };

  const inPane = (query) => [...pane.querySelectorAll(query)].filter(visible);

  const h1 = inPane('h1').map((element) => snippet(element.textContent)).find(Boolean) || '';
  const buttons = new Set(inPane('button, input[type="button"], input[type="submit"], input[type="reset"], [role="button"]'));
  const fields = inPane('input, select, textarea');
  // 卡片是这套 UI 里的面板/指标/样图等容器；用常见类名汇总，避免依赖单一 class。
  const cards = new Set(inPane('.panel, .kpi, .summary-card, .dataset-card, .sample-card, .sample-prompt-card, .thumb-card, .widget-card'));

  const html = document.documentElement;
  const docHeight = Math.max(html.scrollHeight, document.body ? document.body.scrollHeight : 0);
  const overflowHtml = Math.max(0, html.scrollWidth - html.clientWidth);
  const overflowPane = Math.max(0, pane.scrollWidth - pane.clientWidth);
  const paneId = [...document.querySelectorAll('.tabpane')].filter(visible).map((element) => element.id).join(',') || '';

  // 文本级截断：从文本节点向上找最近的 hidden/clip 容器；ellipsis、line-clamp、title 都算“有意截断”。
  const clipped = [];
  const clippedSeen = new Set();
  const walker = document.createTreeWalker(pane, NodeFilter.SHOW_TEXT);
  while (walker.nextNode() && clipped.length < 200) {
    const node = walker.currentNode;
    const text = snippet(node.nodeValue);
    if (!text) continue;
    const host = node.parentElement;
    if (!host || !visible(host)) continue;
    if (host.closest('script, style, noscript, template, svg, canvas')) continue;

    let clippedBy = null;
    let axis = '';
    for (let element = host, depth = 0; element && element !== pane && depth < 5; element = element.parentElement, depth += 1) {
      const style = getComputedStyle(element);
      const clipX = style.overflowX === 'hidden' || style.overflowX === 'clip';
      const clipY = style.overflowY === 'hidden' || style.overflowY === 'clip';
      const overX = element.scrollWidth > element.clientWidth + 1;
      const overY = element.scrollHeight > element.clientHeight + 1;
      if ((clipX && overX) || (clipY && overY)) {
        clippedBy = element;
        axis = clipX && overX ? 'x' : 'y';
        break;
      }
    }
    if (!clippedBy) continue;

    const style = getComputedStyle(clippedBy);
    const clamp = style.webkitLineClamp && style.webkitLineClamp !== 'none';
    const intentional = style.textOverflow === 'ellipsis' || clamp;
    const titled = clippedBy.hasAttribute('title') || host.hasAttribute('title') || Boolean(clippedBy.closest('[title]'));
    if (intentional || titled) continue;

    const key = `${selectorOf(clippedBy)}|${text}`;
    if (clippedSeen.has(key)) continue;
    clippedSeen.add(key);
    clipped.push({ selector: selectorOf(clippedBy), text, axis });
  }

  // data-i18n 目标渲染为空，通常意味着翻译缺失或误删；title/placeholder/aria 只报真正取到空串的。
  const emptyI18n = [];
  for (const element of pane.querySelectorAll('[data-i18n], [data-i18n-title], [data-i18n-placeholder], [data-i18n-aria]')) {
    if (!visible(element)) continue;
    if (element.hasAttribute('data-i18n') && !String(element.textContent || '').trim()) {
      emptyI18n.push({ key: element.getAttribute('data-i18n'), target: 'text', selector: selectorOf(element) });
    }
    for (const [attribute, target] of [['data-i18n-title', 'title'], ['data-i18n-placeholder', 'placeholder'], ['data-i18n-aria', 'aria-label']]) {
      const key = element.getAttribute(attribute);
      if (key && element.getAttribute(target) === '') {
        emptyI18n.push({ key, target, selector: selectorOf(element) });
      }
    }
    if (emptyI18n.length >= 20) break;
  }

  // 裸 key 形如 nav.dashboard；排除代码块等有意展示 key 的位置。
  const bareKeys = [];
  const bareSeen = new Set();
  const bareWalker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (bareWalker.nextNode() && bareKeys.length < 20) {
    const node = bareWalker.currentNode;
    const text = String(node.nodeValue || '').trim();
    if (!text || text.length > 120) continue;
    if (!/^[a-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)+$/.test(text)) continue;
    const host = node.parentElement;
    if (!host || !visible(host)) continue;
    // .mono 用于展示路径 / 文件名（index.json、settings.yaml），不是漏翻的 key。
    if (host.closest('script, style, noscript, template, pre, code, .mono, svg, canvas')) continue;
    if (bareSeen.has(text)) continue;
    bareSeen.add(text);
    // 顶栏（GPU / 任务 chip）不在 tabpane 里，标出来好定位。
    const region = pane.contains(host) ? 'pane' : (host.closest('header, .activity, .topbar') ? 'chrome' : 'outside');
    bareKeys.push({ text, selector: selectorOf(host), region });
  }

  return {
    tab,
    paneId,
    lang: document.documentElement.lang || '',
    title: document.title,
    h1,
    visibleButtons: buttons.size,
    visibleFields: fields.length,
    visibleCards: cards.size,
    docHeight,
    overflow: { html: overflowHtml, pane: overflowPane },
    clippedText: clipped.slice(0, 8),
    emptyI18n: emptyI18n.slice(0, 5),
    bareKeys: bareKeys.slice(0, 5),
    totals: { clippedText: clipped.length, emptyI18n: emptyI18n.length, bareKeys: bareKeys.length },
  };
}

async function navigate(cdp, url) {
  const loaded = cdp.waitEvent('Page.loadEventFired', 45000);
  const result = await cdp.send('Page.navigate', { url });
  if (result.errorText) throw new Error(`导航失败: ${result.errorText}`);
  await loaded;
}

async function waitFor(cdp, expression, timeout, label = expression) {
  const deadline = Date.now() + timeout;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      if (await cdp.evaluate(expression)) return;
    } catch (error) {
      lastError = error; // 导航切换执行上下文期间的瞬时报错，继续轮询。
    }
    await sleep(120);
  }
  throw new Error(`等待超时: ${label}${lastError ? ` (${lastError.message})` : ''}`);
}

/** 结构签名连续稳定一段时间才采集，避免把骨架屏/加载中的中间态写进基线。 */
async function waitSettled(cdp, tab) {
  const minWait = SLOW_TABS.has(tab) ? 2500 : 700;
  const stableFor = 700;
  const hardLimit = SLOW_TABS.has(tab) ? 15000 : 9000;
  const started = Date.now();
  let previous = '';
  let changedAt = Date.now();
  while (Date.now() - started < hardLimit) {
    const signature = await cdp.evaluate(`(() => {
      const pane = document.getElementById('tab-${tab}');
      if (!pane) return '';
      const visiblePane = [...document.querySelectorAll('.tabpane')].find((item) => item.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }));
      return JSON.stringify([
        visiblePane ? visiblePane.id : '',
        document.readyState,
        document.documentElement.scrollHeight,
        pane.querySelectorAll('*').length,
        pane.querySelectorAll('button, input, select, textarea').length,
        pane.scrollWidth,
        document.documentElement.scrollWidth,
      ]);
    })()`);
    if (signature !== previous) {
      previous = signature;
      changedAt = Date.now();
    }
    if (Date.now() - started >= minWait && Date.now() - changedAt >= stableFor) return;
    await sleep(200);
  }
}

async function screenshot(cdp, filePath) {
  // 只截当前视口（1440×1000）：整页 captureBeyondViewport 在长页面（样图 2 万像素）
  // 上容易把无 GPU 的 Chromium 合成器挂住，视口截图稳定且足以做逐 tab 视觉对照。
  await cdp.send('Page.bringToFront');
  const shot = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    fromSurface: true,
    captureBeyondViewport: false,
  }, 20000);
  fs.writeFileSync(filePath, Buffer.from(shot.data, 'base64'));
}

async function restoreChinese(base) {
  await setLanguage(base, 'zh');
}

async function run(options) {
  const startedAt = new Date().toISOString();
  if (options.shots) fs.mkdirSync(options.shots, { recursive: true });

  const cdp = await Cdp.connect(options.cdp, options.base);
  const result = {
    base: options.base,
    langs: options.langs,
    viewport: { width: VIEWPORT.width, height: VIEWPORT.height },
    startedAt,
    finishedAt: null,
    tabs: {},
    events: [],
  };

  try {
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Log.enable');
    await cdp.send('Network.enable');
    await cdp.send('Emulation.setDeviceMetricsOverride', VIEWPORT);

    for (const lang of options.langs) {
      await setLanguage(options.base, lang);
      // 换语言后整页导航，确保服务端设置与页面 i18n 状态都刷新。
      await navigate(cdp, `${options.base}/?ui-smoke=${Date.now()}#dashboard`);
      await waitFor(
        cdp,
        `document.readyState === 'complete' && Boolean(window.__mossUI) && window.__mossUI.state?.i18n?.current === ${JSON.stringify(lang)}`,
        30000,
        `语言 ${lang} 加载完成`,
      );
      await cdp.evaluate('document.fonts ? document.fonts.ready.then(() => true) : true');

      for (const tab of TABS) {
        // 每一组样本从激活 tab 开始记录事件，避免把 reload / 上一组尾部算进来。
        cdp.capture = true;
        cdp.takeEvents();
        await cdp.evaluate(`location.hash = ${JSON.stringify(`#${tab}`)}`);
        await waitFor(cdp, `Boolean(document.getElementById('tab-${tab}'))`, 10000, `面板 #tab-${tab} 存在`);
        await waitFor(
          cdp,
          `!document.getElementById('tab-${tab}').classList.contains('hidden') && document.getElementById('tab-${tab}').checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })`,
          10000,
          `面板 #tab-${tab} 可见`,
        );
        await waitSettled(cdp, tab);

        // 截图从页首开始，保证同一 tab 不同语言之间可比。
        await cdp.evaluate('window.scrollTo(0, 0)');
        await sleep(200);
        const key = `${lang}/${tab}`;
        if (options.shots) {
          await screenshot(cdp, path.join(options.shots, `${lang}-${tab}.png`));
        }
        const metrics = await cdp.evaluate(`(${browserMetrics.toString()})(${JSON.stringify(tab)})`);
        const events = cdp.takeEvents();
        cdp.capture = false;

        const tabEvents = events.map((event) => ({ ...event, lang, tab }));
        result.tabs[key] = { metrics, events: tabEvents };
        result.events.push(...tabEvents);
      }
    }
    result.finishedAt = new Date().toISOString();
  } finally {
    cdp.capture = false;
    try {
      await restoreChinese(options.base);
    } catch (error) {
      process.stderr.write(`[smoke] 恢复中文失败: ${error.message}\n`);
    }
    try {
      cdp.ws.close();
    } catch {
      // 连接已断开时无需处理。
    }
  }

  const serialized = `${JSON.stringify(result, null, 2)}\n`;
  if (options.out) {
    fs.mkdirSync(path.dirname(options.out), { recursive: true });
    fs.writeFileSync(options.out, serialized);
  } else {
    process.stdout.write(serialized);
  }
  return result;
}

async function main() {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`[smoke] ${error.message}\n\n${HELP}`);
    process.exit(2);
    return;
  }
  if (options.help) {
    process.stdout.write(HELP);
    return;
  }
  try {
    await run(options);
  } catch (error) {
    process.stderr.write(`[smoke] 失败: ${error?.stack || error}\n`);
    try {
      await restoreChinese(options.base);
    } catch (restoreError) {
      process.stderr.write(`[smoke] 兜底恢复中文失败: ${restoreError.message}\n`);
    }
    process.exit(1);
  }
}

await main();
