'use strict';

/* =========================================================================
 * 页面片段加载
 *
 * index.html 只保留外壳（顶栏、任务条、dialog、modal）；9 个 tab 的 markup
 * 放在 static/pages/<tab>.html，启动时并发取回、按 TABS 顺序插到 #main 的
 * 第一个 dialog 之前 —— 保持 dialog 始终在 DOM 里（showModal() 依赖它）。
 *
 * 全部片段就位后才允许 bindEvents()/页面加载函数运行，否则 $() 会拿到 null。
 * ========================================================================= */

const FRAGMENT_ROOT = 'pages';

const fragmentCache = new Map();

/**
 * An error a translator can render: the code names an i18n key and the message
 * stays a neutral technical string for the case where no pack loaded at all.
 */
function fragmentError(code, params, message) {
  const error = new Error(message);
  error.code = code;
  error.params = params;
  return error;
}

/** 取回单个片段文本（同一 tab 只请求一次）。 */
async function fetchFragment(tab) {
  if (fragmentCache.has(tab)) return fragmentCache.get(tab);
  const response = await fetch(`${FRAGMENT_ROOT}/${tab}.html`, { cache: 'no-cache' });
  if (!response.ok) {
    throw fragmentError('app.fragmentHttp', { tab, status: response.status },
      `fragment ${tab}: HTTP ${response.status}`);
  }
  const html = await response.text();
  if (!html.trim()) throw fragmentError('app.fragmentEmpty', { tab }, `fragment ${tab} is empty`);
  fragmentCache.set(tab, html);
  return html;
}

/**
 * 并发取回并挂载所有片段。
 * @param {string[]} tabs 挂载顺序
 * @param {Element} [host] 挂载点，默认 #main
 * @returns {Promise<Element[]>} 按顺序排列的 section 节点
 */
export async function mountFragments(tabs, host = document.getElementById('main')) {
  if (!host) throw fragmentError('app.fragmentNoHost', {}, 'no #main to mount into');
  const htmls = await Promise.all(tabs.map((tab) => fetchFragment(tab)));
  const anchor = [...host.children].find((child) => child.tagName === 'DIALOG') || null;
  const mounted = [];
  tabs.forEach((tab, index) => {
    const template = document.createElement('template');
    template.innerHTML = htmls[index];
    const node = template.content.firstElementChild;
    if (!node) throw fragmentError('app.fragmentNoRoot', { tab }, `fragment ${tab} has no root element`);
    if (node.id !== `tab-${tab}`) {
      throw fragmentError('app.fragmentWrongRoot', { tab, id: node.id },
        `fragment ${tab} has root #${node.id}`);
    }
    host.insertBefore(node, anchor);
    mounted.push(node);
  });
  return mounted;
}
