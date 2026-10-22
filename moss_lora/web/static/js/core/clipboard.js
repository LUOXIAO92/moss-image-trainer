/* =========================================================================
 * js/core/clipboard.js — Clipboard
 * split from web/static/app.js, source 8576-8594; behaviour unchanged
 * ========================================================================= */

import { showAlert, showSuccess } from './notify.js';

export async function copyText(text, message = t('common.msg55')) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.append(area);
      area.select();
      document.execCommand('copy');
      area.remove();
    }
    showSuccess(message, '', [], 'copied');
  } catch (error) {
    showAlert({ type: 'warn', title: t('common.msg56'), text: error.message, timeout: 4000 });
  }
}
