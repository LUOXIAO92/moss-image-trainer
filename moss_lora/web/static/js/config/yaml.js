/* =========================================================================
 * js/config/yaml.js — YAML subset parser / dumper
 * split from web/static/app.js, source 610-866 (8 non-contiguous spans, see .tmp/js-map.json); behaviour unchanged
 * ========================================================================= */

import { t } from '../core/i18n.js';

function stripComment(line) {
  let out = '';
  let quote = null;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quote) {
      out += char;
      if (char === quote && line[i - 1] !== '\\') quote = null;
    } else if ((char === '"' || char === "'")
      && (i === 0 || /[\s:[{,\-]/.test(line[i - 1]))) {
      quote = char;
      out += char;
    } else if (char === '#' && (i === 0 || /\s/.test(line[i - 1]))) {
      break;
    } else {
      out += char;
    }
  }
  return out;
}

function splitInline(text) {
  const parts = [];
  let current = '';
  let quote = null;
  let depth = 0;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quote) {
      current += char;
      if (char === quote && text[i - 1] !== '\\') quote = null;
    } else if (char === '"' || char === "'") {
      quote = char;
      current += char;
    } else if (char === '[' || char === '{') {
      depth += 1;
      current += char;
    } else if (char === ']' || char === '}') {
      depth -= 1;
      current += char;
    } else if (char === ',' && depth === 0) {
      parts.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

function parseScalar(raw) {
  const value = String(raw).trim();
  if (value === '' || value === 'null' || value === '~') return null;
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (/^-?\d+$/.test(value)) return parseInt(value, 10);
  if (/^-?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$/.test(value)) return parseFloat(value);
  if (value.startsWith('"') && value.endsWith('"') && value.length >= 2) {
    try { return JSON.parse(value); } catch (_) { return value.slice(1, -1); }
  }
  if (value.startsWith("'") && value.endsWith("'") && value.length >= 2) {
    return value.slice(1, -1).replace(/''/g, "'");
  }
  if (value === '{}') return {};
  if (value.startsWith('[') && value.endsWith(']')) {
    const inner = value.slice(1, -1).trim();
    return inner ? splitInline(inner).map(parseScalar) : [];
  }
  return value;
}

function yamlFailure(message, line, column) {
  const error = new Error(message);
  error.line = line || null;
  error.column = column || null;
  return error;
}

function validateYamlLine(text, lineNumber) {
  let value = text.trim().replace(/^-\s+/, '');
  const colon = value.indexOf(':');
  if (colon >= 0) value = value.slice(colon + 1).trim();
  // Only inspect flow/quoted scalars. Apostrophes in plain Chinese/English
  // captions ("don't") must not be mistaken for an opening quote.
  if (!/^["'{[]/.test(value)) return;
  let quote = null;
  let square = 0;
  let curly = 0;
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    if (quote) {
      if (char === quote && value[i - 1] !== '\\') quote = null;
    } else if (char === '"' || char === "'") {
      quote = char;
    } else if (char === '[') {
      square += 1;
    } else if (char === ']') {
      square -= 1;
    } else if (char === '{') {
      curly += 1;
    } else if (char === '}') {
      curly -= 1;
    }
    if (square < 0 || curly < 0) throw yamlFailure(t('train.msg1'), lineNumber, i + 1);
  }
  if (quote) throw yamlFailure(t('train.msg2'), lineNumber, value.length);
  if (square !== 0 || curly !== 0) throw yamlFailure(t('train.msg3'), lineNumber, value.length);
}

export function parseYaml(text) {
  const rawLines = String(text == null ? '' : text).replace(/\r\n?/g, '\n').split('\n');
  const lines = [];
  for (let index = 0; index < rawLines.length; index += 1) {
    const raw = rawLines[index];
    if (/^ *\t/.test(raw)) throw yamlFailure(t('train.msg4'), index + 1, raw.indexOf('\t') + 1);
    const cleaned = stripComment(raw).replace(/\s+$/, '');
    if (!cleaned.trim()) continue;
    validateYamlLine(cleaned, index + 1);
    const indent = cleaned.match(/^ */)[0].length;
    lines.push({ indent, text: cleaned.slice(indent), line: index + 1 });
  }
  if (!lines.length) return {};
  let cursor = 0;

  function fail(line, message) { throw yamlFailure(message, line.line, line.indent + 1); }

  function parseBlock(indent) {
    if (cursor < lines.length && lines[cursor].indent === indent && lines[cursor].text.startsWith('-')) {
      return parseList(indent);
    }
    return parseMap(indent);
  }

  function parseList(indent) {
    const array = [];
    while (cursor < lines.length && lines[cursor].indent === indent && lines[cursor].text.startsWith('-')) {
      const current = lines[cursor];
      const item = current.text.replace(/^-\s?/, '');
      cursor += 1;
      if (!item) {
        array.push(cursor < lines.length && lines[cursor].indent > indent ? parseBlock(lines[cursor].indent) : null);
      } else if (item.includes(': ') && !/^["']/.test(item)) {
        const object = {};
        const colon = item.indexOf(':');
        object[item.slice(0, colon).trim()] = parseScalar(item.slice(colon + 1));
        array.push(object);
      } else {
        let scalar = item;
        // PyYAML wraps long plain scalars with a deeper continuation line;
        // join it back into the same scalar instead of treating it as a
        // malformed nested item.
        while (cursor < lines.length && lines[cursor].indent > indent && !lines[cursor].text.startsWith('-')) {
          scalar += ` ${lines[cursor].text.trim()}`;
          cursor += 1;
        }
        array.push(parseScalar(scalar));
      }
      if (cursor < lines.length && lines[cursor].indent > indent) fail(lines[cursor], t('common.msg3'));
    }
    return array;
  }

  function parseMap(indent) {
    const object = {};
    while (cursor < lines.length && lines[cursor].indent === indent && !lines[cursor].text.startsWith('-')) {
      const current = lines[cursor];
      const colon = current.text.indexOf(':');
      if (colon < 0) fail(current, t('common.msg4'));
      const key = current.text.slice(0, colon).trim().replace(/^["']|["']$/g, '');
      if (!key) fail(current, t('common.msg5'));
      const rest = current.text.slice(colon + 1).trim();
      cursor += 1;
      if (rest === '') {
        if (cursor < lines.length && lines[cursor].indent > indent) {
          object[key] = parseBlock(lines[cursor].indent);
        } else if (cursor < lines.length && lines[cursor].indent === indent && lines[cursor].text.startsWith('-')) {
          // PyYAML emits block lists at the same indent as their key.  The
          // form parser must accept both that and the more common deeper form.
          object[key] = parseList(indent);
        } else {
          object[key] = null;
        }
      } else {
        let scalar = rest;
        while (cursor < lines.length && lines[cursor].indent > indent && !lines[cursor].text.startsWith('-')) {
          scalar += ` ${lines[cursor].text.trim()}`;
          cursor += 1;
        }
        object[key] = parseScalar(scalar);
      }
    }
    if (cursor < lines.length && lines[cursor].indent > indent) fail(lines[cursor], t('common.msg6'));
    return object;
  }

  const result = parseBlock(lines[0].indent);
  if (cursor < lines.length) fail(lines[cursor], t('common.msg7'));
  if (result == null || typeof result !== 'object' || Array.isArray(result)) {
    throw yamlFailure(t('common.msg8'), lines[0].line, 1);
  }
  return result;
}

function scalarString(value) {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  const text = String(value);
  if (text === '') return '""';
  const needsQuote =
    /^\s|\s$/.test(text) ||
    /^(true|false|null|~|yes|no)$/i.test(text) ||
    /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text) ||
    /^[\[\]{}&*!|>'"%@`#,:-]/.test(text) ||
    /: | #/.test(text) ||
    text.includes('\n');
  return needsQuote ? JSON.stringify(text) : text;
}

export function dumpYaml(value, indent = 0) {
  const pad = '  '.repeat(indent);
  let out = '';
  if (value === null || typeof value !== 'object') return `${pad}${scalarString(value)}\n`;
  if (Array.isArray(value)) {
    if (!value.length) return `${pad}[]\n`;
    for (const item of value) {
      if (item && typeof item === 'object' && !Array.isArray(item)) {
        if (!Object.keys(item).length) out += `${pad}- {}\n`;
        else out += `${pad}-\n${dumpYaml(item, indent + 1)}`;
      } else if (Array.isArray(item) && !item.length) {
        out += `${pad}- []\n`;
      } else {
        out += `${pad}- ${scalarString(item)}\n`;
      }
    }
    return out;
  }
  if (!Object.keys(value).length) return `${pad}{}\n`;
  for (const [key, item] of Object.entries(value)) {
    if (item === undefined) continue;
    if (item && typeof item === 'object' && !Array.isArray(item)) {
      if (!Object.keys(item).length) out += `${pad}${key}: {}\n`;
      else out += `${pad}${key}:\n${dumpYaml(item, indent + 1)}`;
    } else if (Array.isArray(item)) {
      const allScalar = item.every((x) => x === null || typeof x !== 'object');
      const inline = item.map(scalarString).join(', ');
      const flowSafe = item.every((x) => !String(x).includes(','));
      if (!item.length) out += `${pad}${key}: []\n`;
      else if (allScalar && flowSafe && inline.length < 68) out += `${pad}${key}: [${inline}]\n`;
      else out += `${pad}${key}:\n${dumpYaml(item, indent + 1)}`;
    } else {
      out += `${pad}${key}: ${scalarString(item)}\n`;
    }
  }
  return out;
}
