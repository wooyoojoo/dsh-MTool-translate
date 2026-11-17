/**
 * `translate_mtool_file`: translate an MTool `ManualTransFile.json` with the
 * profile's own model route, and write a file MTool can load back.
 *
 * MTool's paid translation is this round trip: export the originals
 * (翻译 → 导出游戏原文 → `ManualTransFile.json`), send them somewhere, load the
 * result back (翻译 → 加载翻译文件). This plugin replaces "somewhere" with
 * `ctx.llm`, so no second program, API key, or Python runtime is installed.
 *
 * The text-processing and validation behaviour is ported from AiNiee
 * (github.com/NEKOparapa/AiNiee), file by file; each block below names its
 * source. What stays local is the delivery: DSH tool registration, the model
 * route, batching, caching, checkpointing.
 *
 * Ported from AiNiee:
 *  - `Backend/resources/regex/regex.json`      RPG Maker control-code patterns
 *  - `Backend/domain/text/text_processor.py`   [P n] placeholders, restore order
 *  - `Backend/domain/text/filters/general_text_filter.py`  skip rules
 *  - `Backend/domain/responses/response_extractor.py`      <textarea> + `n.` lines
 *  - `Backend/domain/responses/advanced_checks.py`         placeholder / newline /
 *                                                          returned-original / residual
 *  - `Backend/domain/prompts/glossary_helper.py`           term matching
 *  - `Backend/domain/text/text_symbol_repair.py`           quote and punctuation repair
 *  - `Backend/resources/prompts/Translate/common_system_zh.txt`  system prompt
 *
 * The file is a flat JSON object whose key is the original string and whose
 * value is the current translation. MTool pre-fills the value with the
 * original; other exporters leave it empty. An entry is *pending* when its
 * value is empty or still equals its key (same rule as `m_tool_reader.py`).
 *
 * Imports only Node builtins, so a Harness upgrade cannot break module
 * resolution, and the same file runs from a checkout, an installed profile,
 * and a packed tarball.
 *
 * @module dsh-mtool-translate
 */

import { createHash } from 'node:crypto'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { basename, dirname, extname, join, resolve } from 'node:path'

/** Cordis plugin name used by Loader diagnostics. */
export const name = 'mtoolTranslate'

/** The tool registry this plugin contributes to. */
export const inject = ['tools']

/** Target language used when neither config nor the call names one. */
export const DEFAULT_TARGET = '简体中文'

/** Lines one model request may carry. */
export const DEFAULT_BATCH_LINES = 30

/** Source characters one model request may carry. */
export const DEFAULT_BATCH_CHARS = 1200

/** Model requests in flight at once. */
export const DEFAULT_CONCURRENCY = 3

/** Attempts per batch beyond the first. */
export const DEFAULT_MAX_RETRIES = 2

/** Cooperative per-call budget; a big file needs room for many batches. */
export const DEFAULT_TIMEOUT_MS = 900_000

/** Output-token ceiling for one batch reply. */
export const DEFAULT_MAX_TOKENS = 4096

/**
 * Reasoning effort for the routed model. Translation is formatting, not
 * reasoning; `low` keeps latency sane while still resolving grammar.
 */
export const DEFAULT_REASONING_EFFORT = 'low'

/** Cache schema version; a mismatch discards old entries. */
export const CACHE_VERSION = 2

/** Longest single original the plugin will send; anything longer is skipped. */
export const MAX_LINE_CHARS = 4000

/** Name of the cache file kept beside an exported text file. */
export const CACHE_FILE = '.mtool-translate-cache.json'

/** `[P1]`, `[P2]`, … placeholders (ported from text_processor.py). */
const PLACEHOLDER = /\[P(\d+)\]/g

/**
 * RPG Maker / Wolf / KRKR control codes, in the order AiNiee applies them.
 * Composite patterns come first so `\C[3]` is never split by the generic
 * `\X[digits]` rule. Ported verbatim from `Backend/resources/regex/regex.json`.
 */
export const CODE_PATTERNS = [
  { source: '\\\\font\\[\\d+\\]\\\\c\\[\\d+\\]\\\\f\\[\\d+\\]A+', description: '\\font[n]\\c[n]\\f[n]\\A+' },
  { source: 'if\\(.{0,5}[vs]\\[\\d+\\].{0,10}\\)', description: 'if(v[n] …)' },
  { source: 'en\\(.{0,5}[vs]\\[\\d+\\].{0,10}\\)', description: 'en(v[n] …)' },
  { source: '[\\\\/][a-z]{1,5}<[a-z\\d]{0,10}>', description: '/C<y> \\FS<12>' },
  { source: '[\\\\/][a-z]{1,5}\\[[a-z\\d]{0,10}\\]', description: '/C[x] \\FS[12]' },
  { source: '[\\\\/][a-z]{1,5}(?=<[^a-z\\d]{0,10}>)', description: '/C<…> prefix' },
  { source: '[\\\\/][a-z]{1,5}(?=\\[[^a-z\\d]{0,10}\\])', description: '/C[…] prefix' },
  { source: '\\\\fr', description: '\\fr reset' },
  { source: '\\\\fb', description: '\\fb bold' },
  { source: '\\\\fi', description: '\\fi italic' },
  { source: '\\\\g', description: '\\G currency' },
  { source: '\\\\\\$', description: '\\$ gold window' },
  { source: '\\\\\\|', description: '\\| wait 1s' },
  { source: '\\\\\\.', description: '\\. wait 0.25s' },
  { source: '\\\\!', description: '\\! wait for input' },
  { source: '\\\\>', description: '\\> same line' },
  { source: '\\\\^', description: '\\^ no wait' },
  { source: '\\\\<br>', description: '\\<br>' },
  { source: '\\\\SE\\[.{0,15}?\\]', description: '\\SE[…]' },
  { source: '【\\\\[A-Za-z]+\\[[^\\]]*】\\\\SE\\[[^\\]]*\\]', description: '【\\N[1]】\\SE[…]' },
  { source: '<[A-Za-z]+:\\d+>', description: '<Tag:1>' },
  { source: '\\\\cself\\[\\d+\\]', description: '\\cself[40]' },
  { source: '\\\\[A-Za-z]\\[\\d+\\]>', description: '\\X[n]>' },
  { source: '\\\\[A-Za-z]\\[\\d+\\]', description: '\\X[n]' },
  { source: '\\\\h\\[[^\\]]*\\]\\\\name', description: '\\h[…]\\name' },
  { source: '\\\\name', description: '\\name' },
  { source: '\\\\gold', description: '\\gold' },
  { source: '\\\\count', description: '\\count' },
  { source: '\\\\AA\\[F\\]', description: '\\AA[F]' },
  { source: '@\\d+', description: '@123' },
  { source: '\\\\ws', description: '\\ws' },
  { source: '\\\\wm\\\\w\\[\\d+\\]', description: '\\wm\\w[180]' },
  { source: '\\\\FA\\[[^\\]]*\\]\\\\SE\\[[^\\]]*\\]', description: '\\FA[…]\\SE[…]' },
  { source: '<ill:\\d+', description: '<ill:1' },
  { source: '<plus:\\\\C\\[\\d+\\]', description: '<plus:\\C[1]' },
  { source: '<minus:\\\\C\\[\\d+\\]', description: '<minus:\\C[1]' },
  { source: '<br:\\\\I\\[\\d+\\]', description: '<br:\\I[1]' },
  { source: '\\{w=wt\\d+\\}', description: '{w=wt1}' },
  { source: '\\{image=[^}]*\\}', description: '{image=…}' },
  { source: '\\{color=[^}]*\\}', description: '{color=…}' },
  { source: '\\{/color\\}', description: '{/color}' },
  { source: '\\{p\\}', description: '{p}' },
  { source: '\\{p\\}\\{p\\}.*?', description: '{p}{p}…' },
  { source: '\\{size=\\d+\\}', description: '{size=1}' },
  { source: '\\{/size\\}', description: '{/size}' },
  { source: '\\{nm\\}[\\s\\S]*?\\{/nm\\}', description: '{nm}…{/nm}' },
  { source: '\\{fast\\}', description: '{fast}' },
  { source: '\\{a=[^}]*\\}', description: '{a=…}' },
  { source: '\\\\A+', description: '\\A+ prefix' },
  { source: '\\\\n', description: 'literal \\n line break' },
  // Two consecutive real newlines are one structural unit; single \r last.
  { source: '(?:\\r?\\n){2}', description: 'double newline' },
  { source: '\\r', description: 'carriage return' },
].map(entry => ({ ...entry, pattern: new RegExp(entry.source, 'g') }))

/**
 * Characters that count as punctuation only (ported from general_text_filter.py).
 */
const PUNCTUATION = new Set([
  ' ', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '，', '。',
  ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '—', '・',
  '？', '↑', '←', '↓', '→', '「', '」', '『', '』', '【', '】', '《', '》', '…', '‥',
  '！', '＂', '＃', '＄', '％', '＆', '＇', '（', '）', '＊', '＋', '－', '．', '／', '：', '；',
  '＜', '＝', '＞', '＠',
])

/** File suffixes that mean "this is a path, not prose" (general_text_filter.py). */
const EXCLUDE_SUFFIX = new Set([
  '.mp3', '.wav', '.png', '.jpg', '.gif', '.rar', '.zip', '.json', '.ogg', '.txt', '.mps', '.woff',
  '.webp', '.doc', '.html', '.bmp', '.pic', '.aac', '.flac', '.avi', '.py', '.c', '.cpp', '.js',
  '.java', '.css', '.xml', '.jpeg', '.mov', '.mkv', '.flv',
])

/** Leading fragments that mean "this is data, not prose" (general_text_filter.py). */
const EXCLUDE_PREFIX = ['MapData/', 'SE/', 'BGS', '0=', 'BGM/', 'FIcon/', '<input type=', 'width:', '<div ', 'EV0', '\\img']

/**
 * Characters that count as "the source language", used to strip code from the
 * edges of a line before translating (text_processor.py JAPANESE_CHAR_SET).
 */
const JAPANESE_CONTENT = /[\u3040-\u309F\u30A0-\u30FF\u30FB-\u30FE\uFF65-\uFF9F\u4E00-\u9FFF\u3400-\u4DBF\u3001-\u303F\uFF01-\uFF5E]/

/** Japanese kana, used by the residual-original check (advanced_checks.py). */
const KANA = /[\u3041-\u3096\u30A0-\u30FF]+/gu

/** Whether `value` is a plain object: not null, not an array. */
function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/** Whether `value` is a non-empty string. */
function isFilledString(value) {
  return typeof value === 'string' && value.trim().length > 0
}

/**
 * Validate and normalize the Loader-supplied config, filling documented
 * defaults.
 *
 * @param config - raw config from the plugin row; every field is optional.
 * @returns The resolved config.
 * @throws When a supplied value is unusable.
 */
export function resolveConfig(config) {
  const raw = config ?? {}
  if (!isPlainObject(raw)) throw new Error('mtoolTranslate: config must be an object')

  const integer = (field, fallback, minimum) => {
    const value = raw[field] ?? fallback
    if (!Number.isInteger(value) || value < minimum) {
      throw new Error(`mtoolTranslate: config.${field} must be an integer >= ${minimum}`)
    }
    return value
  }
  const boolean = (field, fallback) => {
    const value = raw[field] ?? fallback
    if (typeof value !== 'boolean') throw new Error(`mtoolTranslate: config.${field} must be a boolean`)
    return value
  }
  const stringMap = (field) => {
    const value = raw[field]
    if (value === undefined) return {}
    if (!isPlainObject(value)) throw new Error(`mtoolTranslate: config.${field} must be an object of strings`)
    return value
  }

  for (const field of ['target', 'provider', 'model', 'reasoningEffort', 'outputDir', 'sourceLanguage']) {
    if (raw[field] !== undefined && !isFilledString(raw[field])) {
      throw new Error(`mtoolTranslate: config.${field} must be a non-empty string when set`)
    }
  }
  if (raw.glossary !== undefined && !Array.isArray(raw.glossary) && !isPlainObject(raw.glossary)) {
    throw new Error('mtoolTranslate: config.glossary must be an array of pairs or an object map')
  }
  if (raw.exclusion !== undefined && !Array.isArray(raw.exclusion)) {
    throw new Error('mtoolTranslate: config.exclusion must be an array of strings')
  }

  return {
    target: raw.target ?? DEFAULT_TARGET,
    sourceLanguage: raw.sourceLanguage ?? '日语',
    provider: raw.provider,
    model: raw.model,
    reasoningEffort: raw.reasoningEffort ?? DEFAULT_REASONING_EFFORT,
    batchLines: integer('batchLines', DEFAULT_BATCH_LINES, 1),
    batchChars: integer('batchChars', DEFAULT_BATCH_CHARS, 200),
    concurrency: integer('concurrency', DEFAULT_CONCURRENCY, 1),
    maxRetries: integer('maxRetries', DEFAULT_MAX_RETRIES, 0),
    timeoutMs: integer('timeoutMs', DEFAULT_TIMEOUT_MS, 1000),
    maxTokens: integer('maxTokens', DEFAULT_MAX_TOKENS, 256),
    limit: integer('limit', 0, 0),
    outputDir: raw.outputDir,
    glossary: raw.glossary,
    exclusion: raw.exclusion ?? [],
    preReplace: stringMap('preReplace'),
    postReplace: stringMap('postReplace'),
    // AiNiee exposes these as switches; placeholder/newline/empty checks are
    // always on, these three are opt-in and off by default.
    symbolRepair: boolean('symbolRepair', true),
    checkReturnedOriginal: boolean('checkReturnedOriginal', false),
    checkResidualOriginal: boolean('checkResidualOriginal', false),
  }
}

/**
 * Read and parse one `ManualTransFile.json`.
 *
 * @param path - absolute path to the exported file.
 * @returns The parsed object and the raw text.
 * @throws When the file is unreadable or is not a flat JSON object.
 */
export async function readManualTrans(path) {
  let raw
  try {
    raw = await readFile(path, 'utf8')
  } catch (error) {
    throw new Error(`mtoolTranslate: cannot read ${path}: ${error.message}`)
  }
  const text = raw.charCodeAt(0) === 0xfeff ? raw.slice(1) : raw
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new Error(`mtoolTranslate: ${path} is not valid JSON: ${error.message}`)
  }
  if (!isPlainObject(parsed)) {
    throw new Error(`mtoolTranslate: ${path} must be a JSON object mapping original text to translations`)
  }
  return { entries: parsed, text }
}

/**
 * Whether one value still needs a translation.
 *
 * Same rule as AiNiee's `m_tool_reader.py`: a value counts as translated only
 * when it is non-empty and differs from its key.
 *
 * @param key - the original text.
 * @param value - the value stored in the file.
 * @param force - re-translate even a finished entry.
 * @returns Whether the entry is pending.
 */
export function isPending(key, value, force = false) {
  if (typeof value !== 'string') return false
  if (force) return true
  return !(value.trim() !== '' && key !== value)
}

/**
 * Decide whether a string should reach the model at all.
 *
 * Ported from `general_text_filter.py`, plus a length guard and a whole-line
 * exclusion-list check.
 *
 * @param text - one original string.
 * @param exclusion - whole-line terms that must stay untranslated.
 * @returns A skip reason, or `undefined` when the string should be translated.
 */
export function skipReason(text, exclusion = []) {
  if (typeof text !== 'string') return 'not-a-string'
  if (text.length > MAX_LINE_CHARS) return 'too-long'
  if (/^\d+$/u.test(text)) return 'numeric'
  if (text.trim() === '') return 'empty'
  if (['\n', '\\n', '\r', '\\r'].includes(text.trim())) return 'newline-only'
  if ([...text].every(char => PUNCTUATION.has(char))) return 'punctuation-only'
  const suffix = text.trim().endsWith('.') ? '' : text.trim().slice(text.trim().lastIndexOf('.'))
  if (suffix !== '' && EXCLUDE_SUFFIX.has(suffix.toLowerCase())) return 'file-suffix'
  if (EXCLUDE_PREFIX.some(prefix => text.startsWith(prefix))) return 'code-prefix'
  if (exclusion.some(term => term.trim() !== '' && text.trim() === term.trim())) return 'excluded'
  return undefined
}

/**
 * Collect the entries of one file that need work, in file order.
 *
 * Every entry is classified once: already translated, cached, skipped, or
 * pending.
 *
 * @param entries - parsed `ManualTransFile.json`.
 * @param options - `force`, `limit`, `target`, `cache`, `exclusion`.
 * @returns The classified items and the counts behind them.
 */
export function collectWork(entries, options = {}) {
  const { force = false, limit = 0, target = DEFAULT_TARGET, cache = new Map(), exclusion = [] } = options
  const items = []
  let finished = 0
  let cached = 0
  let skipped = 0
  let taken = 0
  for (const [key, value] of Object.entries(entries)) {
    if (typeof value !== 'string') continue
    if (!isPending(key, value, force)) {
      finished += 1
      continue
    }
    const reason = skipReason(key, exclusion)
    if (reason !== undefined) {
      skipped += 1
      items.push({ key, text: key, skip: reason })
      continue
    }
    const hit = force ? undefined : cache.get(cacheKey(key, target))
    if (hit !== undefined) {
      cached += 1
      items.push({ key, text: key, cached: hit })
      continue
    }
    if (limit > 0 && taken >= limit) break
    taken += 1
    items.push({ key, text: key })
  }
  return { items, finished, cached, skipped }
}

/**
 * Apply the user's pre-translation replacement table.
 *
 * Ported from `text_processor.replace_before_translation`; the table is a plain
 * `{ from: to }` object rather than AiNiee's regex-capable rows.
 *
 * @param text - one original string.
 * @param rules - `{ from: to }`.
 * @returns The replaced text.
 */
export function applyReplacements(text, rules) {
  let out = text
  for (const [from, to] of Object.entries(rules ?? {})) {
    if (from !== '' && typeof to === 'string') out = out.split(from).join(to)
  }
  return out
}

/**
 * Strip the code segments that sit outside the Japanese text.
 *
 * Ported from `text_processor.strip_and_record_affixes`: a leading and trailing
 * run with no Japanese character is code or decoration, so it travels with the
 * line untouched and only the core reaches the model.
 *
 * @param text - one original string.
 * @returns The core text and the pieces to put back around it.
 */
export function stripAffixes(text) {
  if (text === '') return { core: '', prefix: '', suffix: '' }
  const first = text.search(JAPANESE_CONTENT)
  if (first < 0) return { core: text, prefix: '', suffix: '' }
  let last = text.length - 1
  while (last > first && !JAPANESE_CONTENT.test(text[last])) last -= 1
  return { core: text.slice(first, last + 1), prefix: text.slice(0, first), suffix: text.slice(last + 1) }
}

/**
 * Mask control codes into `[Pn]` placeholders.
 *
 * Ported from `text_processor._replace_special_placeholders`: the counter is
 * shared across one request, so a placeholder moved to a different line is
 * caught instead of silently restoring the wrong code.
 *
 * @param text - the core text of one line.
 * @param state - the request-wide `{ counter, entries }` accumulator.
 * @returns The masked text.
 */
export function maskCodes(text, state) {
  let masked = text
  for (const { pattern } of CODE_PATTERNS) {
    masked = masked.replace(pattern, (match) => {
      state.counter += 1
      const placeholder = `[P${state.counter}]`
      state.entries.push({ placeholder, original: match })
      return placeholder
    })
  }
  return masked
}

/**
 * Put masked codes back.
 *
 * Ported from `text_processor._restore_special_placeholders`: reverse order,
 * first occurrence only, so nested or repeated codes land where they were.
 *
 * @param text - the translated text with placeholders.
 * @param entries - the placeholders recorded for this line.
 * @returns The game-ready text.
 */
export function unmaskCodes(text, entries) {
  let restored = text
  for (const { placeholder, original } of [...entries].reverse()) {
    restored = restored.replace(placeholder, () => original)
  }
  return restored
}

/**
 * Pack work items into model requests.
 *
 * @param items - pending work items.
 * @param options - `batchLines` and `batchChars` caps.
 * @returns One array per request, each in file order.
 */
export function buildBatches(items, options = {}) {
  const { batchLines = DEFAULT_BATCH_LINES, batchChars = DEFAULT_BATCH_CHARS } = options
  const batches = []
  let current = []
  let chars = 0
  for (const item of items) {
    const length = item.text.length
    if (current.length > 0 && (current.length >= batchLines || chars + length > batchChars)) {
      batches.push(current)
      current = []
      chars = 0
    }
    current.push(item)
    chars += length
  }
  if (current.length > 0) batches.push(current)
  return batches
}

/**
 * Normalize every glossary form the plugin accepts.
 *
 * @param value - object map, array of `{source,target}`, array of
 *   `{term,translation}`, or array of `source=>target` strings.
 * @returns Normalized pairs; unusable entries are dropped.
 */
export function normalizeGlossary(value) {
  if (value === undefined || value === null) return []
  const pairs = []
  const add = (source, target) => {
    if (isFilledString(source) && isFilledString(target)) pairs.push({ source: source.trim(), target: target.trim() })
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      if (typeof entry === 'string') {
        const separator = entry.includes('=>') ? '=>' : '='
        const at = entry.indexOf(separator)
        if (at > 0) add(entry.slice(0, at), entry.slice(at + separator.length))
      } else if (isPlainObject(entry)) {
        add(entry.source ?? entry.term ?? entry.text, entry.target ?? entry.translation ?? entry.value)
      }
    }
  } else if (isPlainObject(value)) {
    for (const [source, target] of Object.entries(value)) add(source, target)
  }
  return pairs
}

/**
 * Resolve a glossary supplied inline, as a path, or not at all.
 *
 * @param value - an inline glossary, a path to a JSON file, or undefined.
 * @returns The normalized pairs.
 * @throws When a named file cannot be read or parsed.
 */
export async function resolveGlossary(value) {
  if (value === undefined || value === null) return []
  if (typeof value !== 'string') return normalizeGlossary(value)
  const trimmed = value.trim()
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) return normalizeGlossary(JSON.parse(trimmed))
  const path = resolve(trimmed)
  let text
  try {
    text = await readFile(path, 'utf8')
  } catch (error) {
    throw new Error(`mtoolTranslate: cannot read glossary ${path}: ${error.message}`)
  }
  try {
    return normalizeGlossary(JSON.parse(text.charCodeAt(0) === 0xfeff ? text.slice(1) : text))
  } catch (error) {
    throw new Error(`mtoolTranslate: glossary ${path} is not valid JSON: ${error.message}`)
  }
}

/**
 * Whether one glossary term applies to a line.
 *
 * Ported from `glossary_helper.GlossaryHelper`: a term matches its whole text,
 * or any of its space/dot-separated parts, case-insensitively — so "エルフ・
 * 森" still catches a line that only says "エルフ".
 *
 * @param term - the glossary source text.
 * @param text - the line to test.
 * @returns Whether the term applies.
 */
export function glossaryMatches(term, text) {
  const haystack = text.toLowerCase()
  const needle = term.toLowerCase()
  if (haystack.includes(needle)) return true
  return needle.split(/[\s·・･∙⋅‧⸱]+/u).filter(Boolean).some(part => haystack.includes(part))
}

/**
 * Attach the glossary terms each line actually contains.
 *
 * @param items - work items.
 * @param glossary - normalized pairs.
 * @returns The same items, each with its `glossary` matches.
 */
export function attachGlossary(items, glossary) {
  if (glossary.length === 0) return items
  for (const item of items) {
    item.glossary = glossary.filter(entry => glossaryMatches(entry.source, item.text))
  }
  return items
}

/**
 * Compose the system prompt.
 *
 * Structure and tone follow AiNiee's
 * `resources/prompts/Translate/common_system_zh.txt`; the `<textarea>` and
 * `n.` conventions are what `response_extractor.py` parses back.
 *
 * @param options - target language, source language, glossary, exclusion list.
 * @returns The system message.
 */
export function buildSystemPrompt(options) {
  const { target, sourceLanguage, glossary = [], exclusion = [] } = options
  const lines = [
    `你是一名专业的游戏本地化译者。把${sourceLanguage}文本逐行翻译成${target}。`,
    '逐行翻译，不要合并；原样保留序号、标记符、占位符（形如 [P1]）、换行符、转义符等特殊内容，保持原来的格式。',
    '',
    '### 翻译原则',
    '忠实准确地进行翻译。原作是艺术作品，允许使用直白或粗俗的描述，不应随意删减或篡改，不应回避或淡化，以保持作品的原貌。',
    '译文中不要出现解释、注释或原文读音；角色名、地名、道具名前后保持一致。',
  ]
  if (glossary.length > 0) {
    lines.push('', '### 术语表，必须严格使用以下译法', '原文|译文')
    for (const entry of glossary) lines.push(`${entry.source}|${entry.target}`)
  }
  if (exclusion.length > 0) {
    lines.push('', '### 禁翻表，以下原文保持不变', '原文')
    for (const term of exclusion) lines.push(term)
  }
  lines.push('', '### 以 textarea 标签输出译文', '<textarea>', `1.${target}文本`, '</textarea>')
  return lines.join('\n')
}

/**
 * Compose the user message: numbered source lines.
 *
 * @param batch - work items carrying `masked`.
 * @returns The user message.
 */
export function buildUserPrompt(batch) {
  return batch.map((item, index) => `${index + 1}.${item.masked}`).join('\n')
}

/**
 * Split a `<textarea>` reply into an ordered list of translations.
 *
 * Ported from `response_extractor.py`: take the last textarea, split on lines
 * that start a new `n.` item, drop leading prose the model wrote before the
 * first numbered line, and strip the numbering.
 *
 * @param reply - the assistant text.
 * @returns One string per numbered item, in order.
 * @throws When no numbered content is present.
 */
export function extractTranslation(reply) {
  const text = String(reply ?? '')
  const areas = [...text.matchAll(/<textarea[^>]*>([\s\S]*?)<\/textarea>/giu)]
  const body = areas.length > 0 ? areas[areas.length - 1][1] : text
  const trimmed = body.trim()
  if (trimmed === '') throw new Error('empty reply')

  const blocks = trimmed.split(/\n(?=\s*\d+\.)/u)
  const items = []
  let started = false
  for (const block of blocks) {
    const line = block.trim()
    if (line === '') continue
    const match = /^(\d+)\.\s*([\s\S]*)$/u.exec(line)
    if (match === null) {
      if (!started) continue
      if (items.length > 0) items[items.length - 1] += `\n${line}`
      continue
    }
    started = true
    items.push(match[2].trim())
  }
  if (items.length === 0) throw new Error(`reply carried no numbered lines: ${trimmed.slice(0, 200)}`)
  return items
}

/**
 * Parse a JSON-array reply.
 *
 * Kept as a fallback for models that ignore the `<textarea>` request; the
 * primary format is the numbered list, which cannot break on newlines.
 *
 * @param reply - the assistant text.
 * @returns One string per item, in order.
 * @throws When no JSON payload is present.
 */
export function parseJsonReply(reply) {
  const text = String(reply ?? '').trim()
  const fenced = text.replace(/^```[a-zA-Z]*\s*/u, '').replace(/```\s*$/u, '').trim()
  const start = fenced.indexOf('[')
  const end = fenced.lastIndexOf(']')
  if (start < 0 || end <= start) throw new Error('no JSON array in reply')
  const parsed = JSON.parse(fenced.slice(start, end + 1))
  if (!Array.isArray(parsed)) throw new Error('reply is not a JSON array')
  return parsed.map((entry) => {
    if (typeof entry === 'string') return entry
    if (isPlainObject(entry)) return String(entry.text ?? entry.t ?? entry.translation ?? entry.target ?? '')
    throw new Error('unexpected JSON array entry')
  })
}

/**
 * Turn one reply into translations aligned with the batch.
 *
 * @param reply - assistant text.
 * @param expected - how many lines the batch asked for.
 * @returns The translations, in batch order.
 * @throws When neither protocol yields the right number of lines.
 */
export function readReply(reply, expected) {
  let firstError
  for (const extract of [extractTranslation, parseJsonReply]) {
    try {
      const items = extract(reply)
      if (items.length === expected) return items
      firstError ??= new Error(`reply had ${items.length} lines, expected ${expected}`)
      if (items.length > expected) return items.slice(0, expected)
    } catch (error) {
      firstError ??= error
    }
  }
  throw firstError ?? new Error('reply could not be read')
}

/**
 * Count line breaks, including the literal `\n` escape MTool exports.
 *
 * Ported from `advanced_checks.check_multiline_text`.
 *
 * @param text - one line.
 * @returns Break count.
 */
export function countNewlines(text) {
  return (text.match(/\n/gu) ?? []).length + (text.match(/\\n/gu) ?? []).length
}

/**
 * Whether a translation is still the source, judged by character overlap.
 *
 * Adapted from `advanced_checks.check_dicts_equal`: AiNiee averages a Jaccard
 * similarity over a whole batch, but this already knows one line at a time, so
 * the score is normalized by the source's own non-kanji characters. Kanji are
 * excluded because Chinese output legitimately reuses them.
 *
 * @param source - the original line, placeholders stripped.
 * @param translated - the translated line.
 * @returns Whether the line looks untranslated.
 */
export function looksLikeOriginal(source, translated) {
  const left = new Set([...source].filter(char => !(char >= '\u4E00' && char <= '\u9FFF')))
  if (left.size === 0) return false
  const right = new Set(translated)
  let intersection = 0
  for (const char of left) if (right.has(char)) intersection += 1
  return intersection / left.size >= 0.9
}

/**
 * Whether the translation still carries source kana.
 *
 * Ported from `advanced_checks.detecting_remaining_original_text`.
 *
 * @param source - the original line.
 * @param translated - the translated line.
 * @returns Whether source kana survived.
 */
export function hasResidualOriginal(source, translated) {
  for (const run of translated.match(KANA) ?? []) {
    if (source.includes(run)) return true
  }
  return false
}

/**
 * Validate one translated line against its source.
 *
 * Ported from `response_checker.check_response_items` and its helpers. A defect
 * means "retry this line"; it never means "write it back".
 *
 * @param item - the work item (carrying `masked`, `placeholders`, `text`).
 * @param translated - the reply for that line.
 * @param options - switches for the opt-in checks.
 * @returns `undefined` when acceptable, otherwise the defect.
 */
export function lineDefect(item, translated, options = {}) {
  if (!isFilledString(translated)) return '【空译文】'
  const missing = item.placeholders.find(entry => !translated.includes(entry.placeholder))
  if (missing !== undefined) return `【占位符缺失】 ${missing.placeholder}`
  if (countNewlines(item.masked) !== countNewlines(translated)) return '【换行符数】'
  const semanticSource = item.masked.replace(PLACEHOLDER, '')
  const semanticTranslated = translated.replace(PLACEHOLDER, '')
  if (options.checkReturnedOriginal === true && looksLikeOriginal(semanticSource, semanticTranslated)) {
    return '【返回原文】'
  }
  if (options.checkResidualOriginal === true && hasResidualOriginal(semanticSource, semanticTranslated)) {
    return '【翻译残留】'
  }
  return undefined
}

/**
 * Repair quotation and punctuation the model rendered in the wrong style.
 *
 * Ported from `text_symbol_repair.repair_text_symbols` stages 0-3: paired
 * corner brackets, internal quotes, and a small punctuation map. Stage 4
 * (per-line quotes in multi-line blocks) is not ported.
 *
 * @param original - the source line.
 * @param translated - the translated line.
 * @returns The repaired line.
 */
export function repairSymbols(original, translated) {
  if (typeof original !== 'string' || typeof translated !== 'string') return translated
  const leading = /^(\s*)/u.exec(original)?.[1] ?? ''
  const trailing = /(\s*)$/u.exec(original)?.[1] ?? ''
  const sourceCore = original.trim()
  let target = translated.trim()
  if (sourceCore === '' || target === '') return translated

  const pairs = [
    { open: '「', close: '」', altOpen: ['“', '‘', '"'], altClose: ['”', '’', '"'] },
    { open: '『', close: '』', altOpen: ['“', '‘', '"'], altClose: ['”', '’', '"'] },
    { open: '“', close: '”', altOpen: ['‘', '「', '"'], altClose: ['’', '」', '"'] },
    { open: '‘', close: '’', altOpen: ['“', '「', '"'], altClose: ['”', '」', '"'] },
  ]
  for (const pair of pairs) {
    const wrapped = sourceCore.startsWith(pair.open) && sourceCore.endsWith(pair.close)
    const single = sourceCore.split(pair.open).length === 2 && sourceCore.split(pair.close).length === 2
    if (!wrapped || !single) continue
    let fixed = false
    for (let index = 0; index < pair.altOpen.length; index += 1) {
      const open = pair.altOpen[index]
      const close = pair.altClose[index]
      if (target.startsWith(open) && target.endsWith(close)) {
        target = pair.open + target.slice(open.length, target.length - close.length) + pair.close
        fixed = true
        break
      }
    }
    if (fixed) break
  }

  // Internal quotes: source uses 「」 n times, the reply uses " 2n times.
  const opens = sourceCore.split('「').length - 1
  const closes = sourceCore.split('」').length - 1
  const quotes = target.split('"').length - 1
  if (opens > 0 && opens === closes && quotes > 0 && quotes % 2 === 0 && opens === quotes / 2) {
    let open = true
    target = [...target].map((char) => {
      if (char !== '"') return char
      const replacement = open ? '「' : '」'
      open = !open
      return replacement
    }).join('')
  }

  target = target
    .split('...').join('…')
    .split('。。。').join('…')
    .split('——').join('—')
    .split('--').join('—')
    .split('?').join('？')

  return leading + target + trailing
}

/**
 * Identify one cached line.
 *
 * @param text - the original text.
 * @param target - the target language.
 * @returns A stable hex key.
 */
export function cacheKey(text, target) {
  return createHash('sha1').update(`${target}\u0000${text}`, 'utf8').digest('hex')
}

/**
 * Read the cache that sits beside an exported file.
 *
 * @param cachePath - cache file path.
 * @param target - target language the cache must match.
 * @returns Cached lines keyed by hash; empty when absent or stale.
 */
export async function readCache(cachePath, target) {
  let text
  try {
    text = await readFile(cachePath, 'utf8')
  } catch {
    return new Map()
  }
  try {
    const parsed = JSON.parse(text)
    if (parsed?.version !== CACHE_VERSION || parsed?.target !== target || !isPlainObject(parsed?.entries)) {
      return new Map()
    }
    return new Map(Object.entries(parsed.entries))
  } catch {
    return new Map()
  }
}

/**
 * Write a file through a sibling temporary, so a crash cannot truncate it.
 *
 * @param path - destination.
 * @param text - contents.
 */
export async function writeAtomic(path, text) {
  await mkdir(dirname(path), { recursive: true })
  const temporary = `${path}.tmp-${process.pid}`
  await writeFile(temporary, text, 'utf8')
  await rename(temporary, path)
}

/**
 * Write the cache beside an exported file.
 *
 * @param cachePath - cache file path.
 * @param target - target language.
 * @param entries - cached lines keyed by hash.
 */
export async function writeCache(cachePath, target, entries) {
  await writeAtomic(cachePath, JSON.stringify({ version: CACHE_VERSION, target, entries }))
}

/**
 * Ask the routed model for one request and return its raw text.
 *
 * @param llm - the `llm` service.
 * @param route - `provider`, `model`, `reasoningEffort`, `maxTokens`.
 * @param system - system message.
 * @param user - user message.
 * @param signal - cancellation from the tool call.
 * @returns The concatenated assistant text and token usage.
 * @throws When the provider fails or the call aborts.
 */
export async function complete(llm, route, system, user, signal) {
  const stream = llm.stream({
    provider: route.provider,
    model: route.model,
    ...route.reasoningEffort === undefined ? {} : { reasoningEffort: route.reasoningEffort },
    ...route.maxTokens === undefined ? {} : { maxTokens: route.maxTokens },
    system,
    messages: [{ role: 'user', content: [{ type: 'text', text: user }] }],
    signal,
  })
  let text = ''
  let usage
  let failure
  for await (const chunk of stream) {
    if (chunk.type === 'text-delta') {
      text += chunk.text
    } else if (chunk.type === 'usage') {
      usage = chunk.usage
    } else if (chunk.type === 'finish' && (chunk.reason.kind === 'error' || chunk.reason.kind === 'aborted')) {
      failure = chunk.reason.failure
    }
  }
  if (failure !== undefined) {
    throw new Error(`model call failed: ${failure.message}${failure.status === undefined ? '' : ` (HTTP ${failure.status})`}`)
  }
  return { text, usage }
}

/**
 * Translate one batch, retrying the batch and then its individual lines.
 *
 * Placeholders are numbered per batch (`prepareBatch`), which is what lets the
 * retry reuse the same masking instead of re-deriving it.
 *
 * @param batch - work items already prepared by `prepareBatch`.
 * @param context - llm, route, prompts, target, retry budget, signal.
 * @returns Translated lines by source text, plus the failures.
 */
export async function translateBatch(batch, context) {
  const { llm, route, system, maxRetries, signal, onUsage, options } = context
  const done = new Map()
  const failures = []
  let pending = batch.slice()
  let lastError

  for (let attempt = 0; attempt <= maxRetries && pending.length > 0; attempt += 1) {
    if (signal?.aborted) throw new Error('cancelled')
    let reply
    try {
      const result = await complete(llm, route, system, buildUserPrompt(pending), signal)
      onUsage?.(result.usage)
      reply = result.text
    } catch (error) {
      if (signal?.aborted) throw error
      lastError = error
      continue
    }
    let translations
    try {
      translations = readReply(reply, pending.length)
    } catch (error) {
      lastError = error
      continue
    }
    const still = []
    pending.forEach((item, index) => {
      const translated = translations[index] ?? ''
      const defect = lineDefect(item, translated, options)
      if (defect !== undefined) {
        lastError = new Error(`${defect} ${JSON.stringify(item.masked.slice(0, 40))}`)
        still.push(item)
        return
      }
      done.set(item.key, item.restore(translated))
    })
    pending = still
    // Once only a couple of lines are missing, one request per line is cheaper
    // than resending the batch they came from.
    if (pending.length > 0 && pending.length <= 2) break
  }

  for (const item of pending) {
    let settled = false
    for (let attempt = 0; attempt <= maxRetries && !settled; attempt += 1) {
      if (signal?.aborted) throw new Error('cancelled')
      try {
        const result = await complete(llm, route, system, buildUserPrompt([item]), signal)
        onUsage?.(result.usage)
        const [translated] = readReply(result.text, 1)
        const defect = lineDefect(item, translated ?? '', options)
        if (defect !== undefined) throw new Error(defect)
        done.set(item.key, item.restore(translated))
        settled = true
      } catch (error) {
        if (signal?.aborted) throw error
        lastError = error
        if (attempt === maxRetries) failures.push({ item, reason: error.message })
      }
    }
  }

  if (pending.length > 0 && failures.length === 0) {
    for (const item of pending) failures.push({ item, reason: lastError?.message ?? 'not translated' })
  }
  return { done, failures }
}

/**
 * Run every batch with a bounded worker pool.
 *
 * @param batches - prepared batches.
 * @param context - translation context plus `concurrency` and `onBatch`.
 * @returns Merged translations and failures.
 */
export async function runBatches(batches, context) {
  const { concurrency, onBatch } = context
  const translations = new Map()
  const failures = []
  let cursor = 0
  const workers = Array.from({ length: Math.min(concurrency, batches.length) }, async () => {
    for (;;) {
      const index = cursor
      cursor += 1
      if (index >= batches.length) return
      if (context.signal?.aborted) throw new Error('cancelled')
      const result = await translateBatch(batches[index], context)
      for (const [key, value] of result.done) translations.set(key, value)
      failures.push(...result.failures)
      await onBatch?.(result, index)
    }
  })
  await Promise.all(workers)
  return { translations, failures }
}

/**
 * Prepare one batch: strip affixes, mask codes, attach the restoration step.
 *
 * The placeholder counter is deliberately shared by the whole batch, as in
 * `text_processor._replace_special_placeholders`, so a code that the model
 * moves across lines fails validation instead of landing on the wrong string.
 *
 * @param batch - pending work items.
 * @param options - `preReplace`, `postReplace`, `symbolRepair`, `sourceLanguage`.
 * @returns The prepared batch.
 */
export function prepareBatch(batch, options = {}) {
  const state = { counter: 0, entries: [] }
  return batch.map((item) => {
    const pre = applyReplacements(item.text, options.preReplace)
    const affixes = stripAffixes(pre)
    // Only the placeholders this line created: the counter is shared by the
    // batch, so slicing from the current end keeps the per-line restoration
    // list from inheriting the lines that came before it.
    const start = state.entries.length
    const masked = maskCodes(affixes.core, state)
    const placeholders = state.entries.slice(start)
    return {
      ...item,
      masked,
      placeholders,
      restore(translated) {
        let out = unmaskCodes(translated, placeholders)
        out = options.postReplace === undefined ? out : applyReplacements(out, options.postReplace)
        if (options.symbolRepair !== false) out = repairSymbols(affixes.core, out)
        return affixes.prefix + out + affixes.suffix
      },
    }
  })
}

/**
 * Pick the model route for this call.
 *
 * @param ctx - plugin context.
 * @param config - resolved config.
 * @param override - a per-call `{provider, model}`.
 * @returns The route to pass to `llm.stream`.
 * @throws When no route can be resolved.
 */
export function resolveRoute(ctx, config, override = {}) {
  const selection = ctx.get?.('agentDefaultModel')?.currentSelection?.()
  const provider = override.provider ?? config.provider ?? selection?.provider
  const model = override.model ?? config.model ?? selection?.model
  if (!isFilledString(provider) || !isFilledString(model)) {
    throw new Error('mtoolTranslate: no model route — set config.provider and config.model, or configure a default model in this profile')
  }
  return {
    provider,
    model,
    reasoningEffort: config.reasoningEffort,
    maxTokens: config.maxTokens,
  }
}

/**
 * Derive the output and cache paths for one input file.
 *
 * @param inputPath - the exported file.
 * @param config - resolved config.
 * @param requested - a per-call override.
 * @returns The absolute output path and the cache path.
 */
export function resolveOutput(inputPath, config, requested) {
  const cachePath = join(dirname(inputPath), CACHE_FILE)
  if (isFilledString(requested)) {
    const path = resolve(requested)
    return { path, cachePath }
  }
  const directory = config.outputDir === undefined ? dirname(inputPath) : resolve(config.outputDir)
  const stem = basename(inputPath, extname(inputPath))
  return { path: join(directory, `${stem}.translated.json`), cachePath }
}

/**
 * Project one result onto the content the model and the UI see.
 *
 * @param value - the tool result.
 * @returns The result content blocks.
 */
export function renderContent(value) {
  const lines = [
    `Translated ${value.totals.translated} of ${value.totals.pending} pending line(s) into ${value.target} with ${value.model}`
      + ` (${value.totals.cached} from cache, ${value.totals.failed} failed).`,
    `Output: ${value.output}`,
  ]
  if (value.totals.remaining > 0) {
    lines.push(`${value.totals.remaining} line(s) still pending — call this tool again with the same path to continue.`)
  }
  if (value.failures.length > 0) {
    lines.push('Failures:')
    for (const failure of value.failures.slice(0, 5)) lines.push(`- ${JSON.stringify(failure.source.slice(0, 80))}: ${failure.reason}`)
  }
  lines.push('Load the output in MTool with 翻译 → 加载翻译文件.')
  return [{ type: 'text', text: lines.join('\n') }]
}

/**
 * The `translate_mtool_file` tool definition.
 *
 * @param ctx - plugin context supplying the model registry.
 * @param config - resolved config.
 * @returns The definition passed to `ctx.tools.register`.
 */
export function defineTranslateTool(ctx, config) {
  return {
    name: 'translate_mtool_file',
    description: 'Translate an MTool game-text export (ManualTransFile.json) into Chinese with this profile\'s model, '
      + 'and write a JSON file that MTool can load back (翻译 → 加载翻译文件). '
      + 'RPG Maker control codes are masked as [P1] placeholders and validated, replies are checked line by line, '
      + 'finished lines are cached, and progress is checkpointed, so a long run can be resumed by calling again with the same path. '
      + 'Add a glossary for character and item names when the user cares about consistent naming.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'Path to the ManualTransFile.json exported by MTool (翻译 → 导出游戏原文).' },
        target: { type: 'string', description: `Target language. Defaults to ${DEFAULT_TARGET}.` },
        glossary: {
          type: 'array',
          items: { type: 'string' },
          description: 'Terminology to enforce, one "原文=>译文" per entry. A path to a JSON file is also accepted.',
        },
        outputPath: { type: 'string', description: 'Where to write the translated JSON. Defaults to <input>.translated.json next to the input.' },
        limit: { type: 'integer', description: 'Translate at most this many lines this call; use it for a cheap sample of a big file.' },
        maxBatches: { type: 'integer', description: 'Stop after this many model requests this call, leaving the rest for a later call.' },
        force: { type: 'boolean', description: 'Re-translate lines that already have a translation instead of skipping them.' },
        dryRun: { type: 'boolean', description: 'Only report how many lines are pending, cached, finished or skipped. Spends no model calls.' },
      },
      required: ['path'],
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          input: { type: 'string' },
          output: { type: 'string' },
          target: { type: 'string' },
          model: { type: 'string' },
          dryRun: { type: 'boolean' },
          done: { type: 'boolean' },
          totals: {
            type: 'object',
            additionalProperties: false,
            properties: {
              entries: { type: 'integer' },
              pending: { type: 'integer' },
              translated: { type: 'integer' },
              cached: { type: 'integer' },
              finished: { type: 'integer' },
              skipped: { type: 'integer' },
              failed: { type: 'integer' },
              remaining: { type: 'integer' },
            },
            required: ['entries', 'pending', 'translated', 'failed', 'remaining'],
          },
          batches: {
            type: 'object',
            additionalProperties: false,
            properties: { total: { type: 'integer' }, run: { type: 'integer' } },
            required: ['total', 'run'],
          },
          usage: {
            type: 'object',
            additionalProperties: false,
            properties: { inputTokens: { type: 'integer' }, outputTokens: { type: 'integer' } },
          },
          durationMs: { type: 'integer' },
          failures: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: { source: { type: 'string' }, reason: { type: 'string' } },
              required: ['source', 'reason'],
            },
          },
        },
        required: ['input', 'target', 'model', 'totals', 'done'],
      },
      render: (_args, value) => renderContent(value),
    },
    timeoutMs: config.timeoutMs,
    // Two runs over one file would race on the same output and cache.
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      return executeTranslate(ctx, config, args, exec.signal)
    },
  }
}

/**
 * One call: read, plan, translate, checkpoint, and report.
 *
 * @param ctx - plugin context.
 * @param config - resolved config.
 * @param args - tool arguments.
 * @param signal - cancellation.
 * @returns The tool result.
 */
export async function executeTranslate(ctx, config, args, signal) {
  const startedAt = Date.now()
  if (!isPlainObject(args) || !isFilledString(args.path)) {
    throw new Error('mtoolTranslate: `path` must name the exported ManualTransFile.json')
  }
  const inputPath = resolve(args.path)
  const target = isFilledString(args.target) ? args.target : config.target
  const limit = Number.isInteger(args.limit) && args.limit > 0 ? args.limit : config.limit
  const force = args.force === true
  const dryRun = args.dryRun === true
  const maxBatches = Number.isInteger(args.maxBatches) && args.maxBatches > 0 ? args.maxBatches : 0

  // Read the file before anything else: a bad path is the likeliest mistake,
  // and its message should not be preempted by a model-route complaint.
  const { entries } = await readManualTrans(inputPath)
  const llm = ctx.get?.('llm')
  if (!dryRun && (llm === undefined || typeof llm.stream !== 'function')) {
    throw new Error('mtoolTranslate: the `llm` service is unavailable in this deployment')
  }
  const route = dryRun ? { provider: 'dry-run', model: 'dry-run' } : resolveRoute(ctx, config, args)
  const glossary = await resolveGlossary(args.glossary ?? config.glossary)
  const exclusion = config.exclusion ?? []

  const { path: outputPath, cachePath } = resolveOutput(inputPath, config, args.outputPath)
  const cache = await readCache(cachePath, target)
  const { items, finished, cached, skipped } = collectWork(entries, { force, limit, cache, target, exclusion })
  const pending = items.filter(item => item.skip === undefined && item.cached === undefined)
  const totals = {
    entries: Object.keys(entries).length,
    pending: pending.length,
    translated: 0,
    cached,
    finished,
    skipped,
    failed: 0,
    remaining: pending.length,
  }

  if (dryRun) {
    return {
      input: inputPath,
      output: outputPath,
      target,
      model: route.model,
      dryRun: true,
      done: pending.length === 0,
      totals,
      batches: { total: buildBatches(pending, config).length, run: 0 },
      usage: { inputTokens: 0, outputTokens: 0 },
      durationMs: Date.now() - startedAt,
      failures: [],
    }
  }

  attachGlossary(pending, glossary)
  const batches = buildBatches(pending, config)
  const toRun = maxBatches > 0 ? batches.slice(0, maxBatches) : batches
  const system = buildSystemPrompt({
    target,
    sourceLanguage: config.sourceLanguage,
    glossary,
    exclusion,
  })
  const prepared = toRun.map(batch => prepareBatch(batch, config))

  for (const item of items) {
    if (item.cached !== undefined) entries[item.key] = item.cached
  }

  const translations = new Map()
  const failures = []
  const usage = { inputTokens: 0, outputTokens: 0 }
  const onUsage = (chunk) => {
    if (chunk === undefined) return
    usage.inputTokens += chunk.inputTokens ?? 0
    usage.outputTokens += chunk.outputTokens ?? 0
  }

  // One writer at a time: batches finish out of order, and two atomic renames
  // of the same file must not interleave.
  let checkpointChain = Promise.resolve()
  const checkpoint = () => {
    checkpointChain = checkpointChain.then(async () => {
      for (const [key, value] of translations) entries[key] = value
      await writeAtomic(outputPath, JSON.stringify(entries, null, 4))
      const merged = new Map(cache)
      for (const [key, value] of translations) merged.set(cacheKey(key, target), value)
      await writeCache(cachePath, target, Object.fromEntries(merged))
    })
    return checkpointChain
  }

  if (prepared.length > 0) {
    const result = await runBatches(prepared, {
      llm,
      route,
      system,
      options: config,
      maxRetries: config.maxRetries,
      concurrency: config.concurrency,
      signal,
      onUsage,
      onBatch: async (batchResult) => {
        for (const [key, value] of batchResult.done) translations.set(key, value)
        failures.push(...batchResult.failures)
        await checkpoint()
      },
    })
    for (const [key, value] of result.translations) translations.set(key, value)
    failures.push(...result.failures)
  }
  await checkpoint()

  totals.translated = translations.size
  totals.failed = failures.length
  totals.remaining = pending.length - translations.size
  return {
    input: inputPath,
    output: outputPath,
    target,
    model: route.model,
    dryRun: false,
    done: totals.remaining === 0,
    totals,
    batches: { total: batches.length, run: toRun.length },
    usage,
    durationMs: Date.now() - startedAt,
    failures: failures.slice(0, 20).map(failure => ({
      source: failure.item.key,
      reason: failure.reason,
    })),
  }
}

/**
 * Register `translate_mtool_file` on the tool registry.
 *
 * @param ctx - plugin context; `tools` is required, `llm` is read per call.
 * @param config - plugin row config, validated here.
 */
export function apply(ctx, config) {
  ctx.tools.register(defineTranslateTool(ctx, resolveConfig(config)))
}
