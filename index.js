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
import { homedir } from 'node:os'
import { basename, dirname, extname, join, resolve } from 'node:path'

import z from '@deepseek-ai/schemastery'

/** Cordis plugin name used by Loader diagnostics. */
export const name = 'mtoolTranslate'

/** The tool registry this plugin contributes to. */
export const inject = ['tools']

/**
 * Settings a user may change: the two knobs that are worth a decision.
 *
 * Everything else is fixed in code — source language is detected from the
 * file, the target language is named per call by the user ("日译中"), the model
 * route follows the profile default, and batch shape, retries, and output
 * location are engineering constants. `Config` and the config tool both build
 * from this table, so the Plugins page and the agent see the same surface.
 */
export const SETTING_FIELDS = {
  reasoningEffort: { kind: 'enum', values: ['off', 'low', 'high', 'max'], default: 'low', description: '推理强度：off 最快，low 术语更稳，high/max 更慢' },
  cacheEnabled: { kind: 'boolean', default: true, description: '保留增量缓存（.mtool-translate-cache.json），关掉后不再读写' },
}

/**
 * Build the Loader schema for one setting.
 *
 * @param field - one `SETTING_FIELDS` entry.
 * @returns The Schemastery schema.
 */
function fieldSchema(field) {
  const description = field.description ?? ''
  if (field.kind === 'string') {
    const schema = z.string().description(description)
    return field.default === undefined ? schema : schema.default(field.default)
  }
  if (field.kind === 'boolean') return z.boolean().default(field.default).description(description)
  if (field.kind === 'enum') {
    return z.union(field.values.map(value => z.const(value))).default(field.default).description(description)
  }
  return z.number().step(1).min(field.min).max(field.max).default(field.default).description(description)
}

/** Loader schema for the plugin row: the form the Plugins page renders. */
export const Config = z.object(Object.fromEntries(
  Object.entries(SETTING_FIELDS).map(([key, field]) => [key, fieldSchema(field)]),
))

/**
 * Validate a partial settings object against `SETTING_FIELDS`.
 *
 * @param values - the candidate settings.
 * @returns The accepted settings.
 * @throws When a key is unknown or a value is out of range.
 */
export function validateSettings(values) {
  if (!isPlainObject(values)) throw new Error('mtoolTranslate: settings must be an object')
  const out = {}
  for (const [key, value] of Object.entries(values)) {
    const field = SETTING_FIELDS[key]
    if (field === undefined) throw new Error(`mtoolTranslate: unknown setting ${JSON.stringify(key)}`)
    if (value === null || value === undefined) continue
    if (field.kind === 'string') {
      if (typeof value !== 'string') throw new Error(`mtoolTranslate: ${key} must be a string`)
    } else if (field.kind === 'boolean') {
      if (typeof value !== 'boolean') throw new Error(`mtoolTranslate: ${key} must be a boolean`)
    } else if (field.kind === 'enum') {
      if (!field.values.includes(value)) throw new Error(`mtoolTranslate: ${key} must be one of ${field.values.join(', ')}`)
    } else {
      if (!Number.isInteger(value) || value < field.min || value > field.max) {
        throw new Error(`mtoolTranslate: ${key} must be an integer in ${field.min}..${field.max}`)
      }
    }
    out[key] = value
  }
  return out
}

/** Target language used when neither config nor the call names one. */
export const DEFAULT_TARGET = '简体中文'

/** Lines one model request may carry. */
export const DEFAULT_BATCH_LINES = 40

/** Source characters one model request may carry. */
export const DEFAULT_BATCH_CHARS = 1600

/** Model requests in flight at once. */
export const DEFAULT_CONCURRENCY = 8

/** Attempts per batch beyond the first. */
export const DEFAULT_MAX_RETRIES = 2

/** Cooperative per-call budget; a big file needs room for many batches. */
export const DEFAULT_TIMEOUT_MS = 900_000

/** Output-token ceiling for one batch reply. */
export const DEFAULT_MAX_TOKENS = 8192

/**
 * Reasoning effort for the routed model. Translation is formatting, not
 * reasoning: `low` measured the same quality band as `high` at a third of the
 * latency, while `off` occasionally lost domain terms ("イベ" became 活动
 * instead of 事件). `off` stays available per call for bulk drafts.
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
  { source: '\\\\$', description: '\\$ gold window' },
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
  { source: '\\{i\\}', description: '{i} combo tag' },
  { source: '\\{/i\\}', description: '{/i} closing tag' },
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
 * Characters that count as "the source language", used to split a line into a
 * translatable core and the code around it (text_processor.py JAPANESE_CHAR_SET).
 */
const JAPANESE_CLASS = '\\u3040-\\u309F\\u30A0-\\u30FF\\u30FB-\\u30FE\\uFF65-\\uFF9F\\u4E00-\\u9FFF\\u3400-\\u4DBF\\u3001-\\u303F\\uFF01-\\uFF5E'

/**
 * The `common` system prompt, verbatim from
 * `resources/prompts/Translate/common_system_zh.txt`.
 */
const COMMON_SYSTEM = [
  '你是一名专业的翻译家，你的任务是把{source_language}文本翻译成{target_language}，逐行翻译，不要合并，原始保留文本中序号、标记符、占位符、换行符、转义符、代码调用过程等特殊内容，保持原来的格式。',
  '',
  '### 翻译原则',
  '忠实准确地进行翻译。原作是伟大艺术作品，允许使用直白或粗俗的描述，不应随意删减或篡改，不应回避或淡化，以保持作品的原貌。',
  '',
  '### 以textarea标签输出译文',
  '<textarea>',
  '1.{target_language}文本',
  '</textarea>',
].join('\n')

/** The `cot` system prompt, verbatim from `cot_system_zh.txt`. */
const COT_SYSTEM = [
  '你是一名专业的翻译家，请你按照以下流程进行翻译：',
  '第一步：初步直译',
  '    将{source_language}文本逐行直译成{target_language}文本，原始保留文本中序号、标记符、占位符、换行符、转义符、代码调用过程等特殊内容，保持原来的格式。',
  '',
  '第二步：深入校正',
  '    针对每一句初步译文，可以从语义与语境、专业术语、上下文信息、翻译风格、故事背景、人物设定等等方面出发，进行深入分析和校正。',
  '',
  '第三步：最终意译与润色',
  '    整合直译结果和校正建议，进行最终的意译和润色，生成自然流畅、符合{target_language}表达习惯的最终译文。',
  '',
  '### 翻译原则',
  '忠实准确地进行翻译。原作是伟大艺术作品，允许使用直白或粗俗的描述，不应随意删减或篡改，不应回避或淡化，以保持作品的原貌。',
  '',
  '### 以textarea标签输出译文',
  '<textarea>',
  '1.{target_language}文本',
  '</textarea>',
].join('\n')

/** `^(\s*)(.*?)(\s*)$` — the whitespace-only affix rule (RE_WHITESPACE_AFFIX). */
const RE_WHITESPACE_AFFIX = /^(\s*)([\s\S]*?)(\s*)$/u

/** The Japanese affix rule (RE_JA_AFFIX): everything outside the Japanese ranges. */
const RE_JA_AFFIX = new RegExp(`^([^${JAPANESE_CLASS}]*)([\\s\\S]*?)([^${JAPANESE_CLASS}]*)$`, 'u')

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
    previousLines: integer('previousLines', 0, 0),
    outputDir: raw.outputDir,
    glossary: raw.glossary,
    // Rows may be `{src, dst, regex}` objects or a plain `{from: to}` map.
    exclusion: raw.exclusion ?? [],
    preReplace: raw.preReplace,
    postReplace: raw.postReplace,
    characters: Array.isArray(raw.characters) ? raw.characters : [],
    worldBuilding: typeof raw.worldBuilding === 'string' ? raw.worldBuilding : '',
    writingStyle: typeof raw.writingStyle === 'string' ? raw.writingStyle : '',
    translationExample: Array.isArray(raw.translationExample) ? raw.translationExample : [],
    targetPlatform: raw.targetPlatform === 'sakura' ? 'sakura' : 'default',
    promptPreset: raw.promptPreset === 'cot' ? 'cot' : 'common',
    // AiNiee exposes these as switches; placeholder/newline/empty checks are
    // always on, these three are opt-in and off by default.
    symbolRepair: boolean('symbolRepair', true),
    cacheEnabled: boolean('cacheEnabled', true),
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
  // MTool's own metadata line must survive untouched; translating it breaks
  // the mode switch on load.
  if (text.startsWith('MTool_Meta_DO_NOT_REMOVE')) return 'mtool-meta'
  if (text.length > MAX_LINE_CHARS) return 'too-long'
  if (/^\d+$/u.test(text)) return 'numeric'
  if (text.trim() === '') return 'empty'
  if (['\n', '\\n', '\r', '\\r'].includes(text.trim())) return 'newline-only'
  if ([...text].every(char => PUNCTUATION.has(char))) return 'punctuation-only'
  const suffix = text.trim().endsWith('.') ? '' : text.trim().slice(text.trim().lastIndexOf('.'))
  if (suffix !== '' && EXCLUDE_SUFFIX.has(suffix.toLowerCase())) return 'file-suffix'
  if (EXCLUDE_PREFIX.some(prefix => text.startsWith(prefix))) return 'code-prefix'
  const excluded = exclusion.some(entry => (entry.regex === true
    ? (entry.pattern?.test(text) ?? false)
    : text.trim() === entry.source.trim()))
  if (excluded) return 'excluded'
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
 * Build one regex, returning `undefined` when the pattern is invalid.
 *
 * @param source - the pattern text.
 * @param flags - regex flags.
 * @returns The regex, or undefined.
 */
export function safeRegex(source, flags = '') {
  try {
    return new RegExp(source, flags)
  } catch {
    return undefined
  }
}

/**
 * Normalize a replacement table into compiled rules.
 *
 * Ported from `text_processor._compile_translation_rules`: a row may be a
 * plain `from -> to` pair or a regex row (`regex: true`); an invalid pattern is
 * dropped instead of aborting the run.
 *
 * @param rules - `{from: to}` map or `[{src, dst, regex}]` rows.
 * @returns Compiled rules.
 */
export function compileReplacements(rules) {
  if (rules === undefined || rules === null) return []
  const rows = []
  const add = (src, dst, regex) => {
    if (!isFilledString(src) || typeof dst !== 'string') return
    rows.push({ src: src.trim(), dst, regex: regex === true })
  }
  if (Array.isArray(rules)) {
    for (const row of rules) {
      if (isPlainObject(row)) add(row.src ?? row.from, row.dst ?? row.to ?? '', row.regex)
    }
  } else if (isPlainObject(rules)) {
    for (const [src, dst] of Object.entries(rules)) add(src, dst, false)
  }
  return rows.map(row => ({ ...row, pattern: row.regex ? safeRegex(row.src) : undefined }))
}

/**
 * Normalize the exclusion list.
 *
 * Ported from `regex.RegexSwitchHelper.normalize_exclusion_rows` +
 * `build_re_pattern`: each row is an exact marker or, with `regex: true`, a
 * pattern.
 *
 * @param rows - strings or `{markers, info, regex}` rows.
 * @returns Compiled entries.
 */
export function compileExclusion(rows) {
  const list = []
  for (const row of rows ?? []) {
    if (typeof row === 'string') {
      if (isFilledString(row)) list.push({ source: row.trim(), info: '', regex: false })
      continue
    }
    if (!isPlainObject(row)) continue
    const source = String(row.markers ?? row.source ?? row.text ?? '').trim()
    if (source === '') continue
    list.push({ source, info: String(row.info ?? row.note ?? ''), regex: row.regex === true })
  }
  return list.map(entry => ({ ...entry, pattern: entry.regex ? safeRegex(entry.source) : undefined }))
}

/**
 * Find the exclusion markers present in one batch.
 *
 * Ported from `prompt_builder.build_ntl_prompt`: only markers that actually
 * occur in the batch are sent, so the model never sees an unrelated list.
 *
 * @param text - the batch's source text.
 * @param exclusion - compiled entries.
 * @returns `[marker, info]` pairs.
 */
export function collectExclusionMatches(text, exclusion) {
  const hits = new Map()
  for (const entry of exclusion ?? []) {
    if (entry.regex === true) {
      if (entry.pattern === undefined) continue
      for (const match of text.matchAll(new RegExp(entry.pattern.source, 'gu'))) {
        if (match[0] !== '') hits.set(match[0], entry.info)
      }
    } else if (text.includes(entry.source)) {
      hits.set(entry.source, entry.info)
    }
  }
  return [...hits.entries()]
}

/**
 * Apply the compiled replacement rules.
 *
 * @param text - one string.
 * @param rules - compiled rules from `compileReplacements`.
 * @returns The replaced text.
 */
export function applyReplacements(text, rules) {
  let out = text
  for (const rule of rules ?? []) {
    if (rule.regex === true) {
      if (rule.pattern === undefined) continue
      out = out.replace(new RegExp(rule.pattern.source, 'gu'), rule.dst)
    } else if (rule.src !== '') {
      out = out.split(rule.src).join(rule.dst)
    }
  }
  return out
}

/**
 * Normalize line breaks to `\n`, remembering what each one was.
 *
 * Ported from `text_processor._normalize_line_endings`: `<br>` tags, CRLF, CR
 * and LF all become one break, and the original form is kept so the finished
 * text can put it back.
 *
 * @param text - one original string.
 * @returns The normalized text and the break kinds in order.
 */
export function normalizeLineEndings(text) {
  const endings = []
  let out = ''
  let index = 0
  while (index < text.length) {
    if (text.slice(index, index + 3).toLowerCase() === '<br') {
      const close = text.indexOf('>', index)
      if (close !== -1) {
        endings.push(text.slice(index, close + 1))
        out += '\n'
        index = close + 1
        continue
      }
    }
    if (text.slice(index, index + 2) === '\r\n') {
      endings.push('\r\n')
      out += '\n'
      index += 2
      continue
    }
    if (text[index] === '\r') {
      endings.push('\r')
      out += '\n'
      index += 1
      continue
    }
    if (text[index] === '\n') {
      endings.push('\n')
      out += '\n'
      index += 1
      continue
    }
    out += text[index]
    index += 1
  }
  return { text: out, endings }
}

/**
 * Whether a fragment is only ASCII letters and spaces.
 *
 * Ported from `text_processor._is_pure_english`.
 *
 * @param text - the fragment.
 * @returns Whether it is pure English.
 */
export function isPureEnglish(text) {
  if (text === '' || text.trim() === '') return false
  return [...text].every(char => (/[A-Za-z]/.test(char) && char.charCodeAt(0) < 0x80) || /\s/.test(char))
}

/**
 * Move bracket edges and short fragments back next to the core.
 *
 * Ported from `text_processor._handle_special_characters`: an opening bracket
 * belongs with the text it opens, a closing bracket with the text it closes, a
 * digit run is a label, and a pure-English fragment is probably part of the
 * sentence rather than decoration.
 *
 * @param prefix - the leading fragment.
 * @param core - the translatable text.
 * @param suffix - the trailing fragment.
 * @returns The adjusted triple.
 */
export function handleSpecialCharacters(prefix, core, suffix) {
  let head = prefix
  let body = core
  let tail = suffix

  if (head.endsWith('[')) { body = `[${body}`; head = head.slice(0, -1) }
  else if (head.endsWith('{')) { body = `{${body}`; head = head.slice(0, -1) }
  else if (head.endsWith('（')) { body = `（${body}`; head = head.slice(0, -1) }
  else if (head.endsWith('(')) { body = `(${body}`; head = head.slice(0, -1) }

  if (tail.startsWith(']')) { body = `${body}]`; tail = tail.slice(1) }
  else if (tail.startsWith('}')) { body = `${body}}`; tail = tail.slice(1) }
  else if (tail.startsWith('）')) { body = `${body}）`; tail = tail.slice(1) }
  else if (tail.startsWith(')')) { body = `${body})`; tail = tail.slice(1) }

  if (tail !== '' && /^\d+$/u.test(tail)) { body += tail; tail = '' }

  if (head !== '' && isPureEnglish(head)) { body = head + body; head = '' }
  if (tail !== '' && isPureEnglish(tail)) { body += tail; tail = '' }

  return { prefix: head, core: body, suffix: tail }
}

/**
 * Split one line into the part worth translating and the code around it.
 *
 * Ported from `text_processor._process_multiline_text`: the Japanese mode keeps
 * everything outside the Japanese ranges as affix; the whitespace mode only
 * keeps surrounding whitespace.
 *
 * @param line - one line.
 * @param jaMode - whether the source is Japanese.
 * @returns The line's parts, or `empty` for a blank line.
 */
export function splitLineAffix(line, jaMode) {
  if (line === '' || line.trim() === '') {
    return { empty: true, whitespace: line, prefix: '', core: '', suffix: '' }
  }
  const match = (jaMode ? RE_JA_AFFIX : RE_WHITESPACE_AFFIX).exec(line)
  if (match === null) return { empty: false, prefix: '', core: line, suffix: '' }

  let prefix = match[1]
  let core = match[2]
  let suffix = match[3]
  const adjusted = handleSpecialCharacters(prefix, core, suffix)
  prefix = adjusted.prefix
  core = adjusted.core
  suffix = adjusted.suffix

  if (core.trim() === '' && line.trim() !== '') {
    prefix = ''
    core = line
    suffix = ''
  }

  if (/^\d+$/u.test(prefix.trim())) {
    const leading = prefix.slice(0, prefix.length - prefix.trimStart().length)
    core = prefix.slice(leading.length) + core
    prefix = leading
  }
  if (/^\d+$/u.test(suffix.trim())) {
    const trailing = suffix.slice(suffix.trimEnd().length)
    core = core + suffix.slice(0, suffix.length - trailing.length)
    suffix = trailing
  }

  return { empty: false, prefix, core, suffix }
}

/**
 * Prepare one string for the model.
 *
 * Ported from `text_processor.strip_and_record_affixes`: normalize the breaks,
 * drop blank lines from the payload, and record every line's affixes so the
 * finished text can be rebuilt exactly.
 *
 * @param text - one original string.
 * @param options - `jaMode`.
 * @returns The payload plus what `restoreText` needs.
 */
export function prepareText(text, options = {}) {
  const normalized = normalizeLineEndings(text)
  const lines = normalized.text.split('\n')
  const infos = lines.map(line => splitLineAffix(line, options.jaMode === true))
  const payload = infos.filter(info => info.empty === false).map(info => info.core).join('\n')
  return { payload, infos, endings: normalized.endings }
}

/**
 * Rebuild one string from its translation.
 *
 * Ported from `text_processor._restore_multiline_text`.
 *
 * @param translated - the translated payload.
 * @param prepared - the value `prepareText` returned.
 * @returns The text with affixes and original line breaks back.
 */
export function restoreText(translated, prepared) {
  const lines = String(translated ?? '').split('\n')
  const out = []
  let index = 0
  for (const info of prepared.infos) {
    if (info.empty === true) {
      out.push(info.whitespace)
      continue
    }
    out.push(info.prefix + (lines[index] ?? '') + info.suffix)
    index += 1
  }
  let joined = out[0] ?? ''
  for (let next = 1; next < out.length; next += 1) {
    joined += (prepared.endings[next - 1] ?? '\n') + out[next]
  }
  return joined
}

/**
 * Guard a leading `1.` so the model does not read it as a list number.
 *
 * Ported from `text_processor.digital_sequence_preprocessing`: `1.标题`
 * becomes `【1】标题` before the request.
 *
 * @param text - the payload.
 * @returns The payload with the label bracketed.
 */
export function digitalSequence(text) {
  return text.replace(/^(\d+)\./u, '【$1】')
}

/**
 * Undo `digitalSequence`.
 *
 * @param text - the translated payload.
 * @returns The label restored.
 */
export function digitalSequenceRestore(text) {
  return text.replace(/^【(\d+)】/u, '$1.')
}

/**
 * Strip control codes that sit at the very edges of the payload.
 *
 * Ported from `text_processor._process_affixes`: repeated prefix matches are
 * removed from the front, and matches that end exactly at the end are removed
 * from the back. Codes inside the text are handled later by `maskCodes`.
 *
 * @param payload - the prepared payload.
 * @returns The trimmed text and the codes to put back.
 */
export function stripCodeAffixes(payload) {
  const prefixes = []
  const suffixes = []
  let text = payload
  for (const { pattern } of CODE_PATTERNS) {
    for (;;) {
      pattern.lastIndex = 0
      const match = pattern.exec(text)
      if (match === null || match.index !== 0 || match[0] === '') break
      prefixes.push(match[0])
      text = text.slice(match[0].length)
    }
  }
  for (const { pattern } of CODE_PATTERNS) {
    for (;;) {
      // `matchAll` starts at the pattern's `lastIndex`, which an earlier `exec`
      // may have left mid-string; without this reset a trailing code is skipped.
      pattern.lastIndex = 0
      let best = null
      for (const match of text.matchAll(pattern)) {
        if (match[0] !== '' && match.index + match[0].length === text.length) best = match
      }
      if (best === null) break
      suffixes.unshift(best[0])
      text = text.slice(0, best.index)
    }
  }
  return { text, prefixes, suffixes }
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
      // Sakura's local models take runs of ↓ instead of bracketed tokens.
      const placeholder = state.sakura === true ? '↓'.repeat(state.counter) : `[P${state.counter}]`
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
  const {
    target,
    sourceLanguage,
    glossary = [],
    exclusion = [],
    batchText = '',
    characters = [],
    worldBuilding = '',
    writingStyle = '',
    translationExample = [],
    preset = 'common',
  } = options
  let out = (preset === 'cot' ? COT_SYSTEM : COMMON_SYSTEM)
    .replaceAll('{source_language}', sourceLanguage)
    .replaceAll('{target_language}', target)

  // Appended in the order `generate_prompt` appends the project tables.
  if (glossary.length > 0) {
    out += '\n###术语表\n原文|译文|备注'
    for (const entry of glossary) out += `\n${entry.source}|${entry.target}| `
  }
  const nonTranslate = collectExclusionMatches(batchText, exclusion)
  if (nonTranslate.length > 0) {
    out += '\n###禁翻表，以下原文保持不变\n原文|备注'
    for (const [marker, info] of nonTranslate) out += `\n${marker}|${info}`
  }
  const present = characters.filter((character) => {
    const name = character.original_name ?? character.name ?? ''
    return isFilledString(name) && batchText.includes(name)
  })
  if (present.length > 0) {
    out += '\n###角色介绍'
    for (const character of present) {
      out += `\n【${character.original_name ?? character.name}】`
      const fields = [
        ['译名', character.translated_name],
        ['性别', character.gender],
        ['年龄', character.age],
        ['性格', character.personality],
        ['说话方式', character.speech_style],
        ['补充信息', character.additional_info],
      ]
      for (const [label, value] of fields) {
        if (isFilledString(value)) out += `\n- ${label}：${value}`
      }
      out += '\n'
    }
  }
  if (isFilledString(worldBuilding)) out += `\n###背景设定\n${worldBuilding}\n`
  if (isFilledString(writingStyle)) out += `\n###翻译风格\n${writingStyle}\n`
  if (translationExample.length > 0) {
    out += '\n###翻译示例\n'
    translationExample.forEach((pair, index) => {
      if (index > 0) out += '\n'
      out += `  -原文${index + 1}：${pair.src ?? pair.original ?? ''}\n  -译文${index + 1}：${pair.dst ?? pair.translated ?? ''}`
    })
  }
  return out
}

/**
 * Compose the numbered source block.
 *
 * Ported from `prompt_builder.build_source_text`: a single-line item is `N.text`,
 * and a multi-line item becomes a bracketed block whose rows carry a
 * `N.total-sub` sub-number, so a text box cannot be mistaken for several items.
 *
 * @param items - work items carrying `masked`.
 * @returns The numbered source text.
 */
export function buildSourceText(items) {
  return items.map((item, index) => {
    const number = index + 1
    const text = String(item.masked ?? '')
    if (!text.includes('\n')) return `${number}.${text}`
    const lines = text.split('\n')
    const total = lines.length
    const rows = lines.map((line, sub) => {
      // AiNiee drops one trailing space when it is the only trailing whitespace.
      const adjusted = /.*[^ ] $/u.test(line) ? line.slice(0, -1) : line
      return `"${number}.${total - sub}.,${adjusted}",`
    })
    return `${number}.[\n${rows.join('\n').replace(/,$/u, '')}\n]`
  }).join('\n')
}

/**
 * Read one bracketed multi-line block back into its lines.
 *
 * Ported from `response_extractor.extract_multiline_content`, reduced to the
 * row form this plugin emits: `"N.k.,text",`.
 *
 * @param body - the block body including its brackets.
 * @returns The item's lines joined by a newline.
 */
export function parseBracketBlock(body) {
  const inner = body.slice(1, -1)
  const rows = []
  for (const raw of inner.split('\n')) {
    let line = raw.trim()
    if (line === '') continue
    line = line.replace(/^["“]/u, '')
    line = line.replace(/["”],?$/u, '')
    line = line.replace(/^\d+\.\d+\.,?\s*/u, '')
    rows.push(line)
  }
  if (rows.length === 1 && /["“”]\s*,\s*["“”]/u.test(inner)) {
    return inner.split(/["“”]\s*,\s*["“”]/u).map(part => part
      .replace(/^["“]/u, '')
      .replace(/["”],?$/u, '')
      .replace(/^\d+\.\d+\.,?\s*/u, '')
      .trim()).join('\n')
  }
  return rows.join('\n')
}

/**
 * Compose the user message: optional context, the task line, and the numbered
 * source inside a `<textarea>`, exactly as `generate_prompt` builds it.
 *
 * @param batch - work items carrying `masked`.
 * @param options - `previous` source lines, used as context.
 * @returns The user message.
 */
export function buildUserPrompt(batch, options = {}) {
  const numbered = buildSourceText(batch)
  const previous = options.previous ?? []
  const head = previous.length > 0
    ? `###上文内容\n<previous>\n${previous.join('\n')}\n</previous>\n`
    : ''
  return `${head} ###这是你接下来的翻译任务，原文文本如下\n<textarea>\n${numbered}\n</textarea>`
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
    // `base_checks.check_dict_order`: a reply whose numbers are not 1..n in
    // order has slid a line, so every later mapping would be wrong.
    if (Number(match[1]) !== items.length + 1) {
      throw new Error(`【行数错误】 - 出现错行串行 (期望 ${items.length + 1}，实际 ${match[1]})`)
    }
    const payload = match[2].trim()
    items.push(payload.startsWith('[') && payload.endsWith(']') ? parseBracketBlock(payload) : payload)
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
      // A count mismatch means every later line would be attributed to the
      // wrong source, so this is a retry, not a truncation.
      firstError ??= new Error(`reply had ${items.length} lines, expected ${expected}`)
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

  // Stage 4 of `repair_text_symbols`: a model that quotes every translated
  // line in a multi-line block, when the source only quotes the block.
  target = adjustParagraphQuotes(sourceCore, target)

  return leading + target + trailing
}

/**
 * Drop the per-line straight quotes a model adds to a multi-line block.
 *
 * Ported from `text_symbol_repair.check_and_adjust_quotes`: a line keeps its
 * quote only when the matching source line starts or ends with one.
 *
 * @param original - the source text.
 * @param translated - the translated text.
 * @returns The translated text with stray line quotes removed.
 */
export function adjustParagraphQuotes(original, translated) {
  const originalLines = original.split('\n')
  const translatedLines = translated.split('\n')
  if (originalLines.length !== translatedLines.length) return translated
  return translatedLines.map((line, index) => {
    const sourceLine = originalLines[index] ?? ''
    let out = line
    if (out.length >= 2 && out.startsWith('"')) {
      const first = sourceLine[0] ?? ''
      if (!['"', '“', '「', "'"].includes(first)) out = out.slice(1)
    }
    if (out.length >= 2 && out.endsWith('"')) {
      const last = sourceLine[sourceLine.length - 1] ?? ''
      if (!['"', '”', '」', "'"].includes(last)) out = out.slice(0, -1)
    }
    return out
  }).join('\n')
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
export async function translateBatch(preparedBatch, context) {
  const { llm, route, maxRetries, signal, onUsage, options } = context
  const { items, previous, system } = preparedBatch
  const done = new Map()
  const failures = []
  const defects = new Map()
  let pending = items.slice()
  let lastError

  for (let attempt = 0; attempt <= maxRetries && pending.length > 0; attempt += 1) {
    if (signal?.aborted) throw new Error('cancelled')
    let reply
    try {
      const result = await complete(llm, route, system, buildUserPrompt(pending, { previous }), signal)
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
        // Record the defect against its own line: a batch-level retry used to
        // report the last defect for every line, which mislabelled the cause.
        defects.set(item.key, defect)
        lastError = new Error(defect)
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

  if (pending.length > 0) {
    const reported = new Set(failures.map(failure => failure.item.key))
    for (const item of pending) {
      if (reported.has(item.key)) continue
      failures.push({ item, reason: defects.get(item.key) ?? lastError?.message ?? 'not translated' })
    }
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
  const state = { counter: 0, entries: [], sakura: options.targetPlatform === 'sakura' }
  const jaMode = isJapaneseSource(options.sourceLanguage)
  const items = batch.map((item) => {
    const pre = applyReplacements(item.text, options.preReplaceRules ?? [])
    const prepared = prepareText(pre, { jaMode })
    const stripped = stripCodeAffixes(prepared.payload)
    const payload = digitalSequence(stripped.text)
    // Only the placeholders this line created: the counter is shared by the
    // batch, so slicing from the current end keeps the per-line restoration
    // list from inheriting the lines that came before it.
    const start = state.entries.length
    const masked = maskCodes(payload, state)
    const placeholders = state.entries.slice(start)
    const sourceCore = digitalSequenceRestore(payload)
    return {
      ...item,
      masked,
      placeholders,
      restore(translated) {
        // Mirror `text_processor.restore_all`, in the same order.
        let out = unmaskCodes(translated, placeholders)
        out = stripped.prefixes.join('') + out + stripped.suffixes.join('')
        if (options.postReplaceRules !== undefined) out = applyReplacements(out, options.postReplaceRules)
        out = digitalSequenceRestore(out)
        if (options.symbolRepair !== false) out = repairSymbols(sourceCore, out)
        return restoreText(out, prepared)
      },
    }
  })
  const terms = new Map()
  for (const item of batch) {
    for (const entry of item.glossary ?? []) terms.set(entry.source, entry.target)
  }
  const system = buildSystemPrompt({
    target: options.target,
    sourceLanguage: options.sourceLanguage,
    glossary: [...terms].map(([source, target]) => ({ source, target })),
    exclusion: options.exclusion ?? [],
    batchText: batch.map(item => item.text).join('\n'),
    characters: options.characters ?? [],
    worldBuilding: options.worldBuilding ?? '',
    writingStyle: options.writingStyle ?? '',
    translationExample: options.translationExample ?? [],
    preset: options.promptPreset ?? 'common',
  })
  return { previous: options.previous ?? [], items, system }
}

/**
 * Whether the configured source language is Japanese.
 *
 * @param language - the config value.
 * @returns Whether the Japanese affix rule applies.
 */
export function isJapaneseSource(language) {
  if (typeof language !== 'string') return true
  return ['ja', 'japanese', '日语', '日文', '日本語', '日本语'].includes(language.trim().toLowerCase())
}

/**
 * Guess a file's source language from a sample of its pending lines.
 *
 * Kana wins over Han so a Japanese file is never mistaken for Chinese just
 * because a sample happened to hold only kanji. The answer feeds the prompt's
 * `{source_language}` slot and the Japanese affix rule.
 *
 * @param texts - the source strings to sample.
 * @returns A language name; `英语` when nothing matches.
 */
export function detectSourceLanguage(texts) {
  const sample = texts.slice(0, 500)
  let kana = 0
  let hangul = 0
  let cyrillic = 0
  let han = 0
  for (const text of sample) {
    if (/[\u3041-\u3096\u30a0-\u30ff]/u.test(text)) kana += 1
    else if (/[\uac00-\ud7af]/u.test(text)) hangul += 1
    else if (/[\u0400-\u04ff]/u.test(text)) cyrillic += 1
    else if (/[\u4e00-\u9fff]/u.test(text)) han += 1
  }
  if (kana > 0) return '日语'
  if (hangul > 0) return '韩语'
  if (cyrillic > 0) return '俄语'
  if (han > 0) return '中文'
  return '英语'
}

/**
 * Whether a language name means Chinese.
 *
 * @param language - a language name or code.
 * @returns Whether it names Chinese.
 */
export function isChineseLanguage(language) {
  return typeof language === 'string' && /中文|汉语|漢語|chinese|^zh/iu.test(language.trim())
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
    reasoningEffort: override.reasoningEffort ?? config.reasoningEffort,
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
      + ` (${value.totals.cached} from cache, ${value.totals.failed} failed, ${(value.durationMs / 1000).toFixed(1)}s).`,
    value.source === undefined ? undefined : `Detected source language: ${value.source}`,
    `Output: ${value.output}`,
    value.note,
  ].filter(line => line !== undefined)
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
export function defineTranslateTool(ctx, config, runtime) {
  return {
    name: 'translate_mtool_file',
    description: 'Translate an MTool game-text export (ManualTransFile.json) with this profile\'s model, '
      + 'and write a JSON file that MTool can load back (翻译 → 加载翻译文件). '
      + 'The source language is detected from the file; the target defaults to 简体中文 and should be set explicitly when the user names a pair such as 日译中. '
      + 'RPG Maker control codes are masked as [P1] placeholders and validated, replies are checked line by line, '
      + 'finished lines are cached, and progress is checkpointed, so a long run can be resumed by calling again with the same path. '
      + 'Add a glossary for character and item names when the user cares about consistent naming.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'Path to the ManualTransFile.json exported by MTool (翻译 → 导出游戏原文).' },
        target: { type: 'string', description: `Target language named by the user, such as 简体中文, 繁體中文, or English. Defaults to ${DEFAULT_TARGET} when the user names none.` },
        reasoningEffort: {
          type: 'string',
          enum: ['off', 'low', 'high', 'max'],
          description: 'Reasoning effort for this call only: off is fastest and is what plain translation needs; high/max cost more time and tokens. Defaults to the plugin setting.',
        },
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
          source: { type: 'string' },
          note: { type: 'string' },
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
      return executeTranslate(ctx, config, args, exec.signal, runtime)
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
export async function executeTranslate(ctx, baseConfig, args, signal, runtime) {
  const startedAt = Date.now()
  // Settings saved from the Plugins page or the config tool live beside the
  // profile, not in the Loader row, so they apply without a restart.
  const override = runtime === undefined ? {} : await readStoredSettings(runtime.stateDir)
  const config = { ...baseConfig, ...override }
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
  const exclusion = compileExclusion(config.exclusion)

  const { path: outputPath, cachePath } = resolveOutput(inputPath, config, args.outputPath)
  const cache = config.cacheEnabled ? await readCache(cachePath, target) : new Map()
  const { items, finished, cached, skipped } = collectWork(entries, { force, limit, cache, target, exclusion })
  const pending = items.filter(item => item.skip === undefined && item.cached === undefined)
  // The file decides its own source language; the caller only names the target.
  const source = detectSourceLanguage(pending.map(item => item.key))
  const sameLanguage = isChineseLanguage(source) && isChineseLanguage(target)
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
      source,
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

  if (sameLanguage) {
    // Translating Chinese into Chinese would spend tokens to echo the file.
    return {
      input: inputPath,
      output: outputPath,
      target,
      source,
      model: route.model,
      dryRun: false,
      done: true,
      note: `${source}文本译成${target}没有意义，已跳过：文件看起来已经是目标语言。`,
      totals: { ...totals, pending: 0, remaining: 0, skipped: totals.skipped + pending.length },
      batches: { total: 0, run: 0 },
      usage: { inputTokens: 0, outputTokens: 0 },
      durationMs: Date.now() - startedAt,
      failures: [],
    }
  }

  attachGlossary(pending, glossary)
  const batches = buildBatches(pending, config)
  const toRun = maxBatches > 0 ? batches.slice(0, maxBatches) : batches
  const compiled = {
    ...config,
    target,
    sourceLanguage: source,
    exclusion,
    preReplaceRules: compileReplacements(config.preReplace),
    postReplaceRules: compileReplacements(config.postReplace),
  }
  const prepared = toRun.map((batch, index) => {
    const previous = config.previousLines > 0 && index > 0
      ? batches[index - 1].slice(-config.previousLines).map(item => item.text)
      : []
    return prepareBatch(batch, { ...compiled, previous })
  })

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
      if (config.cacheEnabled === false) return
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
      options: compiled,
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
    source,
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
 * Project a resolved config onto the settable fields.
 *
 * @param config - a resolved (or partial) config.
 * @returns The settable values.
 */
export function pickSettings(config) {
  const out = {}
  for (const key of Object.keys(SETTING_FIELDS)) {
    if (config?.[key] !== undefined) out[key] = config[key]
  }
  return out
}

/** Directory under `DSH_HOME` that holds this plugin's own settings file. */
export const STATE_DIR_NAME = 'mtool-translate'

/** File the Plugins page and `mtool_translate_config` read and write. */
export const SETTINGS_FILE = 'settings.json'

/** Route prefix the Plugins page calls; it sits under `/api`. */
export const ROUTE_PREFIX = '/api/mtool-translate'

/**
 * Absolute state directory for this plugin: `<DSH_HOME>/mtool-translate`.
 *
 * Same rule as the imagegen plugin, so both live beside the profile data and
 * neither depends on the process working directory.
 *
 * @param env - environment to read `DSH_HOME` from.
 * @param home - home directory used when `DSH_HOME` states none.
 * @returns The absolute state directory.
 */
export function resolveStateDir(env = process.env, home = homedir()) {
  const configured = env.DSH_HOME
  const base = isFilledString(configured) ? configured : join(home, '.dsh')
  return join(base, STATE_DIR_NAME)
}

/** The message of a thrown value, whatever was thrown. */
function messageOf(error) {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Read the settings this page or the tool saved.
 *
 * Unknown keys are dropped instead of rejecting the file, so a setting removed
 * by a later version cannot lock the page out of every saved value.
 *
 * @param stateDir - the plugin's state directory.
 * @returns The stored settings; empty when none exist.
 */
export async function readStoredSettings(stateDir) {
  let text
  try {
    text = await readFile(join(stateDir, SETTINGS_FILE), 'utf8')
  } catch (error) {
    if (error.code === 'ENOENT') return {}
    throw error
  }
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch {
    return {}
  }
  if (!isPlainObject(parsed)) return {}
  const known = {}
  for (const [key, value] of Object.entries(parsed)) {
    if (SETTING_FIELDS[key] !== undefined) known[key] = value
  }
  try {
    return validateSettings(known)
  } catch {
    return {}
  }
}

/**
 * Persist settings, replacing the file atomically.
 *
 * @param stateDir - the plugin's state directory.
 * @param values - the complete settings to store.
 */
export async function writeStoredSettings(stateDir, values) {
  await writeAtomic(join(stateDir, SETTINGS_FILE), `${JSON.stringify(values, null, 2)}\n`)
}

/** A JSON response the Plugins page can read. */
function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  })
}

/** Read a JSON body, or undefined when it is absent or malformed. */
async function readJsonBody(request) {
  try {
    return await request.json()
  } catch {
    return undefined
  }
}

/**
 * The two routes the Plugins page calls.
 *
 * They sit under `/api`, where the connection service has already applied its
 * host/origin fence and browser authentication, so neither accepts a path or
 * touches a caller-supplied location.
 *
 * @param runtime - the plugin runtime carrying the state directory.
 * @returns The routes to register on the connection's fetch table.
 */
export function buildSettingsRoutes(runtime) {
  const status = async () => ({
    settings: pickSettings(resolveConfig(await readStoredSettings(runtime.stateDir))),
    fields: Object.entries(SETTING_FIELDS).map(([key, field]) => ({ key, ...field })),
  })
  return [
    {
      path: `${ROUTE_PREFIX}/status`,
      methods: ['GET'],
      requestBody: 'buffered',
      fetch: async () => {
        try {
          return jsonResponse({ ok: true, ...(await status()) })
        } catch (error) {
          return jsonResponse({ ok: false, error: messageOf(error) }, 500)
        }
      },
    },
    {
      path: `${ROUTE_PREFIX}/update`,
      methods: ['POST'],
      requestBody: 'buffered',
      fetch: async (request) => {
        try {
          const body = await readJsonBody(request)
          const wanted = isPlainObject(body?.settings) ? body.settings : {}
          const unset = Array.isArray(body?.unset)
            ? body.unset.filter(key => SETTING_FIELDS[key] !== undefined)
            : []
          if (Object.keys(wanted).length === 0 && unset.length === 0) {
            return jsonResponse({ ok: false, error: 'a JSON body with settings or unset is required' }, 400)
          }
          const values = validateSettings(wanted)
          const merged = { ...(await readStoredSettings(runtime.stateDir)), ...values }
          for (const key of unset) delete merged[key]
          await writeStoredSettings(runtime.stateDir, merged)
          return jsonResponse({ ok: true, ...(await status()) })
        } catch (error) {
          return jsonResponse({ ok: false, error: messageOf(error) }, 400)
        }
      },
    },
  ]
}

/**
 * Mount the Plugins page routes on the Web connection.
 *
 * The child fiber waits for a Web composition instead of declaring the service
 * on the plugin itself: `connection` is absent from a headless profile, where
 * the tools must still register.
 *
 * @param ctx - plugin context.
 * @param runtime - the plugin runtime.
 */
function mountRoutes(ctx, runtime) {
  ctx.inject(['connection'], (scope) => {
    scope.effect(() => {
      const disposers = buildSettingsRoutes(runtime).map(route => scope.connection.fetch.register(route))
      return () => Promise.all(disposers.map(dispose => dispose())).then(() => undefined)
    }, 'mtoolTranslate: Plugins-page routes')
  })
}

/**
 * The `mtool_translate_config` tool definition.
 *
 * @param ctx - plugin context.
 * @param runtime - the plugin runtime carrying the state directory.
 * @returns The definition passed to `ctx.tools.register`.
 */
export function defineConfigTool(ctx, runtime) {
  return {
    name: 'mtool_translate_config',
    description: 'Read or change the MTool translation plugin\'s settings (reasoning effort, batch size, concurrency, model route, and the rest). '
      + 'Call it with no settings to see the current values; call it with a partial object to save changes. '
      + 'Values apply to the next translate_mtool_file call, and the same settings are editable on the plugin page.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        settings: {
          type: 'object',
          additionalProperties: true,
          description: 'Partial settings to persist, for example {"concurrency": 6, "reasoningEffort": "high"}. Omit to read the current values.',
        },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          applied: { type: 'boolean' },
          settings: { type: 'object', additionalProperties: true },
          fields: { type: 'array', items: { type: 'object', additionalProperties: true } },
        },
        required: ['applied', 'settings'],
      },
      render: (_args, value) => {
        const lines = [value.applied ? 'Settings saved; the next translation uses them.' : 'Current MTool translation settings:']
        for (const [key, setting] of Object.entries(value.settings)) {
          lines.push(`- ${key}: ${JSON.stringify(setting)}  ${SETTING_FIELDS[key]?.description ?? ''}`.trimEnd())
        }
        lines.push('')
        lines.push('Settable: ' + Object.keys(SETTING_FIELDS).join(', '))
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    isConcurrencySafe: () => false,
    async execute(args) {
      if (args?.settings === undefined) {
        return {
          applied: false,
          settings: pickSettings(resolveConfig(await readStoredSettings(runtime.stateDir))),
          fields: Object.entries(SETTING_FIELDS).map(([key, field]) => ({ key, ...field })),
        }
      }
      const values = validateSettings(args.settings)
      const merged = { ...(await readStoredSettings(runtime.stateDir)), ...values }
      await writeStoredSettings(runtime.stateDir, merged)
      return {
        applied: true,
        settings: pickSettings(resolveConfig(merged)),
        fields: Object.entries(SETTING_FIELDS).map(([key, field]) => ({ key, ...field })),
      }
    },
  }
}

/**
 * Register the tools, and mount the Plugins-page routes on a Web composition.
 *
 * @param ctx - plugin context; `tools` is required, `llm` is read per call.
 * @param config - plugin row config, validated here.
 */
export function apply(ctx, config) {
  const runtime = { stateDir: resolveStateDir() }
  ctx.tools.register(defineTranslateTool(ctx, resolveConfig(config), runtime))
  ctx.tools.register(defineConfigTool(ctx, runtime))
  mountRoutes(ctx, runtime)
}
