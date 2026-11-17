/**
 * Self-test for `dsh-mtool-translate`: no network, no dependencies.
 *
 *   node smoke.mjs
 *
 * Covers the ported AiNiee layers (code masking, skip rules, reply extraction,
 * line validation, symbol repair) and then drives the real `executeTranslate`
 * against a stub `llm` service that answers in the ported `<textarea>` + `n.`
 * protocol, so the checkpoint, cache, retry and file-writing paths are
 * exercised without spending a model call.
 */
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  buildBatches,
  buildSourceText,
  cacheKey,
  CODE_PATTERNS,
  collectWork,
  countNewlines,
  digitalSequence,
  digitalSequenceRestore,
  executeTranslate,
  extractTranslation,
  isPending,
  lineDefect,
  looksLikeOriginal,
  maskCodes,
  normalizeLineEndings,
  parseJsonReply,
  prepareBatch,
  prepareText,
  readReply,
  repairSymbols,
  resolveConfig,
  restoreText,
  skipReason,
  stripCodeAffixes,
  unmaskCodes,
} from './index.js'

let failures = 0
let checks = 0

function check(name, fn) {
  checks += 1
  try {
    fn()
    console.log(`  ok   ${name}`)
  } catch (error) {
    failures += 1
    console.error(`  FAIL ${name}\n       ${error.message}`)
  }
}

async function checkAsync(name, fn) {
  checks += 1
  try {
    await fn()
    console.log(`  ok   ${name}`)
  } catch (error) {
    failures += 1
    console.error(`  FAIL ${name}\n       ${error.message}`)
  }
}

/**
 * A stand-in `llm` service answering in the ported protocol: it reads the
 * numbered source lines and writes them back inside `<textarea>`.
 */
function stubLlm(translate, options = {}) {
  const calls = []
  return {
    calls,
    stream(request) {
      const user = request.messages[0].content[0].text
      // The user message wraps the source in <textarea>, like AiNiee's does.
      const body = /<textarea>([\s\S]*)<\/textarea>/u.exec(user)?.[1] ?? user
      const sourceItems = extractTranslation(body)
      calls.push(sourceItems)
      const translated = sourceItems.map((text, index) => translate(text, index, calls.length))
      const reply = `<textarea>\n${buildSourceText(translated.map(masked => ({ masked })))}\n</textarea>`
      return (async function* () {
        yield { type: 'text-delta', index: 0, text: options.wrap === undefined ? reply : options.wrap(reply) }
        yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } }
        yield { type: 'finish', reason: { kind: 'stop' } }
      })()
    },
  }
}

function stubCtx(llm) {
  const services = new Map()
  if (llm !== undefined) services.set('llm', llm)
  return { get: name => services.get(name) }
}

const keep = text => `译${text}`
const config = resolveConfig({ provider: 'stub', model: 'stub-model' })

console.log('ported code masking (regex.json + text_processor.py)')
check('the pattern table covers the common RPG Maker codes', () => {
  assert.equal(CODE_PATTERNS.length >= 45, true)
  assert.equal(CODE_PATTERNS.every(entry => entry.pattern instanceof RegExp), true)
})
check('codes become [Pn] placeholders and restore in place', () => {
  const state = { counter: 0, entries: [] }
  const masked = maskCodes('\\C[3]勇者\\C[0]よ\\n', state)
  assert.equal(masked, '[P1]勇者[P2]よ[P3]')
  assert.equal(unmaskCodes(masked, state.entries), '\\C[3]勇者\\C[0]よ\\n')
})
check('composite codes are not split by the generic rule', () => {
  const state = { counter: 0, entries: [] }
  assert.equal(maskCodes('\\SE[se_1]', state), '[P1]')
  assert.equal(maskCodes('\\N[3]', state), '[P2]')
})
check('the counter is shared across a batch, so moving a code across lines fails', () => {
  const state = { counter: 0, entries: [] }
  maskCodes('\\C[1]甲', state)
  maskCodes('\\C[2]乙', state)
  assert.deepEqual(state.entries.map(entry => entry.placeholder), ['[P1]', '[P2]'])
})
check('affixes outside the Japanese core are stripped and restored', () => {
  const prepared = prepareText('[SFX]\\C[3]こんにちは！', { jaMode: true })
  assert.equal(prepared.payload, 'こんにちは！')
  assert.equal(prepared.infos[0].prefix, '[SFX]\\C[3]')
  assert.equal(restoreText('你好！', prepared), '[SFX]\\C[3]你好！')
})
check('a pure-English affix stays in the translatable text', () => {
  const prepared = prepareText('abc こんにちは', { jaMode: true })
  assert.equal(prepared.payload, 'abc こんにちは')
  assert.equal(prepared.infos[0].prefix, '')
})
check('line breaks are normalized and restored in their original form', () => {
  const prepared = prepareText('おはよう\r\nこんばんは', { jaMode: true })
  assert.equal(prepared.payload, 'おはよう\nこんばんは')
  assert.equal(restoreText('早上好\n晚上好', prepared), '早上好\r\n晚上好')
  assert.equal(normalizeLineEndings('a<br>b').text, 'a\nb')
})
check('blank lines are dropped from the payload and put back', () => {
  const prepared = prepareText('一行目\n\n二行目', { jaMode: true })
  assert.equal(prepared.payload, '一行目\n二行目')
  assert.equal(restoreText('甲\n乙', prepared), '甲\n\n乙')
})
check('a leading list number is bracketed so the reply keeps numbering clean', () => {
  assert.equal(digitalSequence('1.これは'), '【1】これは')
  assert.equal(digitalSequenceRestore('【1】这是'), '1.这是')
})
check('a multi-line item becomes a bracketed sub-numbered block and comes back', () => {
  const items = [
    { masked: '一行目' },
    { masked: 'スキル１番は［攻撃］コマンドを\n選択したときに使用されます。' },
  ]
  const source = buildSourceText(items)
  assert.equal(
    source,
    '1.一行目\n2.[\n"2.2.,スキル１番は［攻撃］コマンドを",\n"2.1.,選択したときに使用されます。"\n]',
  )
  assert.deepEqual(extractTranslation(source), [
    '一行目',
    'スキル１番は［攻撃］コマンドを\n選択したときに使用されます。',
  ])
})
check('edge codes are stripped from the payload and reattached', () => {
  const stripped = stripCodeAffixes('\\C[3]勇者\\C[0]')
  assert.equal(stripped.text, '勇者')
  assert.deepEqual(stripped.prefixes, ['\\C[3]'])
  assert.deepEqual(stripped.suffixes, ['\\C[0]'])
})

console.log('ported skip rules (general_text_filter.py)')
check('the obvious non-translatable shapes are skipped', () => {
  assert.equal(skipReason('123'), 'numeric')
  assert.equal(skipReason('   '), 'empty')
  assert.equal(skipReason('\\n'), 'newline-only')
  assert.equal(skipReason('……'), 'punctuation-only')
  assert.equal(skipReason('img/pictures/face.png'), 'file-suffix')
  assert.equal(skipReason('EV001'), 'code-prefix')
  assert.equal(skipReason('こんにちは'), undefined)
})
check('the exclusion list skips an exact line', () => {
  assert.equal(skipReason('艾莉丝', ['艾莉丝']), 'excluded')
  assert.equal(skipReason('艾莉丝大人', ['艾莉丝']), undefined)
})
check('MTool metadata is never translated', () => {
  assert.equal(skipReason('MTool_Meta_DO_NOT_REMOVE {"v":1}'), 'mtool-meta')
})
check('an entry counts as translated only when it differs from its key', () => {
  assert.equal(isPending('a', 'a'), true)
  assert.equal(isPending('a', ''), true)
  assert.equal(isPending('a', '甲'), false)
})

console.log('ported reply extraction (response_extractor.py)')
check('textarea with numbered lines', () => {
  assert.deepEqual(extractTranslation('<textarea>\n1.甲\n2.乙\n</textarea>'), ['甲', '乙'])
})
check('leading prose before the first numbered line is dropped', () => {
  assert.deepEqual(extractTranslation('好的，这是译文：\n<textarea>\n1.甲\n2.乙\n</textarea>'), ['甲', '乙'])
})
check('a reply without textarea is still read', () => {
  assert.deepEqual(extractTranslation('1.甲\n2.乙'), ['甲', '乙'])
})
check('JSON is accepted as a fallback', () => {
  assert.deepEqual(parseJsonReply('[{"text":"甲"},{"text":"乙"}]'), ['甲', '乙'])
})
check('readReply prefers the numbered protocol and reports a count mismatch', () => {
  assert.deepEqual(readReply('<textarea>1.甲\n2.乙</textarea>', 2), ['甲', '乙'])
  assert.throws(() => readReply('<textarea>1.甲</textarea>', 3), /expected 3/)
})
check('a reply with extra lines is a retry, not a truncation', () => {
  assert.throws(() => readReply('<textarea>1.甲\n2.乙\n3.丙</textarea>', 2), /expected 2/)
})
check('out-of-order numbering is rejected like check_dict_order', () => {
  assert.throws(() => extractTranslation('<textarea>1.甲\n3.乙</textarea>'), /错行串行/)
})

console.log('ported line checks (advanced_checks.py + response_checker.py)')
check('newlines count both real breaks and the literal escape', () => {
  assert.equal(countNewlines('a\\nb\nc'), 2)
})
check('a dropped placeholder is a defect', () => {
  const item = { masked: '[P1]勇者', placeholders: [{ placeholder: '[P1]', original: '\\C[3]' }] }
  assert.match(lineDefect(item, '勇者'), /占位符/)
  assert.equal(lineDefect(item, '[P1]勇者'), undefined)
})
check('newline mismatch is a defect', () => {
  const item = { masked: 'a\\nb\\nc', placeholders: [] }
  assert.match(lineDefect(item, '甲乙'), /换行符/)
})
check('returned-original detection matches character overlap, not equality', () => {
  assert.equal(looksLikeOriginal('こんにちは、勇者さま。', 'こんにちは、勇者さま。'), true)
  assert.equal(looksLikeOriginal('こんにちは、勇者さま。', '你好，勇者大人。'), false)
})

console.log('ported symbol repair (text_symbol_repair.py)')
check('corner quotes replace straight quotes', () => {
  assert.equal(repairSymbols('「こんにちは」', '"你好"'), '「你好」')
})
check('ellipsis and question mark are repaired', () => {
  assert.equal(repairSymbols('そうか…', '这样啊...'), '这样啊…')
  assert.equal(repairSymbols('元気？', '还好?'), '还好？')
})

console.log('batch preparation')
check('prepare and restore round-trips a line', () => {
  const { items: [prepared] } = prepareBatch([{ key: 'k', text: '\\C[3]勇者\\C[0]だ' }], { symbolRepair: true })
  // The leading colour code is an affix, so only the inner code is masked.
  assert.equal(prepared.masked, '勇者[P1]だ')
  assert.equal(prepared.restore('译勇者[P1]だ'), '\\C[3]译勇者\\C[0]だ')
})
check('placeholders do not leak from one line to the next in a batch', () => {
  const { items: prepared } = prepareBatch([
    { key: 'a', text: 'あ\\C[1]い' },
    { key: 'b', text: 'はい' },
  ], { symbolRepair: true })
  assert.equal(prepared[0].placeholders.length, 1)
  assert.equal(prepared[0].masked, 'あ[P1]い')
  assert.equal(prepared[1].placeholders.length, 0)
  assert.equal(prepared[1].masked, 'はい')
  assert.equal(prepared[1].restore('是'), '是')
})

console.log('config')
check('defaults are filled and bad values rejected', () => {
  const resolved = resolveConfig({})
  assert.equal(resolved.target, '简体中文')
  assert.equal(resolved.symbolRepair, true)
  assert.equal(resolved.checkResidualOriginal, false)
  assert.throws(() => resolveConfig({ batchLines: 0 }), /batchLines/)
  assert.throws(() => resolveConfig({ symbolRepair: 'yes' }), /symbolRepair/)
})

console.log('batching')
check('batches respect the line and character caps', () => {
  const items = Array.from({ length: 7 }, (_, i) => ({ key: `k${i}`, text: 'x' }))
  assert.equal(buildBatches(items, { batchLines: 3, batchChars: 1000 }).length, 3)
  const wide = Array.from({ length: 4 }, (_, i) => ({ key: `k${i}`, text: 'x'.repeat(400) }))
  assert.equal(buildBatches(wide, { batchLines: 30, batchChars: 1000 }).length, 2)
})

console.log('end to end against a stub llm')
const dir = await mkdtemp(join(tmpdir(), 'dsh-mtool-translate-test-'))
const inputPath = join(dir, 'ManualTransFile.json')
await writeFile(inputPath, JSON.stringify({
  こんにちは: 'こんにちは',
  アイテムを手に入れた: '',
  '\\C[3]勇者\\C[0]だ': '\\C[3]勇者\\C[0]だ',
  'img/pictures/face.png': 'img/pictures/face.png',
  もう翻訳済み: '你已经翻好了',
}, null, 2))

try {
  await checkAsync('a run translates pending lines, keeps finished ones, and writes valid JSON', async () => {
    const llm = stubLlm(keep)
    const result = await executeTranslate(stubCtx(llm), config, { path: inputPath }, undefined)
    assert.equal(result.done, true)
    assert.equal(result.totals.translated, 3)
    assert.equal(result.totals.finished, 1)
    assert.equal(result.totals.skipped, 1)
    const output = JSON.parse(await readFile(result.output, 'utf8'))
    assert.equal(output.こんにちは, '译こんにちは')
    assert.equal(output['\\C[3]勇者\\C[0]だ'], '\\C[3]译勇者\\C[0]だ')
    assert.equal(output['img/pictures/face.png'], 'img/pictures/face.png')
    assert.equal(output.もう翻訳済み, '你已经翻好了')
    assert.equal(Object.keys(output).length, 5)
    // The model saw placeholders, never raw codes.
    assert.equal(llm.calls.flat().some(text => text.includes('\\C[')), false)
  })

  await checkAsync('a second run spends no model call and reports cache hits', async () => {
    const llm = stubLlm(keep)
    const result = await executeTranslate(stubCtx(llm), config, { path: inputPath }, undefined)
    assert.equal(llm.calls.length, 0)
    assert.equal(result.totals.cached, 3)
    assert.equal(result.done, true)
  })

  await checkAsync('a dropped placeholder is retried instead of written back', async () => {
    const file = join(dir, 'retry.json')
    // The code sits inside the line, so it becomes a placeholder the reply can drop.
    await writeFile(file, JSON.stringify({ '勇者\\C[3]だ': '勇者\\C[3]だ' }))
    let attempt = 0
    const llm = stubLlm((text) => {
      attempt += 1
      return attempt === 1 ? '译勇者' : `译${text}`
    })
    const result = await executeTranslate(stubCtx(llm), config, { path: file }, undefined)
    assert.equal(result.done, true)
    assert.equal(llm.calls.length >= 2, true)
    const output = JSON.parse(await readFile(result.output, 'utf8'))
    assert.equal(output['勇者\\C[3]だ'], '译勇者\\C[3]だ')
  })

  await checkAsync('dryRun reports the plan without calling the model', async () => {
    const file = join(dir, 'dry.json')
    await writeFile(file, JSON.stringify({ ああ: 'ああ' }))
    const result = await executeTranslate(stubCtx(undefined), config, { path: file, dryRun: true }, undefined)
    assert.equal(result.dryRun, true)
    assert.equal(result.totals.pending, 1)
    assert.equal(result.batches.total, 1)
  })

  await checkAsync('limit translates a sample only and maxBatches resumes', async () => {
    const file = join(dir, 'chunks.json')
    const entries = Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`k${i}`, `k${i}`]))
    await writeFile(file, JSON.stringify(entries))
    const small = resolveConfig({ provider: 'stub', model: 'stub-model', batchLines: 5, concurrency: 1 })
    const first = await executeTranslate(stubCtx(stubLlm(keep)), small, { path: file, maxBatches: 2 }, undefined)
    assert.equal(first.done, false)
    assert.equal(first.batches.run, 2)
    assert.equal(first.totals.translated, 10)
    const second = await executeTranslate(stubCtx(stubLlm(keep)), small, { path: file }, undefined)
    assert.equal(second.done, true)
    assert.equal(second.totals.translated, 30)

    const sample = join(dir, 'sample.json')
    await writeFile(sample, JSON.stringify({ あ: 'あ', い: 'い', う: 'う' }))
    const limited = await executeTranslate(stubCtx(stubLlm(keep)), config, { path: sample, limit: 1 }, undefined)
    assert.equal(limited.totals.translated, 1)
  })

  await checkAsync('a missing file fails with an actionable message', async () => {
    await assert.rejects(
      () => executeTranslate(stubCtx(undefined), config, { path: join(dir, 'nope.json') }, undefined),
      /cannot read/,
    )
  })
} finally {
  await rm(dir, { recursive: true, force: true })
}

console.log(`\n${checks - failures}/${checks} checks passed`)
if (failures > 0) process.exitCode = 1
