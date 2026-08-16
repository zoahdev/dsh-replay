/**
 * dsh-replay engine: decode a DeepSeek Harness session log and reconstruct its
 * full trajectory without depending on @deepseek-ai/dsh.
 *
 * The session artifact is a concatenated-Zstandard container of JSONL lines.
 * Each line is either a single event or a packed "chunk row" that expands to
 * several `assistant/chunk` events (the packed-row wire format is owned by
 * @deepseek-ai/dsh-session's chunk-rows module; this decoder mirrors it so the
 * tool stays zero-dependency).
 *
 * @module dsh-replay/engine
 */

import { readFileSync } from 'node:fs'
import { zstdDecompressSync } from 'node:zlib'

const ZSTD_MAGIC = 0xFD2FB528
const ZSTD_FILE_MAGIC = Buffer.from([0x28, 0xb5, 0x2f, 0xfd])

/** Inclusive/exclusive byte range of one structurally complete Zstandard frame. */
function scanZstdFrames(buffer) {
  const frames = []
  let offset = 0
  while (offset < buffer.length) {
    const start = offset
    if (buffer.length - offset < 4) return { frames, tornStart: start }
    if (buffer.readUInt32LE(offset) !== ZSTD_MAGIC) {
      throw new Error(`corrupt Zstandard session log: invalid frame magic at byte ${offset}`)
    }
    offset += 4
    const descriptor = buffer.readUInt8(offset++)
    if ((descriptor & 0x18) !== 0) {
      throw new Error('corrupt Zstandard session log: reserved frame-header bit')
    }
    const contentSizeFlag = descriptor >>> 6
    const singleSegment = (descriptor & 0x20) !== 0
    const checksum = (descriptor & 0x04) !== 0
    const dictionaryFlag = descriptor & 0x03
    const dictionaryBytes = dictionaryFlag === 3 ? 4 : dictionaryFlag
    const contentSizeBytes = contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : (1 << contentSizeFlag)
    const remainingHeaderBytes = (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes
    if (buffer.length - offset < remainingHeaderBytes) return { frames, tornStart: start }
    offset += remainingHeaderBytes
    for (;;) {
      if (buffer.length - offset < 3) return { frames, tornStart: start }
      const blockHeader = buffer.readUIntLE(offset, 3)
      offset += 3
      const lastBlock = (blockHeader & 1) !== 0
      const blockType = (blockHeader >>> 1) & 0x03
      const blockSize = blockHeader >>> 3
      if (blockType === 0x03) throw new Error('corrupt Zstandard session log: reserved block type')
      const payloadBytes = blockType === 0x01 ? 1 : blockSize
      if (buffer.length - offset < payloadBytes) return { frames, tornStart: start }
      offset += payloadBytes
      if (lastBlock) break
    }
    if (checksum) {
      if (buffer.length - offset < 4) return { frames, tornStart: start }
      offset += 4
    }
    frames.push({ start, end: offset })
  }
  return { frames }
}

/** Decompress a concatenated-frame zstd container into its plaintext. */
function decompressContainer(buffer) {
  const { frames } = scanZstdFrames(buffer)
  let out = ''
  for (const frame of frames) {
    out += zstdDecompressSync(buffer.subarray(frame.start, frame.end)).toString('utf8')
  }
  return out
}

/** Expand one JSONL line value into the events it stores (chunk-row aware). */
function expandStorageRecord(value) {
  if (value === null || typeof value !== 'object') return [value]
  const tag = value.type
  if (tag !== 'text-chunks' && tag !== 'reasoning-chunks' && tag !== 'tool-call-chunks') {
    return [value]
  }
  const d = value.data ?? {}
  const members = tag === 'tool-call-chunks' ? d.args : d.texts
  const events = []
  let time = value.time0
  for (let k = 0; k < members.length; k++) {
    if (k > 0) time += d.dt[k - 1]
    let chunk
    if (tag === 'text-chunks') {
      chunk = { type: 'text-delta', index: d.index, text: members[k] }
    } else if (tag === 'reasoning-chunks') {
      chunk = { type: 'reasoning-delta', index: d.index, text: members[k] }
    } else {
      chunk = { type: 'tool-call-delta', index: d.index, id: d.id, argumentsDelta: members[k] }
    }
    events.push({
      type: 'assistant/chunk',
      seq: value.seq0 + k,
      time,
      data: { turn: d.turn, step: d.step, chunk },
    })
  }
  return events
}

/** Decode a session artifact path into `{ header, events }`. */
export function decodeSession(file) {
  const buffer = readFileSync(file)
  let plain
  if (buffer.length >= 4 && buffer.subarray(0, 4).equals(ZSTD_FILE_MAGIC)) {
    plain = decompressContainer(buffer)
  } else {
    plain = buffer.toString('utf8')
  }
  const lines = plain.split(/\r?\n/u).filter(line => line.trim() !== '')
  const events = []
  let header
  for (const line of lines) {
    let value
    try {
      value = JSON.parse(line)
    } catch {
      continue // torn tail bytes are ignored for replay
    }
    if (header === undefined && value?.type === 'session') {
      header = value
      continue
    }
    events.push(...expandStorageRecord(value))
  }
  return { header, events }
}

/** Join every `text` block of a content array into one string. */
function textOf(content) {
  if (!Array.isArray(content)) return ''
  return content
    .filter(block => block?.type === 'text' && typeof block.text === 'string')
    .map(block => block.text)
    .join('\n')
}

/** Reconstruct a trajectory: turns -> steps -> reasoning/text/tool calls. */
export function reconstruct(events) {
  const turns = []
  let turn = null
  let step = null

  const ensureTurn = (n) => {
    if (turn === null || turn.turn !== n) {
      turn = { turn: n, steps: [] }
      turns.push(turn)
      step = null
    }
  }
  const ensureStep = (n) => {
    ensureTurn(turn === null ? 0 : turn.turn)
    if (step === null || step.step !== n) {
      step = { step: n, reasoning: '', text: '', toolCalls: [], userMessages: [] }
      turn.steps.push(step)
    }
  }

  for (const event of events) {
    const type = event.type
    const data = event.data ?? {}
    if (type === 'turn/start') {
      ensureTurn(data.turn)
    } else if (type === 'step/start') {
      ensureTurn(data.turn)
      ensureStep(data.step)
    } else if (type === 'user/message') {
      const text = textOf(data.content)
      if (text !== '') {
        ensureTurn(turn === null ? 0 : turn.turn)
        if (step === null) ensureStep(0)
        step.userMessages.push({ text, source: data.source?.kind })
      }
    } else if (type === 'assistant/message') {
      ensureTurn(data.turn)
      ensureStep(data.step)
      for (const block of data.message?.content ?? []) {
        if (block?.type === 'reasoning' && typeof block.text === 'string') step.reasoning += block.text
        else if (block?.type === 'text' && typeof block.text === 'string') step.text += block.text
        else if (block?.type === 'tool-call') {
          step.toolCalls.push({
            callId: block.callId ?? block.id,
            name: block.name ?? block.function?.name ?? 'unknown',
            arguments: block.arguments ?? block.input ?? '',
          })
        }
      }
    } else if (type === 'tool/call') {
      ensureTurn(data.turn)
      ensureStep(data.step)
      const call = step.toolCalls.find(c => c.callId === data.callId)
      if (call) call.time = event.time
    } else if (type === 'tool/result') {
      ensureTurn(data.turn)
      ensureStep(data.step)
      const resultContent = data.message?.content ?? []
      const callId = resultContent[0]?.toolCallId ?? resultContent[0]?.content?.[0]?.toolCallId
      const isError = resultContent[0]?.isError === true || resultContent[0]?.content?.[0]?.isError === true
      const text = textOf(resultContent[0]?.content) || textOf(resultContent)
      const call = step.toolCalls.find(c => c.callId === callId) ?? step.toolCalls.at(-1)
      if (call) {
        call.result = { text, isError }
        if (call.time !== undefined) call.durationMs = event.time - call.time
      }
    }
  }
  return turns
}

/** Aggregate stats for a reconstructed trajectory. */
export function analyze(turns, header) {
  const steps = turns.flatMap(t => t.steps)
  const calls = steps.flatMap(s => s.toolCalls)
  const byTool = {}
  for (const call of calls) byTool[call.name] = (byTool[call.name] ?? 0) + 1
  const durations = calls.filter(c => c.durationMs !== undefined).map(c => c.durationMs)
  return {
    title: header?.title ?? header?.id,
    turns: turns.length,
    steps: steps.length,
    toolCalls: calls.length,
    errors: calls.filter(c => c.result?.isError === true).length,
    tools: Object.entries(byTool).sort((a, b) => b[1] - a[1]).slice(0, 12),
    maxToolMs: durations.length > 0 ? Math.round(Math.max(...durations)) : 0,
    meanToolMs: durations.length > 0 ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length) : 0,
  }
}

function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function renderToolArgs(args) {
  let parsed = args
  if (typeof args === 'string') {
    try { parsed = JSON.parse(args) } catch { return `<pre>${esc(args)}</pre>` }
  }
  return `<pre>${esc(JSON.stringify(parsed, null, 2))}</pre>`
}

/** Render the trajectory to a self-contained HTML document. */
export function renderHtml({ header, turns }, stats = analyze(turns, header)) {
  const title = header?.title || header?.id || 'session'
  const meta = [
    ['id', header?.id],
    ['preset', header?.agentPreset],
    ['cwd', header?.cwd],
    ['created', header?.createdAt ? new Date(header.createdAt).toISOString() : undefined],
  ]
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `<div class="meta"><span>${esc(k)}</span><code>${esc(v)}</code></div>`)
    .join('')

  const turnHtml = turns.map((t) => {
    const stepsHtml = t.steps.map((s) => {
      const reasoning = s.reasoning.trim()
        ? `<details class="reasoning"><summary>reasoning</summary><pre>${esc(s.reasoning)}</pre></details>`
        : ''
      const text = s.text.trim() ? `<div class="text"><pre>${esc(s.text)}</pre></div>` : ''
      const users = s.userMessages.map(m =>
        `<div class="user"><div class="tag">user</div><pre>${esc(m.text)}</pre></div>`).join('')
      const tools = s.toolCalls.map(c => {
        const r = c.result
        const duration = c.durationMs !== undefined ? `<span class="dur">${c.durationMs}ms</span>` : ''
        const result = r
          ? `<div class="result ${r.isError ? 'err' : 'ok'}"><div class="tag">${r.isError ? 'error' : 'ok'}</div><pre>${esc(r.text)}</pre></div>`
          : ''
        return `<div class="tool"><div class="call"><span class="fn">${esc(c.name)}</span>${duration}${renderToolArgs(c.arguments)}</div>${result}</div>`
      }).join('')
      return `<section class="step"><div class="step-head">step ${esc(s.step)}</div>${users}${reasoning}${text}${tools}</section>`
    }).join('')
    return `<section class="turn"><h3>turn ${esc(t.turn)}</h3>${stepsHtml}</section>`
  }).join('')

  const toolCounts = stats.tools.map(([name, count]) =>
    `<span class="chip"><code>${esc(name)}</code> ×${count}</span>`).join('')
  const statsHtml = `
  <section class="stats">
    <div class="card"><span class="k">turns</span><span class="v">${stats.turns}</span></div>
    <div class="card"><span class="k">steps</span><span class="v">${stats.steps}</span></div>
    <div class="card"><span class="k">tool calls</span><span class="v">${stats.toolCalls}</span></div>
    <div class="card"><span class="k">errors</span><span class="v ${stats.errors > 0 ? 'bad' : ''}">${stats.errors}</span></div>
    <div class="card"><span class="k">mean tool</span><span class="v">${stats.meanToolMs}ms</span></div>
    <div class="card"><span class="k">max tool</span><span class="v">${stats.maxToolMs}ms</span></div>
  </section>
  <section class="tools">${toolCounts}</section>`

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(title)} · dsh-replay</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; background:#0d1117; color:#e6edf3; }
  header { padding:20px 24px; border-bottom:1px solid #30363d; background:#161b22; }
  h1 { margin:0 0 8px; font-size:18px; font-weight:600; }
  .meta { display:inline-block; margin-right:16px; font-size:12px; color:#8b949e; }
  .meta code { margin-left:6px; color:#e6edf3; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:10px; margin-bottom:16px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px 12px; }
  .card .k { display:block; font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.04em; }
  .card .v { font-size:20px; font-weight:600; color:#e6edf3; }
  .card .v.bad { color:#f85149; }
  .tools { margin-bottom:16px; }
  .chip { display:inline-block; margin:0 6px 6px 0; padding:3px 8px; background:#21262d; border:1px solid #30363d; border-radius:12px; font-size:12px; color:#e6edf3; }
  main { max-width:980px; margin:0 auto; padding:24px; }
  .turn { margin-bottom:24px; }
  h3 { margin:0 0 8px; font-size:15px; color:#58a6ff; }
  .step { border:1px solid #30363d; border-radius:8px; padding:12px 14px; margin-bottom:12px; background:#11161d; }
  .step-head { font-size:12px; color:#8b949e; margin-bottom:8px; font-weight:600; }
  .tag { display:inline-block; font-size:10px; text-transform:uppercase; letter-spacing:.05em; padding:2px 6px; border-radius:10px; background:#30363d; color:#8b949e; margin-bottom:6px; }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .reasoning { margin:8px 0; border-left:3px solid #6e7681; padding-left:12px; }
  .reasoning summary { cursor:pointer; color:#8b949e; font-size:12px; }
  .reasoning pre { color:#8b949e; margin-top:6px; }
  .text pre { color:#e6edf3; }
  .user { margin:8px 0; border-left:3px solid #58a6ff; padding-left:12px; }
  .user .tag { background:#1f6feb; color:#fff; }
  .tool { margin:10px 0; border:1px solid #30363d; border-radius:6px; }
  .call { padding:8px 10px; }
  .fn { font:600 13px ui-monospace,Menlo,monospace; color:#d2a8ff; display:block; margin-bottom:4px; }
  .dur { font-size:11px; color:#8b949e; float:right; }
  .result { padding:8px 10px; border-top:1px solid #30363d; }
  .result.ok { background:#0f2a1a; } .result.err { background:#2a1515; }
  .result .tag { font-size:10px; }
  .result.ok .tag { background:#238636; color:#fff; } .result.err .tag { background:#da3633; color:#fff; }
</style></head><body>
<header><h1>${esc(title)}</h1>${meta}</header>
<main>${statsHtml}${turnHtml}</main>
</body></html>\n`
}

/** One-line per-turn summary of two trajectories for a quick diff. */
export function diffTrajectories(a, b) {
  const rows = []
  const max = Math.max(a.length, b.length)
  for (let i = 0; i < max; i++) {
    const ta = a[i]
    const tb = b[i]
    rows.push({ turn: i + 1, a: ta ? summarizeTurn(ta) : null, b: tb ? summarizeTurn(tb) : null })
  }
  return rows
}

function summarizeTurn(t) {
  const calls = t.steps.flatMap(s => s.toolCalls.map(c => c.name))
  const text = t.steps.map(s => s.text.trim()).filter(Boolean).join('\n')
  return { calls, text }
}

/** Render a side-by-side HTML diff of two trajectories. */
export function renderDiffHtml(a, b, aName = 'A', bName = 'B') {
  const rows = diffTrajectories(a, b)
  const max = Math.max(a.length, b.length)
  const body = []
  for (let i = 0; i < max; i++) {
    const ta = a[i]
    const tb = b[i]
    const same = ta && tb && JSON.stringify(ta.steps.flatMap(s => s.toolCalls.map(c => c.name)))
      === JSON.stringify(tb.steps.flatMap(s => s.toolCalls.map(c => c.name)))
    const cls = same ? 'same' : 'diff'
    body.push(`<tr class="${cls}"><td>turn ${i + 1}</td>`)
    body.push(`<td>${ta ? ta.steps.flatMap(s => s.toolCalls.map(c => esc(c.name))).join('<br>') || '—' : '—'}</td>`)
    body.push(`<td>${tb ? tb.steps.flatMap(s => s.toolCalls.map(c => esc(c.name))).join('<br>') || '—' : '—'}</td></tr>`)
  }
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>diff · dsh-replay</title>
<style>
  :root{color-scheme:dark} body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#0d1117;color:#e6edf3}
  header{padding:20px 24px;border-bottom:1px solid #30363d;background:#161b22}
  h1{margin:0;font-size:18px} table{width:100%;border-collapse:collapse;max-width:980px;margin:24px auto}
  th,td{padding:8px 12px;border:1px solid #30363d;text-align:left;vertical-align:top}
  th{background:#161b22;color:#8b949e;font-size:12px;text-transform:uppercase}
  tr.diff td{background:#2a1515} tr.same td{background:#0f2a1a}
</style></head><body>
<header><h1>trajectory diff · ${esc(aName)} vs ${esc(bName)}</h1></header>
<table><thead><tr><th>turn</th><th>${esc(aName)}</th><th>${esc(bName)}</th></tr></thead><tbody>${body.join('')}</tbody></table>
</body></html>\n`
}
