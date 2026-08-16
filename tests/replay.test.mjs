import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { decodeSession, reconstruct, renderHtml, diffTrajectories, analyze } from '../engine/replay.js'

test('decodes a plain JSONL session with a packed chunk row', () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-replay-'))
  const file = join(dir, 'session.jsonl')
  try {
    writeFileSync(file, [
      JSON.stringify({ type: 'session', version: 0, id: 's1', cwd: '/w', agentPreset: 'standard', createdAt: 0 }),
      JSON.stringify({ type: 'turn/start', seq: 0, time: 0, data: { turn: 1 } }),
      JSON.stringify({ type: 'step/start', seq: 1, time: 1, data: { turn: 1, step: 1 } }),
      JSON.stringify({ type: 'user/message', seq: 2, time: 2, data: { role: 'user', content: [{ type: 'text', text: 'hello' }] } }),
      // packed reasoning chunks: one storage row -> three assistant/chunk events
      JSON.stringify({ type: 'reasoning-chunks', seq0: 3, time0: 10, data: { turn: 1, step: 1, index: 0, dt: [2, 2], texts: ['thi', 'nki', 'ng'] } }),
      JSON.stringify({ type: 'assistant/message', seq: 6, time: 20, data: { turn: 1, step: 1, message: { role: 'assistant', content: [{ type: 'text', text: 'hi there' }] } } }),
      JSON.stringify({ type: 'step/end', seq: 7, time: 21, data: { turn: 1, step: 1 } }),
      '',
    ].join('\n'))

    const { header, events } = decodeSession(file)
    assert.equal(header.id, 's1')
    assert.equal(events.length, 8) // 5 plain events + 1 packed row expanding to 3
    assert.equal(events.filter(e => e.type === 'assistant/chunk').length, 3)
    assert.deepEqual(
      events.filter(e => e.type === 'assistant/chunk').map(e => e.data.chunk.text),
      ['thi', 'nki', 'ng'],
    )

    const turns = reconstruct(events)
    assert.equal(turns.length, 1)
    assert.equal(turns[0].steps[0].userMessages[0].text, 'hello')
    assert.equal(turns[0].steps[0].text, 'hi there')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('reconstructs tool calls with results and renders HTML', () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-replay-'))
  const file = join(dir, 'session.jsonl')
  try {
    writeFileSync(file, [
      JSON.stringify({ type: 'session', version: 0, id: 's2', cwd: '/w', createdAt: 0 }),
      JSON.stringify({ type: 'turn/start', seq: 0, time: 0, data: { turn: 1 } }),
      JSON.stringify({ type: 'step/start', seq: 1, time: 1, data: { turn: 1, step: 1 } }),
      JSON.stringify({ type: 'assistant/message', seq: 2, time: 2, data: { turn: 1, step: 1, message: { role: 'assistant', content: [{ type: 'tool-call', callId: 'c1', name: 'bash', arguments: '{"command":"ls"}' }] } } }),
      JSON.stringify({ type: 'tool/call', seq: 3, time: 3, data: { turn: 1, step: 1, callId: 'c1', name: 'bash', arguments: '{"command":"ls"}' } }),
      JSON.stringify({ type: 'tool/result', seq: 4, time: 4, data: { turn: 1, step: 1, message: { content: [{ type: 'tool-result', toolCallId: 'c1', content: [{ type: 'text', text: 'a.txt' }] }] } } }),
      '',
    ].join('\n'))
    const { header, events } = decodeSession(file)
    const turns = reconstruct(events)
    const call = turns[0].steps[0].toolCalls[0]
    assert.equal(call.name, 'bash')
    assert.equal(call.result.text, 'a.txt')
    assert.equal(call.result.isError, false)
    const html = renderHtml({ header, turns })
    assert.ok(html.includes('bash'))
    assert.ok(html.includes('a.txt'))
    assert.ok(html.includes('<!doctype html>'))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('diffTrajectories summarizes per-turn tool calls', () => {
  const a = [{ turn: 1, steps: [{ step: 1, reasoning: '', text: '', toolCalls: [{ name: 'bash', arguments: '' }], userMessages: [] }] }]
  const b = [{ turn: 1, steps: [{ step: 1, reasoning: '', text: '', toolCalls: [{ name: 'fs_write', arguments: '' }], userMessages: [] }] }]
  const rows = diffTrajectories(a, b)
  assert.equal(rows.length, 1)
  assert.deepEqual(rows[0].a.calls, ['bash'])
  assert.deepEqual(rows[0].b.calls, ['fs_write'])
})

test('analyze counts broken tool calls (declared without a paired result)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-replay-'))
  const file = join(dir, 'session.jsonl')
  try {
    writeFileSync(file, [
      JSON.stringify({ type: 'session', version: 0, id: 's3', cwd: '/w', createdAt: 0 }),
      JSON.stringify({ type: 'turn/start', seq: 0, time: 0, data: { turn: 1 } }),
      JSON.stringify({ type: 'step/start', seq: 1, time: 1, data: { turn: 1, step: 1 } }),
      // declared tool call with NO matching tool/result -> broken sequence (#2334)
      JSON.stringify({ type: 'assistant/message', seq: 2, time: 2, data: { turn: 1, step: 1, message: { role: 'assistant', content: [{ type: 'tool-call', callId: 'c1', name: 'read', arguments: '{}' }] } } }),
      '',
    ].join('\n'))
    const { header, events } = decodeSession(file)
    const stats = analyze(reconstruct(events), header)
    assert.equal(stats.brokenCalls, 1)
    assert.equal(stats.toolCalls, 1)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
