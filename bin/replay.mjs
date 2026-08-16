#!/usr/bin/env node
import { existsSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { decodeSession, reconstruct, renderHtml, diffTrajectories } from '../engine/replay.js'

function usage() {
  process.stderr.write(`dsh-replay — time-travel debugger for DeepSeek Harness sessions

Usage:
  dsh-replay <session-id> [--out replay.html]   render a session's full trajectory
  dsh-replay --file <session.jsonl.zstd> [--out replay.html]
  dsh-replay diff <id-a> <id-b> [--root ~/.dsh/sessions]

Options:
  --root <dir>    sessions root (default: $DSH_HOME/sessions or ~/.dsh/sessions)
  --out <file>    HTML output path (default: <session-id>.replay.html)
`)
  process.exit(2)
}

function resolveRoot() {
  if (process.env.DSH_HOME) return join(process.env.DSH_HOME, 'sessions')
  const home = process.env.USERPROFILE || process.env.HOME
  if (home) return join(home, '.dsh', 'sessions')
  return '.'
}

function findSession(root, id) {
  if (existsSync(id)) return id
  const direct = join(root, id, 'session.jsonl.zstd')
  if (existsSync(direct)) return direct
  const plain = join(root, id, 'session.jsonl')
  if (existsSync(plain)) return plain
  // fall back to a recursive search (best-effort)
  const walk = (dir, depth) => {
    if (depth > 8) return undefined
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) {
        const hit = walk(full, depth + 1)
        if (hit) return hit
      } else if (entry === 'session.jsonl.zstd' || entry === 'session.jsonl') {
        if (full.includes(id)) return full
      }
    }
    return undefined
  }
  return walk(root, 0)
}

const args = process.argv.slice(2)
if (args.length === 0 || args[0] === '--help' || args[0] === '-h') usage()

const argv = {}
for (let i = 0; i < args.length; i++) {
  const a = args[i]
  if (a === '--out' || a === '--root' || a === '--file') argv[a.slice(2)] = args[++i]
}

const root = argv.root ?? resolveRoot()

function load(idOrFile) {
  const file = idOrFile.startsWith('.') || idOrFile.includes('/') || idOrFile.includes('\\')
    ? idOrFile
    : findSession(root, idOrFile)
  if (!file) {
    process.stderr.write(`dsh-replay: no session artifact found for "${idOrFile}" under ${root}\n`)
    process.exit(1)
  }
  return { file, ...decodeSession(file) }
}

function write(out, html) {
  writeFileSync(out, html)
  process.stdout.write(`${out}\n`)
}

if (args[0] === 'diff') {
  const [, , idA, idB] = args
  if (!idA || !idB) usage()
  const a = reconstruct(load(idA).events)
  const b = reconstruct(load(idB).events)
  const rows = diffTrajectories(a, b)
  for (const row of rows) {
    const fmt = (side) => side === null ? '(missing)' : `[${side.calls.join(', ')}]`
    process.stdout.write(`turn ${row.turn}: A ${fmt(row.a)} | B ${fmt(row.b)}\n`)
  }
  process.exit(0)
}

const id = args.find(a => !a.startsWith('--') && a !== 'diff')
if (!id) usage()

const { file, header, events } = load(id)
const turns = reconstruct(events)
const out = argv.out ?? `${(header?.id ?? id).replaceAll(':', '-')}.replay.html`
write(out, renderHtml({ header, turns }))
