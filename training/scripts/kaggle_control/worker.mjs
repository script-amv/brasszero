// SPDX-License-Identifier: Apache-2.0
// Per-run capabilities are registered through the account's KV API.
const phases = ['bootstrap', 'training', 'ack_stopping', 'checkpoint_saved', 'exporting', 'export_ready']
const reasons = ['none', 'manual', 'deadline', 'capacity', 'control_unavailable', 'trainer_error', 'setup_error', 'trainer_ended']
const reply = (data, status = 200) => Response.json(data, {status, headers: {'Cache-Control': 'no-store'}})
const denied = () => reply({error: 'unauthorized'}, 403)

async function matches(token, expected) {
  if (!expected || !/^[a-f0-9]{64}$/.test(expected)) return false
  const actual = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token)))
  let difference = 0
  for (let i = 0; i < actual.length; i++) difference |= actual[i] ^ parseInt(expected.slice(2*i, 2*i+2), 16)
  return difference === 0
}

async function progressBody(request) {
  const reader = request.body?.getReader()
  if (!reader) return null
  const bytes = new Uint8Array(256)
  let offset = 0
  try {
    while (true) {
      const {done, value} = await reader.read()
      if (done) break
      if (offset + value.length > bytes.length) {
        await reader.cancel()
        return null
      }
      bytes.set(value, offset)
      offset += value.length
    }
    return JSON.parse(new TextDecoder().decode(bytes.subarray(0, offset)))
  } catch {
    return null
  } finally {
    reader.releaseLock()
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    const match = /^\/runs\/([a-f0-9]{32})\/(poll|status|stop|progress)$/.exec(url.pathname)
    const auth = /^Bearer ([A-Za-z0-9_-]{43})$/.exec(request.headers.get('Authorization') || '')
    if (!match || url.search || !auth) return denied()
    const [, id, action] = match
    const kv = env.CONTROL
    const config = await kv.get(id+'/config', 'json')
    if (!config || config.run !== id || config.expiry < Date.now()/1000 + 60) return denied()
    const owner = await matches(auth[1], config.owner_hash)
    const runner = await matches(auth[1], config.runner_hash)
    if (!(owner || runner) || (action === 'stop' && !owner) || (action === 'progress' && !runner)) return denied()
    const get = action === 'poll' || action === 'status'
    if (request.method !== (get ? 'GET' : 'POST')) return reply({error:'method'}, 405)
    const options = {expiration: config.expiry}
    if (action === 'stop') {
      // Same bytes on every retry. Progress never writes this key.
      const intent = await progressBody(request)
      if (!Number.isSafeInteger(intent?.requested_at) || intent.requested_at < 1 || intent.requested_at > Date.now()+120000) return reply({error:'timestamp'},400)
      const prior = await kv.get(id+'/stop')
      // All owner retries send the same locally durable intent timestamp.
      if (prior === null) await kv.put(id+'/stop', String(intent.requested_at), options)
      return reply({run:id, requested:true, requested_at:Number(prior ?? intent.requested_at)})
    }
    if (action === 'progress') {
      if (!(Number(request.headers.get('Content-Length')) > 0 && Number(request.headers.get('Content-Length')) <= 256)) return reply({error:'size'}, 413)
      const row = await progressBody(request)
      if (!row || (row.ack_at != null && (!Number.isSafeInteger(row.ack_at) || row.ack_at < 0 || row.ack_at > config.expiry*1000)) || !phases.includes(row.phase) || !reasons.includes(row.reason) ||
          !Number.isSafeInteger(row.update) || row.update < 0 || row.update > 999999 ||
          !Number.isSafeInteger(row.seq) || row.seq < 0 || row.seq > 999999) return reply({error:'progress'}, 400)
      // Lexical minimum is the latest phase/sequence even for delayed writes from other edges.
      const key = id+'/phase/'+(9-phases.indexOf(row.phase))+'-'+String(999999-row.seq).padStart(6,'0')
      const metadata = {phase:row.phase, reason:row.reason, update:row.update, seq:row.seq, ack_at:row.ack_at ?? null, at:Math.floor(Date.now()/1000)}
      await kv.put(key, '1', {...options, metadata})
      return reply({accepted:true})
    }
    const stop = await kv.get(id+'/stop')
    const result = {run:id, requested:stop !== null, requested_at:stop === null ? null : Number(stop)}
    if (action === 'status') {
      const listed = await kv.list({prefix:id+'/phase/',limit:1})
      result.progress = listed.keys[0]?.metadata || null
    }
    return reply(result)
  }
}
