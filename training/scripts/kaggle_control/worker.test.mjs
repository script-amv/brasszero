// SPDX-License-Identifier: Apache-2.0
import test from 'node:test'
import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import worker from './worker.mjs'

test('separate run capabilities, irreversible stop and out-of-order phase delivery', async () => {
  const id = 'a'.repeat(32), owner = 'o'.repeat(43), runner = 'r'.repeat(43)
  const hash = t => createHash('sha256').update(t).digest('hex')
  const config = {run:id, owner_hash:hash(owner),runner_hash:hash(runner),expiry:Math.floor(Date.now()/1000)+1000}
  const records = new Map([[id+'/config',{value:config}]])
  const env = {CONTROL:{
    get:async k => records.get(k)?.value ?? null,
    put:async (k,value,options) => records.set(k,{value,...options}),
    list:async ({prefix,limit}) => ({keys:[...records.keys()].filter(k=>k.startsWith(prefix)).sort().slice(0,limit)
      .map(k=>({metadata:records.get(k).metadata}))})
  }}
  const call = (action,token=runner,body,run=id) => {
    const data = body === undefined ? null : JSON.stringify(body)
    return worker.fetch(new Request('https://test/runs/'+run+'/'+action,
      {method:['poll','status'].includes(action)?'GET':'POST',body:data,
       headers:{Authorization:'Bearer '+token,'Content-Length':String(data?.length||0)}}),env)
  }
  assert.equal((await call('poll',owner,undefined,'b'.repeat(32))).status,403)
  assert.equal((await call('stop',runner)).status,403)
  assert.equal((await call('poll','bad')).status,403)
  assert.equal((await (await call('poll')).json()).requested,false)
  assert.equal((await call('stop',owner,{requested_at:12345})).status,200)
  assert.equal((await call('stop',owner,{requested_at:12345})).status,200)
  for (const [phase,seq] of [['export_ready',4],['training',1],['checkpoint_saved',2]]) {
    assert.equal((await call('progress',runner,{phase,seq,update:844,reason:'manual'})).status,200)
  }
  const result = await (await call('status',owner)).json()
  assert.equal(result.requested,true)
  assert.equal(result.progress.phase,'export_ready')
  assert.equal((await call('progress',owner,{phase:'training',seq:8,update:845,reason:'none'})).status,403)
  assert.equal((await call('progress',runner,{phase:'command',seq:8,update:845,reason:'none'})).status,400)
  const oversized = new Request('https://test/runs/'+id+'/progress',
    {method:'POST',headers:{Authorization:'Bearer '+runner,'Content-Length':'25'},body:' '.repeat(512)})
  assert.equal((await worker.fetch(oversized,env)).status,400)
  config.expiry = 0
  assert.equal((await call('status',owner)).status,403)
})
