// Synthetic adapter check. The caller provides the canonical Python-mapped fixtures.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const [sdk, fixture] = process.argv.slice(2);
const {toInternal, toPublic} = await import(pathToFileURL(resolve(sdk, 'src/compiler.js')).href);
const normalize = value => value instanceof Map ? Object.fromEntries([...value].map(([k,v])=>[k,normalize(v)])) : Array.isArray(value) ? value.map(normalize) : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).map(([k,v])=>[k,normalize(v)])) : value;
const payload = JSON.parse(readFileSync(0,'utf8'));
for (const {body,internal,response,public:expected} of payload) {
  const before=structuredClone(body);
  assert.deepEqual(normalize(toInternal(body)), internal);
  assert.deepEqual(toPublic(body,response),expected);
  assert.deepEqual(body,before);
}
for (const {id,body} of JSON.parse(readFileSync(fixture,'utf8')).invalid) {
  assert.throws(()=>toInternal(body),undefined,`Invalid request accepted: ${id}`);
}
console.log(`Verified ${payload.length} synthetic mapping variants; no model execution.`);
