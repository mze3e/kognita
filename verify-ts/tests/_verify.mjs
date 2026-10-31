import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {_parse,_verify} from '../dist/_verify.js';
const fixtures = new URL('../../tests/fixtures/verify/',import.meta.url);
const cases = JSON.parse(readFileSync(new URL('cases.json',fixtures),'utf8'));
for (const item of cases) test(item.file, () => {
  const artifact = _parse(readFileSync(new URL(item.file,fixtures),'utf8'));
  const key = item.public_key ? readFileSync(new URL(item.public_key,fixtures),'utf8') : undefined;
  if(item.valid) assert.equal(_verify(artifact,key),Number(artifact.events.length));
  else assert.throws(() => _verify(artifact,key));
  const args = [new URL('../dist/_cli.js',import.meta.url).pathname,new URL(item.file,fixtures).pathname];
  if(item.public_key) args.push('--public-key',new URL(item.public_key,fixtures).pathname);
  const result = spawnSync(process.execPath,args,{encoding:'utf8'});
  assert.equal(result.status,item.valid ? 0 : 1,result.stderr);
  if(!item.valid) assert.match(result.stderr,/T[1345]|R2/);
});
test('CLI rejects unknown flags and malformed input', () => {
  const cli = new URL('../dist/_cli.js',import.meta.url).pathname;
  for (const args of [[],['anything','--unknown'],[new URL('root.pem',fixtures).pathname]]) {
    const result = spawnSync(process.execPath,[cli,...args],{encoding:'utf8'});
    assert.equal(result.status,1); assert.match(result.stderr,/T4:/);
  }
});
