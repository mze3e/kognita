import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';

test('published root import and CommonJS entry points remain compatible', async () => {
  const esm = await import('@kognita/verify');
  const cjs = createRequire(import.meta.url)('@kognita/verify');
  assert.equal(typeof esm._parse, 'function');
  assert.equal(typeof esm._verify, 'function');
  assert.equal(cjs._parse, esm._parse);
  assert.equal(cjs._verify, esm._verify);
  const metadata = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  assert.equal(metadata.private, false);
  assert.equal(metadata.main, 'dist/_verify.js');
  assert.equal(metadata.exports['.'], './dist/_verify.js');
  assert.equal(metadata.bin['kognita-verify'], 'dist/_cli.js');
});
