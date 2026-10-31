#!/usr/bin/env node
import {readFileSync} from 'node:fs';
import {_parse, _verify} from './_verify.js';
try {
  const args = process.argv.slice(2);
  if (!(args.length === 1 || (args.length === 3 && args[1] === '--public-key'))) throw new Error('T4: usage: kognita-verify EXPORT [--public-key PEM]');
  const count = _verify(_parse(readFileSync(args[0],'utf8')),args.length === 3 ? readFileSync(args[2],'utf8') : undefined);
  console.log(JSON.stringify({event_count:count}));
} catch (error) {
  console.error('T4: verification refused: ' + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
}
