# @kognita/verify

Independent TypeScript evidence verifier with zero runtime dependencies. Requires
Node 22 or later for ESM and the CLI. CommonJS `require()` of this ES module
requires Node 22.12 or later, as with the published 0.7.0 module. CI uses the
current Node 22 release. Uses built-in SHA-256 and Ed25519. No network requests are made
by verification. The published 0.7.0 package is available on npm. Source and distribution
metadata target 0.8.0; compilation requires the development tools.

```sh
npm install @kognita/verify
npx kognita-verify export.json --public-key root.pem
```

```
npm ci --prefix verify-ts
npm test --prefix verify-ts
node verify-ts/dist/_cli.js tests/fixtures/verify/signed.json --public-key tests/fixtures/verify/root.pem
```

The installed CLI is `kognita-verify EXPORT [--public-key PEM]`. Success prints
an event count and exits 0; refusal cites T4 and the applicable T1, T3, T5 or R2
reason and exits 1. Supply an independently trusted initial public PEM to verify
signatures, rotations, revocations and reviewer submissions. Without a public
key the command checks hash integrity and signing continuity only. A consistent
rewrite or a changed export checkpoint requires independent trust to detect.
Anchors and recovery bundles are outside this CLI's scope.

Shared golden exports include v1, v2, Unicode, numeric payloads, signed
chains, rotation, revocation, reviews and invalid exports. Python checks the same
files. Numeric JSON source is retained to preserve large integers and Python's
float representation. Only private implementation helpers are exported.

This package has no publication workflow. npm publication uses the authenticated operator release process. Fixture PEM files contain public keys only.
