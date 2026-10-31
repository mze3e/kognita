# @kognita/verify

Independent TypeScript evidence verifier with zero runtime dependencies. Requires
Node 22 or later. Uses built-in SHA-256 and Ed25519. No network requests are made
by verification. Installation and compilation require the development tools.

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

Shared golden exports include legacy v1, v2, Unicode, numeric payloads, signed
chains, rotation, revocation, reviews and invalid exports. Python checks the same
files. Numeric JSON source is retained to preserve large integers and Python's
float representation. Only private implementation helpers are exported.

This package has no publication workflow. npm publication requires Ahmed's
explicit go. Fixture PEM files contain public keys only.
