import {createHash, createPublicKey, verify} from 'node:crypto';

const _number = Symbol('number');
type _Value = any;
function _fail(rule: string, reason: string): never { throw new Error(`${rule}: ${reason}`); }
// Retain integer precision and the distinction between Python int and float.
export function _parse(text: string): _Value {
  return (JSON.parse as any)(text, (_key: string, value: any, context: any) => {
    if (typeof value !== 'number') return value;
    const source = context?.source;
    if (!source) _fail('T4', 'number source unavailable');
    return {[_number]: source};
  });
}
function _numeric(source: string): string {
  if (!/[.eE]/.test(source)) return BigInt(source).toString();
  const n = Number(source);
  if (!Number.isFinite(n)) _fail('T4', 'nonfinite number');
  if (Object.is(n, -0)) return '-0.0';
  const a = Math.abs(n);
  if (a !== 0 && (a < 1e-4 || a >= 1e16)) {
    const [mantissa, exponent] = n.toExponential().split('e');
    const e = Number(exponent);
    return `${mantissa}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  }
  const result = String(n);
  return result.includes('.') ? result : result + '.0';
}
function _canonical(value: _Value): string {
  if (value && typeof value === 'object' && _number in value) return _numeric(value[_number]);
  if (Array.isArray(value)) return '[' + value.map(_canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort((a,b) => {
    const x = Array.from(a, c => c.codePointAt(0)!); const y = Array.from(b, c => c.codePointAt(0)!);
    for (let i=0;i<Math.min(x.length,y.length);i++) if(x[i]!==y[i]) return x[i]-y[i];
    return x.length-y.length;
  }).map(k => _canonical(k)+':'+_canonical(value[k])).join(',') + '}';
  if (typeof value === 'string') return JSON.stringify(value).replace(/[\u007f-\uffff]/g, c => c.charCodeAt(0) < 128 ? c : '\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
  if (typeof value === 'number') { if (!Number.isSafeInteger(value)) _fail('T4','use parsed JSON for numeric fidelity'); }
  if (value === undefined) _fail('T4', 'missing field');
  return JSON.stringify(value);
}
function _hash(value: _Value): string {return createHash('sha256').update(_canonical(value)).digest('hex');}
function _int(value: _Value): number {
  const source = value?.[_number];
  if (source !== undefined && !/^-?\d+$/.test(source)) _fail('T4','expected integer');
  const n = source === undefined ? value : Number(source);
  if (!Number.isSafeInteger(n)) _fail('T4','expected safe integer');
  return n;
}
function _key(pem: string): {pem: string, id: string, revoked: number | null} {
  const key = createPublicKey(pem);
  if (key.asymmetricKeyType !== 'ed25519') _fail('T1', 'expected Ed25519 key');
  const raw = key.export({format:'der',type:'spki'}).subarray(-32);
  return {pem, id:createHash('sha256').update(raw).digest('hex'), revoked:null};
}
function _signature(body: _Value, signature: string, id: string, pem: string): void {
  if (typeof signature !== 'string' || !/^[a-fA-F0-9]{128}$/.test(signature) || _key(pem).id !== id ||
      !verify(null, Buffer.from(_hash(body),'hex'),pem,Buffer.from(signature,'hex'))) _fail('T1','signature verification failed');
}
function _time(text: string): number {
  // Preserve microseconds for revocation comparisons within one millisecond.
  if (typeof text !== 'string') _fail('R2','invalid timestamp');
  const n = Date.parse(text);
  if (!Number.isFinite(n)) _fail('R2','invalid timestamp');
  const fraction = text.match(/\.(\d+)(?:Z|[+-]\d\d:\d\d)$/)?.[1] ?? '';
  return n + Number('0.' + (fraction.slice(3) || '0'));
}
export function _verify(artifact: _Value, publicKey?: string): number {
  if (!artifact || !Array.isArray(artifact.events)) _fail('T4','events required');
  const rows = artifact.events;
  let previous = '0'.repeat(64);
  const required = publicKey !== undefined || rows.some((r: any) => r.signature != null);
  const registry = new Map<string, ReturnType<typeof _key>>();
  let active: string | undefined;
  const signed: {id:string,time:number}[] = [];
  for (const [index,row] of rows.entries()) {
    if (_int(row.sequence) !== index+1) _fail('T4','sequence mismatch');
    const version = row.schema_version === undefined ? 1 : _int(row.schema_version);
    if (![1,2].includes(version)) _fail('T5','unsupported schema_version');
    if (_hash(row.payload) !== row.payload_hash || row.prev_hash !== previous) _fail('T4','payload or previous hash mismatch');
    const header: any = {};
    for (const key of ['sequence','correlation_id','event_type','actor_type','actor_id','classification','payload_hash','prev_hash','recorded_at']) header[key] = row[key];
    if (version === 2) header.schema_version = row.schema_version;
    const lifecycle = ['KEY_ROTATED','KEY_REVOKED'].includes(row.event_type);
    if (row.signature != null || row.key_id != null) {
      if (version !== 2 || typeof row.signature !== 'string' || !row.signature || typeof row.key_id !== 'string' || !row.key_id) _fail('T1','invalid signature metadata');
      header.key_id = row.key_id;
      if (!active) {
        active = row.key_id;
        registry.set(active!, publicKey !== undefined ? _key(publicKey) : {pem:'',id:active!,revoked:null});
      }
      if (row.key_id !== active || registry.get(active!)?.revoked != null) _fail('R2','inactive or revoked key');
      if (publicKey !== undefined) _signature(header,row.signature,row.key_id,registry.get(active!)!.pem);
      signed.push({id:row.key_id,time:_time(row.recorded_at)});
      header.signature = row.signature;
    } else if (required || lifecycle) _fail('T1','unsigned event in signed chain');
    if (_hash(header) !== row.event_hash) _fail('T4','event hash mismatch');
    previous = row.event_hash;
    if (lifecycle) {
      const body = row.payload;
      if (row.event_type === 'KEY_ROTATED') {
        const key = _key(body._public_key);
        if (key.id !== body.key_id || registry.has(key.id) || typeof body._escrow !== 'string' || !body._escrow.trim()) _fail('R2','invalid rotation');
        registry.set(key.id,key); active = key.id;
      } else {
        const key = registry.get(body.key_id);
        if (!key || key.revoked != null) _fail('R2','unknown or revoked key');
        key.revoked = _time(body.recorded_at);
        if (signed.some(s => s.id === body.key_id && s.time >= key.revoked!)) _fail('R2','signature at or after revocation');
      }
    }
  }
  if (publicKey !== undefined) for (const row of rows) {
    if (row.event_type !== 'EXTERNAL_REVIEW_SUBMITTED') continue;
    const body = row.payload.review; const key = registry.get(body.key_id);
    if (!key || (key.revoked != null && _time(row.recorded_at) >= key.revoked)) _fail('T3','unknown or revoked reviewer');
    const start = _int(body.start), end = _int(body.end);
    if (!(1 <= start && start <= end && end < _int(row.sequence)) || rows[end-1].event_hash !== body.head || !Array.isArray(body.annotations)) _fail('T3','invalid review period');
    for (const item of body.annotations) if (!(_int(item.sequence) >= start && _int(item.sequence) <= end) || typeof item.text !== 'string' || !item.text.trim()) _fail('T3','invalid annotation');
    _signature(body,row.payload.signature,body.key_id,key.pem);
  }
  if (_int(artifact.event_count) !== rows.length || (artifact.head_hash ?? '0'.repeat(64)) !== previous) _fail('T4','export count or head mismatch');
  return rows.length;
}
