"""Local operator commands; actor identities must be authenticated by the host."""
import json
import sys
from datetime import datetime
from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser('assurance', help='0.8 runtime assurance and conformance evidence')
    commands = parser.add_subparsers(dest='assurance_command', required=True)
    inputs = {'configure', 'attest', 'components', 'update', 'outcome', 'advisory', 'drill', 'compare'}
    writes = inputs - {'compare'} | {'manifest'}
    for name in sorted(inputs | {'manifest', 'coverage', 'report', 'conformance'}):
        command = commands.add_parser(name)
        command.add_argument('--db', default='kognita.db')
        command.add_argument('--agent', required=name != 'advisory')
        if name in inputs:
            command.add_argument('--input', required=True, help='JSON input file')
        if name in writes:
            command.add_argument('--actor', required=True, help='authenticated local operator')
        if name == 'manifest':
            command.add_argument('--signing-key', required=True, help='Ed25519 PKCS8 PEM file')
        if name == 'compare':
            command.add_argument('--public-key', required=True)
        if name == 'conformance':
            command.add_argument('--profile', required=True, choices=['MAS', 'HKMA-ART', 'FSB', 'OCC', 'SEC', 'FCA'])
            command.add_argument('--as-of', help='ISO timestamp, defaults to now')
        command.set_defaults(func=run)


def run(args):
    from kognita.db import create_all, make_engine, session_scope
    from kognita.evidence import EvidenceWriter
    from kognita.assurance import AssurancePolicy, configure
    from kognita.assurance.runtime import attest, coverage_report
    from kognita.assurance.lifecycle import (approve_components, approve_update, advisory,
                                            record_outcome, assurance_report)
    from kognita.assurance.reports import export_manifest, compare_manifest, conformance_report
    from kognita.signing import EvidenceSigner
    engine = make_engine(args.db)
    try:
        create_all(engine)
        data = json.loads(Path(args.input).read_text()) if hasattr(args, 'input') else None
        with session_scope(engine) as session:
            writer = EvidenceWriter(engine)
            common = {'evidence': writer, 'actor_id': getattr(args, 'actor', '')}
            name = args.assurance_command
            if name == 'configure':
                result = configure(session, args.agent, AssurancePolicy.model_validate(data), **common).model_dump(mode='json')
            elif name == 'attest':
                result = {'digest': attest(session, args.agent, data, **common)}
            elif name == 'components':
                result = approve_components(session, args.agent, data, **common)
            elif name == 'update':
                result = approve_update(session, args.agent, **data, **common)
            elif name == 'outcome':
                result = record_outcome(session, args.agent, data, **common)
            elif name == 'advisory':
                result = advisory(session, **data, **common)
            elif name == 'manifest':
                result = export_manifest(session, args.agent,
                    EvidenceSigner(Path(args.signing_key).read_bytes()), **common)
            elif name == 'compare':
                result = compare_manifest(session, args.agent, data, Path(args.public_key).read_bytes())
            elif name == 'coverage':
                result = coverage_report(session, args.agent)
            elif name == 'report':
                result = assurance_report(session, args.agent)
            elif name == 'conformance':
                result = conformance_report(session, args.agent, args.profile,
                    as_of=datetime.fromisoformat(args.as_of) if args.as_of else None)
            else:
                from kognita.assurance.drills import simulate
                from kognita.envelope import Envelope
                result = simulate(session, args.agent, data['scenario'], Envelope(**data['envelope']), **common)
            print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f'Assurance refused: {exc}', file=sys.stderr)
        return 1
    finally:
        engine.dispose()
