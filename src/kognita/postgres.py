"""T2 Postgres storage guards and run isolation.

Schema creation installs RLS. Reapply it with
``kognita db rls apply --db URL`` for an existing store.
The trusted connection broker must SET LOCAL run.agent_id for each transaction,
using a bound value via set_config('run.agent_id', :value, true). Never hand
agents SQL credentials: custom settings are not authentication. Runtime roles
must be NOSUPERUSER NOBYPASSRLS and cannot own or alter schema objects.
Missing identities see no runs and cannot insert runs. Legacy runs have no
owner and remain inaccessible until an administrator assigns ownership.
Replication, credentials, role provisioning and trusted identity binding are
operator responsibilities. Evidence is a global infrastructure chain; do not
grant agents access to evidence or other infrastructure tables.
"""


def _install(connection):
    connection.exec_driver_sql('''CREATE TABLE IF NOT EXISTS evidence_head (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        sequence INTEGER NOT NULL, event_hash TEXT NOT NULL)''')
    connection.exec_driver_sql('''INSERT INTO evidence_head
        SELECT 1, sequence, event_hash FROM evidence_events
        ORDER BY sequence DESC LIMIT 1 ON CONFLICT (id) DO NOTHING''')
    connection.exec_driver_sql('''CREATE OR REPLACE FUNCTION _evidence_guard()
        RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
        RAISE EXCEPTION 'T2: evidence is append-only'; END; $$''')
    connection.exec_driver_sql('''CREATE OR REPLACE FUNCTION _evidence_advance()
        RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
        INSERT INTO evidence_head VALUES (1, NEW.sequence, NEW.event_hash)
        ON CONFLICT (id) DO UPDATE SET sequence = EXCLUDED.sequence,
            event_hash = EXCLUDED.event_hash;
        RETURN NEW; END; $$''')
    connection.exec_driver_sql('''CREATE OR REPLACE FUNCTION _head_guard()
        RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
        IF TG_OP = 'UPDATE' THEN
            IF NEW.sequence > OLD.sequence THEN RETURN NEW; END IF;
        END IF;
        RAISE EXCEPTION 'T2: the evidence head only moves forward'; END; $$''')
    for table, name, action, function in (
        ('evidence_events', '_evidence_immutable', 'BEFORE UPDATE OR DELETE OR TRUNCATE', '_evidence_guard'),
        ('evidence_events', '_evidence_head_advance', 'AFTER INSERT', '_evidence_advance'),
        ('evidence_head', '_head_immutable', 'BEFORE UPDATE OR DELETE OR TRUNCATE', '_head_guard'),
    ):
        connection.exec_driver_sql(f'DROP TRIGGER IF EXISTS {name} ON {table}')
        # TRUNCATE triggers are statement-level; keep them separate.
        row_action = action.replace(' OR TRUNCATE', '')
        connection.exec_driver_sql(f'CREATE TRIGGER {name} {row_action} ON {table} FOR EACH ROW EXECUTE FUNCTION {function}()')
        if 'TRUNCATE' in action:
            connection.exec_driver_sql(f'DROP TRIGGER IF EXISTS {name}_truncate ON {table}')
            connection.exec_driver_sql(f'CREATE TRIGGER {name}_truncate BEFORE TRUNCATE ON {table} FOR EACH STATEMENT EXECUTE FUNCTION _evidence_guard()')
    _apply_rls(connection)


def _apply_rls(connection):
    if connection.dialect.name != 'postgresql':
        raise ValueError('T2: RLS requires Postgres')
    connection.exec_driver_sql('''ALTER TABLE runs ADD COLUMN IF NOT EXISTS
        _agent_id TEXT''')
    connection.exec_driver_sql("""ALTER TABLE runs ALTER COLUMN _agent_id
        SET DEFAULT NULLIF(current_setting('run.agent_id', true), '')""")
    connection.exec_driver_sql('ALTER TABLE runs ENABLE ROW LEVEL SECURITY')
    connection.exec_driver_sql('ALTER TABLE runs FORCE ROW LEVEL SECURITY')
    connection.exec_driver_sql('DROP POLICY IF EXISTS _run_agent ON runs')
    connection.exec_driver_sql('''CREATE POLICY _run_agent ON runs
        USING (_agent_id = NULLIF(current_setting('run.agent_id', true), ''))
        WITH CHECK (_agent_id = NULLIF(current_setting('run.agent_id', true), ''))''')
