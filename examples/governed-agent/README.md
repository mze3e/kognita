# governed-agent

Created by `kognita scaffold --template governed-agent`. The directory holds this
app and a SQLite policy and evidence store (`kognita.db`).

From the directory:

```bash
python run_demo.py
```

The script runs four scenarios:

1. Allowed query. `dossier-agent` reads subject `2` (Tan Longitudinal Study). The decision and the retrieval are logged.
2. Denied query. The agent asks for subject `1` (Rivera Cohort). The call is denied with citations and nothing is retrieved.
3. Tampering. One evidence row is edited. `kognita evidence verify --db kognita.db` reports the break.
4. AI gateway. A prompt containing `ana@example.org` is sent through the gateway already in Kognita. A local stand-in stands in for the provider, so no provider key is required. The stand-in receives the redacted prompt. `MODEL_CALL` evidence stores the manifest hash, not the prompt.

The gateway call is made before the row is edited, so the manifest is on the chain that verify then reports as broken.
