# Dependency map

`kognita deps map --db STORE` prints a JSON report from local registers and
verified evidence. `kognita deps check --db STORE` prints the same report and
returns 1 if any finding exists, or 0 when no finding exists. Both refuse
analysis when evidence verification fails, citing `R8-UNDECLARED`.

Paths link the business process (or use-case ID), agent, approved model card,
provider, tools and declared data sources. Cloud and vector platform metadata
remain null where unavailable. No infrastructure or network discovery runs.
Agent and use-case declarations both contribute associations. Tool allowlists
and permitted systems are declarations, never proof of a call.

The comparison uses current effective registers and all retained verified
`MODEL_CALL`, `TOOL_CALL` and `RETRIEVAL` events. Model calls explicitly marked
unsent are excluded. Policy-decision envelopes with the same correlation ID
supply use-case and agent attribution; interaction records can supply a missing
use case. Ambiguous or missing attribution is a finding. Retrieval source labels
are compared exactly with approved sources and data-source references; labels
that differ from inventory identifiers require reconciliation.

`R8-UNDECLARED` identifies observed dependencies absent from declarations.
`R8-UNUSED` identifies declarations unobserved in the retained evidence, including
permitted systems without matching destination evidence. It does not prove that
a dependency is safe to remove. Historical calls can diverge from current
registers after an intentional change. This report identifies differences for
review; it does not reconstruct historical approval or prove the absence of
calls outside the gateways.
