"""Decision output for an independently operated execution sandbox."""
from pydantic import BaseModel, ConfigDict, Field, model_validator
from kognita.envelope import Check
from kognita.vocabulary import CheckResult


class _SandboxConstraints(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    credentials: list[str] = Field(default_factory=list)
    network: list[str] = Field(default_factory=list)
    fs_scope: list[str] = Field(default_factory=list)
    permitted_apis: list[str] = Field(default_factory=list)
    permitted_commands: list[str] = Field(default_factory=list)
    data_export: bool = False
    persistence: bool = False
    auto_stop: int = Field(default=30, gt=0, le=3600)
    cpu: int = Field(default=1, gt=0, le=64)
    mem: int = Field(default=256, gt=0, le=65536)
    disk: int = Field(default=64, gt=0, le=65536)

    @model_validator(mode='after')
    def _bounded(self):
        for values in (self.credentials, self.network, self.fs_scope,
                       self.permitted_apis, self.permitted_commands):
            if any(not s.strip() or '*' in s for s in values):
                raise ValueError('S6B-SANDBOX: explicit allow-lists required')
        if any(not p.startswith('/') or '..' in p.split('/') or p == '/' for p in self.fs_scope):
            raise ValueError('S6B-SANDBOX: bounded absolute file paths required')
        return self


def _constraints(envelope, agent):
    permission = (agent.tool_allowlist or {}).get(envelope.tool, {}) if agent else {}
    if not isinstance(permission, dict) or permission.get('verb') != 'execute':
        return None, []
    try:
        # Missing configuration means the restrictive defaults: no credentials,
        # network, filesystem, APIs, commands, export or persistence.
        body = _SandboxConstraints.model_validate(permission.get('_sandbox', {})).model_dump()
        return body, [Check(check='S6B-SANDBOX', citation='S6B-SANDBOX',
                            regime='INTERNAL', result=CheckResult.PASS)]
    except ValueError:
        return None, [Check(check='S6B-SANDBOX', citation='S6B-SANDBOX',
                            regime='INTERNAL', result=CheckResult.FAIL)]
