"""Supervisory alignment and runtime assurance (0.8).

Authenticated administration supplies policies and external trust roots. All
new approvals are immutable evidence records. External boundaries perform the
actual identity verification, sandboxing, monitoring and legal assessment.
"""
from kognita.assurance.records import AssurancePolicy, configure
from kognita.assurance.trust import Assertion, Attestation, sign_record

__all__ = ['AssurancePolicy', 'Assertion', 'Attestation', 'configure', 'sign_record']
