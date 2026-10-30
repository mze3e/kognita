"""Pass the evaluation gate in tests that redeploy a tier 2 or higher agent."""

from kognita.evaluation import METRICS, EvaluationResult, record_evaluation


def evaluation_thresholds():
    """A bar any recorded passing result meets. The gate still requires the result."""
    return {name: (1.0 if name == "hallucination_rate" else 0.0) for name in METRICS}


def satisfy_evaluation(session, evidence, agent):
    """Record a system-level result for this agent's version and model."""
    record_evaluation(
        session,
        EvaluationResult(
            agent_id=agent.name,
            version=agent.version,
            model_version=agent.model,
            level="system",
            metrics={name: (0.0 if name == "hallucination_rate" else 1.0) for name in METRICS},
        ),
        evidence=evidence,
    )
