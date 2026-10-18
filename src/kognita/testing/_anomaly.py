"""Adversarial traffic fixtures for behavioral containment conformance."""


def _attacks():
    # A trusted harness feeds these through the real decision/record boundary.
    return (
        ('client enumeration', {'clients': [str(i) for i in range(100)]}, 'new'),
        ('action loop', {'actions': 1000}, '1'),
        ('document exfiltration', {'documents': 1001}, '1'),
    )
