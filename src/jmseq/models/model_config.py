"""
LGSSMConfig dataclass — controls which parameters are free vs fixed for each
of the five ctsem-equivalent model variants.

| Name               | free_drift | free_diffusion | free_cint |
|--------------------|------------|----------------|-----------|
| model_lmm          | False      | False          | True      |
| model_nolmm        | True       | True           | False     |
| model_lmmdiff      | False      | True           | True      |
| model_lmmdrift     | True       | False          | True      |
| model_lmmdriftdiff | True       | True           | True      |
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LGSSMConfig:
    name: str
    free_drift: bool      # A matrix: True = estimated, False = zero matrix
    free_diffusion: bool  # G matrix: True = estimated, False = zero matrix
    free_cint: bool       # b vector: True = estimated, False = zero vector
    p: int                # state / observation dimension (number of biomarkers)
    n_tipred: int = 0     # number of time-invariant predictors affecting m0


# Five pre-defined variants (p and n_tipred set at runtime via .replace())
VARIANTS = {
    "model_lmm":          LGSSMConfig("model_lmm",          False, False, True,  2),
    "model_nolmm":        LGSSMConfig("model_nolmm",        True,  True,  False, 2),
    "model_lmmdiff":      LGSSMConfig("model_lmmdiff",      False, True,  True,  2),
    "model_lmmdrift":     LGSSMConfig("model_lmmdrift",     True,  False, True,  2),
    "model_lmmdriftdiff": LGSSMConfig("model_lmmdriftdiff", True,  True,  True,  2),
}


def make_config(name: str, p: int, n_tipred: int = 0) -> LGSSMConfig:
    """Return an LGSSMConfig for the named variant with specified dimensions."""
    base = VARIANTS[name]
    import dataclasses
    return dataclasses.replace(base, p=p, n_tipred=n_tipred)
