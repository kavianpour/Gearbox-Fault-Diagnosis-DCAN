"""
Method registry (Section III.C of the paper).

Each deep comparison method is a combination of two switches over the SAME
condition-identification backbone, so they share code and "use the same
hyper-parameters and settings as the proposed method":

  * use_adversarial : include the domain discriminator (gradient reversal)?
  * da_loss         : distribution-matching loss -> 'none' | 'coral' | 'mkmmd'

This reproduces the paper's definitions:

  - CNN            : condition-identification part only, source-only training,
                     classification loss only, no domain adaptation module.
  - C-MKMMD        : the same baseline plus the MKMMD module, using kernels
                     with the five bandwidths {0.001, 0.01, 1, 10, 100}.
  - C-CORAL        : the same baseline plus the CORAL module.
  - Proposed (DCAN): baseline + CORAL + adversarial domain discriminator.

DANN is the shallow adversarial baseline; JDA is the shallow joint-distribution
baseline. Both live in classic.py, together with the note on RWKDCAE.
"""

from dataclasses import dataclass


@dataclass
class MethodSpec:
    name: str
    use_adversarial: bool
    da_loss: str = "none"          # 'none' | 'coral' | 'mmd' | 'mkmmd'


DEEP_METHODS = {
    "CNN":      MethodSpec("CNN",      use_adversarial=False, da_loss="none"),
    "C-MKMMD":  MethodSpec("C-MKMMD",  use_adversarial=False, da_loss="mkmmd"),
    "C-CORAL":  MethodSpec("C-CORAL",  use_adversarial=False, da_loss="coral"),
    "DCAN":     MethodSpec("DCAN",     use_adversarial=True,  da_loss="coral"),
}

PROPOSED = "DCAN"
