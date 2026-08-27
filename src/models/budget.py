"""Parameter budget accounting against the competition's hard <2B limit.

The limit is easy to breach by accident rather than by design. Two ways it
happens here:

* Loading a checkpoint through a generic AutoModel pulls in the text tower of a
  contrastive model. `google/siglip2-so400m-patch14-384` is 1,136,008,498
  parameters in total but only 428,225,600 in the vision tower -- more than half
  the budget spent on weights that never execute.
* Ensembling. Each additional member is cheap to add and its cost is invisible
  until something totals it up.

So the budget is a checked artifact with a test behind it, not a note in the
README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "PARAMETER_LIMIT",
    "BudgetEntry",
    "ParameterBudget",
    "REJECTED_BACKBONES",
]

# The competition rule: models must use fewer than 2 billion parameters.
PARAMETER_LIMIT = 2_000_000_000

# Backbones considered and deliberately not used, kept so the reasoning is
# reviewable rather than folklore.
REJECTED_BACKBONES: dict[str, str] = {
    "PE-Core-G14-448": (
        "1.88B in the vision tower alone (2.35B with the text tower, an "
        "outright violation). Even used correctly it consumes 94% of the "
        "budget, forbidding any ensemble, and needs a custom loader."
    ),
    "DINOv3-ViT-L/16": (
        "303M and otherwise attractive, but gated behind manual approval that "
        "can take days, under a bespoke licence. Evidence also suggests "
        "self-supervised features separate real from generated less well than "
        "multimodal ones when frozen."
    ),
    "siglip2-so400m text tower": (
        "707.7M of never-executed weights (294.9M of it the 256K-token "
        "vocabulary embedding). Avoided by loading the vision tower via timm."
    ),
}


@dataclass(frozen=True)
class BudgetEntry:
    name: str
    parameters: int
    trainable: bool = False
    note: str = ""


@dataclass
class ParameterBudget:
    """Running total of every parameter in the deployed system."""

    entries: list[BudgetEntry] = field(default_factory=list)

    def add(
        self,
        name: str,
        parameters: int,
        trainable: bool = False,
        note: str = "",
    ) -> "ParameterBudget":
        if parameters < 0:
            raise ValueError(f"{name}: parameter count cannot be negative")
        self.entries.append(BudgetEntry(name, parameters, trainable, note))
        return self

    @property
    def total(self) -> int:
        return sum(entry.parameters for entry in self.entries)

    @property
    def trainable(self) -> int:
        return sum(e.parameters for e in self.entries if e.trainable)

    @property
    def frozen(self) -> int:
        return sum(e.parameters for e in self.entries if not e.trainable)

    @property
    def headroom(self) -> int:
        return PARAMETER_LIMIT - self.total

    @property
    def within_limit(self) -> bool:
        return self.total < PARAMETER_LIMIT

    @property
    def utilization(self) -> float:
        return self.total / PARAMETER_LIMIT

    def check(self) -> None:
        """Raise if the budget is breached. Called from the test suite."""
        if not self.within_limit:
            raise ValueError(
                f"parameter budget exceeded: {self.total:,} >= "
                f"{PARAMETER_LIMIT:,} (over by {-self.headroom:,})"
            )

    def to_markdown(self) -> str:
        lines = [
            "| Component | Parameters | State |",
            "|---|---:|---|",
        ]
        for entry in self.entries:
            state = "trainable" if entry.trainable else "frozen"
            lines.append(
                f"| {entry.name} | {entry.parameters:,} | {state} |"
            )
        lines.append(
            f"| **Total** | **{self.total:,}** | "
            f"**{self.utilization:.1%} of the 2B limit** |"
        )
        return "\n".join(lines)
