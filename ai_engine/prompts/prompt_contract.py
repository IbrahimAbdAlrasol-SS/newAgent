"""
Reusable prompt contract builder.

Builds prompts with a consistent structure:
ROLE / INSTRUCTIONS / CONTEXT / INPUT / OUTPUT FORMAT / EXAMPLES / CONSTRAINTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PromptContract:
    """Structured prompt sections for consistent prompt engineering."""

    role: str
    instructions: list[str]
    context: str
    input_data: str
    output_format: str
    constraints: list[str]
    examples: list[str] = field(default_factory=list)
    version: str = "v1"

    def render(self) -> str:
        """Render the contract as a compact markdown prompt."""
        lines: list[str] = [
            f"# ROLE [{self.version}]",
            self.role.strip(),
            "",
            "# INSTRUCTIONS",
            *[f"- {line}" for line in self.instructions if line.strip()],
            "",
            "# CONTEXT",
            self.context.strip() or "N/A",
            "",
            "# INPUT",
            self.input_data.strip(),
            "",
            "# OUTPUT FORMAT",
            self.output_format.strip(),
        ]

        if self.examples:
            lines.extend(
                [
                    "",
                    "# EXAMPLES",
                    *[
                        f"- {idx}. {example}"
                        for idx, example in enumerate(self.examples, start=1)
                        if example.strip()
                    ],
                ]
            )

        lines.extend(
            [
                "",
                "# CONSTRAINTS",
                *[f"- {line}" for line in self.constraints if line.strip()],
            ]
        )
        return "\n".join(lines)
