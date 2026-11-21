"""Shared formatting helpers for model-facing instruction prompts."""

from __future__ import annotations


def render_instruction_template(template: str, instruction: str) -> str:
    """Format an instruction without duplicating its terminal punctuation."""

    rendered_template = template
    if instruction.rstrip().endswith((".", "!", "?")):
        for placeholder in ("{instruction}", "{task}"):
            for punctuation in (".", "!", "?"):
                rendered_template = rendered_template.replace(
                    placeholder + punctuation,
                    placeholder,
                )
    return rendered_template.replace("{instruction}", instruction).replace(
        "{task}", instruction
    )


__all__ = ["render_instruction_template"]
