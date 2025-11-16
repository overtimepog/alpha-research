---
name: Global Commenting
description: Write self-documenting code with minimal, evergreen comments that explain WHY rather than WHAT, avoiding temporary change notes or obvious descriptions. Use this skill when adding comments to any code file to ensure comments provide value by explaining complex logic, business rules, or non-obvious decisions rather than describing what the code does. Apply this skill when writing code that needs clarification, when documenting complex algorithms or business logic, or when removing outdated or redundant comments that describe obvious operations. This skill prioritizes writing clear, expressive code through meaningful naming and structure that minimizes the need for comments, ensures any comments added are helpful and explain intent or reasoning, and avoids leaving comments about recent changes, temporary fixes, or information that belongs in commit messages or documentation.
---

# Global Commenting

This Skill provides Claude Code with specific guidance on how to adhere to coding standards as they relate to how it should handle global commenting.

## When to use this skill

- When writing any code across the project (to determine if comments are needed)
- When adding comments to explain complex algorithms, business logic, or non-obvious decisions
- When documenting WHY code exists rather than WHAT it does
- When refactoring code to make it more self-documenting and reduce comment needs
- When removing outdated, redundant, or obvious comments
- When identifying comments that describe recent changes or fixes (which should be removed)
- When choosing meaningful variable and function names that reduce the need for explanatory comments
- When cleaning up commented-out code blocks (which should be deleted, not kept as comments)
- When ensuring comments remain evergreen and relevant long-term
- When writing docstrings or JSDoc for public APIs, exported functions, or complex interfaces
- When adding TODO or FIXME comments only when absolutely necessary with clear action items
- When reviewing code to verify comments add value and aren't stating the obvious

## Instructions

For details, refer to the information provided in this file:
[global commenting](../../../agent-os/standards/global/commenting.md)
