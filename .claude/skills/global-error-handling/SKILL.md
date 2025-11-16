---
name: Global Error Handling
description: Implement robust error handling with user-friendly messages, fail-fast validation, specific exception types, and proper resource cleanup across the application. Use this skill when writing try-catch blocks, throwing or catching exceptions, implementing error boundaries, handling API errors, or managing error states in any part of the codebase. Apply this skill when validating inputs early (fail fast), providing clear actionable error messages to users without exposing security details, using specific exception types rather than generic errors, implementing centralized error handling at appropriate boundaries, designing for graceful degradation, implementing retry logic with exponential backoff, or ensuring resources are cleaned up in finally blocks. This skill ensures errors are caught and handled appropriately, user experience remains positive even when errors occur, security is maintained by not leaking sensitive information, and systems continue operating or degrade gracefully when non-critical services fail.
---

# Global Error Handling

This Skill provides Claude Code with specific guidance on how to adhere to coding standards as they relate to how it should handle global error handling.

## When to use this skill

- When writing try-catch-finally blocks or exception handling logic
- When throwing custom errors or exceptions with meaningful messages
- When validating inputs and failing fast on invalid data
- When implementing error boundaries in React or error handling middleware
- When handling API errors, network failures, or external service calls
- When providing user-facing error messages that are clear and actionable
- When logging errors with appropriate context (stack traces, user actions, timestamps)
- When implementing retry strategies with exponential backoff for transient failures
- When ensuring resource cleanup (file handles, database connections, network sockets)
- When centralizing error handling at API controllers, service boundaries, or UI error boundaries
- When designing systems to degrade gracefully when non-critical services fail
- When avoiding exposing sensitive information (stack traces, internal paths) to end users
- When using specific exception types (ValidationError, NotFoundError) instead of generic ones
- When implementing proper error propagation without swallowing errors silently

## Instructions

For details, refer to the information provided in this file:
[global error handling](../../../agent-os/standards/global/error-handling.md)
