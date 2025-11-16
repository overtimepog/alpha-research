---
name: Backend Queries
description: Write efficient, secure database queries using ORMs or raw SQL, preventing N+1 problems, SQL injection, and performance issues. Use this skill when writing database queries, implementing data access layers, creating repository patterns, or optimizing query performance in service files, query builders, or data access objects. Apply this skill when using parameterized queries, implementing eager loading to avoid N+1 queries, selecting only needed columns, adding WHERE/JOIN/ORDER BY clauses, or working with query optimization, indexes, and database performance tuning. This skill ensures queries use proper SQL injection prevention, implement transactions for data consistency, cache expensive queries appropriately, and follow best practices for query timeouts, connection pooling, and database resource management.
---

# Backend Queries

This Skill provides Claude Code with specific guidance on how to adhere to coding standards as they relate to how it should handle backend queries.

## When to use this skill

- When writing database queries using ORM methods or query builders
- When implementing raw SQL queries with proper parameterization
- When creating repository patterns or data access layers
- When optimizing queries to prevent N+1 query problems using eager loading
- When adding JOINs, WHERE clauses, or complex query logic
- When selecting specific columns instead of using SELECT * for performance
- When implementing query pagination, filtering, or sorting logic
- When wrapping related database operations in transactions
- When adding query timeouts or implementing query performance monitoring
- When working with query optimization, EXPLAIN plans, or index usage analysis
- When implementing query caching strategies for expensive or frequent queries
- When preventing SQL injection through parameterized queries and input validation
- When working with files like `queries.py`, `repositories/`, `dao/`, or service layer query logic

## Instructions

For details, refer to the information provided in this file:
[backend queries](../../../agent-os/standards/backend/queries.md)
