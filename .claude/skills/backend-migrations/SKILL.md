---
name: Backend Migrations
description: Create and manage database migrations for schema changes, ensuring zero-downtime deployments and data integrity. Use this skill when creating migration files, modifying database schemas, adding or altering tables/columns/indexes, or working with migration tools like Alembic, Flyway, Liquibase, or framework-specific migration systems (Django migrations, Rails migrations, Prisma migrations). Apply this skill when implementing reversible migrations with up/down methods, handling data migrations separately from schema changes, creating indexes on large tables, or planning backwards-compatible schema changes for high-availability systems. This skill ensures migrations are version-controlled, focused, safe to rollback, and compatible with CI/CD pipelines and zero-downtime deployment strategies.
---

# Backend Migrations

This Skill provides Claude Code with specific guidance on how to adhere to coding standards as they relate to how it should handle backend migrations.

## When to use this skill

- When creating new database migration files for schema changes
- When adding, removing, or modifying database tables, columns, or constraints
- When creating or dropping indexes, foreign keys, or unique constraints
- When working with migration tools like Alembic, Flyway, Liquibase, Django migrations, or Rails migrations
- When implementing reversible migrations with proper up/down or forward/backward methods
- When separating schema changes from data migrations for safety
- When planning zero-downtime deployments requiring backwards-compatible changes
- When creating indexes on large tables using concurrent/online index creation
- When modifying migration files in `migrations/`, `db/migrate/`, or similar directories
- When setting up migration version control and naming conventions
- When implementing migration testing in staging environments
- When handling migration rollbacks or fixing failed migrations

## Instructions

For details, refer to the information provided in this file:
[backend migrations](../../../agent-os/standards/backend/migrations.md)
