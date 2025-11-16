---
name: Backend Models
description: Define database models, schemas, and ORM mappings with proper relationships, constraints, and data integrity rules. Use this skill when creating or modifying ORM model classes, database schema definitions, model relationships, validators, or working with files like models.py, schema.prisma, entity classes, or Sequelize/TypeORM/ActiveRecord model definitions. Apply this skill when defining primary keys, foreign keys, indexes, unique constraints, timestamps, or when implementing model validation logic, serializers, and data type mappings. This skill ensures models follow naming conventions, include appropriate database constraints, define clear relationships with cascade behaviors, balance normalization with performance needs, and maintain data integrity at both the model and database levels.
---

# Backend Models

This Skill provides Claude Code with specific guidance on how to adhere to coding standards as they relate to how it should handle backend models.

## When to use this skill

- When creating or modifying ORM model classes (Django models, SQLAlchemy, Prisma, TypeORM, ActiveRecord)
- When defining database schemas, entity classes, or data models
- When establishing relationships between models (one-to-many, many-to-many, foreign keys)
- When adding database constraints (NOT NULL, UNIQUE, CHECK constraints)
- When defining model validators, custom fields, or data type mappings
- When working with files like `models.py`, `schema.prisma`, `entities/`, or model definition files
- When implementing model-level business logic, computed properties, or hooks
- When adding timestamps (created_at, updated_at) or soft delete functionality
- When defining cascade behaviors for relationships (CASCADE, SET NULL, RESTRICT)
- When optimizing model queries with select_related, prefetch_related, or eager loading
- When creating model serializers, schemas, or data transfer objects (DTOs)
- When balancing normalization with denormalization for performance

## Instructions

For details, refer to the information provided in this file:
[backend models](../../../agent-os/standards/backend/models.md)
