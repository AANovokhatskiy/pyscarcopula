# Refactoring artifacts

This directory contains temporary characterization and migration guardrails
created specifically for an active refactoring.

Tests that describe permanent public behavior belong in the regular test
modules. Before deleting an artifact suite, move every still-relevant
compatibility assertion into its permanent test location and verify the full
test suite.

