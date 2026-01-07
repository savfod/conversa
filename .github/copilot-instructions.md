# AI Copilot Instructions: Conversational Language Teacher

## Project Architecture

This is a modular Python application for AI-powered language learning with conversational practice. The codebase follows the following architecture and code conventions to facilitate maintainability and extensibility.

## Update Guidelines

Please follow these guidelines when updating or adding new code:

0. If some prompt approach seems to be not good, please clearly state it and wait for confirmation

1. Write code in a modular, profesional manner.
* Use type hints for all functions and methods
* Include docstrings for all classes and functions using the Google style.
* Follow PEP8 style guide (enforced by ruff in CI)

2. Prefer clarity and maintainability, avoid extra unused code:
* don't use error-catching to hide problems
  * if some functionality is important it should fail loudly when not working
  * don't use fallbacks excluding cases when it is a planned behavior in normal settings (e.g. network failed to send request)
* prefer asserts over "if false: RuntimeError" blocks

3. Make tests in tests directory for any new functionality added. Use pytest framework, avoid unittest unless absolutely necessary. Avoid line-by-line mocking; prefer really testing the logic.
* the tests for file SOME_PATH/FILE.py should be in tests/SOME_PATH/FILE.py
* to run the tests please use the `uv run -m pytest` command, to simplified check use uv run -m pytest tests -m "not slow"

## Github workflow
* feature development should be done in feature branches
* the main branch is protected and requires PR reviews (at least 1 approval) before merging
* the main info about the change should be in the change.md file in the root directory during the development (added as PR description on PR)
* dev branch is used for integration testing of multiple features before merging to main (squash merge)
