# AI Copilot Instructions: Conversational Language Teacher

## Project Architecture

This is a modular Python application for AI-powered language learning with conversational practice. The codebase follows the following architecture and code conventions to facilitate maintainability and extensibility.

## Update Guidelines

Please follow these guidelines when updating or adding new code:

1. The code should be written
  * in a professional manner
  * preferring clarity and maintainability
    * avoid extra unused code:
      * if some functionality is important it should fail loudly when not working
      * don't use error-catching to hide problems
      * don't use fallbacks excluding cases when it is a planned behavior in normal settings (e.g. network failed to send request)
      * prefer asserts over "if false: RuntimeError" blocks
  * if some approach seems to be not the best, please clearly state it and wait for confirmation

2. New files should be added to conversa/generated/ directory

3. Use type hints for all functions and methods

4. Include docstrings for all classes and functions using the Google style.

5. Make tests in tests directory for any new functionality added. Use pytest framework, avoid unittest unless absolutely necessary. Avoid line-by-line mocking; prefer really testing the logic.

## Github workflow
* feature development should be done in feature branches
* the main branch is protected and requires PR reviews (at least 1 approval) before merging
* dev branch is used for integration testing of multiple features before merging to main (squash merge)
