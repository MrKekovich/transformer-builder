# Testing Guide

---

Testing is an integral part of maintaining code quality and ensuring the reliability of our project. This guide outlines
our testing practices and how contributors can test their code contributions.

---

## Testing Frameworks

We rely on two primary testing frameworks for this project:

1. **Pytest**: [Pytest](https://docs.pytest.org/en/7.4.x/) is a powerful and widely-used testing framework for Python.
   It simplifies test discovery, execution, and reporting, making it an excellent choice for writing and running tests.

2. **Hypothesis**: [Hypothesis](https://hypothesis.works/) is a property-based testing tool that helps us explore a
   broader range of scenarios. It generates test cases automatically and is particularly useful for uncovering edge
   cases and unexpected behavior.

---

## Writing Tests

When contributing to this project, it's essential to include tests for your code changes. Follow these guidelines for
writing tests:

1. **Test Coverage**: Aim to cover your code with tests. Write tests for different code paths and scenarios, including
   edge cases.

2. **Descriptive Test Names**: Use descriptive test names that explain the purpose of the test. A well-named test makes
   it easier to understand its purpose.

3. **Arrange-Act-Assert**: Follow the AAA (Arrange-Act-Assert) pattern in your test functions. Arrange the initial
   conditions, act on the code, and assert the expected outcomes.

4. **Mocking**: When necessary, use mocking to isolate the code under test from external dependencies.

5. **Continuous Integration**: Our project uses continuous integration to automatically run tests on pull requests.
   Ensure that your tests pass in the CI environment.

---

## Reporting Test Failures

If you encounter test failures, please open an [issue](https://github.com/MrKekovich/transformer-builder/issues) to
report the problem. Provide detailed information about the failure and steps to reproduce it. This helps us identify and
address issues promptly.

Testing is a crucial part of maintaining the quality and reliability of our project. Your contributions to testing are
highly valued and help us ensure the project's continued effectiveness and reliability.
