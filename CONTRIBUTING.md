# Contributing to Anvil-Embodied-AI

Thank you for your interest in contributing to Anvil-Embodied-AI! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for Python package management
- Docker (for ROS2 development and testing)
- Git

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
   cd anvil-embodied-ai
   ```

2. Run the setup script:
   ```bash
   ./scripts/setup-dev.sh
   ```

   This will:
   - Install uv if not present
   - Create a virtual environment
   - Install all dependencies
   - Set up pre-commit hooks

3. Verify the setup:
   ```bash
   ./scripts/test.sh
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/<description>` - New features
- `fix/<description>` - Bug fixes
- `docs/<description>` - Documentation updates
- `refactor/<description>` - Code refactoring

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes, following the code style guidelines.

3. Run tests locally:
   ```bash
   ./scripts/test.sh
   ```

4. Run linting:
   ```bash
   ./scripts/lint.sh
   ```

5. Commit your changes:
   ```bash
   git add <files>
   git commit -m "feat: description of changes"
   ```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Pull Requests

1. Push your branch:
   ```bash
   git push -u origin feature/my-feature
   ```

2. Open a Pull Request on GitHub.

3. Fill in the PR template with:
   - Description of changes
   - Related issues
   - Testing performed

4. Wait for CI checks to pass.

5. Request review from maintainers.

## Code Style

### Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use ruff for formatting and linting

### Documentation

- Use docstrings for public functions and classes
- Follow Google-style docstring format
- Keep README files up to date

## Testing

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run specific package tests
uv run pytest packages/mcap_converter/tests/

# Run with coverage
uv run pytest --cov=packages/
```

### Writing Tests

- Place tests in `tests/` directory within each package
- Name test files `test_*.py`
- Use pytest fixtures for common setup
- Aim for meaningful test coverage

## ROS2 Development

### Building ROS2 Packages

```bash
# Using Docker (recommended)
./scripts/build-docker.sh

# Or locally with colcon
cd ros2
colcon build
```

### Running Integration Tests

```bash
docker compose -f docker/inference/docker-compose.test.yml up \
  --abort-on-container-exit \
  --exit-code-from test-runner
```

## Questions?

- Open an issue on GitHub for bugs or feature requests
- Check existing issues before creating new ones
- For security issues, please email security@anvil.bot directly

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
