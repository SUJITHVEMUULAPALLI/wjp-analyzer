# CI/CD Monitoring Guide

## Overview

This guide explains how to monitor and troubleshoot the GitHub Actions CI/CD pipeline for WJP Analyzer.

## Quick Access

- **Actions Tab**: https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/actions
- **Workflow File**: https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/blob/master/.github/workflows/tests.yml

## Monitoring Workflow Runs

### 1. View All Runs

1. Go to the repository on GitHub
2. Click the **Actions** tab
3. You'll see a list of all workflow runs

### 2. View Specific Run

1. Click on a workflow run
2. You'll see:
   - **Status**: ✅ Success, ❌ Failure, ⏳ In Progress
   - **Duration**: How long the run took
   - **Jobs**: Individual test jobs (one per Python version)

### 3. View Job Details

1. Click on a job (e.g., "test (3.11)")
2. You'll see:
   - **Steps**: Each step in the workflow
   - **Logs**: Detailed output from each step
   - **Artifacts**: Coverage reports, etc.

### 4. View Test Results

1. Expand the "Run tests with coverage" step
2. You'll see:
   - Test execution output
   - Coverage report
   - Pass/fail status

## Understanding Status

### ✅ Success
- All tests passed
- Coverage meets threshold (≥90%)
- All steps completed

### ❌ Failure
- Tests failed
- Coverage below threshold
- Build/install error

### ⏳ In Progress
- Workflow is currently running
- Wait for completion

## Common Issues

### Tests Failing

**Symptoms**: Red X on workflow run, test failures in logs

**Solutions**:
1. Check test output in logs
2. Run tests locally: `pytest tests/ -v`
3. Check for dependency issues
4. Verify Python version compatibility

### Coverage Below Threshold

**Symptoms**: Workflow fails with "Coverage is below threshold"

**Solutions**:
1. Check coverage report in logs
2. Run locally: `pytest tests/ --cov=wjp_analyser --cov-report=term-missing`
3. Add tests for uncovered code
4. Review coverage report to identify gaps

### Build/Install Errors

**Symptoms**: Failure in "Install dependencies" step

**Solutions**:
1. Check `requirements.txt` for issues
2. Verify all dependencies are available
3. Check Python version compatibility
4. Review error messages in logs

### Timeout Issues

**Symptoms**: Workflow times out or takes too long

**Solutions**:
1. Optimize slow tests
2. Use `@pytest.mark.slow` for long-running tests
3. Run performance tests separately
4. Consider increasing timeout limits

## Workflow Configuration

### Current Setup

- **Trigger**: Push to `master` or `develop`, PRs
- **Python Versions**: 3.10, 3.11, 3.12
- **Coverage Threshold**: 90%
- **Artifacts**: HTML coverage report (30 days retention)

### Modifying Workflow

Edit `.github/workflows/tests.yml`:
- Change Python versions
- Adjust coverage threshold
- Add/remove steps
- Modify test commands

## Best Practices

### Before Pushing

1. ✅ Run tests locally: `pytest tests/ -v`
2. ✅ Check coverage: `pytest tests/ --cov=wjp_analyser --cov-report=term`
3. ✅ Fix any failures
4. ✅ Commit and push

### After Pushing

1. ✅ Check Actions tab immediately
2. ✅ Monitor workflow progress
3. ✅ Review results when complete
4. ✅ Fix any failures before merging

### Regular Monitoring

1. ✅ Check Actions tab weekly
2. ✅ Review coverage trends
3. ✅ Address failing tests promptly
4. ✅ Update dependencies regularly

## Notifications

### GitHub Notifications

Enable notifications in GitHub settings:
1. Go to repository settings
2. Click "Notifications"
3. Enable "Actions" notifications

### Email Notifications

GitHub will email you when:
- Workflow fails
- Workflow succeeds (optional)
- PR status changes

## Troubleshooting Commands

### Local Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=wjp_analyser --cov-report=term-missing

# Run specific test suite
pytest tests/unit/web/ -v
pytest tests/integration/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Check Workflow Syntax

```bash
# Validate YAML syntax
yamllint .github/workflows/tests.yml

# Or use online validator
# https://github.com/actions/runner/blob/main/.github/workflows/workflow-syntax-check.yml
```

## Performance Metrics

### Expected Run Times

- **Unit Tests**: ~10-15 seconds
- **Integration Tests**: ~5-10 seconds
- **Full Suite**: ~20-30 seconds
- **With Coverage**: ~30-45 seconds

### Optimization Tips

1. Use `pytest-xdist` for parallel execution
2. Cache dependencies in workflow
3. Skip slow tests in CI (run separately)
4. Use test markers to organize tests

## Status Badges

Add to README.md:

```markdown
![Tests](https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer/workflows/WJP%20Analyzer%20Tests/badge.svg)
```

---

**Status**: ✅ CI/CD pipeline active and monitoring  
**Last Check**: After workflow push

