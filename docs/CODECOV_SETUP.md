# Codecov Setup Guide

## Overview

Codecov provides enhanced coverage reporting and tracking for the WJP Analyzer project. This guide explains how to set it up.

## Prerequisites

1. GitHub account
2. Repository access to `SUJITHVEMUULAPALLI/wjp-analyzer`
3. Admin access to repository settings

## Setup Steps

### 1. Sign Up for Codecov

1. Go to https://codecov.io
2. Click "Sign up with GitHub"
3. Authorize Codecov to access your repositories
4. Select the `wjp-analyzer` repository

### 2. Get Upload Token

1. After adding the repository, Codecov will provide an upload token
2. Copy the token (it looks like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

### 3. Add Token to GitHub Secrets

1. Go to your repository: https://github.com/SUJITHVEMUULAPALLI/wjp-analyzer
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `CODECOV_TOKEN`
5. Value: Paste your Codecov token
6. Click **Add secret**

### 4. Update GitHub Actions Workflow

The workflow is already configured to upload to Codecov! It will automatically use the token from secrets.

**Current configuration** (`.github/workflows/tests.yml`):
```yaml
- name: Upload coverage reports
  uses: codecov/codecov-action@v3
  if: matrix.python-version == '3.11'
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
    fail_ci_if_error: false
```

### 5. Verify Setup

1. Push a commit to trigger CI/CD
2. Check GitHub Actions - the Codecov upload step should succeed
3. Visit https://codecov.io/gh/SUJITHVEMUULAPALLI/wjp-analyzer
4. You should see coverage reports

## Using Codecov

### View Coverage Reports

- **Dashboard**: https://codecov.io/gh/SUJITHVEMUULAPALLI/wjp-analyzer
- **Coverage Trends**: View coverage over time
- **File-by-File**: See coverage for individual files
- **Pull Requests**: Coverage comments on PRs

### Coverage Badges

Add to your README.md:
```markdown
[![codecov](https://codecov.io/gh/SUJITHVEMUULAPALLI/wjp-analyzer/branch/master/graph/badge.svg)](https://codecov.io/gh/SUJITHVEMUULAPALLI/wjp-analyzer)
```

### Coverage Thresholds

Codecov can enforce coverage thresholds:
- **Minimum Coverage**: Set in Codecov settings
- **PR Comments**: Automatic comments on PRs
- **Status Checks**: Block merges if coverage drops

## Troubleshooting

### Upload Fails

1. **Check Token**: Verify `CODECOV_TOKEN` is set in GitHub Secrets
2. **Check Permissions**: Ensure Codecov has access to the repository
3. **Check Workflow**: Verify the upload step runs (only on Python 3.11)

### No Coverage Data

1. **Check Coverage File**: Ensure `coverage.xml` is generated
2. **Check Path**: Verify the file path in workflow matches actual location
3. **Check Format**: Ensure coverage format is XML

### Coverage Not Updating

1. **Check Branch**: Ensure you're pushing to the correct branch
2. **Check Workflow**: Verify CI/CD is running successfully
3. **Check Codecov**: Visit Codecov dashboard for error messages

## Advanced Configuration

### Multiple Flags

If you want to track coverage for different test suites separately:

```yaml
- name: Upload coverage reports
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: unittests,integration,perf
    name: codecov-umbrella
```

### Coverage Thresholds

Set in Codecov dashboard:
- **Minimum Coverage**: 90%
- **PR Comments**: Enabled
- **Status Checks**: Enabled

### Notifications

Configure in Codecov settings:
- Email notifications
- Slack integration
- GitHub status checks

## Benefits

✅ **Enhanced Reporting**: Better visualization of coverage  
✅ **Trend Tracking**: See coverage changes over time  
✅ **PR Integration**: Automatic coverage comments  
✅ **Badges**: Display coverage in README  
✅ **File-Level**: See which files need more tests  

---

**Status**: ✅ Workflow configured, ready for token setup  
**Next Step**: Add `CODECOV_TOKEN` to GitHub Secrets

