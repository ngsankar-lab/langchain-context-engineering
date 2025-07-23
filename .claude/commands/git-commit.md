# Git Commit Command Reference

This file provides quick reference commands for committing changes following git best practices.

## Quick Commit Workflow

### 1. Check Status and Review Changes
```bash
git status
git diff
```

### 2. Stage Changes
```bash
# Stage specific files
git add <filename>

# Stage all changes
git add .

# Review staged changes
git diff --staged
```

### 3. Commit with Proper Message
```bash
# Quick commit for simple changes
git commit -m "<type>(<scope>): <description>"

# Examples following conventional commit format:
git commit -m "feat(auth): add user login functionality"
git commit -m "fix(ui): correct button alignment on mobile"
git commit -m "docs(readme): update installation instructions"
git commit -m "style(header): improve navigation layout"
```

## Conventional Commit Types

- `feat` - New feature for the user
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Formatting, missing semi colons, etc; no code change
- `refactor` - Refactoring production code
- `test` - Adding tests, refactoring test; no production code change
- `chore` - Updating build tasks, package manager configs, etc; no production code change
- `perf` - Performance improvements
- `ci` - Continuous integration related changes
- `build` - Changes that affect the build system or external dependencies

## Common Commit Scenarios

### Adding New Features
```bash
git add src/components/UserProfile.js
git add src/pages/Profile.js
git commit -m "feat(user): add user profile management

Add comprehensive user profile functionality including:
- Profile editing interface
- Avatar upload capability
- Settings management
- Privacy controls

Updates routing to include new profile pages."
```

### Fixing Bugs
```bash
git add src/utils/validation.js
git commit -m "fix(validation): correct email validation regex

Update email validation to properly handle plus signs and
international domains. Previous regex was too restrictive."
```

### Styling Updates
```bash
git add src/styles/components.css
git commit -m "style(responsive): improve mobile layout

- Adjust grid layout for screens under 768px
- Increase touch target sizes for better usability
- Fix navigation menu overflow on small screens"
```

### Documentation Updates
```bash
git add README.md
git add docs/api.md
git commit -m "docs(api): update API documentation and setup guide

- Add new endpoint documentation
- Update authentication examples
- Clarify environment variable requirements
- Fix broken links in setup instructions"
```

## Pre-Commit Checklist Commands

### Run Tests
```bash
npm test
# or
yarn test
# or
python -m pytest
```

### Check Code Quality
```bash
# Linting
npm run lint
# or
eslint src/
# or
flake8 .
```

### Review Changes
```bash
git diff --staged
```

## Branch and Push Workflow

### Create Feature Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/user-authentication
```

### Make Changes and Commit
```bash
# Work on files
git add .
git commit -m "feat(auth): implement user authentication system"
```

### Push and Create PR
```bash
git push origin feature/user-authentication
# Create pull request through GitHub/GitLab interface
```

## Emergency Hotfix Process

### Critical Bug Fixes
```bash
git checkout main
git pull origin main
git checkout -b hotfix/security-vulnerability

# Make urgent changes
git add src/auth/security.js
git commit -m "fix(security): patch critical authentication bypass

URGENT: Fix vulnerability that allowed unauthorized access.
Updated token validation to prevent bypass attacks."

git push origin hotfix/security-vulnerability
# Create emergency PR for immediate review
```

## Common Git Commands

### Undo Last Commit (if not pushed)
```bash
git commit --amend -m "New commit message"
```

### View Commit History
```bash
git log --oneline
git log --graph --oneline --all
```

### Check What Will Be Committed
```bash
git diff --staged --name-only
```

## File-Specific Commit Patterns

### When updating configuration files
```bash
git commit -m "chore(config): update build configuration

- Add new environment variables
- Update dependency versions
- Configure new deployment settings"
```

### When updating dependencies
```bash
git commit -m "chore(deps): update project dependencies

- Update React to v18.2.0
- Bump security-related packages
- Remove unused dependencies"
```

### When fixing performance issues
```bash
git commit -m "perf(api): optimize database queries

- Add database indexes for frequently queried fields
- Implement query result caching
- Reduce N+1 query problems"
```

## Best Practices

### Commit Message Guidelines
- Keep the first line under 50 characters
- Use present tense ("add" not "added")
- Be specific about the change
- Include context in the body for complex changes
- Reference issues/tickets when applicable

### What to Include in Commits
- Related changes only (single responsibility)
- Complete, working functionality
- Necessary tests for new features
- Updated documentation for public APIs

### What NOT to Commit
- Sensitive information (passwords, API keys)
- Large binary files (unless using Git LFS)
- Temporary development files
- IDE-specific configuration files
- Dependencies (use package managers)

## Interactive Rebase for Commit History Cleanup

### Squash Multiple Commits
```bash
# Combine last 3 commits into one
git rebase -i HEAD~3

# In the editor, change 'pick' to 'squash' for commits to combine
# pick abc123 First commit
# squash def456 Second commit  
# squash ghi789 Third commit
```

### Reorder or Edit Commits
```bash
git rebase -i HEAD~5
# Reorder lines to change commit order
# Use 'edit' to modify a specific commit
# Use 'drop' to remove a commit entirely
```

## Commit Message Templates

### Create a Template File
```bash
# Create ~/.gitmessage template
git config --global commit.template ~/.gitmessage
```

### Sample Template Content
```
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>

# Type can be:
# feat     (new feature)
# fix      (bug fix)
# docs     (documentation)
# style    (formatting, no code change)
# refactor (refactoring code)
# test     (adding tests)
# chore    (maintenance)
# perf     (performance improvement)
# ci       (CI/CD changes)
# build    (build system changes)
```

## Advanced Commit Scenarios

### Breaking Changes
```bash
git commit -m "feat(api): redesign authentication system

BREAKING CHANGE: API endpoints now require Bearer token authentication.
Migration guide available in docs/migration-v2.md

Closes #456"
```

### Multi-line Commit with Co-authors
```bash
git commit -m "feat(payment): integrate Stripe payment processing

Add comprehensive payment processing with:
- Credit card validation
- Subscription management  
- Webhook handling for payment events
- Error handling and retry logic

Co-authored-by: Jane Doe <jane@example.com>
Co-authored-by: John Smith <john@example.com>

Fixes #123, #456"
```

### Commit with References
```bash
git commit -m "fix(security): patch XSS vulnerability in user input

Sanitize user input in comment forms to prevent script injection.
Added input validation and output encoding.

References: CVE-2023-12345
See: https://owasp.org/www-community/attacks/xss/
Fixes #789"
```

## Git Hooks for Commit Quality

### Pre-commit Hook Example
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run linter
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Fix errors before committing."
    exit 1
fi

# Run tests
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Fix tests before committing."
    exit 1
fi

# Check commit message format (if using commit-msg hook)
echo "Pre-commit checks passed!"
```

### Commit Message Hook
```bash
#!/bin/sh
# .git/hooks/commit-msg

# Check if commit message follows conventional format
commit_regex='^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Use: <type>(<scope>): <description>"
    echo "Example: feat(auth): add user login functionality"
    exit 1
fi
```

## Troubleshooting Commits

### Fix Wrong Commit Message (Before Push)
```bash
# Change the last commit message
git commit --amend -m "corrected commit message"

# Edit the last commit interactively
git commit --amend
```

### Add Files to Last Commit
```bash
# Stage forgotten files and add to last commit
git add forgotten-file.js
git commit --amend --no-edit
```

### Split Large Commit
```bash
# Reset to before the large commit but keep changes
git reset HEAD~1

# Stage and commit changes in smaller, logical chunks
git add file1.js
git commit -m "feat(auth): add login validation"

git add file2.js
git commit -m "feat(auth): add logout functionality"
```

### Undo Commits (Various Scenarios)
```bash
# Undo last commit, keep changes staged
git reset --soft HEAD~1

# Undo last commit, keep changes unstaged
git reset HEAD~1

# Undo last commit, discard all changes
git reset --hard HEAD~1

# Revert a commit that's already pushed (creates new commit)
git revert <commit-hash>
```

## Performance and Efficiency Tips

### Staging Partial Changes
```bash
# Stage only parts of a file
git add -p file.js

# Interactive staging
git add -i
```

### Useful Git Aliases
```bash
# Add to ~/.gitconfig
[alias]
    co = checkout
    br = branch
    ci = commit
    st = status
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = !gitk
    tree = log --graph --oneline --all
    amend = commit --amend --no-edit
    pushf = push --force-with-lease
```

### View Commit Information
```bash
# Show detailed commit information
git show <commit-hash>

# Show files changed in commit
git diff-tree --no-commit-id --name-only -r <commit-hash>

# Show commit statistics
git log --stat

# Search commits by message
git log --grep="search term"

# Search commits by author
git log --author="John Doe"

# Search commits by file
git log -- path/to/file.js
```

## Quick Reference

**Format**: `<type>(<scope>): <description>`

**Examples**:
- `feat(api): add user registration endpoint`
- `fix(ui): resolve mobile menu toggle issue`
- `docs(readme): add contribution guidelines`
- `test(auth): add unit tests for login flow`
- `refactor(utils): simplify date formatting functions`

**Daily Commands**:
```bash
git status                    # Check current state
git add .                     # Stage all changes
git commit -m "message"       # Commit with message
git push origin branch-name   # Push to remote
git pull origin main          # Get latest changes
```

**Remember**:
- Test your changes before committing
- Keep commits atomic and focused
- Write clear, descriptive commit messages
- Use branches for features and fixes
- Review staged changes before committing
- Use conventional commit format for consistency
- Set up git hooks for automated quality checks