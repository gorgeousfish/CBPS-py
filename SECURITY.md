# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue.
2. Email the maintainers directly:
   - Xuanyu Cai: xuanyuCAI@outlook.com
   - Wenli Xu: wlxu@cityu.edu.mo
3. Include:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if any)

We will:
- Acknowledge receipt within 48 hours
- Provide an estimated timeline for a fix within 7 days
- Credit the reporter in the security advisory (unless anonymity is requested)

## Security Practices

- Dependencies are monitored via [Dependabot](https://github.com/gorgeousfish/CBPS-py/security/dependabot)
- Code is scanned with [Bandit](https://github.com/PyCQA/bandit) for common security issues
- All CI runs use pinned action versions to prevent supply chain attacks
