# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

---

## [0.1.0] - 2023-10-30

This release adds MultiHeadAttention, SelfAttention, and ResidualConnection modules to build transformer architectures.

### Added

- MultiHeadAttention module composes multiple SelfAttention layers in parallel
- SelfAttention provides configurable query, key, and value transformations
- ResidualConnection wraps modules with residual and normalization
- Unit tests for new modules

This release is intended for testing.
Please file any issues found!

---