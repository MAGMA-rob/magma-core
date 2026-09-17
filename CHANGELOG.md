# Changelog

All notable changes to MAGMA Core are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0b3] - 2026-09-17

### Added

- Added the shared top-level `magma_agent_timeout` setting for clients of the
  MAGMA agent HTTP API.

### Changed

- Allowed protocol-v2 agent decisions to contain both a user-facing message and
  tool calls, leaving each runner responsible for applying its own policy.

## [2.0.0b2]

### Changed

- Published the second v2 beta.

## [2.0.0b1]

### Added

- Published the first v2 beta.

[Unreleased]: https://github.com/MAGMA-rob/magma-core/compare/v2.0.0b3...HEAD
[2.0.0b3]: https://github.com/MAGMA-rob/magma-core/compare/v2.0.0b2...v2.0.0b3
[2.0.0b2]: https://github.com/MAGMA-rob/magma-core/compare/v2.0.0b1...v2.0.0b2
[2.0.0b1]: https://github.com/MAGMA-rob/magma-core/releases/tag/v2.0.0b1
