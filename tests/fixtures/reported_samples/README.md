# Reported sample fixtures

These public screenshots are regression fixtures copied from the repository's
`debug-reports` branch. The reporting UI and issue bodies state that uploads are
intentionally stored in the public repository for debugging.

| Fixture | GitHub report | Debug artifact commit | SHA-256 |
| --- | --- | --- | --- |
| `issue-7-input.png` | [Issue #7](https://github.com/EthanMcKanna/map-boundary-builder/issues/7) | `a153c14db98611292e3bf13c3bfc41bfa4802262` | `fc1c001af1e816b9581e2650fb3b7e9c0555d96f67e3296922fdceca2f1be1b0` |
| `issue-13-input.jpeg` | [Issue #13](https://github.com/EthanMcKanna/map-boundary-builder/issues/13) | `36422a4c6123f8115611c30375a7c7f5e3cbd560` | `61419a04d7c69d391a0ad3af0442d4a549fe03c0c1dc97d251a0efb7753b1d9f` |

The tests run with external map services disabled, so these cases also verify
that the bundled read-only geocoder seed contains the evidence needed for a
cold deployment.
