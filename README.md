# Atlas Claude

An independently built local voice-analytics assistant with deterministic calculations, numeric grounding, synthetic knowledge retrieval and a restricted developer mode.

**Maturity:** Demo implementation. 75 tests pass; seven local-model checks skip when embedding services are unavailable. The clean copy can regenerate its corpus in explicitly labeled keyword mode. Prior latency claims are not reasserted as current measurements.

## How it works

Vanilla browser UI and WebGL orb → FastAPI → SQLite synthetic data and NumPy analytics. Local brain chooses tools and narrates verified facts. A separate decision model, speech runtime, person detector and embedding endpoint serve optional features. Knowledge graphs preserve row provenance; developer sandbox limits filesystem/network/process access.

## Getting started

Read [installation, configuration and shutdown](docs/OPERATIONS.md) and [model/data retrieval](docs/MODELS.md). Review [dependency evidence](docs/DEPENDENCIES.md) before choosing an environment. The safe review launcher uses loopback and does not install services or load models.

## Security and data

Read [security and safe updates](SECURITY.md) and [ownership/license status](docs/LICENSE-STATUS.md). Runtime originals are preserved separately. This is a sanitized source snapshot with new history: no account state, business records, model weights, unknown media assets or original logs/history are distributed. Third-party notices included in this copy remain applicable.

## Verification

See [verification evidence and limitations](docs/VERIFICATION.md). Tests distinguish synthetic/mocked workflows from live integrations. Original demo screenshots and document recordings are excluded; use the synthetic workflow in the operations guide for your own reviewed demonstration.

## Roadmap

Add an asset installer based on pinned receipts, harden sandbox isolation portability and make optional-model status explicit. Verify acoustic timing and camera behavior on dedicated hardware.

For route/source details see [architecture](docs/ARCHITECTURE.md).
