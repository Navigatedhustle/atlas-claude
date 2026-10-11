# Architecture

Vanilla browser UI and WebGL orb → FastAPI → SQLite synthetic data and NumPy analytics. Local brain chooses tools and narrates verified facts. A separate decision model, speech runtime, person detector and embedding endpoint serve optional features. Knowledge graphs preserve row provenance; developer sandbox limits filesystem/network/process access.

## Implemented endpoints inspected from source

| Method | Route | Source |
| --- | --- | --- |
| GET | `/health` | `atlas/vision_server.py` |
| POST | `/detect` | `atlas/vision_server.py` |
| GET | `/` | `atlas/server.py` |
| GET | `/api/config` | `atlas/server.py` |
| GET | `/api/stats` | `atlas/server.py` |
| POST | `/api/dev/run` | `atlas/server.py` |
| GET | `/api/dev/endpoint` | `atlas/server.py` |
| GET | `/api/brain/graph` | `atlas/server.py` |
| GET | `/api/brain/node/{node_id:path}` | `atlas/server.py` |
| GET | `/api/brain/knows` | `atlas/server.py` |
| POST | `/api/brain/feed` | `atlas/server.py` |
| POST | `/api/brain/feed/reset` | `atlas/server.py` |
| POST | `/api/net/reset` | `atlas/server.py` |
| POST | `/api/latency` | `atlas/server.py` |
| POST | `/api/brief` | `atlas/server.py` |
| POST | `/api/say` | `atlas/server.py` |
| POST | `/api/vision` | `atlas/server.py` |
| WEBSOCKET | `/ws` | `atlas/server.py` |
| WEBSOCKET | `/ws/asr` | `atlas/server.py` |
| GET | `/health` | `atlas/embed_server.py` |
| POST | `/embed` | `atlas/embed_server.py` |

These routes establish implementation shape. They do not establish successful provider integration or model quality. Generated state and data stores are excluded from the repository.
