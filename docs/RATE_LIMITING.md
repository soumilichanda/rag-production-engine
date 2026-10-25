# Phase 13: Distributed Rate Limiting & Throughput Protection

### Operational Policies & Token Bucket Headers

| Header | Description | Default Threshold |
| :--- | :--- | :--- |
| `X-RateLimit-Limit` | Maximum allowable requests per sliding window | 30 req / client |
| `X-RateLimit-Remaining` | Remaining request quota in the active sliding window | Evaluated in real-time |
| `Retry-After` | Cooldown period before requests are accepted again | Calculated from oldest burst stamp |

- **Security & Integrity:** Rate limits are enforced on downstream retrieval and vector search endpoints while keeping internal `/health` and documentation routes exempt.
