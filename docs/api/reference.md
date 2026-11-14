# REST API

Install the package's `api` extra and run `orion100m serve` or `orion300m serve`.
The servers bind localhost by default. Set `ORION_API_TOKEN` to require an
`x-api-key` on non-health endpoints. No request is sent to an external service.

## ORION-100M

- `GET /health`, `GET /v1/health`, `GET /v1/info`
- `POST /v1/classify` with `{ "document": string | DocumentPayload, "registry": CourseProfile[], "top_k": 3 }`
- `POST /v1/predict:batch` with `{ "documents": [...], "batch_size": 32, "top_k": 3 }`
- `POST /v1/extract` with `{ "document": ..., "fields": [...] }`
- `POST /v1/embed` with `{ "texts": [...], "dimension": 640 | 256 | 128 }`

Request bodies are capped at 2.5 MB. Pydantic rejects unknown fields. Error
responses use `{ "error": { "code", "type", "message", "details" } }`.

## ORION-300M

- `GET /health`, `GET /v1/health`, `GET /v1/info`
- `POST /v1/generate`, `POST /v1/generate:batch`, and `POST /v1/generate/stream`

Generation accepts prompt, task, supplied context documents, length and sampling
controls. Request bodies are capped at 3.5 MB. The stream endpoint uses SSE and
forwards incrementally decoded SDK generation events. Consumers apply `delta`
events to append text and `replace` events to replace the current text; the final
`done` event contains status and citations.

The batch endpoint accepts 1–32 request objects with the single-generation
schema and returns results in input order. It evaluates each request
independently; batch throughput has not been benchmarked.

Citation marker mappings identify input documents only. API results label
citations as unverified; support verification belongs to the host application.
