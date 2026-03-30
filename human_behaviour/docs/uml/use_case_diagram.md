# Use case diagram — analyst vs external LLM

**Actor:** a human **Analyst** (or operator) using the HTTP API / React UI.  
**Secondary actor:** **Ollama** (or any HTTP LLM the app calls)—only involved when a feature actually invokes the model (RAG scoring, `/analyze`, `/ask`, optional pipeline question).

The diagram stays at **user goals**, not internal steps (YOLO/OSNet/LLaVA are implementation details inside “Run pipeline”).

```mermaid
flowchart TB
    Analyst((Analyst))

    subgraph System["Protest leader app"]
        UC1[Authenticate\nJWT]
        UC2[Run pipeline\nlocal paths]
        UC3[Upload videos\nand run pipeline]
        UC4[List / open runs\nand details]
        UC5[Analyze run\nLLM scoring]
        UC6[Ask question\nabout a run]
    end

    LLM((Ollama LLM))

    Analyst --> UC1
    Analyst --> UC2
    Analyst --> UC3
    Analyst --> UC4
    Analyst --> UC5
    Analyst --> UC6

    UC5 -.->|calls| LLM
    UC6 -.->|calls| LLM
    UC2 -.->|optional,\nif scoring / Q in pipeline| LLM
```

**Mapping to FastAPI (rough):**

| Use case | Typical endpoint |
|----------|------------------|
| Authenticate | `POST /auth/token`, `GET /auth/me` |
| Run pipeline | `POST /pipeline/run` |
| Upload and run | `POST /pipeline/upload` |
| List / open runs | `GET /runs`, `GET /runs/{id}`, persons, crops, query history |
| Analyze run | `POST /analyze/{run_id}` |
| Ask question | `POST /ask/{run_id}` |

**Not drawn:** PostgreSQL/SQLite is infrastructure the API uses to persist runs; the analyst does not talk to the DB directly.
