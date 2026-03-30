# Class diagram — domain model (what the code actually persists)

The backend stores data with **SQLAlchemy** models in `src/database/models.py`. The multi-camera workflow is **not** a `MulticamPipeline` class: it is the function `run_multicam_pipeline()` plus plain functions (`run_tracking`, `run_description_pipeline`, etc.). This diagram only shows **real ORM types** and relationships.

```mermaid
classDiagram
    direction TB

    class PipelineRun {
        +String run_name
        +String status
        +JSON manifest
        +String scoring_backend
        +DateTime created_at
        +DateTime completed_at
    }

    class Video {
        +int run_id FK
        +String camera_id
        +String filepath
    }

    class Person {
        +int run_id FK
        +int person_id
        +Float leader_score
        +Float heuristic_score
        +Text reasoning
    }

    class Description {
        +int person_db_id FK
        +int frame_idx
        +Text description_text
        +JSON signals
    }

    class LeaderScore {
        +int run_id FK
        +int person_id
        +Float leader_score
        +Text reasoning
        +String backend_used
    }

    class Query {
        +int run_id FK
        +Text question
        +Text answer
        +String generator_backend
    }

    PipelineRun "1" --> "*" Video
    PipelineRun "1" --> "*" Person
    PipelineRun "1" --> "*" Query
    Person "1" --> "*" Description : descriptions
    PipelineRun ..> LeaderScore : FK run_id only
```

**Note:** `LeaderScore` references `pipeline_runs.id`, but `PipelineRun` does not declare a `relationship()` to `LeaderScore` in code—only the foreign key exists on the `leader_scores` table.

---

## Optional: description backends (small OO slice)

`llava_descriptor.py` uses a strategy-style backend, not “Llava inherits Mock”.

```mermaid
classDiagram
    direction LR
    class DescriptorBackend {
        <<abstract>>
    }
    class MockDescriptorBackend
    class LlavaDescriptorBackend
    DescriptorBackend <|-- MockDescriptorBackend
    DescriptorBackend <|-- LlavaDescriptorBackend
```
