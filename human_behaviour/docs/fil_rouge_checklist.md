# Fil Rouge Checklist — Protest Leader Detection

This checklist maps implementation status to the official YouCode fil rouge requirements.

## 1) Analyse des exigences
- [ ] Final cahier des charges document (functional + technical requirements)
- [ ] User stories with acceptance criteria
- [ ] Risk/constraints section
- Evidence now: `birefcontext.md`, `Architecture.md`

## 2) Organisation & modélisation
- [ ] Trello/Jira export
- [ ] UML diagrams (use-case, classes, sequence)
- [ ] MCD/MPD
- Evidence now: updated architecture in `Architecture.md`

## 3) Sources de données
- [x] Dataset tooling for Market-1501 done
- [ ] Protest dataset data card (format, quality, legal constraints)
- [ ] Final ETL note for production data intake
- Evidence now: `src/data/market1501_dataset.py`, `scripts/organize_market1501.py`

## 4) Pipelines & Data Engineering
- [x] End-to-end orchestrated pipeline implemented
- [ ] Optional scheduling/orchestration evidence (Airflow or equivalent rationale)
- Evidence now: `src/pipeline/run_multicam_pipeline.py`

## 5) Développement IA
- [x] Re-ID training + inference done
- [x] Detection baseline + tracking done
- [x] Description + scoring modules done
- [ ] Real-world benchmark report (not only smoke)
- Evidence now: `src/training`, `src/detection`, `src/tracking`, `src/description`, `src/scoring`

## 6) Intégration services externes
- [x] LLaVA integration path
- [x] MLflow Docker integration
- [ ] Comparative benchmark/recommendation write-up
- Evidence now: `src/description/llava_descriptor.py`, MLflow runs/logs

## 7) Développement API
- [x] FastAPI endpoints operational
- [x] Swagger/OpenAPI available
- [x] JWT auth baseline added
- [ ] OAuth2/JWT policy hardening and API tests
- Evidence now: `src/interface/api.py`

## 8) Interface & visualisation
- [x] Streamlit drag/drop interface implemented
- [ ] User documentation screenshots and operator guide
- Evidence now: `src/interface/streamlit_app.py`

## 9) Déploiement & MLOps
- [x] MLflow experiment tracking integrated
- [ ] CI/CD workflow (GitHub Actions/GitLab CI)
- [ ] Docker compose/deployment recipe for full app
- [ ] Drift/performance monitoring plan
- Evidence now: scoring/training MLflow integration, Docker MLflow server usage

## 10) Tests & qualité
- [x] Automated E2E smoke script added
- [ ] Unit tests for key modules (tracking/scoring/api)
- [ ] Test coverage report integrated in CI
- Evidence now: `scripts/test_pipeline_smoke.py`

## 11) Sécurité & conformité
- [x] JWT mechanism present in API
- [ ] Access control policy and secret management doc
- [ ] RGPD/AI Act compliance section and retention/anonymization rules
- [ ] Audit logging policy
- Evidence now: auth in `src/interface/api.py`

## 12) Rapport & soutenance
- [ ] Final report (all chapters + metrics + comparisons)
- [ ] Slides + live demo script (15 min)
- [ ] Code review prep notes + incident/debug scenario prep
- Deadline context: fil rouge target date `30/03/2026`

---

## Recommended next execution order (practical)
1. Run full pipeline on real multi-camera protest videos and freeze settings.
2. Produce final benchmark tables (tracking stability, scoring behavior, retrieval quality).
3. Add CI workflow + minimal unit tests.
4. Finalize security/compliance documentation.
5. Prepare report + slides + demo script.
