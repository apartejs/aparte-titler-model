# Changelog

## 1.0.0 — 2026-09

First release.

- `@aparte/titler` — the runtime: one ESM file, no dependency, reads model files of format 1.
- `@aparte/titler-latin` — runtime + default model, 17 European languages, int3, 133 KB.
- `@aparte/titler-latin-mini` — same languages, smaller vocabulary, 96 KB.
- `@aparte/titler-efigsp` — English, French, Spanish, German, Portuguese, Italian, 77 KB.
- Models (20 variants × fp32 / int8 / int4 / int3), scores and charts on [Hugging Face](https://huggingface.co/apartejs/aparte-titler).
- Python: reference reader (numpy only), evaluation and the full training pipeline.
