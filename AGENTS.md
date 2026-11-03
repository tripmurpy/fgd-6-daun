# Repository Guidelines

## Project Context

This project classifies Indonesian herbal leaves from the Indonesian Herb Leaf Dataset 3500 (3,500 images across 10 classes) using CRISP-DM. The research measures and analyzes how much classification performance and computational efficiency differ between DenseNet-121 and Lightweight DenseNet (LWDN).

## Scope and Responsibilities

The current focus is Modeling and Evaluation. A different engineer owns data cleaning and preparation; use their agreed handoff rather than rebuilding or changing those steps. Continue in the shared [FGD-6 Google Colab notebook](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL). Its cells, settings, and outputs are evolving and provisional until run and checked. The local notebook under `source code/` differs from the shared notebook and has additional DenseNet-121 cells; do not assume those cells are already in Colab or copy them without reconciling the current notebook first. Streamlit deployment comes after the comparison and team model selection. The original LWDN paper is being reviewed as a candidate architecture source. Its design and adaptation to this 10-class task are not approved until the user/team agrees; do not invent or implement a partial variant.

## Colab Development Workflow

Write notebook code in a professional, concise machine-learning engineering style. Prefer the smallest correct implementation; avoid decorative code, unnecessary abstractions, and dependencies. Before adding a cell, inspect the whole relevant notebook flow and reuse existing variables, preprocessing, and helpers. Never repeat a completed phase or duplicate code. Structure new work as Agile increments: one clear goal and one responsibility per code cell, preceded by a short Markdown note stating its purpose and expected output. Use four-space indentation, `snake_case`, explicit configuration, and checks at important data/model boundaries. The local DenseNet settings are a draft, not approved. When a required project decision or specification is missing, ask the user or team before writing dependent code; do not guess. Review the original LWDN paper together before implementation, and do not treat the paper's reported metrics as targets for this dataset. Keep the shared notebook in Colab; do not recreate it as a local project clone.

When editing DenseNet-121 fine-tuning cells, read and follow [`style.md`](style.md) for cell order and presentation. This style guide does not change the modeling or evaluation requirements in this file.

Work in this order: (1) fine-tune DenseNet-121 and assess/select it using validation data; (2) fine-tune LWDN only after the paper-based architecture adaptation and required training settings are agreed, then assess/select it using the same validation data; (3) evaluate the selected checkpoints side by side on the agreed test set once. Do not use test results to tune or select either model. Report accuracy, precision, recall, macro F1-score, per-class results, confusion matrices, parameter count, model size, training time, and measured inference latency. Explicitly quantify the difference between models for each metric, state its direction and units, and analyze class-level errors. Draw conclusions from measured results; do not assume either model is superior.

Record seeds, model settings, preprocessing, selected checkpoints, and Colab runtime/device details needed to interpret and reproduce results. Run notebook cells in order and check that reported results match the saved outputs.

## Deployment

The local Streamlit prototype does not run inference. Defer model integration until Modeling and Evaluation are complete and the team selects a model. Do not claim prediction or deployment readiness from the UI prototype alone.

