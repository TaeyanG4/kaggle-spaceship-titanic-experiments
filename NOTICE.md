# Attribution and reuse

Kaggle Spaceship Titanic Experiments records a multi-agent research project on the Kaggle competition Spaceship Titanic. Experiments were run by AI coding agents (Codex, Claude Code, ChatGPT) under human direction, with all models trained locally.

All modelling code in `src/`, `scripts/`, `tools/` and `kaggle_notebook/` was written for this project. One runner (`scripts/viktor_runner.py`) re-implements the modelling approach of the public notebook *Space Titanic* by Viktor Taran from its published source in order to evaluate it under group-aware validation; it is credited as a re-implementation, not an original method. Public notebooks were read, not executed, and none of their code bodies or prediction files are included.

The final models use the pretrained TabPFN v3.5 weights from Prior Labs, which are subject to the Prior Labs license and require accepting it with a Prior Labs account. The weights are not included here. Other third-party libraries are used under their own licenses.

Raw competition CSVs, OOF prediction files, model weights, credentials and virtual environments are excluded. Files in `submissions/` contain this project's predictions, not hidden test answers. Use of the competition data follows the Kaggle competition rules.

This repository does not grant a license over third-party material. Check the upstream source and terms before reuse.
