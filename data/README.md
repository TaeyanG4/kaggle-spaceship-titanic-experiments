# Data

The official competition files are not included. Join [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic), accept the rules, then:

```powershell
kaggle competitions download -c spaceship-titanic -p data\raw
```

Unzip into `data/raw/` (`train.csv`, `test.csv`, `sample_submission.csv`). Expected SHA256 hashes are in [docs/evidence/data-fingerprints.json](../docs/evidence/data-fingerprints.json). `data/processed/` holds the frozen fold assignments written by the runners (not committed).
