# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
cd /src/mosaicfm
pip install -e . --no-deps
cd scripts
composer train.py /src/mosaicfm/gcloud/mosaicfm-70m-merged.yaml
```
