# Data

The Context is Key data is not committed here. Training and evaluation read it through TimeNet (or straight
from Hugging Face). The demo needs a local copy at `data/cik_test.parquet`:

```bash
huggingface-cli download ServiceNow/context-is-key data/test-00000-of-00001.parquet --repo-type dataset --local-dir /tmp/cik
cp /tmp/cik/data/test-00000-of-00001.parquet data/cik_test.parquet
```
