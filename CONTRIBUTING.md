# Contributing to cfdbender

Write a failing test for the intended behavior first, then implement it. Install
the development dependencies with `pip install -e ".[dev]"` and run:

```console
pytest
ruff check .
python tools/check_public_text.py
ruff format --check .
mypy src
python tools/check_architecture.py
```

Do not commit solver outputs or private geometry. Keep contributions original
or compatible with the MIT license.
