## Removed

- `cfd-matrix rebuild --inputs-from` is gone: it only fed pyflightstream's comparison of the executed script with the rendered one, which has no solver-neutral counterpart. It was accepted and then refused; the parser now rejects it.
