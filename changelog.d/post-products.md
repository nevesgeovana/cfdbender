## Added

- `collect` also writes `post/<matrix>/products.json`, listing each product written and each skipped with its reason and a `complete` flag, and `post/<matrix>/post.log`; `cfdbender.matrix.post.post_workspace` rebuilds them from runs.json and the saved outputs without running a solver.
