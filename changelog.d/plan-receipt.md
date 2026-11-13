## Added

- A plan receipt, `post/<matrix>/plan.json`: the matrix digest and every planned point's input digests, written by `cfdbender.matrix.receipt.write_receipt` and checked by `require_receipt`, which refuses a missing or stale receipt with a message naming what changed.
