## Changed

- An entry that does not apply to a row is now warned and skipped instead of silently ignored or refused: a GEOMETRY cell for a solver that does not read geometry (it was refused), and REF or PPROC keys that XFOIL and QPROP never read. The warning names the row, the file, the entry and why.
