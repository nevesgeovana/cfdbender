## Added

- `cfdbender.post.vtu` reads an unstructured-grid `.vtu` file (ascii, inline base64 and appended
  arrays, zlib compressed or not, 32 and 64 bit headers, little endian) with numpy alone, and
  refuses another compressor or a big-endian file by name.
- `cfdbender.post.sections` cuts a surface by the planes of a `[[sections.distributions]]` entry
  and writes, per entry, `sections/<point>_cp_<name>.csv` and
  `sections/<point>_sloads_<name>.csv` with the columns of the FlightStream sectional-loads and
  Cp exports (`Offset, Chord, X_QC, Z_QC, Fx, Fz, Moment`, and with `integrate = true`
  `Strip_length, Fx_int, Fz_int, My_int`). The caller names the columns its solver can fill;
  the others are absent from the files. The code is solver agnostic and imports nothing from
  `cfdbender.solvers`.
