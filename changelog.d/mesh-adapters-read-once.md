## Fixed

- OpenFOAM and PyFR planning, running and collecting read a large mesh at most once, when it is
  first seen, and not at all on later commands while the file is unchanged, as SU2 already does.
  The mesh is streamed instead of held in memory. Results, recorded input digests and refusal
  messages are unchanged.
- A PyFR restart no longer reads the saved mesh whole to check it. XFOIL and QPROP share a
  command's read of a large airfoil or blade file instead of reading it again for every call and
  point.
