## Fixed

- Planning with large meshes no longer reads and hashes the mesh once per point: each input file
  is read and hashed once per plan, a large file is streamed in blocks instead of held in memory,
  and SU2 marker groups are checked in the same pass. The recorded input digests are unchanged. A
  file that changes while planning is refused with a message to plan again.
