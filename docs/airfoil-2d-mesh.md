# 2D airfoil mesh

`cfd-mesh airfoil-2d` meshes a 2D airfoil section as a structured C-mesh of quadrilaterals and
writes it in the two forms the SU2 and OpenFOAM workspaces read, with the following names:

| File | Content | Names |
|---|---|---|
| `inputs/mesh/<name>.su2` | SU2 native, `NDIME= 2`, quadrilaterals, counter-clockwise cells | markers `airfoil`, `farfield`, and `airfoil_base` for a blunt edge |
| `inputs/mesh/<name>.msh` | gmsh 2.2 ASCII, the same plane mesh extruded one cell along z by one chord (hexahedra) | patches `airfoil`, `farfield`, `front`, `back`, and `airfoil_base` for a blunt edge; volume `fluid` |
| `inputs/mesh/<name>.su2.provenance.json`, `inputs/mesh/<name>.msh.provenance.json` | how each mesh was made | |

```text
cfd-mesh --workspace my-polars airfoil-2d --naca 0012 --reynolds 6e6
cfd-mesh airfoil-2d --coordinates inputs/sections/e387.dat --reynolds 2e5 --level medium
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--naca DIGITS` | | A NACA 4 or 5 digit designation, for example `0012` or `23012` (`NACA 0012` is read too). |
| `--coordinates FILE` | | A coordinates file in Selig order; a relative path is read from the workspace. |
| `--reynolds RE` | required | The chord Reynolds number the first cell is sized for. |
| `--yplus Y+` | 1 | The target y+ of the first cell. |
| `--level` | `coarse` | `coarse`, `medium` or `fine`: the cell counts below. |
| `--farfield CHORDS` | 500 | The far-field distance in chords, at least 5. |
| `--trailing-edge` | `sharp` | `sharp` collapses the trailing-edge gap; `blunt` keeps it as a base. |
| `--out NAME` | `naca<digits>` or the file's stem | The mesh name without a suffix (`.su2` or `.msh` is accepted and dropped). A bare name lands in `inputs/mesh/`; a name with a folder is written there. |

Exactly one of `--naca` and `--coordinates` is given. The command prints its record as JSON
and exits 0; a refusal exits 2, names what is at fault and writes nothing.

## The section

A NACA designation uses the published thickness formula (coefficient -0.1015), so its trailing
edge has a gap: 0.00252 chord for a NACA 0012. A coordinates file holds an optional title line,
then one `x y` pair per line, from the trailing edge over the upper surface to the leading edge
and back under the lower surface; the opposite direction is accepted. The Lednicer layout
(point counts on the first line) is refused.

The section is normalised to a unit chord: the leading edge is the point farthest from the
trailing-edge midpoint; it is moved to the origin and the midpoint to (1, 0). The sidecar
records the transform. A file is refused when its two ends are more than 0.05 chord apart (an
open contour) or when the contour crosses itself.

## Trailing edge

- `sharp` (default): the gap is collapsed in the mesh. Each side moves by the vector from its
  trailing-edge point to the midpoint, weighted by its arc-length fraction from the leading
  edge, so the leading edge does not move. The sidecar records the gap removed
  (`gap_removed`). The C-mesh wake cut then starts at one node.
- `blunt`: the gap is kept. The base gets a block of its own between the two wake blocks, 8
  evenly spaced cells across, and the base edges belong to a separate `airfoil_base` marker. A closed
  trailing edge cannot be meshed blunt.

## Topology

Four transfinite gmsh blocks: lower wake, lower surface, upper surface and upper wake; the two
wake blocks share the wake cut, so the cut has no duplicate node. A blunt trailing edge adds
the base block. The far field is a semicircle of radius D chords about the trailing edge and
straight lines to the outflow at D chords downstream of it, all in the `farfield` marker. Each
wall node's grid line meets the semicircle close to the direction of the wall normal.

## Levels

The counts are those of the NASA Turbulence Modeling Resource NACA 0012 family, whose grids
have 225, 449 and 897 nodes around the C:

| Level | Cells per side on the wall | Normal cells | Wake cells per side | Leading-edge spacing | Trailing-edge spacing | Quadrilaterals |
|---|---|---|---|---|---|---|
| `coarse` | 64 | 64 | 48 | 4.2740e-04 | 4.2599e-03 | 14 336 |
| `medium` | 128 | 128 | 96 | 2.0573e-04 | 2.0545e-03 | 57 344 |
| `fine` | 256 | 256 | 192 | 1.0094e-04 | 1.0090e-03 | 229 376 |

Spacings are in chords. Along the wall, a two-sided stretching runs from the leading-edge
spacing to the trailing-edge spacing; along the wake, cells grow geometrically from the
trailing-edge spacing to the outflow. A blunt base adds 8 x (wake cells) quadrilaterals.

## First cell and growth

The first cell height comes from the target y+ and the Reynolds number through the turbulent
flat-plate skin friction of Schlichting:

```text
Cf     = (2 log10 Re - 0.65)^-2.3
y1 / c = y+ / (Re sqrt(Cf / 2))
```

| Re | Cf | y1 / c at y+ = 1 |
|---|---|---|
| 1e6 | 0.00375 | 2.311e-05 |
| 3e6 | 0.00311 | 8.452e-06 |
| 6e6 | 0.00279 | 4.465e-06 |

The wall-normal cells grow geometrically from y1 to the far field; the ratio is solved for the
level's normal count (1.3118 at the coarse level for Re 6e6 and D = 500). The wake cut keeps the
wall spacing normal to it all the way to the outflow. A ratio above 1.35 is refused, and the
message names the normal count that would satisfy it. The y+ the solver measures depends on the
flow; the correlation sizes the cell before the run.

## The provenance sidecar

Each mesh has a `<file name>.provenance.json` beside it with:

- `command` (`cfd-mesh airfoil-2d`), `mesh` and its `sha256`, the `cfdbender` version;
- `sources`: the coordinates file with its SHA-256 (empty for a NACA designation);
- `options`: every value after defaults (`naca` or `coordinates`, `reynolds`, `yplus`,
  `first_cell`, `level` and its counts and spacings, `normal_growth`, `wake_growth`,
  `farfield`, `trailing_edge`, `gap_removed`, `base_cells`, `span`, the normalising
  `transform`, the `gmsh` version);
- `outputs`: the measured quality (`cells`, `nodes`, cells per marker, `first_cell` minimum,
  mean and maximum on the airfoil surface, `largest_growth`, far-field distance, `outflow_x`,
  `smallest_area`, `largest_aspect_ratio`, `duplicate_nodes`, `unmarked_boundary_edges`,
  `marked_interior_edges`, and `base_first_cell` for a blunt edge), and both files with their
  SHA-256.

## Using the mesh

SU2 (`inputs/setups/s001.toml`), with the markers the mesh carries:

```toml
mesh = "inputs/mesh/naca0012.su2"
MARKER_HEATFLUX = ["airfoil"]
MARKER_FAR = ["farfield"]
```

For a blunt edge, include `airfoil_base` in `MARKER_HEATFLUX` alongside `airfoil`.
The default monitoring and plotting lists then include both walls. OpenFOAM applies the same
wall conditions and integrates forces over both patches.

OpenFOAM, which imports the `.msh` with `gmshToFoam` and sets `front` and `back` to `empty`:

```toml
mesh = "inputs/mesh/naca0012.msh"
```

The span is one chord, so a reference file with `area = 1.0` and `length = 1.0` is right for
both solvers. In the matrix the mesh can also be named in the `MESH` column.

## Repeatability

The kernel runs with one thread and every node of the block edges is placed by cfdbender, so the
same options write the same bytes. gmsh runs in a separate worker process; a kernel fault ends
the worker and leaves `inputs/mesh/` as it was.
