# Geometry and mesh commands

This page lists every command and option of `cfd-geom` and `cfd-mesh`.
`cfd-geom` builds parametric geometry: it validates a configuration, runs its
stages S0 to S13 into IGES files, loops parameter sets, identifies an original
IGES and checks a rebuilt one. `cfd-mesh` makes and edits surface meshes: it
meshes generated geometry, refines or coarsens a mesh, audits it, edits its
trailing edges, scales, morphs and repitches it, and builds rotor wheels.

```console
cfd-geom validate wing.geom.toml
cfd-geom run wing.geom.toml
cfd-mesh mesh-generated wing.geom.toml --obj wing.obj --component WING
```

Both programs work on the current folder; `--workspace PATH` names another one.

## Where the files go

- `cfd-geom run` writes into `inputs/geometry/<name>/`, where `<name>` is
  `--name`, else the configuration, else the file's stem. The folder holds the
  stage records, the report, the IGES files and `geometry.json`, the sidecar
  that marks the folder as generated geometry. `cfd-geom doe` writes one such
  folder per parameter set, named by the set's `name` (the matrix row the
  geometry feeds), else `set-0001`, `set-0002` and so on. `--out` names
  another folder.
- A `cfd-mesh` verb writes a mesh named without a folder into `inputs/mesh/`,
  and beside it `<file name>.provenance.json`, which records the command, the
  source files with their SHA-256, the options and the digest of the mesh. A
  sidecar is never offered as a mesh when a matrix names a mesh by its stem.
- A verb never writes over its input.

## Exit codes

0 means done, 2 means refused: the reason is printed. In particular, a refused `cfd-geom run` can write
`geom-run.log` and `report.json` in its output folder, and a refusal during a later stage can leave
earlier stage files there. `doe` can likewise leave records for its sets.
`cfd-geom run` and `doe` also return 1 when a stage failed (the report names
the stage and the check) and 3 when a check of the stage table was not
measured; `doe` returns the worst code of its sets. `cfd-mesh mesh` returns 1
when the mesh does not pass its acceptance table.

An export check failure includes the check name, measured value, limit and
location in the report's `message`, with a corrective action. For example,
`area.read_back` names the component, face and file whose area changed on
reimport. Check that component's cap and trim definitions, rebuild it and
export again; inspect `S11.R.json` and the face's IGES file beside the report
to compare the retained and imported areas.
Every failed S11 or S12 check retains the selected component names, including
aggregate sewing checks and failures raised by the kernel. A short error note
is accompanied by the measured value, limit and a rebuild action in the report.

The area comparison keeps its relative limit of `0.001`. Areas are measured
from the live kernel's full-precision surface coefficients and trim topology.
Whole curve coefficients prove UV iso-lines; sampled edge points cannot do so.
Kernel vertex connectivity orders boundaries independently of storage order;
numerical endpoint closure gaps contribute to the area uncertainty.
The integration bounds its numerical uncertainty below `0.00001` relative,
using every surface knot span and retaining curved trims. Unsupported surface
or trim representations fail with a message asking for a supported definition
and a rebuild; a default kernel mass estimate cannot pass this check.
The manifest records each source face's absolute area uncertainty. S11.R
requires the full source and read-back uncertainty intervals to fit inside
`0.001`; an unverified result or a missing certificate fails. Rebuild older
exports to create these certificates before running their read-back checks.
Each retained face also carries an eight-character IGES directory label. The
manifest records that label, its position in each file, and the source centroid
and bounding box in metres. S11.R validates the labels and directory order,
then verifies a unique centroid or bounding-box match within `1e-7` m before
comparing each face's corresponding intervals. Bounding boxes provide an independent
check when the kernel's approximate centroid shifts on reimport. This fallback
also requires the complete trimmed IGES entity and its units to match the
geometry fingerprint recorded at export. A box alone can describe the base
surface even after its trim moves, so an unchanged box is insufficient.
It never sorts areas to pair faces. Missing or duplicated labels, a changed
order, missing geometric data or ambiguous geometric matches fail the check
with the component, face, value, limit and rebuild action. Rebuild older
exports to record these identities. Distinct trims on a shared surface keep
separate labels and area measurements; an ambiguous match is refused.
Both the directory-level component name and the explicit NAME property pointer
must match the source component (or the retained face name in an isolated file).
Missing or changed names fail even when the label and measured area are intact.
The same interval and correspondence checks cover every face of the combined
IGES named by `manifest["iges"]`, every component file and every isolated face
file, all imported without sewing. An intact isolated file cannot certify a
changed component or combined file. A missing combined file also fails S11.R.
An integration warning, nonfinite value, or error allowance above the numerical
limit fails the check. It is never treated as a successful measurement.

An attachment positions a component on its parent. When their faces cross,
declare the intended `[[dominance]]` trimming relationships in the geometry
configuration. `gate.grid` refuses edges shared by more than two faces and
names the affected components and faces, so the junctions can be trimmed
before rebuilding.

## The geometry kernel

The commands marked "uses the gmsh kernel" run gmsh in a worker process, so a
fault of the kernel ends the worker, not the command, and leaves no partial
file. The licence of gmsh is on the [Install](install.md) page.

<!-- BEGIN GENERATED COMMAND REFERENCE -->

## Options before the command name

These go between the program name and the command, for example
`cfd-mesh --workspace campaign refine ...`.

| Option | Meaning |
|---|---|
| `cfd-geom --workspace PATH` | The workspace whose inputs/geometry/ receives the output (default: the current folder) |
| `cfd-mesh --workspace PATH` | The workspace a bare output name is written into (default: the current folder) |

## cfd-geom: parametric geometry

### `cfd-geom run`

Run the stages S0 to S13 and write the report.

```text
cfd-geom run CONFIG [options]
```

| Argument or option | Meaning |
|---|---|
| `CONFIG` | The configuration (TOML) |
| `--out DIR` | Folder of the records and files (default: inputs/geometry/&lt;name&gt;/ of the workspace) |
| `--name NAME` | The geometry's folder name (default: the configuration, else the file's stem) |
| `--set K=V` | Override one value, component.key=value (repeatable; the value is TOML) |
| `--configuration NAME` | Entry of the configuration registry |
| `--variant V` | Variant (default: clean) |
| `--until Sxx` | Last stage to run, S0 to S13 (default S13; S10 and earlier need no kernel) |

### `cfd-geom doe`

One report per parameter set; continues after a failure.

```text
cfd-geom doe CONFIG SETS [options]
```

| Argument or option | Meaning |
|---|---|
| `CONFIG` | The configuration (TOML) |
| `SETS` | TOML with one [[sets]] table per parameter set |
| `--out DIR` | Folder of one sub-folder per set, named by the set's name or set-0001, set-0002, ... (default: inputs/geometry/ of the workspace) |
| `--workers N` | Sets in parallel |
| `--configuration NAME` | Entry of the configuration registry |
| `--variant V` | Variant (default: clean) |
| `--until Sxx` | Last stage to run, S0 to S13 (default S13; S10 and earlier need no kernel) |

### `cfd-geom validate`

Resolve the configuration and print every value with its source.

```text
cfd-geom validate CONFIG
```

| Argument or option | Meaning |
|---|---|
| `CONFIG` | The configuration (TOML) |

### `cfd-geom configurations`

List the configurations, variants and members.

```text
cfd-geom configurations CONFIG
```

| Argument or option | Meaning |
|---|---|
| `CONFIG` | The configuration (TOML) |

### `cfd-geom stages`

List the stages, their checks and limits.

```text
cfd-geom stages
```

### `cfd-geom identify`

Cuts the trimmed faces of one component of an IGES file (named on its level) by planes normal to its span axis, finds the clean band of sections, builds the reference planform between the root and the outermost clean cut and fits a CST section to the root, the middle and the tip. The root is the symmetry plane, the host surface or the span extent of the original's own boundary curves. A coarse and a fine pass must agree for the result to be accepted. Prints the result; --config-out writes a configuration that rebuilds the planform.

```text
cfd-geom identify IGES COMPONENT [options]
```

| Argument or option | Meaning |
|---|---|
| `IGES` | The original IGES, read only |
| `COMPONENT` | The component to identify |
| `--host NAME` | A component the surface sits on, NAME or FILE#NAME (repeatable); the root is where the extended leading edge meets it |
| `--root {auto,plane,host,boundary}` | Where the planform starts: the symmetry plane, a host, or the span extent of the face boundary curves (default: host when --host is given, else plane) |
| `--mirrored, --no-mirrored` | Whether the surface has a mirror image across its root, doubling span and area (default: yes for a plane root, no otherwise) |
| `--cut {plain,bridge_te,mirror}` | Bridge_te closes an open blunt trailing-edge slit; mirror adds the reflection of a half-body original (default: plain) |
| `--seed NACA` | The NACA designation the section fit starts from (default: NACA0012) |
| `--span-axis X Y Z X Y Z X Y Z` | First estimate of the span axis (default: y or z, the larger extent) |
| `--stations N` | Stations of the fine pass (default: 161) |
| `--grid N` | Cut-grid points per side of each face (default: 257) |
| `--section-tolerance-mm MM` | Section residual tolerance in mm (default: 0.5 mm or 0.1 percent of the chord) |
| `--section-degree N` | CST degree a section fit starts at (default: 5) |
| `--section-max-degree N` | Highest CST degree a section fit may reach (default: 20) |
| `--host-tolerance M` | Largest gap in m at which the extended leading edge meets a host (default: 5e-5) |
| `--no-converge` | Skip the coarse pass that must agree with the fine one |
| `--out JSON` | Also write the result to this file |
| `--config-out TOML` | Write a configuration (TOML) that rebuilds the identified surface |

### `cfd-geom check`

Samples the original and the rebuilt faces and measures the distance of each to the other: forward (the original to the rebuilt, everywhere) and reverse (the rebuilt to the original, inside its trimmed boundary). The limit is the named profile applied to the local chord. The distances are sampled upper bounds, not continuous Hausdorff distances, as the record's method says. Exit 0 accepted, 1 rejected, 2 refused. Give the local chord as --chord-m, or as --stations-m with --chords-m.

```text
cfd-geom check REBUILT ORIGINAL [options]
```

| Argument or option | Meaning |
|---|---|
| `REBUILT` | The rebuilt IGES, FILE or FILE#NAME |
| `ORIGINAL` | The original IGES, FILE or FILE#NAME |
| `--chord-m M` | One local chord in m |
| `--stations-m S [S ...]` | Increasing stations in m along --axis, with --chords-m |
| `--chords-m C [C ...]` | The chord in m at each station |
| `--axis {0,1,2}` | The axis of --stations-m, 0 x, 1 y, 2 z (default: 1) |
| `--component NAME` | The component of a file both surfaces are named in |
| `--exclude-face INDEX` | Leave this face of the original out, by index (repeatable) |
| `--forward-count N` | Samples per side on each original face (default: 45) |
| `--reverse-count N` | Samples per side on each rebuilt face (default: 31) |
| `--edge-count N` | Samples on each trim edge (default: 101) |
| `--fold-y` | The original is a half body: fold the rebuilt mirror half onto the modelled side |
| `--profile PROFILE` | Named deviation profile (default: study_default, 0.5 mm or 0.1 percent of chord) |
| `--out JSON` | Also write the record to this file |

### `cfd-geom schema`

Prints the draft 2020-12 JSON Schema of the component parameter record (schema 0.2.0): the contract between a reconstruction and this library. With --validate the record file is parsed and its structural rules checked instead, and with --asset-root every file it pins is re-read against its SHA-256. Needs no gmsh.

```text
cfd-geom schema [options]
```

| Argument or option | Meaning |
|---|---|
| `--out JSON` | Write the schema to this file |
| `--validate RECORD` | Validate this record (JSON) |
| `--asset-root DIR` | With --validate: the folder the record's file paths resolve against, to check every pin |

### `cfd-geom iges-convert`

Every face of the IGES files is converted to a B-spline surface and written to one IGES, each component on its own level with one NAME, so a CAD import and this package read the same components back. The first file's length unit is kept. Writes the IGES and the record &lt;out&gt;.json.

```text
cfd-geom iges-convert IGES [IGES ...] [options]
```

| Argument or option | Meaning |
|---|---|
| `IGES [IGES ...]` | IGES files, read only |
| `--out OUT` | New .igs |
| `--name-encoding NAME_ENCODING` | Text encoding of the NAME bytes in the IGES files (default latin-1, one character per byte; use utf-8 for UTF-8 names). Never guessed: a name invalid in the encoding is refused |

## cfd-mesh: mesh generated geometry

### `cfd-mesh mesh`

Validates the mesh controls against the configuration (errors E01 to E16 refuse by name, W01 warns), plans every component's counts, meshes the configuration with strategy 1, or strategy 2 (full grid) when mesh.strategy = 2, and measures the acceptance table of that strategy on the realized mesh. With --dry-run it prints the plan and the warnings and writes nothing. Otherwise it writes &lt;descriptor&gt;.obj and &lt;descriptor&gt;.mesh.json into --out (default: inputs/mesh/ of the workspace), with the provenance sidecar of the OBJ.

```text
cfd-mesh mesh CONFIG [options]
```

| Argument or option | Meaning |
|---|---|
| `CONFIG` | The configuration (TOML) |
| `--controls TOML` | The mesh controls |
| `--dry-run` | Validate and print the planned counts only |
| `--out DIR` | Folder of the written mesh and record (default: inputs/mesh/ of the workspace) |
| `--descriptor NAME` | File stem (mesh.output.descriptor) |

### `cfd-mesh mesh-generated`

Resolves the configuration and meshes its patches from their role: constant spanwise width with a refined tip, chordwise growth from the leading edge, constant azimuth and axial step on a body, a planar all-quad flat tip cap and, for a wing on a body, the fairing as one quad-dominant transition ring. The orientation is outward. The record (counts, the acceptance checks, the spacing rules) is printed; the OBJ holds one group per component. A bare --obj name is written into inputs/mesh/ of the workspace, with its provenance sidecar.

```text
cfd-mesh mesh-generated CONFIG [options]
```

| Argument or option | Meaning |
|---|---|
| `CONFIG` | The configuration (TOML) |
| `--obj OBJ` | New .obj |
| `--component NAME` | Mesh this body or lifting surface alone, in its own frame (default: the assembly) |
| `--controls TOML` | Mesh controls overriding the defaults, by table: lifting_surface, body, checks, assembly |
| `--out JSON` | Also write the record to this file (a file, unlike mesh --out, which is a folder) |

## cfd-mesh: refine, coarsen and audit

### `cfd-mesh refine`

Refines or coarsens a panel mesh from its OBJ into a new geometry folder &lt;out&gt;/&lt;stem&gt;_&lt;tag&gt;/ holding the OBJ, the trailing-edge points file, the boundaries file, &lt;stem&gt;_&lt;tag&gt;.refine.json and the audit against the source. FACTOR multiplies the intervals of each index direction of a grid family and divides the target edge length of a remeshed family; a FACTOR below 1 coarsens. --parallel and --normal replace it along and across the reference perimeter of the grid families ([[perimeter]] node lists of the refinement file, else the family's longest boundary side). Without FACTOR, --parallel or --normal, the factors come from the refinement file (--config, or &lt;stem&gt;.refine.toml beside the mesh), whose [components], [refine] tag and elements (triangles or quad-dominant) are read whenever it exists. Faces are written in the source's order. The source and its folder are never modified. Exits 0 when the level is written, even when its audit fails: each failed gate or check is a warning and the audit is written beside the level, while audit exits 1 on the same audit. Exits 2 when the request is refused, and then writes nothing.

```text
cfd-mesh refine MESH [FACTOR] [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | The source OBJ |
| `[FACTOR]` | The factor of every selected family; below 1 coarsens |
| `--families FAMILIES` | The families to change, comma separated (default: all) |
| `--parallel PARALLEL` | The factor of the grid families along their reference perimeter |
| `--normal NORMAL` | The factor of the grid families across their reference perimeter |
| `--config FILE` | The refinement file (default: &lt;stem&gt;.refine.toml) |
| `--accept-remesh` | Remesh with FACTOR a family given --parallel or --normal, or method = grid, whose grid is not recovered, with a warning, in place of refusing it |
| `--accept-unknown` | Warn and skip, in place of refusing, an unknown table or key of the refinement file, a value it cannot read (the key takes its default), an implicit refinement file that cannot be read, a family table or component member the mesh does not hold, and a --families name it does not hold |
| `--strict-factors` | Refuse a grid factor below the least its grid allows (1/m for its m intervals, the least a tube's caps carry); by default it is raised to that least with a warning |
| `--out DIR` | Where the level folder goes (default: inputs/mesh/ of the workspace) |
| `--overwrite` | Replace an existing level folder |
| `--visual-inspect` | Write &lt;level&gt;.refine-governors.pdf beside its OBJ: the governing families per direction, then the patches and directions pages |
| `--keep-within-component` | Share directional governors across each declared component's grid patches |
| `--quiet` | Do not print the progress lines on the error stream |
| `--audit-off` | Skip the level audit and print how to run it later |

### `cfd-mesh sector-remesh`

Remeshes the selected families of a sector OBJ using the generic surface remesher, including grids and families at factor 1. FACTOR divides the local target edge length; below 1 coarsens. The refinement file's element modes choose triangles or quad-dominant cells. Free nodes are projected onto the source surface; boundaries and trailing edges follow their source curves, and interfaces with unchanged families are frozen. Boundary nodes can be resampled, so check periodic seams before sector-to-wheel. No saved simulation or solver is needed. Writes a level folder under inputs/mesh/ by default, with boundary and point files, a refinement record, audit and provenance. --out chooses its parent folder. Other options follow refine.

```text
cfd-mesh sector-remesh MESH [FACTOR] [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | The source OBJ |
| `[FACTOR]` | The factor of every selected family; below 1 coarsens |
| `--families FAMILIES` | The families to change, comma separated (default: all) |
| `--parallel PARALLEL` | The factor of the grid families along their reference perimeter |
| `--normal NORMAL` | The factor of the grid families across their reference perimeter |
| `--config FILE` | The refinement file (default: &lt;stem&gt;.refine.toml) |
| `--accept-remesh` | Remesh with FACTOR a family given --parallel or --normal, or method = grid, whose grid is not recovered, with a warning, in place of refusing it |
| `--accept-unknown` | Warn and skip, in place of refusing, an unknown table or key of the refinement file, a value it cannot read (the key takes its default), an implicit refinement file that cannot be read, a family table or component member the mesh does not hold, and a --families name it does not hold |
| `--strict-factors` | Refuse a grid factor below the least its grid allows (1/m for its m intervals, the least a tube's caps carry); by default it is raised to that least with a warning |
| `--out DIR` | Where the level folder goes (default: inputs/mesh/ of the workspace) |
| `--overwrite` | Replace an existing level folder |
| `--visual-inspect` | Write &lt;level&gt;.refine-governors.pdf beside its OBJ: the governing families per direction, then the patches and directions pages |
| `--keep-within-component` | Share directional governors across each declared component's grid patches |
| `--quiet` | Do not print the progress lines on the error stream |
| `--audit-off` | Skip the level audit and print how to run it later |

### `cfd-mesh prepare-to-refine`

Optional step before refine. MESH is an OBJ, whose families are cut at their sharp edges, or an ANSA NASTRAN export (.nas, .bdf, .dat) written with the FE perimeters on, whose families are cut along those perimeters. Writes into inputs/mesh/ of the workspace, or --out, &lt;stem&gt;_patches.obj (each family cut into &lt;family&gt;_&lt;NNN&gt;_grid or _mixed patches, numbered within the family from the largest extent along y), &lt;stem&gt;_patches.refine.toml (each patch at factor 1, grid or remesh, [components] putting each family back together) and &lt;stem&gt;_patches.nodes.txt (the rearmost line of each grid patch, its edges within --max-angle of the plane across --aft, as a node list).

```text
cfd-mesh prepare-to-refine MESH [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | The source OBJ or NASTRAN export; it is not modified |
| `--max-angle DEG` | The largest angle of a listed edge with the plane across --aft (default 45) |
| `--aft X Y Z X Y Z X Y Z` | The aft direction (default +x) |
| `--overwrite` | Replace the prepared files |
| `--out DIR` | Output folder (default: inputs/mesh/ of the workspace) |
| `--unit UNIT` | A NASTRAN mesh's length unit (default MILLIMETER); an OBJ is written as read |
| `--obj-unit OBJ_UNIT` | The unit a NASTRAN mesh is written in to the patches OBJ (default METER) |
| `--visual-inspect` | Also write &lt;stem&gt;_patches.inspect.pdf: one page per original family, each patch colored and named, the governing patches labelled GOVERNING |
| `--keep-within-component` | Share directional governors across each original family's grid patches |

### `cfd-mesh audit`

Audits an OBJ and writes &lt;stem&gt;.audit.json beside it. The gates G1 to G6 judge the topology, the orientation, the open boundary, the trailing edge and the nodes families share; the relative checks compare the 95th percentiles of the skewness, the warp and the size growth with the source's; the other figures are reported and never judged. Without --against only G1, G2 and G4 are judged. The trailing-edge points file is the one &lt;stem&gt;.boundaries.toml names, or else &lt;stem&gt;.te.txt. A gate or check that fails is a warning, and the audit is still written. Exits 0 when every gate and check passes, 1 when one fails, 2 when the input is refused or when the --out CSV cannot be written after the audit was.

```text
cfd-mesh audit OBJ [options]
```

| Argument or option | Meaning |
|---|---|
| `OBJ` | The OBJ to audit |
| `--against SOURCE` | The OBJ the mesh was refined or coarsened from; the gates and checks that compare with a source are judged against it |
| `--out FILE` | Write the figures as CSV, one row per family and figure; the audit JSON is always written beside the OBJ |

### `cfd-mesh inventory`

Writes the boundary inventory of an OBJ beside it: its groups that hold a face, in the order of the file, with the face counts and the digest of the file. An OBJ's existing sidecar is never rewritten, because beside a raw mesh it also carries the [import] table and the trailing edges.

```text
cfd-mesh inventory GEOMETRY [options]
```

| Argument or option | Meaning |
|---|---|
| `GEOMETRY` | An OBJ mesh |
| `--overwrite` | Accepted for the other verbs' sake; an OBJ's existing sidecar is refused either way |

### `cfd-mesh degenerate`

Derives the thin blade of a blade mesh, the sheet midway between its two sides from the root to the tip, with the root moved outward along the span by --root-offset so it does not cross the spinner, and writes into inputs/mesh/ of the workspace, or --out, as &lt;stem&gt;_thin_blade.obj with its boundary inventory &lt;stem&gt;_thin_blade.boundaries.toml. The root is the end of the blade nearer the origin of the mesh's own coordinates. The source is read only; a file holding more than one boundary needs --boundary. The derived geometry is a modelling choice: a solver run confirms what it makes of it.

```text
cfd-mesh degenerate GEOMETRY [options]
```

| Argument or option | Meaning |
|---|---|
| `GEOMETRY` | A blade mesh: an OBJ |
| `--out DIR` | Output folder (default: inputs/mesh/ of the workspace) |
| `--kind {thin-blade}` | The degenerate geometry to derive; thin-blade is the supported kind (default: thin-blade) |
| `--root-offset LENGTH` | How far the root of the thin blade is moved outward along the span, a positive length in the mesh's own unit, shorter than the blade |
| `--boundary NAME` | The boundary (an OBJ's group) that is the blade, for a file that also holds a spinner, a nacelle or other bodies; only its faces are read and the output is named &lt;stem&gt;_&lt;NAME&gt;_thin_blade; without it a file holding more than one boundary is refused |
| `--overwrite` | Rewrite a thin blade and its inventory that already exist; without it an existing one is refused |

### `cfd-mesh merge`

Joins two trimmed OBJ along the open boundaries that lie within --threshold of each other. The part --adapt names takes the other's nodes along that interface; the other part is kept unchanged. Writes the OBJ, &lt;out stem&gt;.merge.json, its audit and its inventory. The boundary declarations of the two sources are not merged. A threshold that finds no interface warns, writes nothing and exits 1.

```text
cfd-mesh merge OBJ OBJ [options]
```

| Argument or option | Meaning |
|---|---|
| `OBJ OBJ` | The two trimmed parts |
| `--threshold LENGTH` | The largest distance across the interface, in the OBJ's own length unit |
| `--adapt OBJ` | Which of the two parts is adapted, by its path or its file name; the other keeps its nodes and faces |
| `--out OBJ` | The merged OBJ to write; a name of another kind is warned and takes .obj; a bare name lands in the workspace's inputs/mesh/ |
| `--overwrite` | Replace an existing output |

## cfd-mesh: trailing edges

### `cfd-mesh trailing-edge`

Writes a trailing-edge points file, the mid-point of every mesh edge on the trailing edge under a unit line. Either from a blade mesh (MESH OUTPUT with --axis, --hub and --unit: the aftmost vertex of each chordwise section, joined along the aft ridge) or from an ordered node-window export (--from-nodes, no mesh needed, --out names the file).

```text
cfd-mesh trailing-edge [MESH] [OUTPUT] [options]
```

| Argument or option | Meaning |
|---|---|
| `[MESH]` | The blade surface OBJ |
| `[OUTPUT]` | The points file to write |
| `--from-nodes FROM_NODES` | Ordered node-window export; no mesh needed |
| `--out OUT` | Destination for --from-nodes; default: &lt;input stem&gt;.te.txt |
| `--axis X Y Z X Y Z X Y Z` |  |
| `--hub X Y Z X Y Z X Y Z` |  |
| `--unit UNIT` | Coordinate unit; METER by default for --from-nodes |
| `--sections SECTIONS` |  |
| `--overwrite` |  |

### `cfd-mesh convert2midnodes`

Reads the node list a mesh-preparation tool saves from a node set (Id, Name, X, Y, Z), puts the nodes in order along their lines (a long jump starts a new line, so one list may hold every trailing edge) and writes one mid-point per neighbouring pair.

```text
cfd-mesh convert2midnodes NODES OUTPUT [options]
```

| Argument or option | Meaning |
|---|---|
| `NODES` | The node list saved from the node set |
| `OUTPUT` | The .te.txt to write |
| `--unit UNIT` | The coordinates' unit (default: METER) |
| `--overwrite` | Replace an existing OUTPUT |
| `--accept-empty` | A list with fewer than two nodes, no usable node or no edge of MESH is warned and nothing is written, exit 0, in place of refusing it |
| `--per-family MESH` | The OBJ the nodes belong to: write &lt;OUTPUT stem&gt;.&lt;family&gt;.te.txt per family from its edges joining two listed nodes, in place of OUTPUT |

### `cfd-mesh identify-te`

A trailing edge is a family's rearmost line: the edges its surface lies forward of. Without NODES they are found from the mesh alone (each family cut at its sharp edges, an edge kept when it lies within --max-angle of the plane across --aft, the yz plane by default; a sharp trailing edge is followed to a twisted tip and a closed tip cap's contour is left out; a perimeter line found for more than half its edges is taken whole, to its corners) and the nodes are also written as &lt;mesh stem&gt;.nodes.txt, the node list a refine file reads. With NODES (a node-set list, every perimeter in one file, any order) they are taken among its lines, on the OBJ or a refined level of it. Written beside the mesh: &lt;mesh stem&gt;.te.txt, the points file a boundaries file names, for every mapped family; or &lt;mesh stem&gt;.&lt;family&gt;.te.txt per family with --families. Without --families only quad families are mapped (quadrilaterals at least 80 percent of the faces); --all-families maps the others too.

```text
cfd-mesh identify-te OBJ [NODES] [options]
```

| Argument or option | Meaning |
|---|---|
| `OBJ` | The mesh, with its families |
| `[NODES]` | The node list saved from the node set (default: the edges are found from the mesh) |
| `--families A,B` | The families, a points file each (default: every quad family, one file) |
| `--all-families` | Without --families, map every family, not only the quad families |
| `--max-angle DEG` | Without NODES, the largest angle of an edge with the plane across --aft (default 45) |
| `--aft X Y Z X Y Z X Y Z` | The aft direction (default +x) |
| `--unit UNIT` | The coordinates' unit (default: the mesh's, else METER) |
| `--overwrite` | Replace existing files |
| `--accept-empty` | A NODES list with fewer than two nodes, or no trailing edge found, is warned and nothing is written, exit 0, in place of refusing it |

### `cfd-mesh collapse-te`

MESH is an OBJ. The [collapse_te] table of the morphing file names the blades; each pair of rim nodes across a blade's one-cell base strip merges at its midpoint and the strip leaves. Every other node and face keeps its place; the boundaries file gets one trailing-edge line per blade and the checks are written to &lt;out&gt;.collapse.json. Quality findings (a face the merge turns over) are warned with their numbers and the mesh is written; a broken input or invariant refuses and nothing is written.

```text
cfd-mesh collapse-te MESH [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | Source .obj, read only |
| `--morphing MORPHING` | TOML with a [collapse_te] table |
| `--out OUT` | New .obj; a bare name lands in the workspace's inputs/mesh/ |

## cfd-mesh: rotors and whole meshes

### `cfd-mesh scale`

FACTOR is a positive number (0.5 halves every length). Every vertex of the OBJ given by --mesh is scaled about the origin, or about --about x,y,z; face order, groups, normals and comments stay byte for byte. --out names the scaled OBJ (default &lt;stem&gt;_scaled&lt;factor&gt;.obj in the workspace's inputs/mesh); &lt;name&gt;.scale.json beside it holds the input digest, factor, centre, counts and the bounding boxes before and after. The input is never overwritten; an existing output is refused unless --overwrite.

```text
cfd-mesh scale FACTOR [options]
```

| Argument or option | Meaning |
|---|---|
| `FACTOR` | Positive uniform scale factor, for example 0.5 |
| `--mesh MESH` | Source OBJ, read only |
| `--out OUT` | The scaled OBJ; default &lt;stem&gt;_scaled&lt;factor&gt;.obj |
| `--about ABOUT` | Scale centre x,y,z (default 0,0,0) |
| `--overwrite` | Replace the output if it exists |

### `cfd-mesh sector-to-wheel`

SECTOR is an OBJ (with its .obj.boundaries.toml and trailing-edge points files beside it). The sector is copied --copies times about --axis through --center, the nodes of the seams are merged within --tolerance, the surfaces named in --blades are numbered one per copy and every other surface is one group. The wheel is an OBJ with its boundaries file in the named form, one points file per rim per blade and &lt;out stem&gt;.wheel.json. --out names the wheel (default &lt;sector stem&gt;_wheel.obj in the workspace's inputs/mesh). A saved simulation is not accepted.

```text
cfd-mesh sector-to-wheel SECTOR [options]
```

| Argument or option | Meaning |
|---|---|
| `SECTOR` | The sector OBJ |
| `--copies COPIES` | The number of copies, 2 or more |
| `--blades FAMILY [FAMILY ...]` | The OBJ group(s) that are the blade |
| `--axis AXIS` | X, y, z, a signed one of them written --axis=-x, or three comma-separated numbers (default x) |
| `--center X Y Z X Y Z X Y Z` | A point of the axis, in the mesh's unit (default the origin) |
| `--tolerance METRES` | Nodes within this distance, in metres, are one node (default 1e-4) |
| `--out PATH` | The wheel (default &lt;sector stem&gt;_wheel.obj) |
| `--overwrite` | Replace an existing wheel |

### `cfd-mesh repitch`

MESH is an OBJ. The [repitch] table of the morphing file names the blades, the hub and the angle; each blade turns rigidly about its pitch axis; [repitch] hub_method chooses how the hub follows: its complete quad rings move (rings), the band around each turned root is remeshed quad-dominant (remesh), or auto picks one. The checks are written to &lt;out&gt;.repitch.json. Quality findings are warned with their numbers and the mesh is written; a broken input or invariant refuses and nothing is written.

```text
cfd-mesh repitch MESH [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | Source .obj, read only |
| `--morphing MORPHING` | TOML control file of the verb |
| `--out OUT` | New .obj (a bare name lands in the workspace's inputs/mesh) |

### `cfd-mesh hub-rings`

MESH is an OBJ. The [hub_rings] table of the morphing file states rings (and optionally band_width); blades and hub sit at the top or in the table. Per blade, rings rows of quads are marched outward from the root on the hub, the hub faces they reach plus one layer leave, and the band between them is remeshed quad-dominant. Blade nodes, other families and the hub's seams stay to the bit; the checks are written to &lt;out&gt;.hub_rings.json. Quality findings are warned with their numbers and the mesh is written; a broken input or invariant refuses and nothing is written.

```text
cfd-mesh hub-rings MESH [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | Source .obj, read only |
| `--morphing MORPHING` | TOML control file of the verb |
| `--out OUT` | New .obj (a bare name lands in the workspace's inputs/mesh) |

### `cfd-mesh morph`

MESH is an OBJ, a sector or a full wheel. The tables the morphing file holds are applied in one fixed order: [hub_rings], [repitch], [collapse_te], [wheel] (a sector only), each with its own checks; blades and hub sit at the top of the file. One OBJ is written with its boundaries file, its points files and &lt;out&gt;.morph.json. Each step warns of quality with its numbers and goes on; nothing is written when a step refuses a broken input or invariant.

```text
cfd-mesh morph MESH [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | Source .obj, read only |
| `--morphing MORPHING` | TOML control file of the verb |
| `--out OUT` | New .obj (a bare name lands in the workspace's inputs/mesh) |

### `cfd-mesh iges-grid`

Each component the IGES files name (the NAME of a level) becomes one family of the OBJ. Faces are joined so shared edges are conformal; four-sided patches follow the laws of the grid control file and become quads, other patches are quad-dominant at the component's size. Writes the OBJ, &lt;out&gt;.boundaries.toml, one points file per trailing edge found and the record &lt;out&gt;.json. Each junction of two components is measured against the original faces of both. Quality is warned, never refused.

```text
cfd-mesh iges-grid IGES [IGES ...] [options]
```

| Argument or option | Meaning |
|---|---|
| `IGES [IGES ...]` | IGES files, read only |
| `--grid GRID` | Grid control file (TOML) |
| `--out OUT` | New .obj (a bare name lands in the workspace's inputs/mesh) |
| `--component NAME` | Grid only this component, named as the files name it (repeatable); the others leave the model before the join. An unknown or ambiguous name is refused with the names found. Default: every component |
| `--name-encoding NAME_ENCODING` | Text encoding of the NAME bytes in the IGES files (default latin-1, one character per byte; use utf-8 for UTF-8 names). Never guessed: a name invalid in the encoding is refused |

### `cfd-mesh visual-inspect`

Writes &lt;mesh&gt;.inspect.pdf: the top, front, side and isometric views as a technical drawing (one scale, aligned, a scale bar), one color per patch from --cmap, each patch's perimeter in a thick line, the node lists and trailing-edge points files over the surface, and a table of each patch's faces and marked points. MESH is an OBJ or a NASTRAN export.

```text
cfd-mesh visual-inspect MESH [options]
```

| Argument or option | Meaning |
|---|---|
| `MESH` | The OBJ or NASTRAN mesh |
| `--nodes FILE [FILE ...]` | Node lists saved from an ANSA SET; repeat the flag or put MESH first |
| `--te FILE [FILE ...]` | Trailing-edge points files (.te.txt, .nodes.te.txt), one color each; repeat the flag or put MESH first |
| `--cmap CMAP` | The matplotlib colormap of the patches (default: matplotlib's default) |
| `--families A,B` | Draw only these patches |
| `--box XMIN XMAX YMIN YMAX ZMIN ZMAX XMIN XMAX YMIN YMAX ZMIN ZMAX XMIN XMAX YMIN YMAX ZMIN ZMAX XMIN XMAX YMIN YMAX ZMIN ZMAX XMIN XMAX YMIN YMAX ZMIN ZMAX XMIN XMAX YMIN YMAX ZMIN ZMAX` | Draw only the faces and points inside this box |
| `--mesh-unit MESH_UNIT` | The drawing's length unit: an OBJ's own unit, or the unit a NASTRAN mesh is converted to (default METER); the scale bar names it |
| `--points-unit POINTS_UNIT` | The unit of the node lists and points files that state none |
| `--unit UNIT` | A NASTRAN mesh's length unit (default MILLIMETER); not read for an OBJ |
| `--iso-from X Y Z X Y Z X Y Z` | The direction the isometric view is seen from (default 1 -1 1) |
| `--labels` | Write each patch's name on it |
| `--no-edges` | Draw the faces without edges |
| `--perimeter-width PT` | Width of the patch perimeters in points (default 0.9; 0 draws none) |
| `--dpi DPI` | Resolution of the rasterized surface in dots per inch (default 300) |
| `--out OUT` | The PDF (default: &lt;mesh&gt;.inspect.pdf beside the mesh) |
| `--overwrite` | Replace an existing PDF |
| `--png` | Also write each page of views as a PNG beside the PDF |

### `cfd-mesh prop-slice`

Isolates one blade about the axis, cuts it at radial stations by the planes the propeller builder places its sections in, and writes to --out the normalized profile of each station (sections/sNN.dat), the table over r/R of chord, pitch angle, thickness, camber, skew and rake (table.csv), the propeller definition that rebuilds the propeller (propeller.toml) and the record prop-slice.json.

```text
cfd-mesh prop-slice [options]
```

| Argument or option | Meaning |
|---|---|
| `--geometry GEOMETRY` | The propeller, an .obj |
| `--blades BLADES` | The number of blades |
| `--axis DX DY DZ DX DY DZ DX DY DZ` | Thrust direction, the builder's frame.thrust_dir (default -1 0 0) |
| `--origin X Y Z X Y Z X Y Z` | A point of the shaft in mm (default 0 0 0) |
| `--blade-dir DX DY DZ DX DY DZ DX DY DZ` | Reference blade direction; blade 0 is the nearest (default 0 0 1) |
| `--stations R_R [R_R ...]` | R/R of the sections (default 21 steps from the root to the tip, and 0.7) |
| `--unit UNIT` | OBJ length unit: mm (default), cm, m, in, or MILLIMETER, CENTIMETER, METER, INCH |
| `--out OUT` | The folder to write |
| `--overwrite` | Replace the outputs --out already holds (refused without it) |

## cfd-mesh: 2D airfoil volume meshes

### `cfd-mesh airfoil-2d`

Mesh a 2D airfoil section as a structured C-mesh of quadrilaterals and write &lt;name&gt;.su2 (markers airfoil and farfield) and &lt;name&gt;.msh (gmsh 2.2, one cell thick, patches airfoil, farfield, front and back, volume fluid) into the workspace's inputs/mesh/, each with a provenance sidecar. The first cell is sized for a target y+ at the given Reynolds number. A blunt edge adds airfoil_base.

```text
cfd-mesh airfoil-2d [options]
```

| Argument or option | Meaning |
|---|---|
| `--naca DIGITS` | A NACA 4 or 5 digit designation, for example 0012 |
| `--coordinates FILE` | A coordinates file in Selig order (a relative path is read from the workspace) |
| `--reynolds RE` | The chord Reynolds number the first cell is sized for |
| `--yplus Y+` | The target y+ of the first cell (default 1) |
| `--level {coarse,medium,fine}` | The cell counts (default coarse) |
| `--farfield CHORDS` | The far-field distance in chords (default 500) |
| `--trailing-edge {sharp,blunt}` | Collapse the trailing-edge gap or keep it as a base (default sharp) |
| `--out NAME` | The mesh name without a suffix (default: naca&lt;digits&gt; or the file's stem); a name with a folder is written there |

<!-- END GENERATED COMMAND REFERENCE -->
