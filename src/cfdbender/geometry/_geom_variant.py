"""Selected control positions, exported through the existing source and assembly path."""

from itertools import product
from pathlib import Path

from cfdbender._errors import InputArtifactError
from cfdbender.geometry.stages import junctions, unpack, write_record


def control_selections(config, members, variant):
    plain = unpack(config)
    active = plain["variants"][variant]
    choices = {}
    for name in sorted(members):
        for entry in plain["components"][name].get("control_surfaces", []):
            key = f"{name}.{entry['name']}"
            if key in active:
                choices[key] = [
                    f"{entry['name']}-d{str(float(a)).removesuffix('.0')}"
                    for a in sorted(entry["deflection_deg"])
                ]
    return [dict(zip(choices, values, strict=True)) for values in product(*choices.values())]


def generate(config, path, configuration, variant, members):
    from ._geom_export import generate_record
    from .pipeline import _assembly_record

    target = Path(path)
    planned = junctions.dominance_plan(unpack(config).get("dominance", []), members, [])
    written, exports = [], []
    for index, selection in enumerate(control_selections(config, members, variant)):
        output = target if index == 0 else target.parent / f"position-{index}" / target.name
        if planned:
            assembled = _assembly_record(
                config,
                members,
                str(output.parent / "S10-assembly.json"),
                configuration,
                variant,
                selection=selection,
            )
            result = assembled["export"]
            if not result:
                raise InputArtifactError(str(assembled["records"][-1]), kind="geometry")
        else:
            result = generate_record(config, str(output), configuration, variant, selection)
        written.append(result)
        exports.append(
            {
                "selection": selection,
                "manifest": str(Path(result["manifest"]).relative_to(target.parent)),
                "iges": str(Path(result["iges"]).relative_to(target.parent)),
                "surface_id": result["surface_id"],
            }
        )
    return _bundle(written, exports, target.parent)


def _bundle(written, exports, folder):
    source = written[0]
    source["alternatives"] = written[1:]
    source["variant_exports"] = exports
    write_record(
        folder / "variant-exports.json",
        {
            "format": "cfdbender-geom-variant-exports/1",
            "exports": exports,
        },
    )
    return source


def assemble_positions(config, members, out, configuration, variant):
    """Run the existing assembly for every selected position and retain its records."""
    from .pipeline import _assembly_record

    folder = Path(out).parent
    results, written, exports = [], [], []
    for index, selection in enumerate(control_selections(config, members, variant)):
        destination = folder if index == 0 else folder / f"position-{index}"
        result = _assembly_record(
            config,
            members,
            str(destination / "S10-assembly.json"),
            configuration,
            variant,
            selection=selection,
        )
        results.append(result)
        if not result["export"]:
            break
        source = result["export"]
        written.append(source)
        exports.append(
            {
                "selection": selection,
                "manifest": str(Path(source["manifest"]).relative_to(folder)),
                "iges": str(Path(source["iges"]).relative_to(folder)),
                "surface_id": source["surface_id"],
            }
        )
    records = {}
    for index, result in enumerate(results):
        for record in result["records"]:
            combined = records.setdefault(
                record["stage"],
                {
                    **record,
                    "checks": [],
                    "warnings": [],
                    "outputs": {"positions": [], "inputs": {"positions": []}},
                    "wall_s": 0,
                },
            )
            combined["checks"].extend(
                {**c, "where": {**c["where"], "position": index}} for c in record["checks"]
            )
            combined["warnings"].extend(record["warnings"])
            combined["outputs"]["positions"].append(record["outputs"])
            combined["outputs"]["inputs"]["positions"].append(record["outputs"].get("inputs", {}))
            combined["wall_s"] += record["wall_s"]
    complete = all(r["export"] for r in results)
    result = {
        "records": list(records.values()),
        "export": _bundle(written, exports, folder) if complete else {},
    }
    write_record(Path(out), result)
    return result
