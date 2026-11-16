"""Builders of the published geometry input form, shared by the geometry tests."""

from __future__ import annotations


def station(item):
    """Return the ``[[stations]]`` table of a section given as ``{eta, naca|file|bspline|cst}``."""
    out = {"eta": item["eta"]}
    rest = {k: v for k, v in item.items() if k != "eta"}
    if "naca" in rest:
        out["section"] = {"kind": "naca", "code": rest.pop("naca")}
    elif "file" in rest:
        out["section"] = {"kind": "coords", "file": rest.pop("file")}
    elif "bspline" in rest:
        out["section"] = {"kind": "bspline", **rest.pop("bspline")}
    elif "cst_upper" in rest:
        out["section"] = {"kind": "cst", **{k: rest.pop(k) for k in list(rest)}}
    out.update(rest)
    return out


def stations(items):
    """Return the stations of a list of sections."""
    return [station(item) for item in items]


def naca_stations(designations, eta=None):
    """Return stations of NACA sections, evenly spaced from 0 to 1 unless ``eta`` is given."""
    count = len(designations)
    etas = eta if eta is not None else [i / max(1, count - 1) for i in range(count)]
    return [
        {"eta": e, "section": {"kind": "naca", "code": d.removeprefix("NACA")}}
        for e, d in zip(etas, designations, strict=True)
    ]


_SECTION_MM = {
    "width": "half_width_mm",
    "height": "half_height_mm",
    "centre_height": "centre_z_mm",
    "z_c": "centre_z_mm",
    "radius": "radius_mm",
}


def body_section(section):
    """Return a body section of the published form from one written in metres."""
    return {
        _SECTION_MM.get(k, k): round(v * 1000.0, 9) if k in _SECTION_MM else v
        for k, v in section.items()
    }


def body_input(raw):
    """Return the published form of a body written with the resolver's names (metres)."""
    out = dict(raw)
    if "length" in out:
        out["length_mm"] = round(out.pop("length") * 1000.0, 9)
    if isinstance(out.get("rear"), str):
        out["tail"] = {"kind": out.pop("rear")}
    elif "rear" in out:
        out["tail"] = out.pop("rear")
    if isinstance(out.get("nose"), str):
        out["nose"] = {"kind": out["nose"]}
    if "stations" in out and out.get("type", "body") == "body":
        out["parts"] = [0.0, *out.pop("stations"), 1.0]
    if isinstance(out.get("sections"), list):
        out["sections"] = [body_section(s) for s in out["sections"]]
    return out
