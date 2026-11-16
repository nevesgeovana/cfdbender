"""Shared test support: small IGES files written from the fixed eighty-column layout.

Bilinear B-spline patches (128), a surface of revolution (120) on a line axis (110), a free line
(110), and 406 form 15 NAME entities on levels. ``cli`` runs the IGES verbs the way a person types
them: ``geometry iges-grid`` is ``cfd-mesh iges-grid`` and ``geometry iges-convert`` is
``cfd-geom iges-convert``.
"""

import math

from kernel_support import kernel_or_skip

UNIT_FLAG = {"MM": 2, "IN": 1, "M": 6}


def _line(text, section, number):
    return f"{text:<72}{section}{number:>7}"


class Iges:
    """A small IGES writer: entities in directory order, each with its level and status."""

    def __init__(self, unit="MM", *, flag=None, head=",,"):
        self.unit = unit
        # The unit flag (default the one of ``unit``) and the two delimiter fields that open
        # the global section.
        self.flag = UNIT_FLAG[unit] if flag is None else flag
        self.head = head
        self.entities = []

    def add(self, kind, values, *, level=0, form=0, status="00000000"):
        self.entities.append((kind, values, level, form, status))
        return 2 * len(self.entities) - 1

    def name(self, level, text):
        return self.add(406, ["1", f"{len(text)}H{text}"], level=level, form=15)

    def patch(self, corners, *, level, status="00000000"):
        """A bilinear B-spline patch: corners (u0v0, u1v0, u0v1, u1v1)."""
        values = ["1", "1", "1", "1", "0", "0", "1", "0", "0"]
        values += ["0.0", "0.0", "1.0", "1.0", "0.0", "0.0", "1.0", "1.0"] + ["1.0"] * 4
        for point in corners:
            values += [f"{c:.6f}" for c in point]
        values += ["0.0", "1.0", "0.0", "1.0"]
        return self.add(128, values, level=level, status=status)

    def line(self, a, b, *, level=0, status="00000000"):
        return self.add(110, [f"{c:.6f}" for c in (*a, *b)], level=level, status=status)

    def text(self):
        start = [_line("cfdbender test IGES", "S", 1)]
        unit = self.unit
        fields = (
            f"{self.head}4HTEST,8Htest.igs,4HPYFS,3H1.0,32,38,6,308,15,4HTEST,1.0,{self.flag},"
            f"{len(unit)}H{unit},1,1.0,15H20261105.120000,1.0E-6,100000.0,4HPYFS,4HPYFS,11,0,"
            "15H20261105.120000;"
        )
        glob = [
            _line(fields[i : i + 72], "G", n + 1) for n, i in enumerate(range(0, len(fields), 72))
        ]
        directory, parameters = [], []
        for index, (kind, values, level, form, status) in enumerate(self.entities):
            de = 2 * index + 1
            record = ",".join([str(kind), *values]) + ";"
            first = len(parameters) + 1
            for i in range(0, len(record), 64):
                parameters.append(
                    _line(f"{record[i : i + 64]:<64}{de:>8}", "P", len(parameters) + 1)
                )
            count = len(parameters) - first + 1
            one = [str(kind), str(first), "0", "0", str(level), "0", "0", "0", status]
            two = [str(kind), "0", "0", str(count), str(form), "", "", "", "0"]
            directory.append(_line("".join(f"{f:>8}" for f in one), "D", de))
            directory.append(_line("".join(f"{f:>8}" for f in two), "D", de + 1))
        counts = f"S{len(start):>7}G{len(glob):>7}D{len(directory):>7}P{len(parameters):>7}"
        return "\n".join(start + glob + directory + parameters + [_line(counts, "T", 1)]) + "\n"

    def write(self, path):
        path.write_text(self.text(), encoding="latin-1")
        return path


def square(x0, y0, x1, y1, z=0.0):
    return [(x0, y0, z), (x1, y0, z), (x0, y1, z), (x1, y1, z)]


def plate_and_flap(folder, *, flap_gap=0.0, unit="MM", scale=1.0, name="parts.igs"):
    """Plate [0,2]x[0,1] on level 1, flap [0,1]x[1,2] on level 2: the flap shares half an edge."""
    iges = Iges(unit)
    iges.name(1, "Plate|ANSA_Thick=1.")
    iges.name(2, "Flap")
    iges.name(0, "Translate.3")
    iges.patch([tuple(c * scale for c in p) for p in square(0, 0, 2, 1)], level=1)
    iges.patch([tuple(c * scale for c in p) for p in square(0, 1 + flap_gap, 1, 2)], level=2)
    return iges.write(folder / name)


def wedge(folder, *, te_curve=False, name="wing.igs", component="Wing"):
    """A wing wedge on level 1: upper and lower faces meet at a sharp trailing edge at x = 1."""
    iges = Iges()
    iges.name(1, component)
    iges.patch([(0, 0, 0.1), (1, 0, 0), (0, 1, 0.1), (1, 1, 0)], level=1)
    iges.patch([(0, 0, -0.1), (1, 0, 0), (0, 1, -0.1), (1, 1, 0)], level=1)
    iges.patch([(0, 0, -0.1), (0, 0, 0.1), (0, 1, -0.1), (0, 1, 0.1)], level=1)
    if te_curve:
        iges.name(3, "WingTE")
        iges.line((1, 0, 0), (1, 1, 0), level=3)
    return iges.write(folder / name)


def revolution(folder, name="body.igs"):
    """A cone-like surface of revolution (120) of a slanted line about the x axis, on level 4."""
    iges = Iges()
    iges.name(4, "Body")
    axis = iges.line((0, 0, 0), (1, 0, 0), status="00010000")
    generatrix = iges.line((0, 0, 0.2), (1, 0, 0.5), status="00010000")
    iges.add(120, [str(axis), str(generatrix), "0.0", f"{2 * math.pi:.9f}"], level=4)
    return iges.write(folder / name)


def grid_file(folder, text):
    path = folder / "grid.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _main(args):
    """Return the program and its arguments for a command written in the old two-word form."""
    if args[:1] == ["geometry"]:
        args = args[1:]
    if args[:1] == ["iges-convert"]:
        from cfdbender.geometry.cli import main as geom_main

        return geom_main, args
    from cfdbender.mesh.cli import main as mesh_main

    return mesh_main, args


def cli(args, capsys):
    capsys.readouterr()
    main, rest = _main(list(args))
    try:
        code = main(rest)
    except SystemExit as error:
        code = error.code
    out = capsys.readouterr()
    return code, out.out, out.err


def gmsh_or_skip():
    return kernel_or_skip()
