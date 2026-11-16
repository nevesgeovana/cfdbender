Export check failures now explain the failing check, measured value, limit,
location and corrective action in the geometry run report.

migration: An export failure's message now contains a diagnostic instead of an empty string.

Rectangular export faces now use converged surface integration before and
after reimport, avoiding false area failures on rounded caps. The relative
area acceptance limit remains 0.001.

migration: Rectangular face areas in export manifests and read-back records can change through more accurate measurement.

Reader gridding failures now identify crossing components and faces and
explain which trimming relationship the geometry configuration needs.

migration: A gate.grid refusal includes its defect count, zero limit and affected faces.
