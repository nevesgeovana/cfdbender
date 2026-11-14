## Changed

- SU2 decks now write `MARKER_PLOTTING` and `MARKER_MONITORING` with the wall markers of `MARKER_EULER`, `MARKER_HEATFLUX` and `MARKER_ISOTHERMAL` when the setup states neither, so the forces are summed over the walls and `surface.csv` has rows. A list stated in the setup wins.
