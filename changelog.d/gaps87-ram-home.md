## Changed

- The 0.5 GiB free-RAM floor and the host memory measurement have one home, `cfdbender.core.run` (`free_ram_gib`, `check_available_ram`); the adapters' `require_ram` imports it, so the two guards cannot drift.
