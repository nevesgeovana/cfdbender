## Fixed

- SU2 decks now write `FLUID_MODEL= IDEAL_GAS` with `GAS_CONSTANT= 287.05287` and `GAMMA_VALUE= 1.4`. Under SU2's default `STANDARD_AIR` the two constants were replaced by 287.058 and 1.4, so the free stream did not use cfdbender's atmosphere. A `FLUID_MODEL` stated in the setup wins; cfdbender then writes neither constant and warns when the gas differs from the core one.
  migration: a setup that relied on `STANDARD_AIR` states `FLUID_MODEL = "STANDARD_AIR"`.
