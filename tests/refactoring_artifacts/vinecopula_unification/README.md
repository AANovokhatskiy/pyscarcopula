# VineCopula unification artifacts

`test_stage0_contract.py` freezes the pre-unification `RVineCopula` behavior.
It may be removed after the generic `VineCopula` tests cover all permanent
contracts and every intentionally changed assertion is documented in
`VINECOPULA_UNIFICATION_PLAN.md`.

The format-v2 RVine persistence fixture is not temporary. It remains under
`tests/fixtures/persistence/` to guard backward-compatible loading.

