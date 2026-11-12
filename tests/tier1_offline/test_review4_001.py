import pytest
from test_p0000_item3b_plan import plan_api, study


@pytest.mark.parametrize(
    "output", ["post/result.", "post/name /x", "post/FILE~1.TXT", "post/CON.txt"]
)
def test_review4_001_ambiguous_windows_output_is_refused(tmp_path, output):
    path = study(tmp_path)
    path.write_text(path.read_text().replace("-2,0,2", "0"))
    api = plan_api()

    def hook(row, inputs):
        return api.RowRequirements(actions=("solve",), outputs=(output,))

    with pytest.raises(api.PlanError):
        api.plan_matrix(tmp_path, "study", validators={"mock": hook})
