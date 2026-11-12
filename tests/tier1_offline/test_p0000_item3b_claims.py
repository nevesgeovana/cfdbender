"""P0000 item 3B: one point cannot claim a file as a directory or its parent."""

import pytest
from test_p0000_item3b_plan import plan_api, study


@pytest.mark.parametrize(
    "outputs",
    [
        ("post/result", "post/result/child.csv"),
        ("sims/sim_{pol}/datapoints/{point}",),
        ("sims/sim_{pol}",),
    ],
)
def test_p0000_item3b_file_directory_claims_are_refused_even_within_one_point(tmp_path, outputs):
    path = study(tmp_path)
    path.write_text(path.read_text().replace("-2,0,2", "0"))
    api = plan_api()

    def claims(row, inputs):
        return api.RowRequirements(actions=("solve",), outputs=outputs)

    with pytest.raises(api.PlanError, match="point"):
        api.plan_matrix(tmp_path, "study", validators={"mock": claims})
