import pandas as pd
from src.services.hcc_lookup import HCCLookupService


def test_lookup(tmp_path):
    csv = tmp_path / "hcc.csv"
    pd.DataFrame(
        [{"Description": "Diabetes Mellitus", "ICD-10-CM Codes": "E11", "Tags": "True"}]
    ).to_csv(csv, index=False)

    svc = HCCLookupService(str(csv))
    result = svc.lookup("Diabetes Mellitus")

    print(result)
    assert result["code"] == "E11"
