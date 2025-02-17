from mmm_fair import data_uci

def test_data_uci():
    # This might fail if you have no real "credit" dataset loaded from ucimlrepo
    csv_ds = data_uci("credit", target="Y")  
    X = csv_ds.to_features(sensitive=["Gender"])
    assert X.shape[0] > 0, "Expected non-empty features"