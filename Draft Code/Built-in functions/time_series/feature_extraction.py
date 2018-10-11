from tsfresh import extract_features


def get_features(ds, key=None):

    df = ds.build_dataframe()
    df.index.name = 'time'
    df = df.reset_index()

    if key is None:
        key = 'feature_extraction_id'
        df[key] = 1

    features = extract_features(df, column_id=key, column_sort='time')
    return features
