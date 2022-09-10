
import pandas as pd


def get_meta_segments_delta(
    current_meta_segments: pd.DataFrame, old_meta_segments: pd.DataFrame
):
    if old_meta_segments is None:
        return current_meta_segments

    df = old_meta_segments.join(current_meta_segments, lsuffix=".OLD", rsuffix=".NEW")
    for c in ["point", "sequential", "monotonic", "random", "dictionary"]:
        df[f"{c}_accesses.NEW"] = df[f"{c}_accesses.NEW"] - df[f"{c}_accesses.OLD"]

    for c in df.columns:
        if ".OLD" in c:
            df.drop(c, axis=1, inplace=True)

    for c in df.columns:
        if ".NEW" in c:
            df.rename({c: c.replace(".NEW", "")}, axis=1, inplace=True)

    return df
