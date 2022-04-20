import pandas as pd
from typing import List, Union, TypedDict, Dict


class LabelMetrics(TypedDict):
    correct: int
    predicted: int
    true: int


def metrics(test_df: pd.DataFrame, predictions: List[Union[str, int]]) -> Dict[str, LabelMetrics]:
    metrics_dict = {}

    def metrics_increase(label: str, type: str):
        if label not in metrics_dict:
            metrics_dict[label] = {"correct": 0, "predicted": 0, "true": 0}
        metrics_dict[label][type] += 1

    for i in range(len(test_df)):
        actual = test_df.iloc[i, -1]
        predicted = predictions[i]
        metrics_increase(actual, "true")
        if actual == predicted:
            metrics_increase(actual, "correct")
        metrics_increase(predicted, "predicted")

    return metrics_dict
