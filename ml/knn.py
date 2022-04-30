from pandas import DataFrame
from typing import List, Union
import math


class KNN:
    k: int
    verbose: bool

    def __init__(self, k, verbose=False):
        self.k = k
        self.verbose = verbose
        self.data = []

    def fit(self, x: List[float], y: Union[str, int]):
        self.data.append((x, y))

    def train(self, train_df: DataFrame):
        for i in range(len(train_df)):
            self.fit(train_df.iloc[i, :-1], train_df.iloc[i, -1])

    def predict(self, x: List[float]) -> Union[str, int]:
        distances = []
        for i in range(len(self.data)):
            dist = 0
            for j in range(len(x)):
                dist += (x[j] - self.data[i][0][j]) ** 2
            distances.append((dist, self.data[i][1]))
        distances.sort(key=lambda x: x[0])
        votes = {}
        for i in range(self.k):
            if i >= len(distances):
                break
            vote = distances[i][1]
            distance = distances[i][0]
            if distance == 0:
                vote_value = math.inf
            else:
                vote_value = 1 / distance
            if vote in votes:
                votes[vote] += vote_value
            else:
                votes[vote] = vote_value
        sorted_votes = sorted(
            votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]

    def predict_on_df(self, df: DataFrame) -> List[Union[str, int]]:
        predictions = []
        for i in range(len(df)):
            predictions.append(self.predict(df.iloc[i, :-1]))
        return predictions
