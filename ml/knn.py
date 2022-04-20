from pandas import DataFrame
from typing import List, Union


class KNN:
    def __init__(self, k):
        self.k = k
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
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1
        sorted_votes = sorted(
            votes.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return sorted_votes[0][0]

    def predict_on_df(self, df: DataFrame):
        predictions = []
        for i in range(len(df)):
            predictions.append(self.predict(df.iloc[i, :-1]))
        return predictions
