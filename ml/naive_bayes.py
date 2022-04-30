from pandas import DataFrame
from typing import List, Tuple, Set, Dict


class NaiveBayes:
    c: float
    verbose: bool
    value_set: List[Set[str]] = []
    pure_probabilities: Dict[str, float] = {}
    pure_probabilities_desc: Dict[str, str] = {}
    cond_probabilities: Dict[Tuple[str, int, str], float] = {}
    cond_probabilities_desc: Dict[Tuple[str, int, str], str] = {}

    def __init__(self, c: float, verbose=False):
        self.c = c
        self.verbose = verbose

    def train(self, train_df: DataFrame):
        total_num = len(train_df)
        for i in train_df.keys():
            self.value_set.append(set(train_df[i]))
        pure_count_dict: Dict[str, int] = {}
        cond_count_dict: Dict[Tuple[str, int, str], int] = {}
        for i in range(len(train_df)):
            row_x = train_df.iloc[i, :-1]
            row_y = train_df.iloc[i, -1]
            for j in range(len(row_x)):
                x_col = row_x[j]
                key = (x_col, j, row_y)
                if key in cond_count_dict:
                    cond_count_dict[key] += 1
                else:
                    cond_count_dict[key] = 1
            if row_y in pure_count_dict:
                pure_count_dict[row_y] += 1
            else:
                pure_count_dict[row_y] = 1
        label_dom = len(self.value_set[-1])
        for label in sorted(self.value_set[-1]):
            count = pure_count_dict[label] if label in pure_count_dict else 0
            self.pure_probabilities[label] = count / total_num
            self.pure_probabilities_desc[label] = f"{count} / {total_num}"

            for i in range(len(self.value_set) - 1):
                x_col_dom = len(self.value_set[i])

                for x_col in sorted(self.value_set[i]):
                    key = (x_col, i, label)
                    count = cond_count_dict[key] if key in cond_count_dict else 0
                    self.cond_probabilities[key] = (
                        count + self.c) / (pure_count_dict[label] + self.c * x_col_dom)
                    self.cond_probabilities_desc[key] = f"{count + self.c} / {pure_count_dict[label] + self.c * x_col_dom}"

    def calculate_y_prob(self, x: List[str], y: str) -> float:
        if y not in self.value_set[-1]:
            print(f"Warning: label {y} not in training label set")
            return 0
        if self.verbose:
            print(f"P(C={y}) = {self.pure_probabilities_desc[y]}")
        prob = self.pure_probabilities[y]
        if len(x) > len(self.value_set) - 1:
            print(f"Error: x ({x}) has more features than training data")
            exit(1)
        for i, x_col in enumerate(x):
            if x_col not in self.value_set[i]:
                print(
                    f"Warning: x value {x_col} for column #{i+1} not in training set")
                return 0
            key = (x_col, i, y)
            if self.verbose:
                print(
                    f"P(A{i}={x_col} | C={y}) = {self.cond_probabilities_desc[key]}")
            prob *= self.cond_probabilities[key]
        return prob

    def predict(self, x: List[str]) -> str:
        labels: List[str] = []
        probs: List[float] = []
        for label in sorted(self.value_set[-1]):
            prob = self.calculate_y_prob(x, label)
            labels.append(label)
            probs.append(prob)
        for i in range(len(labels)):
            label = labels[i]
            prob = probs[i]
            if self.verbose:
                print(f"P(C={label}) = {prob:.6f}")
        max_prob = max(probs)
        max_index = probs.index(max_prob)
        return labels[max_index]

    def predict_on_df(self, df: DataFrame) -> List[str]:
        predictions = []
        for i in range(len(df)):
            predictions.append(self.predict(df.iloc[i, :-1]))
        return predictions
