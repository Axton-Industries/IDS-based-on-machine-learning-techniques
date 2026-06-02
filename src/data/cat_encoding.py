import pandas as pd


class CustomBenignFirstEncoder:
    def __init__(self):
        self.classes_ = []
        self.mapping_ = {}

    def fit(self, y):
        unique_classes = sorted(list(set(y)))
        if "Benign" in unique_classes:
            unique_classes.remove("Benign")
            self.classes_ = ["Benign"] + unique_classes
        else:
            self.classes_ = unique_classes

        self.mapping_ = {label: idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return [self.mapping_[label] for label in y]

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def get_mapping(self):
        """Returns the dictionary mapping labels to numbers."""
        return self.mapping_
