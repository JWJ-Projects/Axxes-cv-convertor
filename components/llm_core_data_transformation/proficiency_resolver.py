from gliner2 import AutoExtractor
import pandas as pd
import numpy as np
class Sequence_Evaluator():
    def __init__(self):
        self.model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
        self.test_data = pd.read_csv("./data/developer_proficiency_test.csv")
        return
    def load_stage1(self):
        self.model.load_adapter(r"decide_proficiency_finetuned_stage1/best")
        return
    def evaluate(self, dataset=None):
        if dataset is not None:
            df = dataset
        else:
            df = self.test_data
        conf_matrix = np.zeros((4,4)) # Using a 4 by 4 confusion matrix to capture the results
        labels = ["Junior", "Confirmed", "Advanced", "Expert"]
        for i, row in df.iterrows():
            proficiency_test = self.model.classify_text(
                row["description"],
                {"proficiency": [
                    "Junior", "Confirmed", "Advanced", "Expert"
                ]}
            )
            conf_matrix[labels.index(proficiency_test["proficiency"]), labels.index(row["proficiency"])] += 1
        return conf_matrix
class Sequence_Metrics():
    def __init__(self, conf_matrix):
        self.conf_matrix = conf_matrix
        return
    def accuracy(self):
        correct = 0
        for i in range(0,4):
            correct += self.conf_matrix[i,i]
        return correct / self.conf_matrix.sum()
    def recall(self, label: int = 0):
        tp = self.conf_matrix[label, label]
        y_label_true = self.conf_matrix[:, label].sum()
        return tp / y_label_true
    def precision(self, label: int = 0):
        tp = self.conf_matrix[label, label]
        y_label_pred = self.conf_matrix[label, :].sum()
        return tp / y_label_pred
def main():
    seq_eval = Sequence_Evaluator()
    seq_eval.load_stage1()
    conf_matrix = seq_eval.evaluate()
    metrics = Sequence_Metrics(conf_matrix)
    print(f"Accuracy: {metrics.accuracy()}")
    print(f"Recall: {metrics.recall()}")
    print(f"Precision: {metrics.precision()}")
    print(conf_matrix)
    return
def predict():
    return
if __name__ == '__main__':
    main()