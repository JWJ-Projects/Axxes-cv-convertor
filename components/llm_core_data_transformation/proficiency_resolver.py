from gliner2 import AutoExtractor
import pandas as pd

df = pd.read_csv("./data/developer_proficiency_descriptions.csv")
def main():
    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
    correct = 0
    incorrect = 0
    for i, row in df.iterrows():
        proficiency_test = model.classify_text(
            row["description"],
            {"proficiency": [
                "Junior", "Confirmed", "Advanced", "Expert"
            ]}
        )
        if proficiency_test["proficiency"] == row["proficiency"]:
            correct += 1
        else:
            incorrect += 1
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    return
if __name__ == '__main__':
    main()