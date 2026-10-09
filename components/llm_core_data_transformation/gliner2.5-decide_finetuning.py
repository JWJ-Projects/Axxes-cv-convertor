from gliner2 import AutoExtractor
from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorTrainer, ExtractorDataset, TrainingConfig
import pandas as pd
import json
def df_to_json(df: pd.DataFrame, json_path: str = "./data/train.jsonl") -> str:
    json_objects = []
    for i, row in df.iterrows():
        json_object = {
            "input":row["description"],
            "output":{
                "classifications": [{
                    "task": "proficiency",
                    "labels": ["Junior","Confirmed","Advanced", "Expert"],
                    "true_label": "Junior"
                }]
            }
        }
        json_objects.append(json.dumps(json_object) + '\n')
    with open(json_path, 'w') as json_file:
        json_file.writelines(json_objects)
    return json_path

def main():
    df_train = pd.read_csv("./data/developer_proficiency_descriptions.csv")
    json_path = df_to_json(df_train)
    model = AutoExtractor.from_pretrained(
        "fastino/GLiNER2.5-Decide"
    )
    config = TrainingConfig(
        output_dir="./decide_finetuned",
        num_epochs=3,
        batch_size=8,
        encoder_lr=1e-5,
        task_lr=5e-4,
    )

    trainer = ExtractorTrainer(model, config)

    trainer.train(train_data=json_path)
    return
if __name__ == '__main__':
    main()