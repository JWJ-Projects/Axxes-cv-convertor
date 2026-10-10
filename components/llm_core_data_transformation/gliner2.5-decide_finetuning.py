from gliner2 import AutoExtractor
from gliner2.training.data import InputExample
from gliner2.training.trainer import ExtractorTrainer, ExtractorDataset, TrainingConfig
import pandas as pd
import json
import torch
def df_to_json(df: pd.DataFrame, json_train_path: str = "./data/train.jsonl", json_val_path: str = "./data/val.jsonl") -> str:
    json_objects = []
    df = df.sample(random_state=42)
    for i, row in df.iterrows():
        json_object = {
            "input":row["description"],
            "output":{
                "classifications": [{
                    "task": "proficiency",
                    "labels": ["Junior","Confirmed","Advanced", "Expert"],
                    "true_label": [row["proficiency"]]
                }]
            }
        }
        json_objects.append(json.dumps(json_object) + '\n')
    json_train = json_objects[:int(len(json_objects)*0.8)]
    json_val = json_objects[int(0.8*len(json_objects)):]
    with open(json_train_path, 'w') as json_train_file:
        json_train_file.writelines(json_train)
    with open(json_val_path, 'w') as json_val_file:
        json_val_file.writelines(json_val)
    return json_train_path, json_val_path

def main():
    print(torch.cuda.is_available())
    df_train = pd.read_csv("./data/developer_proficiency_descriptions.csv")
    json_train_path, json_val_path = df_to_json(df_train)
    model = AutoExtractor.from_pretrained(
        "fastino/GLiNER2.5-Decide"
    )
    config = TrainingConfig(
        output_dir="./decide_proficiency_finetuned_stage1",
        num_epochs=10,
        batch_size=16,
        eval_batch_size=16,
        encoder_lr=1e-5,
        task_lr=1e-4,
        use_lora=True,
        lora_alpha=64,
        lora_r=32,
        lora_dropout=0,
        early_stopping = True,
        early_stopping_patience = 5,
        num_workers=0
    )

    trainer = ExtractorTrainer(model, config)
    trainer.train(train_data=json_train_path, eval_data=json_val_path)
    return
if __name__ == '__main__':
    main()