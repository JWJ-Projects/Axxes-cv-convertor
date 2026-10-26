from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split


def df_to_json(df: pd.DataFrame, json_train_path: str = "./data/train.jsonl", json_val_path: str = "./data/val.jsonl") -> tuple[str, str]:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["proficiency"],
    )

    for split_df, path in ((train_df, json_train_path), (val_df, json_val_path)):
        with open(path, "w", encoding="utf-8") as file:
            for _, row in split_df.iterrows():
                record = {
                    "input": row["description"],
                    "output": {
                        "classifications": [{
                            "task": "proficiency",
                            "labels": ["Junior", "Confirmed", "Advanced", "Expert"],
                            "true_label": [row["proficiency"]],
                        }]
                    },
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Training class counts:\n", train_df["proficiency"].value_counts())
    print("Validation class counts:\n", val_df["proficiency"].value_counts())
    return json_train_path, json_val_path

def main():
    print(torch.cuda.is_available())
    df_train = pd.read_csv("./data/developer_proficiency_train_val.csv")
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