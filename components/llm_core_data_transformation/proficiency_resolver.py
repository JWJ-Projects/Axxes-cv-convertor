from gliner2 import AutoExtractor
import pandas as pd


def main():
    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
    for row in proficiency_descriptions:
        proficiency_test = model.classify_text(
            desc,
            {"proficiency": [
                "Junior", "Confirmed", "Advanced", "Expert"
            ]}
        )
        print(proficiency_test)
    return
if __name__ == '__main__':
    main()