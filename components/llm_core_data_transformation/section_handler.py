"""Manual GLiNER section-classification experiment.

Importing this module is side-effect free. Run it directly to load the model and
execute the example classification.
"""


def classify_example():
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
    return model.classify_text(
        """I am a highly energetic and passionate Fullstack Developer, dedicated to delivering
clean, maintainable, and scalable solutions. I thrive on immersing myself in new business
contexts and domains, translating this understanding into software that delivers real
value. With a preference for being involved throughout the entire development lifecycle,
I bring hands-on expertise from recent projects at DPG Media and Brussels Airport,
where I focused on Cloud Native applications on AWS using Microservices and
Serverless architectures""",
        {"scope": ["personality", "skills", "projects"]},
    )


def main():
    print(classify_example())


if __name__ == "__main__":
    main()
