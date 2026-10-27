from components.llm_core_data_transformation.proficiency_resolver import predict as predict_proficiency
from components.vlm_core_data_extraction.vlm_extractor import extract_data_from_cv, retrieve_extracted_cv_data

from pathlib import Path


def main():

    pdf_path = Path("components/vlm_core_data_extraction/assets/original_cvs/Joppe_Iemants_CV.pdf")
    extracted_cv_data = retrieve_extracted_cv_data(pdf_path) # This reads already existing extractions based on the original cv path (Use extract_data_from_cv for real extraction(use same pdf_path))
    print(extracted_cv_data.model_dump_json(indent=2))

    #proficiencies = predict_proficiency() 

    return

if __name__ == '__main__':
    main()