from components.llm_core_data_transformation.proficiency_resolver import predict as predict_proficiency

def main():
    # vlm_core_data_extraction entrypoint
    proficiencies = predict_proficiency() 
    return

if __name__ == '__main__':
    main()