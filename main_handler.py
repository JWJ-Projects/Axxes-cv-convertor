"""Current entry point: RawCV JSON -> skills and optional local-LLM introduction.

PDF extraction and full CV rendering remain separate pipeline stages.
"""
from components.llm_core_data_transformation.skills_handler import main

if __name__ == '__main__':
    main()