# Axxes CV Converter

## Description

This is a custom pipeline that uses various NLP type models to parse, classify and rewrite information that would be found in and around the cv of an applicant in the field of IT. The desired output is a standardised cv, used within Axxes IT consultancy. This entire project is developed by Jonas Vandenberghe, Joppe Iemants and William Plancke under the guidance of the Howest AI lecturer team. However it is the intellectual property of Axxes and the entire codebase will be transfered to them when the project is complete.

## Skills transformation

The transformation component now produces renderer-ready skills JSON from extracted
CV data, with optional interview evidence, opt-in GLiNER proficiency inference,
and evidence-grounded introduction generation using a local chat LLM.
See [the component guide](components/llm_core_data_transformation/README.md) for
commands, the output contract, review policies, and tests.

If for some reason cuda isn't available (GPU inference) check your cuda version with
```bash
nvidia-smi
```
Then uninstall `torch` and `torchvision`.
Reinstall both from here https://pytorch.org/get-started/previous-versions/
Example for cuda 13.0 and 13.1:
```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```
Then it should be fixed.
## Infrastructure

To be added...
