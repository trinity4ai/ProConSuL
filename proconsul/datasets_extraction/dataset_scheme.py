from abc import ABC


class DatasetScheme(ABC):
    # Generated keys
    WHOLE_TEXT_KEY = 'whole_text'
    INPUT_IDS_KEY = 'input_ids'
    ATTENTION_MASK_KEY = 'attention_mask'
    LABELS_KEY = 'labels'
    PROMPT_KEY = 'prompt'
    GENSUM_KEY = 'generated_summary'
    GENSUM_INTENT_KEY = 'gensum_intent'
    GEN_CONTEXT_KEY = 'generated_context'
    DATASET_IDS_KEY = 'ids'
    PROMPT_LEN_KEY = 'prompt_len'
    SYNTHETIC_DOC_KEY = 'synth_doc'

    # QA-based metrics keys
    SUFFICIENCY_CLAIMS_KEY = "sufficiency_claims"
    SUFFICIENCY_CLAIMS_SCORE_KEY = "sufficiency_answers_score"
    SUFFICIENCY_CLAIMS_ANSWERS_KEY = "sufficiency_answers"
    ILLEGAL_FACTS_KEY = "illegal_facts"
    ILLEGAL_FACTS_SCORE_KEY = "illegal_facts_answers_score"
    ILLEGAL_FACTS_ANSWERS_KEY = "illegal_facts_answers"
    HALLUCINATION_SUBS_KEY = "hallucination_subs"
    HALLUCINATION_SUBS_SCORE_KEY = "hallucination_subs_score"
    SUFFICIENCY_SUBS_KEY = "sufficiency_subs"
    SUFFICIENCY_SUBS_SCORE_KEY = "sufficiency_subs_score"

    # Original dataset keys
    ID_KEY = 'id'
    CODE_KEY = 'code'
    DOC_KEY = 'doc'
    NAME_KEY = 'name'
    TO_KEY = 'to'
    PATH_KEY = 'path'
    INTENT_KEY = 'intent'
    CONTEXT_KEY = 'context'
    REPO_KEY = 'repository'
    SYSTEM_KEY = 'system'
    MACRO_KEY = 'macro'
    MACRO_EXP_KEY = 'macro_exp'
    MACRO_CODE_KEY = 'macro_code'
    VENDOR_KEY = 'vendor'
    NON_FUNCTION_KEY = 'non_function'
    DECL_KEY = 'decl'