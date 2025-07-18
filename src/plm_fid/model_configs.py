from transformers import (
    EsmModel,
    EsmTokenizer,
    AutoModelForMaskedLM,
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    RoFormerModel,
    RoFormerTokenizer,
)
import re


def space_protein_sequence(seq: str) -> str:
    """Adds spaces between amino acids and handles rare residues."""
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


def antiberta_preprocessor(seq: str) -> str:
    """Detects heavy and light chains and returns them separately."""
    if "|" in seq:
        heavy_chain, light_chain = seq.split("|")
        heavy_chain = space_protein_sequence(heavy_chain)
        light_chain = space_protein_sequence(light_chain) 
        return f"{heavy_chain} [SEP] {light_chain}"
    else:
        return space_protein_sequence(seq)



MODEL_FAMILY_CONFIGS: dict[str, dict] = {
    "esm2": {
        "model_class": EsmModel,
        "tokenizer_class": EsmTokenizer,
        "model_kwargs": {},
        "tokenizer_kwargs": {},
        "preprocessor": None,
        "tokenizer_on_model": False,
        "max_sequence_length": None,
    },
    "protbert": {
        "model_class": BertModel,
        "tokenizer_class": BertTokenizer,
        "model_kwargs": {},
        "tokenizer_kwargs": {"do_lower_case": False},
        "preprocessor": space_protein_sequence,
        "tokenizer_on_model": False,
        "max_sequence_length": None,
    },
    "prott5": {
        "model_class": T5EncoderModel,
        "tokenizer_class": T5Tokenizer,
        "model_kwargs": {},
        "tokenizer_kwargs": {},
        "preprocessor": space_protein_sequence,
        "tokenizer_on_model": False,
        "max_sequence_length": None,
    },
    "esmplusplus": {
        "model_class": AutoModelForMaskedLM,
        "tokenizer_class": None,
        "model_kwargs": {"trust_remote_code": True},
        "tokenizer_kwargs": {},
        "preprocessor": None,
        "tokenizer_on_model": True,
        "max_sequence_length": None,
    },
    "antiberta": {
        "model_class": RoFormerModel,
        "tokenizer_class": RoFormerTokenizer,
        "model_kwargs": {},
        "tokenizer_kwargs": {},
        "preprocessor": antiberta_preprocessor,
        "tokenizer_on_model": False,
        "max_sequence_length": 254,
    },
}

MODEL_CONFIGS: dict[str, dict] = {
    # ESM2 Family
    "facebook/esm2_t6_8M_UR50D": MODEL_FAMILY_CONFIGS["esm2"],
    "facebook/esm2_t12_35M_UR50D": MODEL_FAMILY_CONFIGS["esm2"],
    "facebook/esm2_t30_150M_UR50D": MODEL_FAMILY_CONFIGS["esm2"],
    "facebook/esm2_t33_650M_UR50D": MODEL_FAMILY_CONFIGS["esm2"],
    "facebook/esm2_t36_3B_UR50D": MODEL_FAMILY_CONFIGS["esm2"],
    "facebook/esm2_t48_15B_UR50D": MODEL_FAMILY_CONFIGS["esm2"],
    # ProtBERT Family
    "Rostlab/prot_bert": MODEL_FAMILY_CONFIGS["protbert"],
    "Rostlab/prot_bert_bfd": MODEL_FAMILY_CONFIGS["protbert"],
    # ProtT5 Family
    "Rostlab/prot_t5_xl_uniref50": MODEL_FAMILY_CONFIGS["prott5"],
    # ESM++ Family
    "Synthyra/ESMplusplus_small": MODEL_FAMILY_CONFIGS["esmplusplus"],
    "Synthyra/ESMplusplus_large": MODEL_FAMILY_CONFIGS["esmplusplus"],
    # AntiBERTa Family
    "alchemab/antiberta2-cssp": MODEL_FAMILY_CONFIGS["antiberta"],
}


def get_model_config(model_name: str) -> dict:
    return MODEL_CONFIGS[model_name]
