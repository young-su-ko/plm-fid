from enum import Enum


class PLM(str, Enum):
    ESM2_8M = "facebook/esm2_t6_8M_UR50D"
    ESM2_35M = "facebook/esm2_t12_35M_UR50D"
    ESM2_150M = "facebook/esm2_t30_150M_UR50D"
    ESM2_650M = "facebook/esm2_t33_650M_UR50D"
    ESM2_3B = "facebook/esm2_t36_3B_UR50D"
    ESM2_15B = "facebook/esm2_t48_15B_UR50D"

    PROTBERT = "Rostlab/prot_bert"
    PROTBERT_BFD = "Rostlab/prot_bert_bfd"
    PROTT5 = "Rostlab/prot_t5_xl_uniref50"

    ESMPLUSPLUS_SMALL = "Synthyra/ESMplusplus_small"
    ESMPLUSPLUS_LARGE = "Synthyra/ESMplusplus_large"

    ANTIBERTA2_CSSP = "alchemab/antiberta2-cssp"

    def __str__(self):
        return self.value
