from .base import extract_answer
from .case_io import BaselineCase, CaseWriter, read_cases, already_done_ids
from .fair_config import FairConfig, DEFAULT_FAIR_CONFIG
from .runner_base import BaselineRunner

def _build_registry():
    from .zero_shot import ZeroShotRunner
    from .cot import CoTRunner
    from .bon import BoNRunner
    from .mur_original import MURRunner
    from .prm_bon import PRMBoNRunner
    from .snell_optimal_tts import SnellOptimalTTSRunner
    from .math_shepherd import MathShepherdRunner
    from .phi_decoding import PhiDecodingRunner
    from .caa import CAARunner
    from .repe import RepERunner
    from .loci_runner import (LOCIRunner, LOCIBaseRunner, LOCIProbeOnlyRunner,
                              LOCIRandomHeadsRunner, LOCIFixedWindowRunner)
    return {
        ZeroShotRunner.name:        ZeroShotRunner,
        CoTRunner.name:             CoTRunner,
        BoNRunner.name:             BoNRunner,
        MURRunner.name:             MURRunner,
        PRMBoNRunner.name:          PRMBoNRunner,
        SnellOptimalTTSRunner.name: SnellOptimalTTSRunner,
        MathShepherdRunner.name:    MathShepherdRunner,
        PhiDecodingRunner.name:     PhiDecodingRunner,
        CAARunner.name:             CAARunner,
        RepERunner.name:            RepERunner,
        LOCIRunner.name:             LOCIRunner,
        LOCIBaseRunner.name:         LOCIBaseRunner,
        LOCIProbeOnlyRunner.name:    LOCIProbeOnlyRunner,
        LOCIRandomHeadsRunner.name:  LOCIRandomHeadsRunner,
        LOCIFixedWindowRunner.name:  LOCIFixedWindowRunner,
    }

def get_registry():
    return _build_registry()

__all__ = ['extract_answer', 'BaselineCase', 'CaseWriter', 'read_cases',
           'already_done_ids', 'FairConfig', 'DEFAULT_FAIR_CONFIG',
           'BaselineRunner', 'get_registry']
