from __future__ import annotations
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Dict, List, Optional
STEP_PROMPT_TEMPLATE = "Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer enclosed in \\boxed{{}}. Start your solution with 'Step{step_idx}:'\n"
ONESHOT_PROMPT_TEMPLATE = "Q: {question}\nSolve step by step. Always end your solution with the phrase 'the answer is' followed by your final answer enclosed in \\boxed{{}}."
SYSTEM_PROMPTS = {'math': '', 'science': 'You are a helpful assistant. Here is a question and four candidate answers. You need to reason step by step and choose the most likely answer from the four candidate answers. Answer "A", "B", "C", or "D".', 'logic': ''}

def system_prompt_for(category: str) -> str:
    return SYSTEM_PROMPTS.get(category, '')

@dataclass(frozen=True)
class FairConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 2048
    max_steps: int = 20
    stop_sequences: List[str] = field(default_factory=lambda: ['Step'])
    n_candidates: int = 4
    seed: int = 42
    answer_marker: str = 'the answer is'
    boxed_marker: str = '\\boxed'

    def with_overrides(self, **kwargs: Any) -> 'FairConfig':
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
DEFAULT_FAIR_CONFIG = FairConfig()
