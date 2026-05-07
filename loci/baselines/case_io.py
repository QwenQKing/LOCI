from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class BaselineCase:
    method: str
    dataset: str
    model: str
    example_id: str
    category: str
    level: Optional[str]
    question: str
    ground_truth: str
    predicted: str = ''
    correct: bool = False
    trajectory: str = ''
    raw_outputs: List[str] = field(default_factory=list)
    n_calls: int = 0
    n_policy_output_tokens: int = 0
    n_critic_output_tokens: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    latency_s: float = 0.0
    confidence: Optional[float] = None
    mean_entropy: Optional[float] = None
    step_records: List[Dict[str, Any]] = field(default_factory=list)
    intermediate: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CaseWriter:

    def __init__(self, path: str, fsync_every: int=1) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = None
        self._n_written = 0
        self._fsync_every = max(1, int(fsync_every))

    def __enter__(self) -> 'CaseWriter':
        self._fh = open(self.path, 'a', encoding='utf-8')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            self._fh.close()
            self._fh = None

    def write(self, case: BaselineCase) -> None:
        if self._fh is None:
            raise RuntimeError("CaseWriter not opened (use 'with' block)")
        line = json.dumps(case.to_dict(), ensure_ascii=False)
        self._fh.write(line + '\n')
        self._n_written += 1
        if self._n_written % self._fsync_every == 0:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass

def read_cases(path: str) -> List[BaselineCase]:
    out: List[BaselineCase] = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                import sys
                print(f'[read_cases] WARN: skipping malformed line {ln} in {path}: {e}', file=sys.stderr)
                continue
            out.append(BaselineCase(**data))
    return out

def already_done_ids(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'example_id' in obj:
                    done.add(obj['example_id'])
            except json.JSONDecodeError:
                continue
    return done
