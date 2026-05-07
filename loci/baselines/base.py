import re
from typing import Optional
_THE_ANSWER_IS_RE = re.compile('(?:the\\s+)?answer\\s+is\\s*[:\\-]?\\s*([^\\n]+?)(?:[.!?](?:\\s|$)|\\n|$)', re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile('final\\s+answer\\s*[:\\-]?\\s*([^\\n]+?)(?:[.!?](?:\\s|$)|\\n|$)', re.IGNORECASE)
_MC_LETTER_RE = re.compile('(?:^|[\\s(\\[,.:;])([A-E])(?=$|[\\s)\\].,:;])')
_NUMERIC_RE = re.compile('-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?(?:/\\d+)?')

def _extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind('\\boxed{')
    if idx < 0:
        return None
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return text[start:].strip()

def extract_answer(text: str, is_multiple_choice: bool=False) -> str:
    if not text:
        return ''
    text = text.strip()
    boxed = _extract_boxed(text)
    if boxed:
        if is_multiple_choice:
            letter = _MC_LETTER_RE.findall(' ' + boxed + ' ')
            if letter:
                return letter[-1]
        return boxed
    matches = list(_THE_ANSWER_IS_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip().rstrip('.').strip()
        if is_multiple_choice:
            letter = _MC_LETTER_RE.findall(' ' + candidate + ' ')
            if letter:
                return letter[-1]
        inner_boxed = _extract_boxed(candidate)
        if inner_boxed:
            return inner_boxed
        return candidate
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip().rstrip('.').strip()
        if is_multiple_choice:
            letter = _MC_LETTER_RE.findall(' ' + candidate + ' ')
            if letter:
                return letter[-1]
        inner_boxed = _extract_boxed(candidate)
        if inner_boxed:
            return inner_boxed
        return candidate
    if is_multiple_choice:
        letters = _MC_LETTER_RE.findall(' ' + text + ' ')
        if letters:
            return letters[-1]
    tail = text[-300:]
    nums = _NUMERIC_RE.findall(tail)
    if nums:
        return nums[-1]
    for line in reversed(text.rstrip().split('\n')):
        line = line.strip()
        if line:
            return line[:200]
    return text[:200]
