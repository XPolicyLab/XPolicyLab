from functools import lru_cache
from pathlib import Path
import json

@lru_cache(None)
def protocols():
    return json.loads((Path(__file__).resolve().parents[2]/'data/contracts/original_protocols.json').read_text())

def data_protocol(name):
    return protocols()[name]
