from functools import lru_cache
import json,os,sys
from pathlib import Path,PosixPath

@lru_cache(None)
def aliases():
    return json.loads((Path(__file__).resolve().parents[2]/'data/contracts/path_aliases.json').read_text())

def relocated_path(args):
    original=str(Path(*args))
    root=os.environ.get('CLOUDROBO_WM_DATA_ROOT')
    if root:
        for old,new in sorted(aliases().items(),key=lambda x:-len(x[0])):
            if original==old or original.startswith(old+'/'):
                return str(Path(root)/new)+original[len(old):]
    return original

class DataPath(PosixPath):
    def __new__(cls,*args,**kwargs):
        return super().__new__(cls,relocated_path(args),**kwargs)

    # Path initialization moved from __new__ to __init__ in Python 3.12.
    # Apply the same alias there so either interpreter preserves path semantics.
    if sys.version_info >= (3,12):
        def __init__(self,*args,**kwargs):
            super().__init__(relocated_path(args),**kwargs)
