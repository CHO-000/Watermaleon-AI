from __future__ import annotations

import sys
from pathlib import Path

local_packages = Path(__file__).with_name('.vendor')
if local_packages.exists():
    sys.path.insert(0, str(local_packages))

import uvicorn

if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=8000)
