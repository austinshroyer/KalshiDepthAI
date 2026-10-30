#!/usr/bin/env python
import sys

from kalshi_quant.cli import app

if __name__ == "__main__":
    app(["collect", "all"] + sys.argv[1:])
