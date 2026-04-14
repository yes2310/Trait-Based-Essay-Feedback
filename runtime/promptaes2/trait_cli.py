from __future__ import annotations

import warnings

from promptaes2.pretrain_cli import (
    _validate_parsed_args,
    build_parser,
)
from promptaes2.pretrain_cli import main as _pretrain_main


def main(argv: list[str] | None = None):
    warnings.warn(
        "promptaes2.trait_cli is deprecated and will be removed in a future release. "
        "Use `python -m promptaes2.pretrain_cli ...` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _pretrain_main(argv)


if __name__ == "__main__":
    main()
