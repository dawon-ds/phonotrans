"""Hangul pronunciation to romanization used in the initial pipeline experiment."""

from g2pk import G2p
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

_g2p = G2p()
_transliter = Transliter(academic)


def hangul_to_romanization(text: str) -> str:
    """Apply Korean pronunciation rules and convert the result to romanization."""
    pronounced = _g2p(text)
    return _transliter.translit(pronounced).replace("-", "")


if __name__ == "__main__":
    sample = input("Enter Japanese pronunciation written in Hangul: ").strip()
    print(hangul_to_romanization(sample))
