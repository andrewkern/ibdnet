"""Utility helpers built around :mod:`cyvcf2` for lightweight VCF access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

try:
    from cyvcf2 import Variant, VCF
except ImportError:  # pragma: no cover - cyvcf2 is optional at test time
    Variant = object  # type: ignore
    VCF = object  # type: ignore


@dataclass(slots=True)
class VariantWindow:
    """Container for a window of VCF variants."""

    chrom: str
    positions: List[int]
    genotypes: List[List[int]]  # [variant][sample*2]


def open_vcf(path: str, samples: Optional[Sequence[str]] = None) -> VCF:
    """Open a VCF/BCF file via :class:`cyvcf2.VCF`."""

    try:
        vcf = VCF(path, samples=samples)
    except Exception as exc:  # pragma: no cover - cyvcf2 raises custom errors
        raise RuntimeError(f"Failed to open VCF '{path}': {exc}") from exc
    return vcf


def iter_windows(vcf: VCF, window_size: int) -> Generator[VariantWindow, None, None]:
    """Yield windows of variants of approximately `window_size` length."""

    chrom: Optional[str] = None
    positions: List[int] = []
    gts: List[List[int]] = []

    for variant in vcf:  # type: ignore[attr-defined]
        if chrom is None:
            chrom = variant.CHROM  # type: ignore[attr-defined]
        if variant.CHROM != chrom or len(positions) >= window_size:  # type: ignore[attr-defined]
            yield VariantWindow(chrom=chrom, positions=positions, genotypes=gts)
            chrom = variant.CHROM  # type: ignore[attr-defined]
            positions = []
            gts = []
        positions.append(int(variant.POS))  # type: ignore[attr-defined]
        gts.append(variant.genotypes)  # type: ignore[attr-defined]

    if chrom is not None and positions:
        yield VariantWindow(chrom=chrom, positions=positions, genotypes=gts)


__all__ = ["open_vcf", "iter_windows", "VariantWindow"]
