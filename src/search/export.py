"""Candidate figures for human review: one multi-page PDF per cadence and method."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.search.pipeline import CadenceWindows  # noqa: E402
from src.utils.visualization import plot_candidate  # noqa: E402


def export_candidates(ranked: pd.DataFrame, amaps: List[Optional[np.ndarray]], *, model,
                      windows: CadenceWindows, method: str, top_k: Optional[int], plot_format: str,
                      cad_dir: Path, median: float, mad_sigma: float, cad_idx: int,
                      target: str, freq_step_hz: float, fch1_mhz: float,
                      nchans_total: int, obs_date: str, device: str) -> int:
    """Plot the first ``top_k`` rows of ``ranked`` (the :func:`rank_for_export`
    order; every row when ``top_k`` is None) to ``{method}_candidates.pdf``
    and/or per-candidate PNGs.

    Returns the number of candidates plotted. Backbones without an anomaly map
    get original | reconstruction | error panels instead.
    """
    top = (ranked if top_k is None else ranked.head(top_k)).reset_index(drop=True)
    want_pdf = plot_format in ("pdf", "both")
    want_png = plot_format in ("png", "both")
    pdf = PdfPages(cad_dir / f"{method}_candidates.pdf") if want_pdf and len(top) > 0 else None
    for rank, row in top.iterrows():
        fs = int(row["f_start_peak"])
        score = float(row["peak_score"])
        window = windows.window(fs)
        sigma = (score - median) / mad_sigma if mad_sigma > 0 else 0
        amap = amaps[rank]
        recon = None
        if amap is None:
            x = torch.from_numpy(np.array([window])).float().unsqueeze(1).to(device)
            with torch.no_grad():
                recon = model(x).squeeze(1).cpu().numpy()[0]
        fig = plot_candidate(
            original=window, reconstruction=recon, score=score, sigma=sigma, method=method,
            cad_idx=cad_idx, target=target, f_start=fs, df=freq_step_hz, fch1_mhz=fch1_mhz,
            nchans_total=nchans_total, obs_date=obs_date, anomaly_map=amap,
        )
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        if want_png:
            fig.savefig(cad_dir / f"{method}_rank{rank:02d}_f{fs}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    if pdf is not None:
        pdf.close()
    return len(top)


def subset_pdf(src: Path, dst: Path, pages: List[int], expected_pages: int) -> None:
    """Write the ``pages`` (0-based, in this order) of ``src`` to ``dst``.

    ``expected_pages`` guards against editing a PDF that does not hold the
    candidates the page indices refer to (e.g. one exported with ``top_k``).
    Needs ``pypdf``.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(src))
    if len(reader.pages) != expected_pages:
        raise ValueError(f"{src}: {len(reader.pages)} pages, expected {expected_pages}")
    writer = PdfWriter()
    for k in pages:
        writer.add_page(reader.pages[k])
    with open(dst, "wb") as f:
        writer.write(f)
