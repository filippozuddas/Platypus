"""Spectrogram plots, reconstruction error maps, and candidate stamps."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def add_obs_dividers(ax, n_rows: int, n_obs: int = 6, color: str = "white",
                      lw: float = 0.8, alpha: float = 0.7):
    """Draw horizontal lines marking the boundaries between the ``n_obs``
    stacked observations of a cadence in a (time, freq) waterfall plot.

    ``n_rows`` (e.g. ``frame.tchans``) is the concatenated ABACAD cadence —
    ``n_rows // n_obs`` time bins per observation (see ``configs/data/gbt_fine.yaml``:
    96 = 6 obs x 16 bins). No-op if ``n_rows`` doesn't divide evenly by
    ``n_obs`` (e.g. a single-observation plot).
    """
    if n_rows % n_obs != 0:
        return
    bins_per_obs = n_rows // n_obs
    for row in range(bins_per_obs, n_rows, bins_per_obs):
        ax.axhline(row - 0.5, color=color, lw=lw, alpha=alpha, ls="-")


def add_on_off_labels(ax, n_rows: int, n_obs: int = 6, on_rows=(0, 2, 4),
                       off_rows=(1, 3, 5), on_color: str = "white",
                       off_color: str = "white", fontsize: int = 6):
    """Annotate each of the ``n_obs`` stacked-observation bands with "ON"/"OFF",
    inside ``ax`` at the top-left corner of each band.

    ``on_rows``/``off_rows`` are observation indices within the ABACAD cadence
    (same convention as ``src.search.candidates.on_off_contrast``:
    ``(0, 2, 4)``=ON, ``(1, 3, 5)``=OFF). ``n_rows`` is the axis's total row
    count — pass ``frame.tchans`` for a full-resolution waterfall panel (label
    at the top of each ``n_rows // n_obs``-bin band) or ``nh`` for UDMA's
    native (nh, nw) anomaly-map grid, where each row already is one
    observation (``n_obs`` should equal ``nh`` in that case). Small font and a
    corner placement keep the label from obscuring candidate signal in the
    band. No-op if ``n_rows`` doesn't divide evenly by ``n_obs``.
    """
    if n_rows % n_obs != 0:
        return
    bins_per_obs = n_rows // n_obs
    for obs_idx in range(n_obs):
        if obs_idx in on_rows:
            label, color = "ON", on_color
        elif obs_idx in off_rows:
            label, color = "OFF", off_color
        else:
            continue
        y_top = obs_idx * bins_per_obs - 0.5
        ax.text(0.01, y_top + bins_per_obs * 0.08, label,
                transform=ax.get_yaxis_transform(),
                va="top", ha="left", fontsize=fontsize, color=color,
                fontweight="bold", clip_on=True)


def upsample_map_bilinear(amap: np.ndarray, target_shape) -> np.ndarray:
    """Bilinearly upsample a native (nh,nw) patch-grid map to ``target_shape`` pixels.

    Visualization only — never feed the result back into scoring. The model's
    top-k/mean pooling must always run on the native grid; interpolation here
    just makes the low-resolution grid legible against the full-res spectrogram.
    """
    zh = target_shape[0] / amap.shape[0]
    zw = target_shape[1] / amap.shape[1]
    return zoom(amap, (zh, zw), order=1)


def overlay_anomaly_map(ax, base_img: np.ndarray, amap: np.ndarray, cmap: str = "inferno",
                         alpha: float = 0.45, title: str = None, origin: str = "upper",
                         extent=None):
    """Plot ``base_img`` in grayscale with the bilinearly-upsampled ``amap`` overlaid.

    ``amap`` is the native (nh,nw) patch-grid anomaly map (e.g. UDMA's fused
    map_cob); it is upsampled to ``base_img.shape`` purely for this overlay.
    ``origin`` must match the convention used for ``base_img`` elsewhere in the
    same figure, or the overlay will be flipped relative to its neighbors.
    ``extent`` (left, right, bottom, top), forwarded to both ``imshow`` calls,
    calibrates the x-axis to real frequency (MHz) instead of channel index —
    see ``plot_candidate``.
    """
    up = upsample_map_bilinear(amap, base_img.shape)
    vmin, vmax = np.percentile(base_img, [1, 99])
    ax.imshow(base_img, aspect="auto", origin=origin, cmap="gray", vmin=vmin, vmax=vmax,
              extent=extent)
    im = ax.imshow(up, aspect="auto", origin=origin, cmap=cmap, alpha=alpha, extent=extent)
    if title:
        ax.set_title(title)
    return im


def plot_candidate(original, reconstruction, score, sigma, method, cad_idx,
                    target, f_start, df, fch1_mhz=0.0, nchans_total=0, obs_date="",
                    anomaly_map=None, n_obs=6, show_overlay=True):
    """Build (but don't save) the original|reconstruction|error figure for one
    candidate; caller decides whether to write it to PNG, a per-cadence PDF,
    or both.

    ``reconstruction``/error panels for pixel-decoder backbones; if
    ``reconstruction`` is None (UDMA, no pixel decoder), ``anomaly_map`` — its
    native (nh,nw) disagreement grid — is shown instead (see
    ``UDMA.anomaly_map`` / ``scripts/debug/udma_anomaly_maps.py``). The
    bilinear-overlay panel (native grid resampled onto the full-res waterfall)
    is optional — set ``show_overlay=False`` to drop it and keep just
    original + native anomaly_map (2 panels instead of 3); no effect on the
    ``reconstruction`` path, which always shows all 3 panels.

    Horizontal lines mark the ``n_obs`` observation boundaries (ABACAD
    cadence stacking) on every panel drawn at ``original``'s native time
    resolution — the two full-resolution panels (``original``,
    ``reconstruction``/residual) plus the bilinear anomaly-map overlay.

    Args:
        original: (time, freq) waterfall array.
        reconstruction: (time, freq) reconstructed array (None for anomaly_map backbones).
        score: scalar anomaly score for the suptitle.
        sigma: score's significance in MAD-sigma (None to omit from suptitle).
        method: scoring method name (e.g. "topk", "recon").
        cad_idx: cadence index for the suptitle.
        target: target source name.
        f_start: channel index into the source file's frequency axis; used
            (not shown directly) to compute "Center freq" in the suptitle.
        df: file header's signed ``foff`` (Hz/channel) — negative for the
            typical filterbank convention where channel 0 is the top of the
            band and frequency decreases with increasing channel index, so
            "Center freq" can be lower than the band's start frequency.
        fch1_mhz: file header's absolute sky frequency at channel 0 (MHz);
            0 = channel-0 freq unknown, "Center freq" falls back to an
            in-file offset.
        nchans_total: total channel count of the source file (e.g.
            ``data_cfg["raw"]["nchans"]``, fixed per product), used with
            ``fch1_mhz``/``df`` to find the band's low-frequency edge for
            the suptitle's "Obs start freq" (the lower of ``fch1_mhz`` and
            ``fch1_mhz + nchans_total*df``, since ``fch1_mhz`` alone is the
            top of the band when ``df`` is negative). 0 = suptitle falls
            back to showing ``fch1_mhz`` directly.
        obs_date: observation date, ``YYYYMMDD`` (as returned by
            ``read_cadence_meta``); reformatted to ``YYYY-MM-DD`` in the
            suptitle. "" = omit.
        anomaly_map: (nh, nw) anomaly map from UDMA (shown if reconstruction is None).
        n_obs: number of observations in cadence (for divider lines).
        show_overlay: if True, show bilinear-resampled anomaly map overlay.
    """
    n_rows = original.shape[0]
    fchans = original.shape[-1]
    vmin, vmax = np.percentile(original, [1, 99])

    # Real-frequency (MHz) x-axis for every full-resolution panel, so the
    # "Center freq of snippet" printed in the suptitle is visibly the
    # midpoint of the plotted band rather than a disconnected number next to
    # a bare channel-index axis (channel-axis alone was the source of
    # Vishal's "doesn't look like the center" confusion, 2026-07-30).
    # extent=(left, right, bottom, top); left/right are the snippet's low-
    # and high-channel-index edges converted to MHz — reversed automatically
    # when df<0, so the axis direction always matches the sky.
    freq_axis = fch1_mhz != 0.0 and df != 0.0
    if freq_axis:
        f_edge0_mhz = fch1_mhz + f_start * df / 1e6
        f_edge1_mhz = fch1_mhz + (f_start + fchans) * df / 1e6
        extent = (f_edge0_mhz, f_edge1_mhz, n_rows - 0.5, -0.5)
        freq_xlabel = "Frequency (MHz)"
    else:
        extent = None
        freq_xlabel = "Freq channel"

    show_overlay = show_overlay or reconstruction is not None
    n_panels = 3 if show_overlay else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    im0 = axes[0].imshow(original, aspect="auto", origin="upper",
                          vmin=vmin, vmax=vmax, cmap="viridis", extent=extent)
    axes[0].set_title("Original")
    axes[0].set_ylabel("Time bin")
    axes[0].set_xlabel(freq_xlabel)
    add_obs_dividers(axes[0], n_rows, n_obs)
    add_on_off_labels(axes[0], n_rows, n_obs)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    if reconstruction is not None:
        im1 = axes[1].imshow(reconstruction, aspect="auto", origin="upper",
                              vmin=vmin, vmax=vmax, cmap="viridis", extent=extent)
        axes[1].set_title("Reconstruction")
        axes[1].set_xlabel(freq_xlabel)
        add_obs_dividers(axes[1], n_rows, n_obs)
        add_on_off_labels(axes[1], n_rows, n_obs)
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        error = np.abs(original - reconstruction)
        im2 = axes[2].imshow(error, aspect="auto", origin="upper", cmap="hot", extent=extent)
        axes[2].set_title("Residual |orig - recon|")
        axes[2].set_xlabel(freq_xlabel)
        add_obs_dividers(axes[2], n_rows, n_obs)
        add_on_off_labels(axes[2], n_rows, n_obs)
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    else:
        im1 = axes[1].imshow(anomaly_map, aspect="auto", origin="upper", cmap="viridis")
        axes[1].set_title("anomaly_map (UDMA, native (nh,nw) grid)")
        axes[1].set_xlabel("Freq patch col")
        axes[1].set_ylabel("Time patch row")
        add_on_off_labels(axes[1], anomaly_map.shape[0], anomaly_map.shape[0])
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        if show_overlay:
            overlay_anomaly_map(axes[2], original, anomaly_map,
                                 title="anomaly_map (bilinear overlay)", extent=extent)
            axes[2].set_xlabel(freq_xlabel)
            add_obs_dividers(axes[2], n_rows, n_obs)
            add_on_off_labels(axes[2], n_rows, n_obs)

    f_center_mhz = fch1_mhz + (f_start + fchans / 2) * df / 1e6
    if nchans_total:
        f_other_edge_mhz = fch1_mhz + nchans_total * df / 1e6
        f_obs_start_mhz = min(fch1_mhz, f_other_edge_mhz)
    else:
        f_obs_start_mhz = fch1_mhz
    date_label = f"{obs_date[:4]}-{obs_date[4:6]}-{obs_date[6:8]}" if len(obs_date) == 8 else obs_date
    score_line = f"{method} score={score:.4f}"
    if sigma is not None:
        score_line += f" ({sigma:.1f}s)"
    fig.suptitle(
        f"Candidate: cad={cad_idx} ({target})  date={date_label}\n"
        f"Obs start freq={f_obs_start_mhz:.6f} MHz  Center freq of snippet={f_center_mhz:.6f} MHz\n"
        f"{score_line}",
        fontsize=11,
    )
    plt.tight_layout()
    return fig
