from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from .prv_plot_utils import _plot_sleep_stagegram_on_ax

if TYPE_CHECKING:
    import pandas as pd


def _plot_sleep_stagegram_on_axis(
    ax,
    *,
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
    title: str = "Hypnogram",
    show_stats: bool = True,
):
    ok = _plot_sleep_stagegram_on_ax(
        ax,
        edf_base=edf_base,
        aux_df=aux_df,
        show_title=False,
        show_xlabel=True,
        show_stats_box=show_stats,
    )
    if not ok:
        return False
    ax.set_title(f"{edf_base} - {title}", fontsize=14, pad=12)
    return True


def _build_sleep_stagegram_figure(
    edf_base: str,
    aux_df: Optional["pd.DataFrame"],
):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ok = _plot_sleep_stagegram_on_axis(ax, edf_base=edf_base, aux_df=aux_df)
    if not ok:
        plt.close(fig)
        return None
    fig.tight_layout()
    return fig
