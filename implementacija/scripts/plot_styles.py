"""Dosljedne oznake CPU izvedbi na grafikonima."""

PARALLEL_BACKENDS = ('openmp', 'openmp_simd', 'bh_openmp')
COLORS = {
    'sequential': '#26364f', 'blocked': '#728097', 'simd': '#743eb0',
    'openmp': '#1656b0', 'openmp_simd': '#b32066',
    'bh_sequential': '#c66c19', 'bh_optimized': '#278239', 'bh_openmp': '#007f91',
}
MARKERS = {
    'sequential': 'o', 'blocked': 's', 'simd': 'o',
    'openmp': '^', 'openmp_simd': 'D',
    'bh_sequential': 'P', 'bh_optimized': 'o', 'bh_openmp': 'v',
}


def curve_style(backend):
    linestyle = '--' if backend in PARALLEL_BACKENDS else '-'
    if backend == 'blocked':
        linestyle = ':'
    return dict(color=COLORS[backend], marker=MARKERS[backend],
                linestyle=linestyle,
                markerfacecolor='white' if backend == 'blocked' else COLORS[backend],
                linewidth=1.8, markersize=5, zorder=4 if backend == 'blocked' else 3)
