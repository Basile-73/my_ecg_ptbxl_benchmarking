import argparse
import re
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
import matplotlib.pyplot as plt

from maps import COLOR_MAP, NAME_MAP, EXCLUDE_MODELS, plot_font_sizes

_SCRIPT_DIR = Path(__file__).resolve().parent        # visualisation/
_DATA_DIR   = _SCRIPT_DIR.parent / 'outputs' / 'training_data'

# Register CMU Serif font
_FONT_DIR = _SCRIPT_DIR.parent.parent / "fonts" / "cm-unicode-0.7.0"
for _ttf in _FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(_ttf))
plt.rcParams["font.family"] = "CMU Serif"

font_sizes = {k: v * 1.872 for k, v in plot_font_sizes.items()}

_BOLD_TITLE_FP = FontProperties(fname=str(_FONT_DIR / 'cmunbx.ttf'), size=font_sizes['title'])
_REG_TITLE_FP  = FontProperties(fname=str(_FONT_DIR / 'cmunrm.ttf'), size=font_sizes['title'])

DATASETS = [
    ('European ST-T', 'eu'),
    ('Synthetic',     'syn'),
    ('PTB-XL',        'ptb'),
]

METRICS = [
    ('test',  'Test SNR (dB)'),
    ('train', 'Training Loss'),
]


def get_model_name(col):
    """Parse model name from column like 'model_name - test/SNR'."""
    return col.split(' - ')[0]


def _parse_length_col(col):
    """Parse 'base_model_NNNN - metric' → (base_model_str, split_length_int or None)."""
    model_part = col.split(' - ')[0]
    m = re.match(r'^(.+)_(\d{3,5})$', model_part)
    if m:
        return m.group(1), int(m.group(2))
    return model_part, None


def _lookup_color(model_name):
    """Look up color from COLOR_MAP, stripping trailing _N suffixes as fallback."""
    name = model_name
    while name:
        if name in COLOR_MAP:
            return COLOR_MAP[name]
        m = re.match(r'^(.+)_\d+$', name)
        if m:
            name = m.group(1)
        else:
            break
    return '#C9C9C9'


def _lookup_name(model_name):
    """Look up display name from NAME_MAP, stripping trailing _N suffixes as fallback."""
    name = model_name
    while name:
        if name in NAME_MAP:
            return NAME_MAP[name]
        m = re.match(r'^(.+)_\d+$', name)
        if m:
            name = m.group(1)
        else:
            break
    return model_name


def _lighten_color(hex_color, factor):
    """Blend hex_color towards white. factor=1.0 → original colour, factor=0.0 → white."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * (1 - factor))
    g = int(g + (255 - g) * (1 - factor))
    b = int(b + (255 - b) * (1 - factor))
    return f'#{r:02x}{g:02x}{b:02x}'


def _plot_cols(ax, df, cols, y_label, show_xlabel=True, show_ylabel=True, show_legend=True, legend_ncols=1, legend_slice=None):
    cols = [c for c in cols if get_model_name(c) not in EXCLUDE_MODELS]
    for col in cols:
        model = get_model_name(col)
        ax.plot(
            df['Step'],
            df[col],
            label=NAME_MAP.get(model, model),
            color=COLOR_MAP.get(model, '#C9C9C9'),
            linewidth=1.5,
        )
    if show_xlabel:
        ax.set_xlabel('Step', fontsize=font_sizes['axis_labels'])
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=font_sizes['axis_labels'])
    ax.tick_params(axis='both', labelsize=font_sizes['ticks'])
    ax.grid(True, alpha=0.3)
    if show_legend:
        if legend_slice is not None:
            handles, labels = ax.get_legend_handles_labels()
            start, stop = legend_slice
            ax.legend(handles[start:stop], labels[start:stop],
                      fontsize=font_sizes['legend'], ncol=legend_ncols)
        else:
            ax.legend(fontsize=font_sizes['legend'], ncol=legend_ncols)


def _plot_length_cols(ax, df, cols, y_label):
    """Plot columns with split-length-aware coloring (shorter split → lighter shade)."""
    cols = [c for c in cols if _parse_length_col(c)[0] not in EXCLUDE_MODELS]
    parsed = [_parse_length_col(c) for c in cols]
    lengths = sorted({length for _, length in parsed if length is not None})
    n = len(lengths)

    def _factor(length):
        if n <= 1 or length is None:
            return 1.0
        return 0.3 + 0.7 * lengths.index(length) / (n - 1)  # 1800→0.3 … 14400→1.0

    for col, (model, length) in zip(cols, parsed):
        color = _lighten_color(_lookup_color(model), _factor(length))
        display_name = _lookup_name(model)
        label = f'{display_name} ({length})' if length is not None else display_name
        ax.plot(df['Step'], df[col], label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Step', fontsize=font_sizes['axis_labels'])
    ax.set_ylabel(y_label, fontsize=font_sizes['axis_labels'])
    ax.tick_params(axis='both', labelsize=font_sizes['ticks'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=font_sizes['legend'])


def _save(fig, path, no_save):
    if not no_save:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {path}")


def plot_stages(args):
    datasets = DATASETS
    if args.dataset:
        datasets = [(l, p) for l, p in DATASETS if p == args.dataset]

    output_dir = Path(args.output) if args.output else Path('../outputs/plots')

    # Rows: top = train, bottom = test
    _ROWS = [
        ('train', 'Training Loss'),
        ('test',  'Test SNR (dB)'),
    ]

    for dataset_label, prefix in datasets:
        dfs = {
            suffix: pd.read_csv(_DATA_DIR / f'{prefix}_{suffix}.csv')
            for suffix, _ in _ROWS
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(dataset_label, fontsize=font_sizes['title'], fontweight='bold')

        for row, (suffix, y_label) in enumerate(_ROWS):
            df = dfs[suffix]
            metric_cols = [
                c for c in df.columns
                if c != 'Step' and '__MIN' not in c and '__MAX' not in c
            ]
            non_drnet_cols = [c for c in metric_cols if 'drnet' not in get_model_name(c)]
            drnet_cols     = [c for c in metric_cols if 'drnet' in get_model_name(c)]

            for col_idx, (cols, title) in enumerate([
                (non_drnet_cols, 'Stage 1 models'),
                (drnet_cols,     'Stage 2 models (DRNET)'),
            ]):
                ax = axes[row][col_idx]
                _plot_cols(
                    ax, df, cols, y_label,
                    show_xlabel=(row == 1),
                    show_ylabel=(col_idx == 0),
                    show_legend=(row == 0),
                )
                if row == 0:
                    ax.set_title(title, fontsize=font_sizes['title'], fontweight='bold')
                    if prefix == 'syn':
                        bottom = -0.01 if col_idx == 0 else -0.001
                        ax.set_ylim(bottom=bottom, top=0.4 if col_idx == 0 else 0.05)

        plt.tight_layout()
        _save(fig, output_dir / f'training_{prefix}.png', args.no_save)
        plt.show()

    if not args.dataset:
        _plot_stages_combined(DATASETS, _ROWS, output_dir, args.no_save)


def _plot_stages_combined(datasets, rows, output_dir, no_save):
    letters = ['A', 'B', 'C']
    n_datasets = len(datasets)
    plot_height = 5 * 0.75 * 0.9  # 25% then a further 10% reduction from per-dataset view
    fig = plt.figure(
        figsize=(14, plot_height * len(rows) * n_datasets),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(n_datasets, 1)

    for i, ((dataset_label, prefix), letter) in enumerate(zip(datasets, letters)):
        subfig = subfigs[i]
        # Reserve top space via a placeholder suptitle, then overlay a mixed
        # bold-prefix / regular-suffix title using HPacker.
        subfig.suptitle(' ', fontproperties=_BOLD_TITLE_FP)
        bold_part = TextArea(
            f'{letter}.',
            textprops=dict(fontproperties=_BOLD_TITLE_FP),
        )
        regular_part = TextArea(
            f' {dataset_label}',
            textprops=dict(fontproperties=_REG_TITLE_FP),
        )
        packed = HPacker(children=[bold_part, regular_part],
                         pad=0, sep=0, align='baseline')
        subfig.add_artist(AnchoredOffsetbox(
            loc='upper left',
            child=packed,
            frameon=False,
            pad=0,
            borderpad=0,
            bbox_to_anchor=(0.02, 0.995),
            bbox_transform=subfig.transSubfigure,
        ))

        dfs = {
            suffix: pd.read_csv(_DATA_DIR / f'{prefix}_{suffix}.csv')
            for suffix, _ in rows
        }

        axes = subfig.subplots(len(rows), 2)
        is_last_dataset = (i == n_datasets - 1)

        for row, (suffix, y_label) in enumerate(rows):
            df = dfs[suffix]
            metric_cols = [
                c for c in df.columns
                if c != 'Step' and '__MIN' not in c and '__MAX' not in c
            ]
            non_drnet_cols = [c for c in metric_cols if 'drnet' not in get_model_name(c)]
            drnet_cols     = [c for c in metric_cols if 'drnet' in get_model_name(c)]

            for col_idx, (cols, title) in enumerate([
                (non_drnet_cols, 'Stage 1 models'),
                (drnet_cols,     'Stage 2 models (DRNET)'),
            ]):
                ax = axes[row][col_idx]
                if i == 0 and col_idx == 0:
                    show_legend = (row in (0, 1))
                    legend_slice = (0, 3) if row == 0 else (3, 6)
                elif i == 0 and col_idx == 1 and row == 0:
                    show_legend = True
                    legend_slice = None
                else:
                    show_legend = False
                    legend_slice = None
                _plot_cols(
                    ax, df, cols, y_label,
                    show_xlabel=(row == len(rows) - 1 and is_last_dataset),
                    show_ylabel=(col_idx == 0),
                    show_legend=show_legend,
                    legend_slice=legend_slice,
                )
                if row == 0 and i == 0:
                    ax.set_title(title, fontsize=font_sizes['title'], fontweight='bold')
                if row == 0 and prefix == 'syn':
                    bottom = -0.01 if col_idx == 0 else -0.001
                    ax.set_ylim(bottom=bottom, top=0.4 if col_idx == 0 else 0.05)

    _save(fig, output_dir / 'training_combined.png', no_save)
    plt.show()


def plot_compression(args):
    output_dir = Path(args.output) if args.output else Path('../outputs/plots')
    df = pd.read_csv(_DATA_DIR / 'compression.csv')

    metric_cols = [
        c for c in df.columns
        if c != 'Step' and '__MIN' not in c and '__MAX' not in c
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_cols(ax, df, metric_cols, 'Test RMSE')
    ax.set_title('Compression vs No Compression', fontsize=font_sizes['title'], fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir / 'compression_rmse.png', args.no_save)
    plt.show()


def plot_2file(args):
    output_dir = Path(args.output) if args.output else Path('../outputs/plots')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=args.sharey)

    for ax, csv_path, label in zip(axes, args.files, args.labels):
        df = pd.read_csv(csv_path)
        metric_cols = [
            c for c in df.columns
            if c != 'Step' and '__MIN' not in c and '__MAX' not in c
            and 'train/loss' in c
        ]
        _plot_length_cols(ax, df, metric_cols, args.ylabel)
        ax.set_title(label, fontsize=font_sizes['title'], fontweight='bold')

    plt.tight_layout()
    stem1 = Path(args.files[0]).stem
    stem2 = Path(args.files[1]).stem
    _save(fig, output_dir / f'{stem1}_vs_{stem2}.png', args.no_save)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot ECG denoising training curves.')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # --- stages ---
    p = subparsers.add_parser('stages', help='Stage 1 vs Stage 2 models, all datasets')
    p.add_argument('--dataset', choices=['eu', 'syn', 'ptb'], help='Restrict to one dataset')
    p.add_argument('--no-save', action='store_true', dest='no_save')
    p.add_argument('--output',  help='Output directory (default: ../outputs/plots)')

    # --- compression ---
    p = subparsers.add_parser('compression', help='Compression vs no-compression comparison')
    p.add_argument('--no-save', action='store_true', dest='no_save')
    p.add_argument('--output',  help='Output directory (default: ../outputs/plots)')

    # --- 2file ---
    p = subparsers.add_parser('2file', help='Side-by-side comparison of two CSV files')
    p.add_argument('--files',  nargs=2, required=True, metavar='CSV',   help='Two CSV files to compare')
    p.add_argument('--labels', nargs=2, required=True, metavar='LABEL', help='Subplot title for each file')
    p.add_argument('--ylabel', default='', metavar='LABEL',             help='Y-axis label (default: none)')
    p.add_argument('--sharey', action='store_true',                      help='Share y-axis between subplots')
    p.add_argument('--no-save', action='store_true', dest='no_save')
    p.add_argument('--output',  help='Output directory (default: ../outputs/plots)')

    args = parser.parse_args()

    if args.mode == 'stages':
        plot_stages(args)
    elif args.mode == 'compression':
        plot_compression(args)
    elif args.mode == '2file':
        plot_2file(args)


if __name__ == '__main__':
    main()
