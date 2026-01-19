import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from outputs.duration_plots import plot_results, summarize_results

# plot unet, unet_mamba and imunet_mamba
all_results = summarize_results('../outputs/P0_curriculum_synthetic', ["model.type", "split_length"])
#results = all_results[all_results['model.type']!='imunet']
plot_results(all_results, keys = ["model.type", "split_length"])

# plot imunet and unet
results_curriculum = all_results[
    (all_results['model.type'] == 'unet') |
    (all_results['model.type'] == 'unet_mamba_block')
]

plot_results(results_curriculum, keys = ["model.type", "split_length"])

models = ['unet', 'unet_mamba_block']
durations = [1800, 3600, 7200, 14400]
counter = [1,2,3]

all_results = pd.DataFrame()
for model in models:
    for duration in durations:
        for i in counter:
            results = pd.read_csv(f'../outputs/P0_no_curriculum_mitbih_sin_more_data/{duration}_{model}_{i}/results.csv', index_col=0)
            results['model'] = model
            results['record_length'] = duration
            results['run'] = i
            all_results = pd.concat([all_results, results])

out = (
    all_results
    .pivot(
        index=['model', 'record_length', 'run'],
        columns='metric',
        values='mean'
    )
    .reset_index()
)



plot_results(results_curriculum, keys = ["model.type", "split_length"], extra_df=out)


# plot differences
df = results.set_index(['split_length','model.type'])

diff_mamba = df.xs('unet_mamba', level=1) - df.xs('unet', level=1)
diff_mamba['model.type'] = 'unet_mamba - unet'

diff_bidir = df.xs('unet_mamba_bidir', level=1) - df.xs('unet', level=1)
diff_bidir['model.type'] = 'unet_mamba_bidir - unet'

results_diff = pd.concat([results, diff_mamba.reset_index(), diff_bidir.reset_index()])
results_diff = results_diff[
    (results_diff['model.type'] == 'unet_mamba - unet') |
    (results_diff['model.type'] == 'unet_mamba_bidir - unet')
]

plot_results(results_diff, keys = ["model.type", "split_length"])
