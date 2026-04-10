import pandas as pd
from pathlib import Path

input_path = Path('new_code/data/ptb_xl/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv')
output_dir = Path('new_code/visualisation/output/statement_table')
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path, index_col=0)

# Keep only rows where form == 1
df = df[df['diagnostic'] == 1.0][['description']].copy()
df.index.name = 'Abbreviation'
df.columns = ['Description']

def format_latex_table(df):
    latex_str = df.to_latex(
        escape=True,
        column_format='ll',
    )

    wrapped = (
        "\\begin{table}[htbp]\n"
        "\\caption{ECG Form Statement Labels}\n"
        "\\begin{center}\n"
        f"{latex_str}\n"
        "\\label{tab:form_labels}\n"
        "\\end{center}\n"
        "\\end{table}"
    )
    return wrapped

latex_table = format_latex_table(df)
print(latex_table)

output_path = output_dir / 'statement_table.tex'
with open(output_path, 'w') as f:
    f.write(latex_table)
print(f"\nTable saved to {output_path}")
