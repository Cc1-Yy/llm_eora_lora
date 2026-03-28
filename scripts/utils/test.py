import pandas as pd

csv_path = "outputs/lm/report_exp3/table_exp3_lm_main_summary.csv"
out_path = "outputs/lm/report_exp3/table_exp3_lm_main_summary.tex"

df = pd.read_csv(csv_path)

body = df.to_latex(
    index=False,
    escape=False,
    na_rep="--",
    float_format="%.4f",
    column_format="lcccccc",
)

latex_table = f"""
\\begin{{table}}[H]
    \\centering
    \\small
    \\renewcommand{{\\arraystretch}}{{1.12}}
    \\caption{{Main language-modelling results for Experiment~3 under quantization recovery.}}
    \\label{{tab:exp3_lm_main_summary}}
{body}
\\end{{table}}
"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"Saved to: {out_path}")