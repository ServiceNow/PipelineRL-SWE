"""Generate paper tables from audited, frozen-policy evaluation snapshots."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def interval(value, bounds, decimals, signed=False):
    def number(x):
        return f'{100*x:+.{decimals}f}' if signed else f'{100*x:.{decimals}f}'
    return f'{number(value)} [{number(bounds[0])}, {number(bounds[1])}]'


def table(source, filename, label, deterministic):
    data = json.loads((HERE / 'data' / source).read_text())
    lines = [r'\begin{table*}[t]', r'\centering', r'\footnotesize',
             r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Fresh set & \shortstack{Cal. target\\\%} & \shortstack{Fresh acc.\\ours, \%} & \shortstack{$\Delta$ acc. vs median\\pp [95\% CI]} & \shortstack{Savings vs median\\\% [95\% CI]} & \shortstack{$\Delta$ acc. vs mean\\pp [95\% CI]} & \shortstack{Savings vs mean\\\% [95\% CI]} \\',
             r'\midrule']
    for i, (dataset, shortname) in enumerate([('mmlupro', 'MMLU-Pro'), ('omni500', 'Omni-MATH')]):
        if i:
            lines.append(r'\midrule')
        for target, row in data['datasets'][dataset]['results'].items():
            if not row['arms']:
                continue
            cells = [shortname, f'{float(target)*100:.0f}',
                     f"{row['arms']['learned']['accuracy']*100:.2f}"]
            for baseline in ['median', 'mean']:
                c = row['comparisons'][baseline]
                cells.extend([interval(c['accuracy_delta'], c['accuracy_delta_ci95'], 2, signed=True),
                              interval(c['savings'], c['savings_ci95'], 1)])
            # Math mode avoids text-mode hyphens for negative values.
            cells = cells[:3] + [f'${cell}$' for cell in cells[3:]]
            lines.append(' & '.join(cells) + r' \\')
    if deterministic:
        explanation = ('For each target, each arm selects the cheapest single-$V$ policy reaching at least that accuracy on original calibration. Every query receives one answer; no routing randomization or abstention is used. Fresh accuracy need not equal the target or reference accuracy.')
    else:
        explanation = ('Two-$V$ mixtures and their probabilities are selected on original calibration to attain each target and then frozen. Entries evaluate mixture-expected outcomes; fresh accuracies need not match.')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{\textbf{' + ('Fresh deterministic policies.' if deterministic else 'Fresh randomized-policy sensitivity.') +
              r'} MMLU-Pro has 6,500 new problems and Omni-MATH has 1,000. ' + explanation +
              ' Positive accuracy differences and savings favor learned costs. All reachable targets are shown. Intervals are pointwise paired stratified percentile bootstraps (2,000 resamples), conditional on the fitted heads and calibrated policies. Generation spending excludes encoder overhead.}',
              rf'\label{{{label}}}', r'\end{table*}']
    (HERE / filename).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    table('fresh_deterministic_results.json', 'fresh_results_table.tex', 'tab:fresh', True)
    table('fresh_policy_results.json', 'fresh_randomized_table.tex', 'tab:randomized', False)
