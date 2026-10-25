"""LaTeX tables from results/test_summary.json.
report/idea3_table1.tex: main table (point estimates in %, * = paired difference from base excludes 0).
report/idea3_tableA.tex: appendix table with 95% intervals.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HERE

COLS = [('acc', 'Acc$\\uparrow$'), ('s0', '$S_0$'), ('s_app_tr', '$S^{\\text{tr}}_{\\text{app}}\\downarrow$'),
        ('s_app_ho', '$S^{\\text{ho}}_{\\text{app}}\\downarrow$'), ('r_beh', '$R_{\\text{beh}}\\uparrow$'),
        ('r_state', '$R_{\\text{state}}\\uparrow$'), ('s_face', '$S_{\\text{face}}\\downarrow$')]
NAMES = {'base': 'Base', 'constitution': 'Constitution', 'naive': 'Naive steering', 'refined': 'Refined steering'}


def main():
    s = json.loads((HERE / 'results/test_summary.json').read_text())
    rows = []
    for m in ('base', 'constitution', 'naive', 'refined'):
        cells = []
        for k, _ in COLS:
            v = s['methods'][m][k]
            star = ''
            if m != 'base' and k != 's0':
                d = s['vs_base'][m][k]
                if d and (d[1] > 0 or d[2] < 0):
                    star = '$^{*}$'
            cells.append(f'{100 * v[0]:.1f}{star}' if v else '--')
        rows.append(NAMES[m] + ' & ' + ' & '.join(cells) + r' \\')
    head = 'Method & ' + ' & '.join(h for _, h in COLS) + r' \\'
    (HERE / 'report/idea3_table1.tex').write_text('\n'.join([r'\toprule', head, r'\midrule'] + rows + [r'\bottomrule']) + '\n')
    rowsA = []
    for m in ('base', 'constitution', 'naive', 'refined'):
        cells = [f"{100 * v[0]:.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]" if (v := s['methods'][m][k]) else '--' for k, _ in COLS]
        rowsA.append(NAMES[m] + ' & ' + ' & '.join(cells) + r' \\')
    (HERE / 'report/idea3_tableA.tex').write_text('\n'.join([r'\toprule', head, r'\midrule'] + rowsA + [r'\bottomrule']) + '\n')
    print((HERE / 'report/idea3_table1.tex').read_text())


if __name__ == '__main__':
    main()
