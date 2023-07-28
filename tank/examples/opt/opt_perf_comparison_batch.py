from typing import Iterable, List
import subprocess


def make_commands() -> Iterable[List[str]]:
    command = 'python opt_perf_comparison.py --no-save-json'.split(' ')
    max_seq_lens = [32, 128, 512]
    model_names = ['facebook/opt-' + e for e in ['125m', '350m']]
    for max_seq_len in max_seq_lens:
        for model_name in model_names:
            yield command + [
                f'--max-seq-len={max_seq_len}', f'--model-name={model_name}'
            ]


def main():
    #    command = 'python --version'.split(' ')
    for command in make_commands():
        result = subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
