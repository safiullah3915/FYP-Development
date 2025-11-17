"""
Evaluate ranker calibration by measuring precision@k across score buckets.
Usage:
    python evaluate_ranker.py --data data/ranker_train.csv
"""
import argparse
import pandas as pd


DEFAULT_THRESHOLDS = [
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.0),
]
K_VALUES = [5, 10, 20]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate ranker calibration by score bucket')
    parser.add_argument('--data', type=str, required=True, help='Path to ranker dataset CSV')
    parser.add_argument(
        '--thresholds',
        type=str,
        default='',
        help='Custom score thresholds as comma-separated ranges (e.g., "0.0-0.4,0.4-0.7,0.7-1.0")'
    )
    return parser.parse_args()


def parse_thresholds(arg_value: str):
    if not arg_value:
        return DEFAULT_THRESHOLDS
    thresholds = []
    for token in arg_value.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split('-')
        if len(parts) != 2:
            raise ValueError(f'Invalid threshold format: {token}')
        thresholds.append((float(parts[0]), float(parts[1])))
    return thresholds


def compute_precision_at_k(df: pd.DataFrame, k: int) -> float:
    per_user = []
    grouped = df.groupby('user_id')
    for _, group in grouped:
        group_sorted = group.sort_values('original_score', ascending=False)
        top_k = group_sorted.head(k)
        if top_k.empty:
            continue
        precision = top_k['label'].sum() / len(top_k)
        per_user.append(precision)
    if not per_user:
        return 0.0
    return sum(per_user) / len(per_user)


def evaluate(data_path: str, thresholds):
    df = pd.read_csv(data_path)
    if 'original_score' not in df.columns:
        raise ValueError('Dataset missing original_score column. Regenerate dataset with latest pipeline.')
    if 'label' not in df.columns:
        raise ValueError('Dataset missing label column.')

    results = []
    for lower, upper in thresholds:
        bucket_df = df[(df['original_score'] >= lower) & (df['original_score'] < upper)]
        bucket_label = f'{lower:.1f}-{upper:.1f}'
        bucket_size = len(bucket_df)
        metrics = {'score_range': bucket_label, 'samples': bucket_size}
        if bucket_size == 0:
            for k in K_VALUES:
                metrics[f'precision@{k}'] = 0.0
        else:
            for k in K_VALUES:
                metrics[f'precision@{k}'] = round(compute_precision_at_k(bucket_df, k), 4)
        results.append(metrics)
    return results


def print_results(results):
    if not results:
        print('No data available for evaluation.')
        return
    header = ['Score Range', 'Samples'] + [f'Precision@{k}' for k in K_VALUES]
    print('\t'.join(header))
    for row in results:
        line = [
            row['score_range'],
            str(row['samples']),
        ] + [f"{row[f'precision@{k}']:.4f}" for k in K_VALUES]
        print('\t'.join(line))


def main():
    args = parse_arguments()
    thresholds = parse_thresholds(args.thresholds)
    results = evaluate(args.data, thresholds)
    print_results(results)


if __name__ == '__main__':
    main()

