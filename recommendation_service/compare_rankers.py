"""
Compare baseline vs candidate ranker outputs using replayed sessions.
Each CSV should include columns: session_id, startup_id, rank, score, label (1 for positive, 0 otherwise).
"""
import argparse
import pandas as pd
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline vs candidate ranker performance')
    parser.add_argument('--baseline', required=True, help='CSV with baseline ranking results')
    parser.add_argument('--candidate', required=True, help='CSV with candidate ranking results')
    parser.add_argument('--k', type=int, default=10, help='K for precision/NDCG/MRR (default: 10)')
    return parser.parse_args()


def load_results(path):
    df = pd.read_csv(path)
    required_cols = {'session_id', 'startup_id', 'rank', 'label'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in {path}: {missing}')
    return df


def precision_at_k(group, k):
    top_k = group.nsmallest(k, 'rank')
    if top_k.empty:
        return 0.0
    return top_k['label'].sum() / len(top_k)


def ndcg_at_k(group, k):
    top_k = group.nsmallest(k, 'rank')
    if top_k.empty:
        return 0.0
    dcg = 0.0
    for idx, row in enumerate(top_k.itertuples(index=False), start=1):
        if row.label:
            dcg += 1 / math.log2(idx + 1)
    ideal_labels = sorted(group['label'], reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(ideal_labels, start=1):
        if rel:
            idcg += 1 / math.log2(idx + 1)
    return (dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(group, k):
    top_k = group.nsmallest(k, 'rank')
    positives = top_k[top_k['label'] == 1]
    if positives.empty:
        return 0.0
    best_rank = positives['rank'].min()
    return 1.0 / best_rank if best_rank > 0 else 1.0


def compute_metrics(df, k):
    precision_values = []
    ndcg_values = []
    mrr_values = []

    for _, group in df.groupby('session_id'):
        precision_values.append(precision_at_k(group, k))
        ndcg_values.append(ndcg_at_k(group, k))
        mrr_values.append(mrr_at_k(group, k))

    return {
        'precision@k': sum(precision_values) / len(precision_values) if precision_values else 0.0,
        'ndcg@k': sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0,
        'mrr@k': sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        'sessions': len(precision_values)
    }


def print_comparison(baseline_metrics, candidate_metrics, k):
    print(f'Comparison @ k={k}')
    header = ['Metric', 'Baseline', 'Candidate', 'Delta']
    print('\t'.join(header))
    for metric in ['precision@k', 'ndcg@k', 'mrr@k']:
        base_val = baseline_metrics[metric]
        cand_val = candidate_metrics[metric]
        delta = cand_val - base_val
        print(f'{metric}\t{base_val:.4f}\t{cand_val:.4f}\t{delta:+.4f}')
    print(f'Sessions evaluated: baseline={baseline_metrics["sessions"]}, candidate={candidate_metrics["sessions"]}')


def main():
    args = parse_args()
    baseline_df = load_results(args.baseline)
    candidate_df = load_results(args.candidate)

    baseline_metrics = compute_metrics(baseline_df, args.k)
    candidate_metrics = compute_metrics(candidate_df, args.k)

    print_comparison(baseline_metrics, candidate_metrics, args.k)


if __name__ == '__main__':
    main()

