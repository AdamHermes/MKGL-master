#!/usr/bin/env python3
"""
Convert raw JSONL prediction output to a readable format.

Usage:
    python scripts/format_predictions.py predictions.jsonl -o readable_output.txt
    python scripts/format_predictions.py predictions.jsonl --format csv -o output.csv
    python scripts/format_predictions.py predictions.jsonl --top 10
    python scripts/format_predictions.py predictions.jsonl --names data/names/fb15k237
"""

import json
import argparse
import csv
from pathlib import Path


def load_name_mappings(names_dir):
    """Load entity and relation name mappings from data/names/ directory."""
    names_dir = Path(names_dir)
    entity_names = {}
    relation_names = {}
    
    # Load entity names
    entity_file = names_dir / 'entity.txt'
    if entity_file.exists():
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        entity_id, name = parts
                        entity_names[entity_id] = name
        print(f"Loaded {len(entity_names)} entity names from {entity_file}")
    
    # Load relation names
    relation_file = names_dir / 'relation.txt'
    if relation_file.exists():
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        rel_id, name = parts
                        relation_names[rel_id] = name
        print(f"Loaded {len(relation_names)} relation names from {relation_file}")
    
    return entity_names, relation_names


def load_predictions(filepath):
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def format_entity_name(name, entity_names=None):
    """Clean up entity name for readability."""
    if not name:
        return name
    # Look up in name mapping if available
    if entity_names and name in entity_names:
        return entity_names[name]
    # Remove common prefixes like /m/, /g/, etc.
    if name.startswith('/m/') or name.startswith('/g/'):
        return name
    # Clean up underscore-separated names
    return name.replace('_', ' ')


def format_relation_name(name, relation_names=None):
    """Clean up relation name for readability."""
    if not name:
        return name
    # Look up in name mapping if available
    if relation_names and name in relation_names:
        return relation_names[name]
    # Extract meaningful part from paths like /film/film/genre
    parts = name.strip('/').split('/')
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return name


def to_readable_text(predictions, top_k=10, show_scores=True, entity_names=None, relation_names=None):
    """Convert predictions to readable text format."""
    lines = []
    lines.append("=" * 80)
    lines.append("KNOWLEDGE GRAPH LINK PREDICTION RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    for i, pred in enumerate(predictions):
        query_type = pred.get('query_type', 'unknown')
        ranking = pred.get('ranking', '?')
        
        if query_type == 'tail_prediction':
            head = format_entity_name(pred.get('head', '?'), entity_names)
            relation = format_relation_name(pred.get('relation', '?'), relation_names)
            ground_truth = format_entity_name(pred.get('ground_truth', '?'), entity_names)
            
            lines.append(f"[{i+1}] TAIL PREDICTION | Rank: {ranking}")
            lines.append(f"    Query: ({head}, {relation}, ?)")
            lines.append(f"    Ground Truth: {ground_truth}")
            
        elif query_type == 'head_prediction':
            tail = format_entity_name(pred.get('tail', '?'), entity_names)
            relation = format_relation_name(pred.get('relation', '?'), relation_names)
            ground_truth = format_entity_name(pred.get('ground_truth', '?'), entity_names)
            
            lines.append(f"[{i+1}] HEAD PREDICTION | Rank: {ranking}")
            lines.append(f"    Query: (?, {relation}, {tail})")
            lines.append(f"    Ground Truth: {ground_truth}")
        
        # Top-k predictions
        top_preds = pred.get('top_k_predictions', [])[:top_k]
        lines.append(f"    Top-{len(top_preds)} Predictions:")
        
        for p in top_preds:
            entity = format_entity_name(p.get('entity', '?'), entity_names)
            rank = p.get('rank', '?')
            if show_scores and 'score' in p:
                score = p['score']
                lines.append(f"        {rank}. {entity} (score: {score:.4f})")
            else:
                lines.append(f"        {rank}. {entity}")
        
        lines.append("")
    
    # Summary
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    
    tail_preds = [p for p in predictions if p.get('query_type') == 'tail_prediction']
    head_preds = [p for p in predictions if p.get('query_type') == 'head_prediction']
    
    if tail_preds:
        tail_rankings = [p['ranking'] for p in tail_preds if 'ranking' in p]
        if tail_rankings:
            lines.append(f"Tail Predictions: {len(tail_preds)}")
            lines.append(f"  - MRR: {sum(1/r for r in tail_rankings) / len(tail_rankings):.4f}")
            lines.append(f"  - Hits@1: {sum(1 for r in tail_rankings if r <= 1) / len(tail_rankings):.4f}")
            lines.append(f"  - Hits@10: {sum(1 for r in tail_rankings if r <= 10) / len(tail_rankings):.4f}")
    
    if head_preds:
        head_rankings = [p['ranking'] for p in head_preds if 'ranking' in p]
        if head_rankings:
            lines.append(f"Head Predictions: {len(head_preds)}")
            lines.append(f"  - MRR: {sum(1/r for r in head_rankings) / len(head_rankings):.4f}")
            lines.append(f"  - Hits@1: {sum(1 for r in head_rankings if r <= 1) / len(head_rankings):.4f}")
            lines.append(f"  - Hits@10: {sum(1 for r in head_rankings if r <= 10) / len(head_rankings):.4f}")
    
    all_rankings = [p['ranking'] for p in predictions if 'ranking' in p]
    if all_rankings:
        lines.append(f"Overall ({len(all_rankings)} queries):")
        lines.append(f"  - MRR: {sum(1/r for r in all_rankings) / len(all_rankings):.4f}")
        lines.append(f"  - Hits@1: {sum(1 for r in all_rankings if r <= 1) / len(all_rankings):.4f}")
        lines.append(f"  - Hits@3: {sum(1 for r in all_rankings if r <= 3) / len(all_rankings):.4f}")
        lines.append(f"  - Hits@10: {sum(1 for r in all_rankings if r <= 10) / len(all_rankings):.4f}")
    
    return '\n'.join(lines)


def to_csv(predictions, top_k=10, entity_names=None, relation_names=None):
    """Convert predictions to CSV format."""
    rows = []
    
    for pred in predictions:
        query_type = pred.get('query_type', 'unknown')
        ranking = pred.get('ranking', '')
        relation = format_relation_name(pred.get('relation', ''), relation_names)
        ground_truth = format_entity_name(pred.get('ground_truth', ''), entity_names)
        
        if query_type == 'tail_prediction':
            query_entity = format_entity_name(pred.get('head', ''), entity_names)
            query_str = f"({query_entity}, {relation}, ?)"
        else:
            query_entity = format_entity_name(pred.get('tail', ''), entity_names)
            query_str = f"(?, {relation}, {query_entity})"
        
        # Get top predictions
        top_preds = pred.get('top_k_predictions', [])[:top_k]
        pred_entities = [format_entity_name(p.get('entity', ''), entity_names) for p in top_preds]
        pred_scores = [str(p.get('score', '')) for p in top_preds]
        
        row = {
            'query_type': query_type,
            'query': query_str,
            'ground_truth': ground_truth,
            'ranking': ranking,
        }
        
        for i in range(top_k):
            row[f'pred_{i+1}'] = pred_entities[i] if i < len(pred_entities) else ''
            row[f'score_{i+1}'] = pred_scores[i] if i < len(pred_scores) else ''
        
        rows.append(row)
    
    return rows


def to_compact(predictions, top_k=5, entity_names=None, relation_names=None):
    """Convert to compact one-line-per-query format."""
    lines = []
    lines.append("Query Type | Query | Ground Truth | Rank | Top Predictions")
    lines.append("-" * 120)
    
    for pred in predictions:
        query_type = "TAIL" if pred.get('query_type') == 'tail_prediction' else "HEAD"
        ranking = pred.get('ranking', '?')
        relation = format_relation_name(pred.get('relation', '?'), relation_names)
        ground_truth = format_entity_name(pred.get('ground_truth', '?'), entity_names)
        
        if pred.get('query_type') == 'tail_prediction':
            head = format_entity_name(pred.get('head', '?'), entity_names)
            query = f"({head}, {relation}, ?)"
        else:
            tail = format_entity_name(pred.get('tail', '?'), entity_names)
            query = f"(?, {relation}, {tail})"
        
        top_preds = pred.get('top_k_predictions', [])[:top_k]
        preds_str = ", ".join([format_entity_name(p.get('entity', '?'), entity_names) for p in top_preds])
        
        # Truncate for display
        query_disp = query[:50] + "..." if len(query) > 50 else query
        gt_disp = ground_truth[:25] + "..." if len(ground_truth) > 25 else ground_truth
        preds_disp = preds_str[:60] + "..." if len(preds_str) > 60 else preds_str
        
        lines.append(f"{query_type:4} | {query_disp:53} | {gt_disp:28} | {ranking:4} | {preds_disp}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Format prediction output files')
    parser.add_argument('input', type=str, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=str, help='Output file (default: stdout)')
    parser.add_argument('-f', '--format', type=str, choices=['text', 'csv', 'compact'], 
                        default='text', help='Output format')
    parser.add_argument('-k', '--top', type=int, default=10, 
                        help='Number of top predictions to show')
    parser.add_argument('--no-scores', action='store_true', 
                        help='Hide prediction scores')
    parser.add_argument('--names', type=str, default=None,
                        help='Path to names directory (e.g., data/names/fb15k237)')
    
    args = parser.parse_args()
    
    # Load name mappings if provided
    entity_names = None
    relation_names = None
    if args.names:
        entity_names, relation_names = load_name_mappings(args.names)
    else:
        # Try to auto-detect from input filename
        input_lower = args.input.lower()
        if 'fb15k237' in input_lower:
            default_names = Path('data/names/fb15k237')
            if default_names.exists():
                print(f"Auto-detected FB15k237 dataset, loading names from {default_names}")
                entity_names, relation_names = load_name_mappings(default_names)
        elif 'wn18rr' in input_lower:
            default_names = Path('data/names/wn18rr')
            if default_names.exists():
                print(f"Auto-detected WN18RR dataset, loading names from {default_names}")
                entity_names, relation_names = load_name_mappings(default_names)
    
    # Load predictions
    predictions = load_predictions(args.input)
    print(f"Loaded {len(predictions)} predictions from {args.input}")
    
    # Format output
    if args.format == 'text':
        output = to_readable_text(predictions, top_k=args.top, show_scores=not args.no_scores,
                                  entity_names=entity_names, relation_names=relation_names)
    elif args.format == 'compact':
        output = to_compact(predictions, top_k=args.top, 
                           entity_names=entity_names, relation_names=relation_names)
    elif args.format == 'csv':
        rows = to_csv(predictions, top_k=args.top,
                     entity_names=entity_names, relation_names=relation_names)
        if args.output:
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"Saved to {args.output}")
            return
        else:
            # Print CSV to stdout
            import io
            output_io = io.StringIO()
            if rows:
                writer = csv.DictWriter(output_io, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            output = output_io.getvalue()
    
    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
