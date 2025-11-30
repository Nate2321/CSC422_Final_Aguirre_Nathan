"""
Generate all evaluation visualizations
"""
import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from visualization.plots import (
    plot_training_history,
    plot_confusion_matrix,
    plot_per_class_metrics,
    create_evaluation_report
)

def main():
    # Paths
    results_dir = Path('results')
    eval_dir = results_dir / 'evaluation'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Load metrics
    with open(eval_dir / 'test_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print("Generating visualizations...")
    
    # 1. Training history
    history_path = results_dir / 'models' / 'training_history.json'
    if history_path.exists():
        print("  - Training history curves...")
        plot_training_history(
            str(history_path),
            save_path=str(figures_dir / 'training_history.png')
        )
    else:
        print(f"  ! Training history not found at {history_path}")
    
    # 2. Confusion matrix (raw)
    print("  - Confusion matrix (raw counts)...")
    cm = np.array(metrics['confusion_matrix'])
    class_names = list(metrics['per_class_metrics'].keys())
    
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(figures_dir / 'confusion_matrix_raw.png'),
        normalize=False,
        figsize=(14, 12)
    )
    
    # 3. Confusion matrix (normalized)
    print("  - Confusion matrix (normalized)...")
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(figures_dir / 'confusion_matrix_normalized.png'),
        normalize=True,
        figsize=(14, 12)
    )
    
    # 4. Per-class metrics
    print("  - Per-class metrics...")
    plot_per_class_metrics(
        metrics,
        save_path=str(figures_dir / 'per_class_metrics.png'),
        figsize=(16, 8)
    )
    
    print(f"\n✅ All visualizations saved to {figures_dir}/")
    print("\nGenerated files:")
    for f in sorted(figures_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
