import os
import json

def load_evaluation_results(exp_path):
    results_file = os.path.join(exp_path, 'results.json')
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def print_category_precisions(results):
    if 'category_metrics' not in results:
        print("Category metrics not found in results.")
        return
    
    for category, metrics in results['category_metrics'].items():
        precision = metrics['precision']
        print(f"Category: {category}, Precision: {precision:.4f}")

# 设置实验路径
exp_path = 'autodl-tmp/yolov8/runs/detect/train81/weights/best.pt'

# 加载评估结果
results = load_evaluation_results(exp_path)
if results:
    # 输出每个类别的准确率
    print_category_precisions(results)
