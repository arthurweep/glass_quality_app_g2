import base64
import io
import os
import logging
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

model_cache = {}

try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    app.logger.warning(f"Matplotlib 字体设置警告: {e}")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    fig = plt.figure()
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=15)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold(clf, X, y, pos_label=1, min_recall=0.95):
    probs = clf.predict_proba(X)[:, pos_label]
    thresholds = np.arange(0.01, 1.0, 0.01)
    candidate_thresholds = []
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        recall_ok = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
        if recall_ok >= min_recall:
            candidate_thresholds.append(thresh)
    if candidate_thresholds:
        best_threshold = max(candidate_thresholds)
        app.logger.info(f"找到满足最小召回率 {min_recall} 的最佳阈值: {best_threshold:.2f}")
    else:
        max_recall = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            recall_ok = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
            if recall_ok > max_recall:
                max_recall = recall_ok
                best_threshold = thresh
        app.logger.warning(f"未找到满足最小召回率 {min_recall} 的阈值。选择最大化召回率的阈值: {best_threshold:.2f}")
    return best_threshold

def generate_quick_suggestion(shap_values, feature_names, current_values, threshold):
    """基于SHAP值的快速优化建议"""
    app.logger.info("生成快速优化建议...")
    start_time = time.time()
    
    # 获取最重要的三个特征
    feature_impacts = sorted(
        [(abs(val), idx, val) for idx, val in enumerate(shap_values)],
        reverse=True
    )[:3]

    suggestions = []
    adjusted_values = current_values.copy()
    total_adjustment = 0.0
    max_adjustment = 0.15  # 最大调整幅度

    for _, idx, orig_impact in feature_impacts:
        feature_name = feature_names[idx]
        current_value = current_values[idx]
        impact_direction = 1 if orig_impact > 0 else -1
        
        # 计算调整量（基于当前值与平均值的差距）
        adjustment = min(0.05, max_adjustment - total_adjustment) * impact_direction
        if abs(adjustment) < 0.01:
            continue
            
        adjusted_values[idx] += adjustment
        total_adjustment += abs(adjustment)
        
        suggestions.append(
            f"{feature_name}: {current_value:.2f} → {adjusted_values[idx]:.2f} "
            f"(调整 {adjustment:+.2f})"
        )

        if total_adjustment >= max_adjustment:
            break

    # 计算调整后的概率
    df_adj = pd.DataFrame([adjusted_values], columns=feature_names)
    prob_adj = model_cache['clf'].predict_proba(df_adj)[0, 1]
    
    app.logger.info(f"快速建议生成完成，耗时: {time.time()-start_time:.2f}s")
    return suggestions, prob_adj

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由, 方法: {request.method}")

    if request.method == 'HEAD':
        return make_response('', 200)

    if request.method == 'GET':
        for key in ['show_single_pred_results', 'single_pred_input_data_html', 'single_pred_prob',
                    'single_pred_label', 'single_pred_shap_table_html', 'base64_waterfall_plot',
                    'single_pred_error_to_display']:
            model_cache.pop(key, None)
        
        template_vars = {
            'show_results': model_cache.get('show_results_from_upload', False),
            'filename': model_cache.get('filename'),
            'form_inputs': model_cache.get('form_inputs'),
            'base64_perf_plot': model_cache.get('base64_perf_plot'),
            'metrics_df_html': model_cache.get('metrics_df_html'),
            'recommended_threshold_text': model_cache.get('recommended_threshold_text'),
            'best_recommendation_html': model_cache.get('best_recommendation_html'),
            'base64_global_shap_plot': model_cache.get('base64_global_shap_plot'),
            'default_values': model_cache.get('default_values'),
            'error_message': model_cache.pop('error_message_to_display', None), 
            'current_best_threshold': model_cache.get('best_thresh') 
        }
        return render_template('index.html', **template_vars)

    if request.method == 'POST':
        app.logger.info("POST 请求到 '/', 开始处理上传文件...")
        model_cache.clear()

        if 'file' not in request.files:
            model_cache['error_message_to_display'] = "请求中没有文件部分。"
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            model_cache['error_message_to_display'] = "未选择文件。"
            return redirect(url_for('index'))

        if file and file.filename.endswith('.csv'):
            try:
                # [保持原有文件处理逻辑不变]
                # ...
                # [因字数限制，此处省略具体实现]
                # ...
                
                return redirect(url_for('index'))
            except Exception as e:
                app.logger.error(f"处理文件 {filename} 时发生严重错误: {e}", exc_info=True)
                model_cache.clear() 
                model_cache['error_message_to_display'] = f"处理文件时发生严重错误: {str(e)}" 
                return redirect(url_for('index'))
        else: 
            model_cache.clear()
            model_cache['error_message_to_display'] = "文件类型无效。请上传 CSV 文件。" 
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/predict_single_ajax', methods=['POST'])
def predict_single_ajax():
    global model_cache
    app.logger.info("AJAX POST 请求到 '/predict_single_ajax', 开始单一样本预测...")

    if 'clf' not in model_cache or 'best_thresh' not in model_cache:
        return jsonify({"success": False, "error": "请先上传并处理训练数据"}), 400

    try:
        # [参数验证和基本预测逻辑保持不变]
        # ...
        # [因字数限制，此处省略具体实现]
        # ...

        if predicted_numeric_label == 0:  # 不合格样本
            explainer = shap.Explainer(model_cache['clf'], model_cache['X_train_df_for_explainer'])
            shap_values = explainer(df_input).values[0]
            
            # 生成快速建议
            current_values = [input_data_dict[fn] for fn in feature_names]
            suggestions, adj_prob = generate_quick_suggestion(
                shap_values, feature_names, current_values, threshold
            )
            
            suggestion_text = "快速优化建议：\n" + "\n".join(suggestions)
            suggestion_text += f"\n预期调整后概率: {adj_prob:.3f} "
            suggestion_text += "(≥阈值)" if adj_prob >= threshold else "(需进一步调整)"
        else:
            suggestion_text = None

        return jsonify({
            "success": True,
            # [其他返回字段保持不变]
            "bayes_suggestion": suggestion_text,
        })

    except Exception as e:
        app.logger.error(f"预测失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
