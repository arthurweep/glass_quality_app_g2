import base64
import io
import os
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from scipy.special import expit as sigmoid
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=10)
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
    else:
        max_recall = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            recall_ok = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
            if recall_ok > max_recall:
                max_recall = recall_ok
                best_threshold = thresh
    return best_threshold

def generate_feature_importance_plot(clf, feature_names):
    """生成特征重要性图表"""
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importance)), importance[indices])
    ax.set_title('Feature Importance (XGBoost Built-in)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    if request.method == 'GET':
        return render_template('index.html',
            show_results=model_cache.get('show_results_from_upload', False),
            filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('feature_names', []),
            default_values=model_cache.get('default_values', {}),
            error_message=model_cache.pop('error_message_to_display', None),
            current_best_threshold=model_cache.get('best_thresh', None),
            feature_importance_plot=model_cache.get('feature_importance_plot', None)
        )
    if request.method == 'POST':
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
                df = pd.read_csv(file)
                model_cache['filename'] = file.filename
                if "OK_NG" not in df.columns:
                    model_cache['error_message_to_display'] = "CSV必须包含'OK_NG'列。"
                    return redirect(url_for('index'))
                X = df.drop("OK_NG", axis=1).copy()
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X = X.fillna(X.mean())
                y = pd.to_numeric(df["OK_NG"], errors='coerce').astype(int)
                
                feature_names = list(X.columns)
                model_cache['feature_names'] = feature_names
                model_cache['X_train_df_for_explainer'] = X.copy()
                
                weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
                clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False)
                clf.fit(X, y, sample_weight=weights)
                model_cache['clf'] = clf
                
                best_threshold = find_best_threshold(clf, X, y, pos_label=1, min_recall=0.95)
                model_cache['best_thresh'] = best_threshold
                model_cache['default_values'] = X.mean().to_dict()
                
                # 生成特征重要性图表
                model_cache['feature_importance_plot'] = generate_feature_importance_plot(clf, feature_names)
                
                model_cache['show_results_from_upload'] = True
                return redirect(url_for('index'))
            except Exception as e:
                model_cache.clear()
                model_cache['error_message_to_display'] = f"处理文件时出错: {str(e)}"
                return redirect(url_for('index'))
        else:
            model_cache.clear()
            model_cache['error_message_to_display'] = "文件类型无效。请上传CSV。"
            return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/predict_single_ajax', methods=['POST'])
def predict_single_ajax():
    global model_cache
    if 'clf' not in model_cache or 'best_thresh' not in model_cache:
        return jsonify({"success": False, "error": "请先上传并处理训练数据"}), 400
    
    try:
        clf = model_cache['clf']
        threshold = model_cache['best_thresh']
        X_background = model_cache['X_train_df_for_explainer']
        feature_names = model_cache['feature_names']
        
        input_data_dict = {}
        for fn in feature_names:
            val = request.form.get(fn)
            if val is None or val.strip() == '':
                return jsonify({"success": False, "error": f"特征 '{fn}' 的值不能为空。", "field": fn}), 400
            input_data_dict[fn] = float(val)
        
        df_input = pd.DataFrame([input_data_dict], columns=feature_names)
        prob_ok = clf.predict_proba(df_input)[0, 1]
        pred_label = "✅ 合格 (OK)" if prob_ok >= threshold else "❌ 不合格 (NG)"
        
        is_ng = bool(prob_ok < threshold)
        
        # 获取特征重要性
        feature_importance = clf.feature_importances_
        importance_data = []
        for i, name in enumerate(feature_names):
            importance_data.append({
                "feature": name,
                "importance": float(feature_importance[i]),
                "current_value": float(input_data_dict[name])
            })
        importance_data.sort(key=lambda x: x["importance"], reverse=True)
        
        suggestion_text = ""
        shap_analysis = []
        waterfall_plot = None
        
        if is_ng:
            # SHAP分析
            explainer = shap.Explainer(clf, X_background)
            shap_values = explainer(df_input)
            
            # 生成SHAP waterfall图
            waterfall_plot = generate_shap_waterfall_base64(shap_values[0])
            
            # SHAP值分析
            shap_vals = shap_values.values[0]
            base_value = shap_values.base_values[0]
            
            for i, name in enumerate(feature_names):
                shap_analysis.append({
                    "feature": name,
                    "shap_value": float(shap_vals[i]),
                    "current_value": float(input_data_dict[name]),
                    "contribution": "正面影响" if shap_vals[i] > 0 else "负面影响"
                })
            
            # 按SHAP值绝对值排序
            shap_analysis.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            
            # 生成优化建议
            suggestions = []
            for item in shap_analysis[:3]:  # 取前3个最重要的特征
                if item["shap_value"] < 0:  # 负面影响，建议增加
                    adj = min(0.1, abs(item["shap_value"]) * 0.1)
                    new_val = item["current_value"] + adj
                    suggestions.append(f"建议将 {item['feature']} 从 {item['current_value']:.2f} 调整到 {new_val:.2f} (增加 {adj:.2f})")
                elif item["shap_value"] > 0 and prob_ok < threshold:  # 正面但仍不够
                    adj = min(0.05, item["shap_value"] * 0.1)
                    new_val = item["current_value"] + adj
                    suggestions.append(f"建议将 {item['feature']} 从 {item['current_value']:.2f} 调整到 {new_val:.2f} (微调 +{adj:.2f})")
            
            suggestion_text = "基于SHAP分析的优化建议：\n" + "\n".join(suggestions) if suggestions else "暂无明确优化建议"
        
        return jsonify({
            "success": True,
            "prob_ok": f"{prob_ok:.3f}",
            "label": pred_label,
            "threshold_used": f"{threshold:.2f}",
            "is_ng": is_ng,
            "feature_importance": importance_data,
            "shap_analysis": shap_analysis,
            "waterfall_plot": waterfall_plot,
            "suggestion_text": suggestion_text
        })
        
    except Exception as e:
        app.logger.error(f"预测过程出错: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
