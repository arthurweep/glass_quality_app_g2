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
from sklearn.metrics import classification_report, f1_score, recall_score
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

def quick_suggestion(shap_values, feature_names, current_values, threshold, clf):
    # 只取前3重要特征
    idxs = np.argsort(-np.abs(shap_values))[:3]
    suggestions = []
    adjusted = current_values.copy()
    for idx in idxs:
        direction = 1 if shap_values[idx] < 0 else -1
        adj = 0.05 * direction
        adjusted[idx] += adj
        suggestions.append(f"{feature_names[idx]}: {current_values[idx]:.2f} → {adjusted[idx]:.2f} (调整 {adj:+.2f})")
    df_adj = pd.DataFrame([adjusted], columns=feature_names)
    prob_adj = clf.predict_proba(df_adj)[0, 1]
    return suggestions, prob_adj

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
            current_best_threshold=model_cache.get('best_thresh', None)
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
                model_cache['feature_names'] = list(X.columns)
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
        suggestion_text = ""
        if prob_ok < threshold:
            explainer = shap.Explainer(clf, X_background)
            shap_values = explainer(df_input).values[0]
            current_values = [input_data_dict[fn] for fn in feature_names]
            suggestions, adj_prob = quick_suggestion(shap_values, feature_names, current_values, threshold, clf)
            suggestion_text = "快速优化建议：\n" + "\n".join(suggestions)
            suggestion_text += f"\n预期调整后概率: {adj_prob:.3f} "
            suggestion_text += "(≥阈值)" if adj_prob >= threshold else "(需进一步调整)"
        return jsonify({
            "success": True,
            "prob_ok": f"{prob_ok:.3f}",
            "label": pred_label,
            "threshold_used": f"{threshold:.2f}",
            "bayes_suggestion": suggestion_text,
            "is_ng": prob_ok < threshold
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
