import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

FIELD_LABELS = {
    "F_cut_act": "刀头实际压力",
    "v_cut_act": "切割实际速度",
    "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度",
    "F_wheel_act": "磨轮压紧力",
    "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}

model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation_object, show=False, max_display=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names):
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    num_features_to_display = min(len(feature_names), 15)
    ax.bar(range(num_features_to_display), [float(importance[i]) for i in indices[:num_features_to_display]])
    ax.set_title('特征重要性', fontsize=14)
    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('重要性分数', fontsize=12)
    ax.set_xticks(range(num_features_to_display))
    ax.set_xticklabels([FIELD_LABELS.get(feature_names[i], feature_names[i]) for i in indices[:num_features_to_display]], rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_for_ng_recall(clf, X, y, ng_label=0, ok_label=1, min_target_ng_recall=0.95):
    probs_ok = clf.predict_proba(X)[:, ok_label]
    best_threshold = 0.5
    candidate_thresholds_met_target = []
    for threshold_candidate in np.arange(0.01, 1.0, 0.005):
        y_pred = (probs_ok >= threshold_candidate).astype(int)
        recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
        if recall_ng_current >= min_target_ng_recall:
            precision_ok_current = precision_score(y, y_pred, pos_label=ok_label, zero_division=0)
            candidate_thresholds_met_target.append((threshold_candidate, precision_ok_current, recall_ng_current))
    if candidate_thresholds_met_target:
        candidate_thresholds_met_target.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
        best_threshold = candidate_thresholds_met_target[0][0]
        app.logger.info(f"找到优化NG召回率的平衡阈值: {best_threshold:.3f} (NG召回率 >= {min_target_ng_recall}, OK精确率最高)")
    else:
        max_ng_recall_overall = -1.0
        best_threshold_for_max_ng_recall = 0.5
        for threshold_candidate in np.arange(0.01, 1.0, 0.005):
            y_pred = (probs_ok >= threshold_candidate).astype(int)
            recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
            if recall_ng_current > max_ng_recall_overall:
                max_ng_recall_overall = recall_ng_current
                best_threshold_for_max_ng_recall = threshold_candidate
            elif recall_ng_current == max_ng_recall_overall and threshold_candidate < best_threshold_for_max_ng_recall:
                best_threshold_for_max_ng_recall = threshold_candidate
        best_threshold = best_threshold_for_max_ng_recall
        app.logger.warning(f"未能达到目标NG召回率 {min_target_ng_recall}。选择最大化NG召回率的阈值: {best_threshold:.3f} (此时NG召回率: {max_ng_recall_overall:.2f})")
    return float(best_threshold)

def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names):
    adjustments = {}
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()
    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    required_boost = float(max(threshold_ok_prob - current_prob_ok, 0.0))
    if required_boost <= 1e-4:
        return adjustments, float(current_prob_ok)
    sorted_features_by_shap = sorted(enumerate(shap_values_np), key=lambda x: -abs(x[1]))
    adjusted_values_for_final_check = current_values_np.copy()
    for idx, shap_val_for_feature in sorted_features_by_shap:
        if required_boost <= 1e-4:
            break
        feature_name = feature_names[idx]
        delta = 0.001
        temp_values_plus_delta = current_values_np.copy()
        temp_values_plus_delta[idx] += delta
        prob_after_delta_change = clf.predict_proba(temp_values_plus_delta.reshape(1, -1))[0, 1]
        sensitivity = (prob_after_delta_change - current_prob_ok) / delta
        if abs(sensitivity) < 1e-6:
            continue
        needed_feature_change = required_boost / sensitivity
        max_abs_change_ratio = 0.4
        max_abs_change_value = abs(current_values_np[idx] * max_abs_change_ratio) if current_values_np[idx] != 0 else 0.15
        actual_feature_change = float(np.clip(needed_feature_change, -max_abs_change_value, max_abs_change_value))
        if abs(actual_feature_change) < 1e-5:
            continue
        actual_prob_gain = float(sensitivity * actual_feature_change)
        adjusted_values_for_final_check[idx] += actual_feature_change
        adjustments[feature_name] = {
            'current_value': float(current_values_np[idx]),
            'adjustment': actual_feature_change,
            'new_value': float(adjusted_values_for_final_check[idx]),
            'expected_gain': actual_prob_gain
        }
        required_boost -= actual_prob_gain
    final_prob_after_all_adjustments = float(clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1])
    return adjustments, final_prob_after_all_adjustments

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    if request.method == 'GET':
        metrics = model_cache.get('metrics', {
            'threshold': 0.5, 'accuracy': 0.0, 'recall_ok': 0.0, 'recall_ng': 0.0, 
            'precision_ok': 0.0, 'precision_ng': 0.0, 'f1_ok': 0.0, 'f1_ng': 0.0,
            'trees': 'N/A', 'depth': 'N/A', 'lr': 'N/A'
        })
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                metrics[k] = float(v)
        return render_template('index.html',
            show_results=bool(model_cache.get('show_results', False)),
            filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('features', []),
            default_values=model_cache.get('defaults', {}),
            model_metrics=metrics,
            feature_plot=model_cache.get('feature_plot', None),
            error_msg=model_cache.pop('error', None),
            field_labels=FIELD_LABELS
        )
    if request.method == 'POST':
        model_cache.clear()
        if 'file' not in request.files:
            model_cache['error'] = "未选择文件"
            return redirect(url_for('index'))
        file = request.files['file']
        if not file or file.filename == '':
            model_cache['error'] = "文件无效或文件名为空"
            return redirect(url_for('index'))
        try:
            df = pd.read_csv(file)
            model_cache['filename'] = file.filename
            if "OK_NG" not in df.columns:
                model_cache['error'] = "CSV文件中必须包含 'OK_NG' 列。"
                return redirect(url_for('index'))
            X_raw = df.drop("OK_NG", axis=1)
            X = X_raw.copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(X.mean())
            y = pd.to_numeric(df["OK_NG"], errors='coerce').fillna(0).astype(int)
            features = X.columns.tolist()
            clf = xgb.XGBClassifier(
                n_estimators=250, max_depth=7, learning_rate=0.03,
                subsample=0.6, colsample_bytree=0.6, gamma=0.2,
                min_child_weight=1, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, use_label_encoder=False
            )
            sample_weights = compute_sample_weight(class_weight={0:3.0, 1:1}, y=y)
            clf.fit(X, y, sample_weight=sample_weights)
            best_threshold = find_best_threshold_for_ng_recall(clf, X, y, ng_label=0, ok_label=1, min_target_ng_recall=0.95)
            probs_ok_train = clf.predict_proba(X)[:, 1]
            preds_at_best_thresh_train = (probs_ok_train >= best_threshold).astype(int)
            model_metrics = {
                'trees': int(clf.get_params()['n_estimators']),
                'depth': int(clf.get_params()['max_depth']),
                'lr': float(clf.get_params()['learning_rate']),
                'threshold': float(best_threshold),
                'accuracy': float(accuracy_score(y, preds_at_best_thresh_train)),
                'recall_ok': float(recall_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0)),
                'recall_ng': float(recall_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0)),
                'precision_ok': float(precision_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0)),
                'precision_ng': float(precision_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0)),
                'f1_ok': float(f1_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0)),
                'f1_ng': float(f1_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0)),
            }
            model_cache.update({
                'show_results': True,
                'features': features,
                'defaults': {k: float(v) for k, v in X.mean().to_dict().items()},
                'clf': clf,
                'X_train_df': X.copy(),
                'metrics': model_metrics,
                'feature_plot': generate_feature_importance_plot(clf, features)
            })
        except Exception as e:
            model_cache['error'] = f"处理文件时出错: {str(e)}"
            app.logger.error(f"文件处理或模型训练出错: {e}", exc_info=True)
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    global model_cache
    if 'clf' not in model_cache:
        return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']
        features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        input_data_dict = {}
        for f_name in features:
            val_str = request.form.get(f_name)
            if val_str is None or val_str.strip() == '':
                return jsonify({'error': f'特征 "{FIELD_LABELS.get(f_name, f_name)}" 的值不能为空。'}), 400
            input_data_dict[f_name] = float(val_str)
        df_input = pd.DataFrame([input_data_dict], columns=features)
        prob_ok = clf.predict_proba(df_input)[0, 1]
        is_ng = bool(prob_ok < threshold)
        background_data_df = model_cache['X_train_df']
        explainer = shap.Explainer(clf, background_data_df)
        shap_explanation_obj = explainer(df_input)
        shap_values_for_output = shap_explanation_obj.values[0]
        waterfall_plot_base64 = None
        if is_ng:
            base_val_for_waterfall = explainer.expected_value
            if isinstance(base_val_for_waterfall, (np.ndarray, list)):
                base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
            base_val_for_waterfall = float(base_val_for_waterfall)
            shap_explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_output.astype(float),
                base_values=base_val_for_waterfall,
                data=df_input.iloc[0].values.astype(float),
                feature_names=features
            )
            waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
        response = {
            'prob': float(round(prob_ok, 3)),
            'threshold': float(round(threshold, 3)),
            'is_ng': is_ng,
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output],
            'metrics': model_cache['metrics'],
            'waterfall': waterfall_plot_base64,
            'input_data': input_data_dict
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500

@app.route('/adjust_single', methods=['POST'])
def adjust_single():
    global model_cache
    if 'clf' not in model_cache:
        return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']
        features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': '请求体为空或不是有效的JSON。'}), 400
        input_data_dict = json_data.get('input_data')
        shap_values_list = json_data.get('shap_values')
        if not input_data_dict or not isinstance(input_data_dict, dict):
            return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features):
            return jsonify({'error': '缺少或无效的 shap_values。'}), 400
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float)
        shap_values_np_array = np.array(shap_values_list, dtype=float)
        adjustments, final_prob_after_adjustment = calculate_precise_adjustment(
            clf, current_values_np_array, shap_values_np_array, threshold, features
        )
        return jsonify({
            'adjustments': adjustments,
            'final_prob_after_adjustment': float(final_prob_after_adjustment)
        })
    except Exception as e:
        app.logger.error(f"优化建议接口 (/adjust_single) 出错: {e}", exc_info=True)
        return jsonify({'error': f'优化建议过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
