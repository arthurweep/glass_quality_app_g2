import base64
import io
import os
import logging
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

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
    "F_cut_act": "刀头实际压力", "v_cut_act": "切割实际速度", "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度", "F_wheel_act": "磨轮压紧力", "P_cool_act": "冷却水压力",
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
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    booster = clf.get_booster()
    importance_scores = booster.get_score(importance_type='weight')
    if not importance_scores:
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else:
            return None
    mapped_importance = {}
    if all(k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        for i, f_name_original in enumerate(feature_names_original_english):
            internal_f_key = f"f{i}"
            if internal_f_key in importance_scores:
                mapped_importance[f_name_original] = importance_scores[internal_f_key]
        if not mapped_importance and importance_scores:
            mapped_importance = importance_scores
    else:
        mapped_importance = importance_scores
    if not mapped_importance:
        return None
    sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10)
    top_features_data = sorted_importance[:num_features_to_display]
    feature_labels_for_plot_english = [item[0] for item in top_features_data]
    scores_for_plot = [float(item[1]) for item in top_features_data]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot_english, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score (Weight)', fontsize=12)
    ax.set_title('Feature Importance Ranking', fontsize=16)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_f1(clf, X, y):
    probs_ok = clf.predict_proba(X)[:, 1]
    best_f1_macro, best_thresh = 0.0, 0.5
    best_metrics_at_thresh = {}
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred = (probs_ok >= t).astype(int)
        f1_macro_current = f1_score(y, y_pred, average='macro', zero_division=0)
        if f1_macro_current > best_f1_macro:
            best_f1_macro = f1_macro_current
            best_thresh = t
            best_metrics_at_thresh = {
                'accuracy': accuracy_score(y, y_pred),
                'recall_ok': recall_score(y, y_pred, pos_label=1, zero_division=0),
                'recall_ng': recall_score(y, y_pred, pos_label=0, zero_division=0),
                'precision_ok': precision_score(y, y_pred, pos_label=1, zero_division=0),
                'precision_ng': precision_score(y, y_pred, pos_label=0, zero_division=0),
                'f1_ok': f1_score(y, y_pred, pos_label=1, zero_division=0),
                'f1_ng': f1_score(y, y_pred, pos_label=0, zero_division=0),
                'threshold': t
            }
    if not best_metrics_at_thresh:
        dummy_preds = (probs_ok >= 0.5).astype(int)
        best_metrics_at_thresh = {
            'accuracy': accuracy_score(y, dummy_preds),
            'recall_ok': recall_score(y, dummy_preds, pos_label=1, zero_division=0),
            'recall_ng': recall_score(y, dummy_preds, pos_label=0, zero_division=0),
            'precision_ok': precision_score(y, dummy_preds, pos_label=1, zero_division=0),
            'precision_ng': precision_score(y, dummy_preds, pos_label=0, zero_division=0),
            'f1_ok': f1_score(y, dummy_preds, pos_label=1, zero_division=0),
            'f1_ng': f1_score(y, dummy_preds, pos_label=0, zero_division=0),
            'threshold': 0.5
        }
        best_thresh = 0.5
    final_threshold = float(best_metrics_at_thresh.get('threshold', best_thresh))
    final_metrics = {k: float(v) for k, v in best_metrics_at_thresh.items()}
    final_metrics['threshold'] = final_threshold
    return final_threshold, final_metrics

def calculate_adjustment_refined(clf, current_values, shap_values, threshold, feature_names, initial_is_ng):
    import math
    original_values = np.array(current_values, dtype=float).flatten()
    adjusted_values = original_values.copy()
    max_iterations = 30
    max_step_ratio = 0.2
    max_total_ratio = 5.0
    min_prob_improve = 1e-4
    min_abs_change = 1e-5

    for iteration in range(max_iterations):
        prob_before = clf.predict_proba(adjusted_values.reshape(1, -1))[0, 1]
        if prob_before >= threshold:
            break

        prob_needed = threshold - prob_before
        made_adjustment = False

        sensitivities = []
        delta = 1e-3
        for i in range(len(feature_names)):
            temp_vals = adjusted_values.copy()
            temp_vals[i] += delta
            prob_after = clf.predict_proba(temp_vals.reshape(1, -1))[0, 1]
            sens = (prob_after - prob_before) / delta
            sensitivities.append(sens)

        sorted_indices = np.argsort([-abs(s) for s in sensitivities])

        for idx in sorted_indices:
            sens = sensitivities[idx]
            if abs(sens) < 1e-8 or math.isnan(sens) or math.isinf(sens):
                continue

            needed_change = prob_needed / sens

            orig_val = original_values[idx]
            max_step = abs(orig_val) * max_step_ratio if orig_val != 0 else 0.1
            step_change = np.clip(needed_change, -max_step, max_step)

            total_change = adjusted_values[idx] + step_change - orig_val
            max_total = abs(orig_val) * max_total_ratio if orig_val != 0 else 1.0
            if total_change > max_total:
                step_change = max_total - (adjusted_values[idx] - orig_val)
            elif total_change < -max_total:
                step_change = -max_total - (adjusted_values[idx] - orig_val)

            if abs(step_change) < min_abs_change:
                continue

            adjusted_values[idx] += step_change
            made_adjustment = True

            prob_after_adj = clf.predict_proba(adjusted_values.reshape(1, -1))[0, 1]
            gain = prob_after_adj - prob_before
            if math.isnan(gain) or math.isinf(gain):
                gain = 0.0

            yield_feature = feature_names[idx]
            yield {
                'feature': yield_feature,
                'current_value': float(orig_val),
                'adjustment': float(adjusted_values[idx] - orig_val),
                'new_value': float(adjusted_values[idx]),
                'expected_gain': gain
            }

            if prob_after_adj >= threshold:
                break

        if not made_adjustment:
            break

    final_prob = clf.predict_proba(adjusted_values.reshape(1, -1))[0, 1]
    return adjusted_values, final_prob

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    if request.method == 'GET':
        metrics = model_cache.get('metrics', {})
        default_metrics = {
            'threshold': 0.5, 'accuracy': 0.0, 'recall_ok': 0.0, 'recall_ng': 0.0,
            'precision_ok': 0.0, 'precision_ng': 0.0, 'f1_ok': 0.0, 'f1_ng': 0.0,
            'trees': 'N/A', 'depth': 'N/A', 'lr': 'N/A'
        }
        final_metrics = {**default_metrics, **metrics}
        for k, v in final_metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                final_metrics[k] = float(v)
        return render_template('index.html',
                               show_results=bool(model_cache.get('show_results', False)),
                               filename=model_cache.get('filename', ''),
                               form_inputs=model_cache.get('features', []),
                               default_values=model_cache.get('defaults', {}),
                               model_metrics=final_metrics,
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
                n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False
            )
            sample_weights = compute_sample_weight(class_weight={0: 2.0, 1: 1}, y=y)
            clf.fit(X, y, sample_weight=sample_weights)
            best_threshold, calculated_metrics = find_best_threshold_f1(clf, X, y)
            model_params = clf.get_params()
            final_model_metrics = {
                'trees': int(model_params['n_estimators']), 'depth': int(model_params['max_depth']),
                'lr': float(model_params['learning_rate']), **calculated_metrics
            }
            model_cache.update({
                'show_results': True, 'features': features,
                'defaults': {k: float(v) for k, v in X.mean().to_dict().items()},
                'clf': clf, 'X_train_df': X.copy(), 'metrics': final_model_metrics,
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
            try:
                input_data_dict[f_name] = float(val_str)
            except ValueError:
                return jsonify({'error': f'特征 "{FIELD_LABELS.get(f_name, f_name)}" 的输入值 "{val_str}" 不是有效的数字。'}), 400
        df_input = pd.DataFrame([input_data_dict], columns=features)
        prob_ok = clf.predict_proba(df_input)[0, 1]
        is_ng = bool(prob_ok < threshold)
        background_data_df = model_cache['X_train_df']
        explainer = shap.Explainer(clf, background_data_df)
        shap_explanation_obj = explainer(df_input)
        shap_values_for_output = shap_explanation_obj.values[0]
        base_val_for_waterfall = explainer.expected_value
        if isinstance(base_val_for_waterfall, (np.ndarray, list)):
            base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
        base_val_for_waterfall = float(base_val_for_waterfall)
        shap_explanation_for_plot = shap.Explanation(
            values=shap_values_for_output.astype(float), base_values=base_val_for_waterfall,
            data=df_input.iloc[0].values.astype(float), feature_names=features
        )
        waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_plot)
        response = {
            'prob': float(round(prob_ok, 3)), 'threshold': float(round(threshold, 3)), 'is_ng': is_ng,
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output],
            'metrics': model_cache['metrics'], 'waterfall': waterfall_plot_base64,
            'input_data': input_data_dict, 'initial_is_ng_for_adjustment': is_ng
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
        initial_is_ng = json_data.get('initial_is_ng_for_adjustment', True)
        if not input_data_dict or not isinstance(input_data_dict, dict):
            return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features):
            return jsonify({'error': '缺少或无效的 shap_values。'}), 400
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float)
        shap_values_np_array = np.array(shap_values_list, dtype=float)

        # 调用改进的调整函数
        adjustments_generator = calculate_adjustment_refined(
            clf, current_values_np_array, shap_values_np_array, threshold, features, initial_is_ng
        )
        adjustments = {}
        for adj in adjustments_generator:
            adjustments[adj['feature']] = {
                'current_value': adj['current_value'],
                'adjustment': adj['adjustment'],
                'new_value': adj['new_value'],
                'expected_gain_this_step': adj['expected_gain']
            }
        final_prob_after_adjustment = clf.predict_proba(np.array([adj['new_value'] for adj in adjustments.values()]).reshape(1, -1))[0, 1] if adjustments else clf.predict_proba(current_values_np_array.reshape(1, -1))[0, 1]
        message = "调整建议已生成。" if adjustments else "未能找到有效的调整建议。"
        return jsonify({
            'adjustments': adjustments,
            'final_prob_after_adjustment': float(final_prob_after_adjustment),
            'message': message
        })
    except Exception as e:
        app.logger.error(f"优化建议接口 (/adjust_single) 出错: {e}", exc_info=True)
        return jsonify({'error': f'优化建议过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
