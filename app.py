import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 使用Matplotlib默认英文字体
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

# 字段中文名映射，仅用于HTML文本显示
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

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions)", fontsize=14)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    booster = clf.get_booster()
    # 确保booster的feature_names与训练时一致（通常XGBoost内部会处理）
    # booster.feature_names = feature_names_original_english # 通常不需要，除非get_score返回的是f0, f1...
    
    importance_scores = booster.get_score(importance_type='weight')
    if not importance_scores:
        return None

    # 如果get_score返回的是f0, f1...形式的键，我们需要映射回原始特征名
    # 假设feature_names_original_english的顺序与模型内部特征顺序一致
    mapped_importance = {}
    for i, f_name in enumerate(feature_names_original_english):
        internal_f_name = f"f{i}" # XGBoost内部可能使用的名称
        if internal_f_name in importance_scores:
            mapped_importance[f_name] = importance_scores[internal_f_name]
        elif f_name in importance_scores: # 如果直接返回了原始特征名
             mapped_importance[f_name] = importance_scores[f_name]


    if not mapped_importance: # 如果映射后为空，尝试直接使用importance_scores
        if all(isinstance(k, str) and k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
             # 如果还是f0, f1...形式，记录警告，但继续尝试（可能绘制的是f0, f1...）
            app.logger.warning("Feature importance keys are f0, f1... and could not be mapped to original names for plotting. Plotting with f-scores.")
            mapped_importance = importance_scores # 回退
        else: # 如果键已经是字符串特征名
            mapped_importance = importance_scores


    sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10)
    top_features = sorted_importance[:num_features_to_display]
    
    feature_labels_for_plot_english = [f[0] for f in top_features] 
    scores_for_plot = [float(f[1]) for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 8)) # 增加高度给标签
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot_english, fontsize=9) # 英文Y轴标签
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score (Weight)', fontsize=12)
    ax.set_title('Feature Importance Ranking', fontsize=16)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_f1(clf, X, y):
    # ... (与上一版逻辑一致)
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
            'accuracy': accuracy_score(y, dummy_preds), 'recall_ok': recall_score(y, dummy_preds, pos_label=1, zero_division=0),
            'recall_ng': recall_score(y, dummy_preds, pos_label=0, zero_division=0), 'precision_ok': precision_score(y, dummy_preds, pos_label=1, zero_division=0),
            'precision_ng': precision_score(y, dummy_preds, pos_label=0, zero_division=0), 'f1_ok': f1_score(y, dummy_preds, pos_label=1, zero_division=0),
            'f1_ng': f1_score(y, dummy_preds, pos_label=0, zero_division=0), 'threshold': 0.5
        }
        best_thresh = 0.5
    final_threshold = float(best_metrics_at_thresh.get('threshold', best_thresh))
    final_metrics = {k: float(v) for k, v in best_metrics_at_thresh.items()}
    final_metrics['threshold'] = final_threshold
    return final_threshold, final_metrics


def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names, initial_is_ng):
    adjustments = {}
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()
    
    # 初始预测概率
    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    
    # 即使 initial_is_ng 为 True 且 current_prob_ok 可能已略高于 threshold，
    # 调整目标仍然是使概率达到或超过 threshold_ok_prob。
    # 如果当前已经远超，则可能不需要大幅调整。
    required_boost = float(threshold_ok_prob - current_prob_ok)
    
    # 如果最初判定为OK，并且当前概率已经合格，则不进行调整。
    if not initial_is_ng and current_prob_ok >= threshold_ok_prob:
        return adjustments, float(current_prob_ok), "Sample is already predicted as OK and meets/exceeds threshold."

    # 如果最初判定为NG，但当前概率已经显著高于阈值 (例如，高出0.05)，
    # 也可能不需要进一步“提升”调整，而是可以考虑“巩固性”调整或不调整。
    # 但为了确保总有建议，我们继续，除非调整无法带来任何提升。
    # if initial_is_ng and current_prob_ok > threshold_ok_prob + 0.05:
    #     return adjustments, float(current_prob_ok), "Sample was NG, but current probability significantly exceeds threshold. No further 'improvement' adjustment needed."

    sorted_features_by_shap = sorted(enumerate(shap_values_np), key=lambda x: -abs(x[1]))
    adjusted_values_for_final_check = current_values_np.copy()
    adjustment_made_count = 0

    for idx, shap_val_for_feature in sorted_features_by_shap:
        # 计算当前累积调整后的概率，看是否还需要提升
        prob_after_previous_adjustments = clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1]
        current_required_boost = float(threshold_ok_prob - prob_after_previous_adjustments)

        if current_required_boost <= 0 and adjustment_made_count > 0 and initial_is_ng:
            # 如果最初是NG，且做过调整后已达标，可以考虑停止或做一轮巩固
            break 
        if adjustment_made_count >= 3 and current_required_boost <= 0.01 and initial_is_ng: # 最多调整3个特征如果已接近达标
            break

        feature_name = feature_names[idx]
        delta = 0.001 
        
        # 敏感度计算基于当前已调整的值
        temp_values_plus_delta = adjusted_values_for_final_check.copy()
        temp_values_plus_delta[idx] += delta
        prob_after_delta_on_adjusted = clf.predict_proba(temp_values_plus_delta.reshape(1, -1))[0, 1]
        sensitivity = (prob_after_delta_on_adjusted - prob_after_previous_adjustments) / delta
        
        if abs(sensitivity) < 1e-7: # 提高敏感度阈值，避免除以极小数
            continue
            
        needed_feature_change = current_required_boost / sensitivity
        
        max_abs_change_ratio = 0.40 
        # 注意：这里的current_feature_val应该是原始值，而不是adjusted_values_for_final_check[idx]
        # 因为调整上限是基于原始值的。
        original_feature_val = current_values_np[idx] 
        max_abs_change_value = abs(original_feature_val * max_abs_change_ratio) if original_feature_val != 0 else 0.20
        
        actual_feature_change = float(np.clip(needed_feature_change, -max_abs_change_value, max_abs_change_value))
        
        if abs(actual_feature_change) < 1e-5:
            continue

        # 预估这一步的概率增益
        expected_gain_this_step = float(sensitivity * actual_feature_change)
        
        adjusted_values_for_final_check[idx] += actual_feature_change
        
        adjustments[feature_name] = {
            'current_value': float(current_values_np[idx]), # 始终显示原始值
            'adjustment': actual_feature_change,
            'new_value': float(adjusted_values_for_final_check[idx]), # 累积调整后的值
            'expected_gain': expected_gain_this_step
        }
        adjustment_made_count +=1
            
    final_prob_after_all_adjustments = float(clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1])
    
    message = None
    if not adjustments and initial_is_ng:
        current_final_prob_no_adjust = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
        if current_final_prob_no_adjust >= threshold_ok_prob:
             message = "Sample was initially NG, but its current probability already meets/exceeds threshold. No adjustment needed."
        elif abs(current_final_prob_no_adjust - threshold_ok_prob) < 0.02:
            message = "Sample is very close to the OK threshold. Minor adjustments might not be impactful or practical."
        else:
            message = "Could not compute effective adjustments. Features might be insensitive, at adjustment limits, or model is firm on this sample."
            
    return adjustments, final_prob_after_all_adjustments, message

# --- Routes (与上一版逻辑基本一致, 确保所有从模型或NumPy获取的数值在放入JSON响应前都转换为Python原生类型) ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD': return make_response('', 200)
    if request.method == 'GET':
        metrics = model_cache.get('metrics', {})
        default_metrics = {
            'threshold': 0.5, 'accuracy': 0.0, 'recall_ok': 0.0, 'recall_ng': 0.0, 
            'precision_ok': 0.0, 'precision_ng': 0.0, 'f1_ok': 0.0, 'f1_ng': 0.0,
            'trees': 'N/A', 'depth': 'N/A', 'lr': 'N/A'
        }
        final_metrics = {**default_metrics, **metrics}
        for k, v in final_metrics.items():
            if isinstance(v, (np.float32, np.float64)): final_metrics[k] = float(v)
        return render_template('index.html',
            show_results=bool(model_cache.get('show_results', False)), filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('features', []), default_values=model_cache.get('defaults', {}),
            model_metrics=final_metrics, feature_plot=model_cache.get('feature_plot', None),
            error_msg=model_cache.pop('error', None), field_labels=FIELD_LABELS
        )
    if request.method == 'POST':
        model_cache.clear()
        if 'file' not in request.files:
            model_cache['error'] = "未选择文件"; return redirect(url_for('index'))
        file = request.files['file']
        if not file or file.filename == '':
            model_cache['error'] = "文件无效或文件名为空"; return redirect(url_for('index'))
        try:
            df = pd.read_csv(file); model_cache['filename'] = file.filename
            if "OK_NG" not in df.columns:
                model_cache['error'] = "CSV文件中必须包含 'OK_NG' 列。"; return redirect(url_for('index'))
            X_raw = df.drop("OK_NG", axis=1); X = X_raw.copy()
            for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(X.mean())
            y = pd.to_numeric(df["OK_NG"], errors='coerce').fillna(0).astype(int)
            features = X.columns.tolist()
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, 
                colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False
            )
            sample_weights = compute_sample_weight(class_weight={0:2.0, 1:1}, y=y)
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
    if 'clf' not in model_cache: return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']; features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        input_data_dict = {}
        for f_name in features:
            val_str = request.form.get(f_name)
            if val_str is None or val_str.strip() == '': return jsonify({'error': f'特征 "{FIELD_LABELS.get(f_name, f_name)}" 的值不能为空。'}), 400
            try: input_data_dict[f_name] = float(val_str)
            except ValueError: return jsonify({'error': f'特征 "{FIELD_LABELS.get(f_name, f_name)}" 的输入值 "{val_str}" 不是有效的数字。'}), 400
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
                values=shap_values_for_output.astype(float), base_values=base_val_for_waterfall,
                data=df_input.iloc[0].values.astype(float), feature_names=features # Use English names
            )
            waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
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
    if 'clf' not in model_cache: return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']; features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        json_data = request.get_json()
        if not json_data: return jsonify({'error': '请求体为空或不是有效的JSON。'}), 400
        input_data_dict = json_data.get('input_data')
        shap_values_list = json_data.get('shap_values')
        initial_is_ng = json_data.get('initial_is_ng_for_adjustment', True)
        if not input_data_dict or not isinstance(input_data_dict, dict): return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features): return jsonify({'error': '缺少或无效的 shap_values。'}), 400
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float)
        shap_values_np_array = np.array(shap_values_list, dtype=float)
        adjustments, final_prob_after_adjustment, message = calculate_precise_adjustment(
            clf, current_values_np_array, shap_values_np_array, threshold, features, initial_is_ng
        )
        return jsonify({
            'adjustments': adjustments, # Keys are English feature names
            'final_prob_after_adjustment': float(final_prob_after_adjustment),
            'message': message
        })
    except Exception as e:
        app.logger.error(f"优化建议接口 (/adjust_single) 出错: {e}", exc_info=True)
        return jsonify({'error': f'优化建议过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
