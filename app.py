import base64
import io
import os
import logging
import math # 导入math模块以使用 isnan 和 isinf
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

FIELD_LABELS = { # 仅用于HTML表单标签和表格表头中文显示
    "F_cut_act": "刀头实际压力", "v_cut_act": "切割实际速度", "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度", "F_wheel_act": "磨轮压紧力", "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}
model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    # 图表全部使用英文标签
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14)
    plt.tight_layout(); return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    # 图表全部使用英文标签
    booster = clf.get_booster()
    importance_scores = booster.get_score(importance_type='weight')
    if not importance_scores:
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else: return None
    mapped_importance = {}
    if all(k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        for i, f_name_original in enumerate(feature_names_original_english):
            internal_f_key = f"f{i}"
            if internal_f_key in importance_scores: mapped_importance[f_name_original] = importance_scores[internal_f_key]
        if not mapped_importance and importance_scores: mapped_importance = importance_scores 
    else: mapped_importance = importance_scores
    if not mapped_importance: return None
    sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10)
    top_features_data = sorted_importance[:num_features_to_display]
    feature_labels_for_plot_english = [item[0] for item in top_features_data] 
    scores_for_plot = [float(item[1]) for item in top_features_data]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot))); ax.set_yticklabels(feature_labels_for_plot_english, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('Importance Score (Weight)', fontsize=12)
    ax.set_title('Feature Importance Ranking', fontsize=16); plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_f1(clf, X, y):
    # ... (与上一版 v7 逻辑一致)
    probs_ok = clf.predict_proba(X)[:, 1]
    best_f1_macro, best_thresh = 0.0, 0.5; best_metrics_at_thresh = {}
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred = (probs_ok >= t).astype(int)
        f1_macro_current = f1_score(y, y_pred, average='macro', zero_division=0)
        if f1_macro_current > best_f1_macro:
            best_f1_macro = f1_macro_current; best_thresh = t
            best_metrics_at_thresh = {
                'accuracy': accuracy_score(y, y_pred),
                'recall_ok': recall_score(y, y_pred, pos_label=1, zero_division=0),
                'recall_ng': recall_score(y, y_pred, pos_label=0, zero_division=0),
                'precision_ok': precision_score(y, y_pred, pos_label=1, zero_division=0),
                'precision_ng': precision_score(y, y_pred, pos_label=0, zero_division=0),
                'f1_ok': f1_score(y, y_pred, pos_label=1, zero_division=0),
                'f1_ng': f1_score(y, y_pred, pos_label=0, zero_division=0), 'threshold': t
            }
    if not best_metrics_at_thresh:
        dummy_preds = (probs_ok >= 0.5).astype(int)
        best_metrics_at_thresh = {
            'accuracy': accuracy_score(y, dummy_preds), 'recall_ok': recall_score(y, dummy_preds, pos_label=1, zero_division=0),
            'recall_ng': recall_score(y, dummy_preds, pos_label=0, zero_division=0), 'precision_ok': precision_score(y, dummy_preds, pos_label=1, zero_division=0),
            'precision_ng': precision_score(y, dummy_preds, pos_label=0, zero_division=0), 'f1_ok': f1_score(y, dummy_preds, pos_label=1, zero_division=0),
            'f1_ng': f1_score(y, dummy_preds, pos_label=0, zero_division=0), 'threshold': 0.5
        }; best_thresh = 0.5
    final_threshold = float(best_metrics_at_thresh.get('threshold', best_thresh))
    final_metrics = {k: float(v) for k, v in best_metrics_at_thresh.items()}
    final_metrics['threshold'] = final_threshold
    return final_threshold, final_metrics

def calculate_adjustment_guaranteed_final_v2(clf, current_values_array, shap_values_array, target_ok_prob, feature_names, initial_is_ng):
    """
    “理论可行”的调整算法V2：确保对NG样本给出调整，并处理NaN的预期贡献。
    """
    original_values_np = np.array(current_values_array, dtype=float).flatten()
    current_adjusted_values = original_values_np.copy()
    
    initial_prob_ok = clf.predict_proba(original_values_np.reshape(1, -1))[0, 1]

    if not initial_is_ng and initial_prob_ok >= target_ok_prob:
        return {}, float(initial_prob_ok), "样本当前已为合格状态且满足目标概率，无需调整。"

    effective_target_prob = target_ok_prob + 0.01 if initial_is_ng else target_ok_prob
    effective_target_prob = min(effective_target_prob, 0.999)

    max_iterations_total = 50 # 减少迭代次数，但每轮尝试更多
    max_abs_change_ratio_overall = 5.0 # 特征值最多变为原始值的 (1 +/- 5) 倍
    min_absolute_change_for_zero_original_overall = 100.0
    min_meaningful_prob_change_per_step = 0.0001
    absolute_min_feature_change_step = 1e-5
    
    cumulative_adjustments_info = {}

    for iteration in range(max_iterations_total):
        current_prob_iter = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]

        if current_prob_iter >= effective_target_prob: break
        
        prob_needed_to_reach_target = effective_target_prob - current_prob_iter
        if prob_needed_to_reach_target <= 0 : break
        
        # 基于初始SHAP值排序特征，并结合当前状态决定调整
        # (feature_idx, initial_shap_value)
        # 我们希望每次迭代都重新评估哪些特征调整最有利
        # 这里简化为，每次迭代都按初始SHAP排序尝试调整
        sorted_shap_indices = sorted(range(len(feature_names)), key=lambda k: -abs(shap_values_array[k]))
        
        made_adjustment_this_iteration = False
        
        for idx in sorted_shap_indices:
            feature_name = feature_names[idx]
            original_val = original_values_np[idx]
            current_val_for_step = current_adjusted_values[idx]

            prob_before_this_feature_adjust = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
            if prob_before_this_feature_adjust >= effective_target_prob: # 内部循环中也检查是否已达标
                made_adjustment_this_iteration = True # 标记一下，以便外部循环可以break
                break

            delta = 0.001 
            temp_for_sensitivity = current_adjusted_values.copy()
            temp_for_sensitivity[idx] += delta
            prob_after_delta = clf.predict_proba(temp_for_sensitivity.reshape(1, -1))[0, 1]
            sensitivity = (prob_after_delta - prob_before_this_feature_adjust) / delta

            if abs(sensitivity) < 1e-8 or math.isnan(sensitivity) or math.isinf(sensitivity):
                app.logger.debug(f"迭代 {iteration+1}, 特征 {feature_name}: 敏感度无效 ({sensitivity:.2e}), 跳过。")
                continue

            # 目标是弥补剩余的概率差距，或者至少达到一个最小提升
            target_prob_gain_for_this_feature_step = max(min_meaningful_prob_change_per_step, (effective_target_prob - prob_before_this_feature_adjust) * 0.2) # 尝试弥补20%的差距
            target_prob_gain_for_this_feature_step = min(target_prob_gain_for_this_feature_step, (effective_target_prob - prob_before_this_feature_adjust) )


            needed_feature_change_for_step = target_prob_gain_for_this_feature_step / sensitivity
            
            # 定义此特征的“理论最大”调整边界 (基于原始值)
            lower_bound_overall = original_val - abs(original_val * max_abs_change_ratio_overall) if original_val != 0 else -min_absolute_change_for_zero_original_overall
            upper_bound_overall = original_val + abs(original_val * max_abs_change_ratio_overall) if original_val != 0 else min_absolute_change_for_zero_original_overall

            # 计算单步调整量，并确保调整后的值不超过全局边界
            potential_new_value = current_val_for_step + needed_feature_change_for_step
            clipped_new_value = np.clip(potential_new_value, lower_bound_overall, upper_bound_overall)
            actual_feature_change_this_step = clipped_new_value - current_val_for_step
            
            if abs(actual_feature_change_this_step) < absolute_min_feature_change_step:
                app.logger.debug(f"迭代 {iteration+1}, 特征 {feature_name}: 计算的调整量过小 ({actual_feature_change_this_step:.2e}), 跳过。")
                continue
            
            current_adjusted_values[idx] += actual_feature_change_this_step
            made_adjustment_this_iteration = True
            
            prob_after_this_feature_adjust = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
            actual_gain_this_step = prob_after_this_feature_adjust - prob_before_this_feature_adjust

            display_gain = 0.0
            if not (math.isnan(actual_gain_this_step) or math.isinf(actual_gain_this_step)):
                display_gain = actual_gain_this_step
            else:
                 app.logger.warning(f"特征 {feature_name} 的预期贡献计算为NaN/inf，显示为0。Sensitivity: {sensitivity}, Change: {actual_feature_change_this_step}")


            cumulative_adjustments_info[feature_name] = {
                'current_value': float(original_val),
                'adjustment': float(current_adjusted_values[idx] - original_val),
                'new_value': float(current_adjusted_values[idx]),
                'expected_gain_this_step': display_gain 
            }
        
        if not made_adjustment_this_iteration: break # 如果一轮迭代没有任何有效调整，则停止
            
    final_prob_ok = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
    
    message = ""
    if final_prob_ok >= effective_target_prob:
        message = f"调整建议已生成。调整后样本预测合格概率为: {final_prob_ok:.3f} (目标: ≥{target_ok_prob:.3f})。"
        if not cumulative_adjustments_info and initial_is_ng :
             message = f"样本虽初判为NG（初始概率{initial_prob_ok:.3f}），但其概率已满足或超过目标 {target_ok_prob:.3f}，无需特定调整。"
    else: # 未达到目标
        message = f"已尝试在极大范围内调整特征（{iteration+1}轮迭代）。"
        if cumulative_adjustments_info:
             message += f"调整后样本预测合格概率为: {final_prob_ok:.3f}，仍未达到目标 {target_ok_prob:.3f}。"
        else: # 没有任何调整被记录，但仍未达标
             message += f"未能找到任何有效的调整组合使样本达到合格标准（当前概率{initial_prob_ok:.3f}，目标{target_ok_prob:.3f}）。"
        message += " 这可能表示模型对此特定NG样本的判定非常顽固，或所有特征的调整都无法有效提升其合格概率。建议人工复核此样本，或评估模型对此类样本的泛化能力。"

    return cumulative_adjustments_info, float(final_prob_ok), message

# --- Routes (与上一版v7一致，除了调用新的调整函数 calculate_adjustment_guaranteed_final_v2) ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ... (与上一版v7 GET部分完全一致)
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
        # ... (与上一版v7 POST部分完全一致)
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
    # ... (与上一版v7 predict部分完全一致)
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
    # ... (与上一版v7 adjust_single部分一致，但调用新的 calculate_adjustment_guaranteed_final_v2)
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
        
        adjustments, final_prob_after_adjustment, message = calculate_adjustment_guaranteed_final_v2(
            clf, current_values_np_array, shap_values_np_array, threshold, features, initial_is_ng
        )
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

