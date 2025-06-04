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

FIELD_LABELS = {
    "F_cut_act": "刀头实际压力", "v_cut_act": "切割实际速度", "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度", "F_wheel_act": "磨轮压紧力", "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}
model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14)
    plt.tight_layout(); return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    # ... (与上一版 v6 逻辑一致，确保图表标签为英文)
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
    # ... (与上一版 v6 逻辑一致)
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

def calculate_adjustment_guaranteed(clf, current_values_array, shap_values_array, target_ok_prob, feature_names, initial_is_ng):
    """
    “理论可行”的调整算法：迭代调整，直至达到目标概率，允许较大调整幅度。
    """
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    adjusted_values_iter = current_values_np.copy() # 用于迭代调整的值
    
    # 如果最初不是NG，且当前概率已经合格，则不进行调整
    initial_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    if not initial_is_ng and initial_prob_ok >= target_ok_prob:
        return {}, float(initial_prob_ok), "样本当前已判定为合格且概率高于或等于目标阈值，无需调整。"

    # --- 调整参数 ---
    max_iterations = 20  # 增加迭代次数，给算法更多机会达到目标
    max_abs_change_ratio_per_step = 0.20 # 每一步允许特征值相对原始值变化的最大比例（例如20%）
                                        # 这个限制是针对“单步”的，累积起来可以超过原始值的100%
    min_prob_improvement_per_iteration = 0.005 # 每轮迭代期望的最小概率提升
    absolute_min_feature_change = 1e-5 # 忽略绝对值小于此的特征调整量

    # 按SHAP值绝对大小对特征排序，优先调整影响大的
    # shap_values_array 和 feature_names 应该有相同的顺序
    sorted_shap_indices = sorted(range(len(feature_names)), key=lambda k: -abs(shap_values_array[k]))

    cumulative_adjustments_info = {} # 存储每个特征的累积调整信息

    for iteration in range(max_iterations):
        current_iter_prob_ok = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]

        if current_iter_prob_ok >= target_ok_prob: # 已达到目标
            break
        
        prob_needed_to_reach_target = target_ok_prob - current_iter_prob_ok
        if prob_needed_to_reach_target <= 0: # 以防万一
            break

        made_change_in_this_iteration = False
        
        # 在这一轮迭代中，尝试调整所有特征（按SHAP重要性顺序）
        for original_idx in sorted_shap_indices:
            feature_name = feature_names[original_idx]
            
            # 基于当前已调整的值计算敏感度
            prob_before_this_feature_adjust = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
            
            delta = 0.001 
            temp_for_sensitivity = adjusted_values_iter.copy()
            temp_for_sensitivity[original_idx] += delta
            prob_after_delta = clf.predict_proba(temp_for_sensitivity.reshape(1, -1))[0, 1]
            sensitivity = (prob_after_delta - prob_before_this_feature_adjust) / delta

            if abs(sensitivity) < 1e-8: # 特征不敏感
                continue

            # 理论上，为了弥补全部剩余差距，此特征需要调整的量
            # 我们不一次性调整这么多，而是设定一个单步目标提升量
            target_prob_gain_this_step = max(min_prob_improvement_per_iteration, prob_needed_to_reach_target / (len(sorted_shap_indices) - sorted_shap_indices.index(original_idx) +1e-5) ) # 分摊
            target_prob_gain_this_step = min(target_prob_gain_this_step, prob_needed_to_reach_target) # 不要超过总需求

            needed_feature_change_for_step = target_prob_gain_this_step / sensitivity
            
            # 单步调整幅度限制（基于原始值）
            original_val = current_values_np[original_idx]
            max_step_change_abs = abs(original_val * max_abs_change_ratio_per_step) if original_val != 0 else 0.10 # 如果原始值为0，允许调整0.1
            
            actual_feature_change_this_step = float(np.clip(needed_feature_change_for_step, -max_step_change_abs, max_step_change_abs))

            if abs(actual_feature_change_this_step) < absolute_min_feature_change:
                continue
            
            # 应用调整
            adjusted_values_iter[original_idx] += actual_feature_change_this_step
            made_change_in_this_iteration = True
            
            # 重新计算调整后的概率，看这一步实际带来了多少提升
            prob_after_this_feature_adjust = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
            actual_gain_this_step = prob_after_this_feature_adjust - prob_before_this_feature_adjust

            # 更新累积调整信息
            cumulative_adjustments_info[feature_name] = {
                'current_value': float(original_val),
                'adjustment': float(adjusted_values_iter[original_idx] - original_val), # 总调整量
                'new_value': float(adjusted_values_iter[original_idx]),
                'expected_gain_this_step': actual_gain_this_step # 记录这一步的实际增益
            }
            
            # 如果调整后已经达标，则可以提前结束本轮对其他特征的调整
            if prob_after_this_feature_adjust >= target_ok_prob:
                break 
        
        if not made_change_in_this_iteration and iteration > 0: # 如果一轮下来没任何改变
            break
            
    final_prob_after_all_iterations = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
    
    message = None
    if not cumulative_adjustments_info and initial_is_ng:
        if initial_prob_ok >= target_ok_prob:
             message = "样本虽初判为NG，但其当前概率已达合格标准，无需调整。"
        else:
            message = "关键特征对模型输出不敏感或已达调整尝试极限，无法给出有效的调整建议以达到合格标准。"
    elif cumulative_adjustments_info and initial_is_ng:
        if final_prob_after_all_iterations < target_ok_prob:
            message = f"已尽力调整。调整后预测OK概率为 {final_prob_after_all_iterations:.3f}，仍低于目标阈值 {target_ok_prob:.3f}。可能需要检查模型或放宽调整限制。"
        else:
            message = f"已给出调整建议。调整后预测OK概率为 {final_prob_after_all_iterations:.3f}，已达到或超过合格阈值。"
    elif not initial_is_ng and cumulative_adjustments_info : # 如果本来是OK，但算法还是调整了（不应该发生，除非目标被设高了）
        message = f"对已合格样本尝试了调整。最终概率 {final_prob_after_all_iterations:.3f}."


    return cumulative_adjustments_info, float(final_prob_after_all_iterations), message


# --- Routes (与上一版v6一致，除了调用新的调整函数) ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ... (与上一版v6 GET部分完全一致)
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
        # ... (与上一版v6 POST部分完全一致)
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
    # ... (与上一版v6 predict部分完全一致，确保传递 initial_is_ng_for_adjustment)
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
    # ... (与上一版v6 adjust_single部分一致，但调用新的 calculate_adjustment_guaranteed)
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
        
        # 调用新的“保证调整”函数
        adjustments, final_prob_after_adjustment, message = calculate_adjustment_guaranteed(
            clf, current_values_np_array, shap_values_np_array, threshold, features, initial_is_ng
        )
        return jsonify({
            'adjustments': adjustments, # 键是英文特征名
            'final_prob_after_adjustment': float(final_prob_after_adjustment),
            'message': message # 返回附加消息
        })
    except Exception as e:
        app.logger.error(f"优化建议接口 (/adjust_single) 出错: {e}", exc_info=True)
        return jsonify({'error': f'优化建议过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
