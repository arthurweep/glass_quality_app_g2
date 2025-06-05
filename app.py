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
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14)
    plt.tight_layout(); return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    # ... (与上一版 v9-v10 逻辑一致) ...
    booster = clf.get_booster()
    importance_scores = booster.get_score(importance_type='weight') 
    if not importance_scores:
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else:
            app.logger.warning("无法获取特征重要性分数。")
            return None
    mapped_importance = {}
    if importance_scores and all(isinstance(k, str) and k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        temp_map = {f"f{i}": name for i, name in enumerate(feature_names_original_english)}
        for f_key, score in importance_scores.items():
            if f_key in temp_map: mapped_importance[temp_map[f_key]] = score
            else: mapped_importance[f_key] = score 
        if not mapped_importance and importance_scores : mapped_importance = importance_scores
    elif importance_scores: mapped_importance = importance_scores
    if not mapped_importance: 
        app.logger.warning("最终未能准备好用于绘图的特征重要性数据。")
        return None
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
    # ... (与上一版 v9-v10 逻辑一致) ...
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

def calculate_adjustment_direct_search_v11(clf, current_values_array, shap_values_array, target_ok_prob_original, feature_names, initial_is_ng):
    """
    直接搜索调整算法 V11：更努力地寻找调整方案，简化版爬山。
    """
    original_values_np = np.array(current_values_array, dtype=float).flatten()
    current_adjusted_values = original_values_np.copy()
    
    initial_prob_ok = clf.predict_proba(original_values_np.reshape(1, -1))[0, 1]
    app.logger.info(f"调整(v11)开始：初始NG={initial_is_ng}, 初始OK概率={initial_prob_ok:.4f}, 原始目标={target_ok_prob_original:.4f}")

    if not initial_is_ng and initial_prob_ok >= target_ok_prob_original:
        return {}, float(initial_prob_ok), "样本当前已合格且满足目标概率，无需调整。"

    effective_target_prob = min(target_ok_prob_original + 0.015, 0.999) # 内部调整目标
    app.logger.info(f"内部调整目标概率设为: {effective_target_prob:.4f}")

    # --- 调整参数 ---
    max_total_iterations = 150  # 增加总迭代轮次
    
    # 特征调整的“极宽松”边界
    max_abs_change_ratio_from_original = 10.0 
    abs_change_limit_for_zero_original = 200.0 

    min_significant_prob_gain_for_best_step = 1e-7 # 只有当最佳单步调整的增益大于此值才应用
    exploration_step_ratio = 0.02 # 试探步长为原始值的2%
    min_exploration_abs_step = 0.001 # 最小绝对试探步长

    cumulative_adjustments_made = {}
    
    for iteration_count in range(max_total_iterations):
        prob_at_iter_start = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
        app.logger.info(f"第 {iteration_count + 1} 轮迭代开始，当前OK概率: {prob_at_iter_start:.4f}")

        if prob_at_iter_start >= effective_target_prob:
            app.logger.info(f"已达到目标概率 {effective_target_prob:.4f}。")
            break

        best_move_info = { # 记录本轮迭代中最佳的单步调整
            'feature_idx': -1,
            'applied_change': 0.0,
            'new_prob': prob_at_iter_start, # 初始化为当前概率
            'prob_gain': -1.0 # 初始化为负数
        }

        for feature_idx in range(len(feature_names)):
            original_feature_value = original_values_np[feature_idx]
            current_feature_val_for_test = current_adjusted_values[feature_idx]

            # 定义此特征的全局调整边界
            lower_overall_bound = original_feature_value - abs(original_feature_value * max_abs_change_ratio_from_original) \
                                if original_feature_value != 0 else -abs_change_limit_for_zero_original
            upper_overall_bound = original_feature_value + abs(original_feature_value * max_abs_change_ratio_from_original) \
                                if original_feature_value != 0 else abs_change_limit_for_zero_original

            # 计算试探步长
            step_val = abs(original_feature_value * exploration_step_ratio) if original_feature_value != 0 else min_exploration_abs_step
            step_val = max(step_val, min_exploration_abs_step) # 确保步长不为0

            # 尝试两个方向的调整
            for direction_multiplier in [1, -1]:
                potential_change = direction_multiplier * step_val
                
                # 确保调整后的值在全局边界内
                new_val_candidate = current_feature_val_for_test + potential_change
                new_val_clipped = np.clip(new_val_candidate, lower_overall_bound, upper_overall_bound)
                
                actual_change_to_test = new_val_clipped - current_feature_val_for_test

                if abs(actual_change_to_test) < 1e-9: # 如果实际能调整的量太小（可能已达边界），则忽略此方向
                    continue

                temp_adjusted_values_for_test = current_adjusted_values.copy()
                temp_adjusted_values_for_test[feature_idx] = new_val_clipped # 应用试探调整
                
                prob_after_test_step = clf.predict_proba(temp_adjusted_values_for_test.reshape(1, -1))[0, 1]
                prob_gain_from_test_step = prob_after_test_step - prob_at_iter_start # 相对于本轮迭代开始时的概率

                if prob_gain_from_test_step > best_move_info['prob_gain']:
                    best_move_info['feature_idx'] = feature_idx
                    best_move_info['applied_change'] = actual_change_to_test # 这是实际应用到 current_adjusted_values 的改变量
                    best_move_info['new_prob'] = prob_after_test_step
                    best_move_info['prob_gain'] = prob_gain_from_test_step
        
        # 检查本轮迭代是否找到了有效的提升步骤
        if best_move_info['prob_gain'] > min_significant_prob_gain_for_best_step:
            best_feature_idx = best_move_info['feature_idx']
            change_to_apply = best_move_info['applied_change']
            feature_name_adjusted = feature_names[best_feature_idx]
            
            current_adjusted_values[best_feature_idx] += change_to_apply # 正式应用最佳调整
            
            app.logger.info(f"  应用调整: 特征 '{feature_name_adjusted}' ({FIELD_LABELS.get(feature_name_adjusted, feature_name_adjusted)}) "
                            f"改变 {change_to_apply:+.4f}, 新值为 {current_adjusted_values[best_feature_idx]:.4f}."
                            f" OK概率从 {prob_at_iter_start:.4f} 变为 {best_move_info['new_prob']:.4f} (增益: {best_move_info['prob_gain']:.4f})")
            
            cumulative_adjustments_made[feature_name_adjusted] = {
                'current_value': float(original_values_np[best_feature_idx]),
                'adjustment': float(current_adjusted_values[best_feature_idx] - original_values_np[best_feature_idx]),
                'new_value': float(current_adjusted_values[best_feature_idx]),
                'expected_gain_this_step': best_move_info['prob_gain'] 
            }
        else: # 如果一整轮都找不到任何显著提升
            app.logger.info(f"第 {iteration_count + 1} 轮迭代结束：未能找到任何能带来显著概率提升 ({min_significant_prob_gain_for_best_step:.2e}) 的单步调整，终止总迭代。")
            break
            
    final_prob_ok_overall = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
    app.logger.info(f"调整(v11)结束：最终OK概率={final_prob_ok_overall:.4f} (迭代次数: {iteration_count + 1})")
    
    message_to_user = ""
    # ... (消息逻辑与 v10 calculate_adjustment_persistent_effort 一致) ...
    if final_prob_ok_overall >= effective_target_prob:
        message_to_user = f"调整建议已生成。调整后样本预测合格概率为: {final_prob_ok_overall:.4f} (目标: ≥{target_ok_prob_original:.3f})。"
        if not cumulative_adjustments_made and initial_is_ng : 
             message_to_user = f"样本虽初判为NG（初始概率 {initial_prob_ok:.4f}），但其概率已满足或超过调整目标 {effective_target_prob:.4f}，无需特定参数调整。"
    else: 
        message_to_user = f"已尝试在极大范围内调整特征（共 {iteration_count + 1} 轮迭代）。"
        if cumulative_adjustments_made:
             message_to_user += f"调整后样本预测合格概率为: {final_prob_ok_overall:.4f}，但仍未达到目标 {target_ok_prob_original:.3f}。"
        else:
             message_to_user += f"未能找到任何有效的调整组合使样本达到合格标准（当前实际概率 {initial_prob_ok:.4f}，目标 {target_ok_prob_original:.3f}）。"
        message_to_user += " 这可能表示：1. 模型对此特定NG样本的判定非常“顽固”（即其特征值落入了一个很难通过调整离开的决策区域，这是树模型的特性）。2. 所有特征的调整对于提升合格概率的效果都非常有限。建议：请人工复核此样本的实际情况，或考虑在训练数据中补充更多此类“边界”样本后重新训练模型，以增强模型对此类情况的辨别和调整指导能力。"

    return cumulative_adjustments_made, float(final_prob_ok_overall), message_to_user

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ... (与上一版 v9-v10 GET部分完全一致) ...
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
        # ... (与上一版 v9-v10 POST部分完全一致) ...
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
            X = X.fillna(X.mean()) # 简单填充均值
            y = pd.to_numeric(df["OK_NG"], errors='coerce').fillna(0).astype(int)
            features = X.columns.tolist()
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, 
                colsample_bytree=0.8, gamma=0.1, random_state=42, use_label_encoder=False,
                eval_metric='logloss' # 明确指定评估指标以避免警告
            )
            # 对类别不平衡进行权重处理，NG样本权重更高
            sample_weights = compute_sample_weight(class_weight={0:2.0, 1:1.0}, y=y) 
            clf.fit(X, y, sample_weight=sample_weights)
            best_threshold, calculated_metrics = find_best_threshold_f1(clf, X, y)
            model_params = clf.get_params()
            final_model_metrics = {
                'trees': int(model_params['n_estimators']), 'depth': int(model_params['max_depth']),
                'lr': float(model_params['learning_rate']), **calculated_metrics
            }
            model_cache.update({
                'show_results': True, 'features': features,
                'defaults': {k: float(v) for k, v in X.mean().to_dict().items()}, # 确保是原生float
                'clf': clf, 'X_train_df': X.copy(), 'metrics': final_model_metrics,
                'feature_plot': generate_feature_importance_plot(clf, features)
            })
        except Exception as e:
            model_cache['error'] = f"处理文件时出错: {str(e)}"
            app.logger.error(f"文件处理或模型训练出错: {e}", exc_info=True)
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    # ... (与上一版 v9-v10 predict部分完全一致) ...
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
        explainer = shap.Explainer(clf, background_data_df) # 使用训练数据作为背景
        shap_explanation_obj = explainer(df_input) # 计算单个样本的SHAP值
        shap_values_for_output = shap_explanation_obj.values[0] # 通常是二维，取第一行
        base_val_for_waterfall = explainer.expected_value
        if isinstance(base_val_for_waterfall, (np.ndarray, list)): # expected_value可能是数组
             base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 and clf.n_classes_ == 2 else base_val_for_waterfall[0]
        base_val_for_waterfall = float(base_val_for_waterfall)
        # 为Waterfall图创建Explanation对象，确保使用英文特征名
        shap_explanation_for_plot = shap.Explanation(
            values=shap_values_for_output.astype(float), 
            base_values=base_val_for_waterfall,
            data=df_input.iloc[0].values.astype(float), 
            feature_names=features # 传递原始英文特征名列表
        )
        waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_plot)
        response = {
            'prob': float(round(prob_ok, 3)), 'threshold': float(round(threshold, 3)), 'is_ng': is_ng,
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output], # 确保是Python float列表
            'metrics': model_cache['metrics'], 'waterfall': waterfall_plot_base64,
            'input_data': input_data_dict, 'initial_is_ng_for_adjustment': is_ng
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500

@app.route('/adjust_single', methods=['POST'])
def adjust_single():
    # ... (与上一版v9-v10 adjust_single部分一致，但调用新的 calculate_adjustment_direct_search_v11) ...
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
        
        adjustments, final_prob_after_adjustment, message = calculate_adjustment_direct_search_v11(
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
    port = int(os.environ.get("PORT", 10000)) # Render通常会设置PORT环境变量
    app.run(host='0.0.0.0', port=port, debug=False)
