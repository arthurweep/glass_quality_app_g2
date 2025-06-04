import base64
import io
import os
import logging
import math # 导入math模块以使用 isnan 和 isinf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 后端模式，不显示GUI
import matplotlib.pyplot as plt
# 确保图表使用英文标签，避免字体问题
plt.rcParams['axes.unicode_minus'] = False

import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24) # 用于session等
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO) # 设置日志级别

# 字段英文名到中文名的映射，仅用于HTML模板中的文本显示
FIELD_LABELS = {
    "F_cut_act": "刀头实际压力", "v_cut_act": "切割实际速度", "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度", "F_wheel_act": "磨轮压紧力", "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}
model_cache = {} # 用于存储训练好的模型和相关数据

def fig_to_base64(fig):
    """将Matplotlib图像对象转换为Base64编码的字符串，用于在HTML中嵌入显示"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') # 保存到内存缓冲区
    plt.close(fig) # 关闭图像，释放资源
    buf.seek(0) # 重置缓冲区指针到开头
    return base64.b64encode(buf.getvalue()).decode('utf-8') # 编码并转为字符串

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    """生成SHAP Waterfall图，使用英文标签"""
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14) # 英文标题
    plt.tight_layout() # 自动调整布局
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    """生成特征重要性图，使用英文标签，并更鲁棒地处理特征名"""
    booster = clf.get_booster()
    importance_scores = booster.get_score(importance_type='weight') # 获取原始重要性分数
    
    if not importance_scores: # 如果XGBoost原生接口未返回
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            # 尝试Scikit-learn接口的feature_importances_
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else:
            app.logger.warning("无法获取特征重要性分数。")
            return None # 确实无法获取

    # 检查importance_scores的键是否是f0, f1...形式，如果是，尝试映射回原始英文名
    # 这一步很重要，因为booster.get_score()有时不直接使用原始特征名
    mapped_importance = {}
    if importance_scores and all(isinstance(k, str) and k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        app.logger.info("特征重要性键为f0,f1...形式，尝试映射到原始特征名。")
        temp_map = {f"f{i}": name for i, name in enumerate(feature_names_original_english)}
        for f_key, score in importance_scores.items():
            if f_key in temp_map:
                mapped_importance[temp_map[f_key]] = score
            else: # 如果有无法映射的f-score，也保留，但记录下来
                app.logger.warning(f"无法映射特征重要性键 {f_key} 到原始特征名。")
                mapped_importance[f_key] = score 
        if not mapped_importance and importance_scores : # 如果映射后为空但原始分数存在
             app.logger.warning("映射f0,f1...到原始特征名失败，图表可能使用f0,f1...标签。")
             mapped_importance = importance_scores # 回退，至少显示点什么
    elif importance_scores: # 假设键已经是原始特征名，或不需要映射
        mapped_importance = importance_scores
        
    if not mapped_importance: 
        app.logger.warning("最终未能准备好用于绘图的特征重要性数据。")
        return None

    sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10)
    top_features_data = sorted_importance[:num_features_to_display]
    
    feature_labels_for_plot_english = [item[0] for item in top_features_data] # 特征名（应为英文）
    scores_for_plot = [float(item[1]) for item in top_features_data]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot_english, fontsize=9) # 设置Y轴为英文标签
    ax.invert_yaxis() 
    ax.set_xlabel('Importance Score (Weight)', fontsize=12)
    ax.set_title('Feature Importance Ranking', fontsize=16)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_f1(clf, X, y):
    """使用F1分数最大化原则自动选择最优分类阈值 (此函数逻辑不变)"""
    # ... (与上一版 v8 逻辑一致)
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

def calculate_adjustment_robust_iterative(clf, current_values_array, shap_values_array, target_ok_prob, feature_names, initial_is_ng):
    """
    鲁棒的迭代调整算法，更努力地为NG样本寻找调整方案。
    处理预期贡献中的NaN问题。
    """
    original_values_np = np.array(current_values_array, dtype=float).flatten()
    current_adjusted_values = original_values_np.copy() # 当前迭代中被修改的特征值副本
    
    initial_prob_ok = clf.predict_proba(original_values_np.reshape(1, -1))[0, 1]

    # 如果样本最初就不是NG，并且其初始概率已经合格，则无需调整
    if not initial_is_ng and initial_prob_ok >= target_ok_prob:
        return {}, float(initial_prob_ok), "样本当前已为合格状态且满足目标概率，无需进行调整。"

    # --- 调整参数 ---
    # 对于NG样本，目标是略高于阈值，例如高出0.015，以确保稳定合格
    effective_target_prob = target_ok_prob + 0.015 if initial_is_ng else target_ok_prob
    effective_target_prob = min(effective_target_prob, 0.999) # 概率上限

    max_total_iterations = 75  # 总迭代次数上限，给算法更多机会
    max_feature_loops_per_iteration = 2 # 在一轮大迭代中，完整遍历特征列表的次数
    
    # 特征调整的“极宽松”边界，基于原始值的倍数或一个较大的绝对值
    max_abs_change_ratio_overall = 7.0 # 例如，允许特征值最大变为原始值的 (1 +/- 7) 倍
    min_absolute_change_for_zero_original_overall = 150.0 # 如果原始值为0，允许的绝对变化量

    min_meaningful_prob_gain_per_step = 1e-5 # 单步调整期望的最小有效概率提升
    absolute_min_feature_change_value_step = 1e-6 # 忽略绝对值小于此的特征调整量
    
    cumulative_adjustments_dict = {} # 存储最终返回给用户的调整方案

    for iteration_num in range(max_total_iterations):
        current_prob_at_iteration_start = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]

        if current_prob_at_iteration_start >= effective_target_prob:
            app.logger.info(f"迭代 {iteration_num+1}: 当前概率 {current_prob_at_iteration_start:.4f} 已达到目标 {effective_target_prob:.4f}。")
            break # 已达到目标

        prob_needed_to_reach_target_iter = effective_target_prob - current_prob_at_iteration_start
        if prob_needed_to_reach_target_iter <= 0 : # 以防万一
            app.logger.info(f"迭代 {iteration_num+1}: 无需更多概率提升。")
            break
        
        made_any_adjustment_in_this_iteration = False
        
        # 在一轮大迭代中，可以多次遍历所有特征
        for feature_loop_count in range(max_feature_loops_per_iteration):
            if clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1] >= effective_target_prob: break # 内循环也检查

            # 每次都根据当前状态重新计算敏感度并排序
            sensitivities_current = []
            delta_sens = 1e-3
            prob_base_for_sens = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
            for i in range(len(feature_names)):
                temp_vals_sens = current_adjusted_values.copy()
                temp_vals_sens[i] += delta_sens
                prob_after_delta_sens = clf.predict_proba(temp_vals_sens.reshape(1, -1))[0, 1]
                sens = (prob_after_delta_sens - prob_base_for_sens) / delta_sens
                sensitivities_current.append({'idx': i, 'sens': sens, 'shap': shap_values_array[i]})

            # 优先调整：高SHAP绝对值 + 高敏感度绝对值 + 方向正确
            sensitivities_current.sort(key=lambda x: (-abs(x['shap']), -abs(x['sens'])))
            
            made_adjustment_in_feature_loop = False

            for sens_info in sensitivities_current:
                idx = sens_info['idx']
                sensitivity = sens_info['sens']
                # initial_shap_val = sens_info['shap'] # 可以用来辅助判断调整方向，但敏感度更直接
                feature_name = feature_names[idx]
                original_val = original_values_np[idx]
                current_val_before_this_step = current_adjusted_values[idx]

                prob_at_step_start = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
                if prob_at_step_start >= effective_target_prob: break # 再次检查

                if abs(sensitivity) < 1e-9 or math.isnan(sensitivity) or math.isinf(sensitivity):
                    # app.logger.debug(f"特征 {feature_name} (迭代 {iteration_num+1}, 内循环 {feature_loop_count+1}): 敏感度无效 ({sensitivity:.3e}), 跳过。")
                    continue

                # 目标：在这一小步中，尝试弥补一部分剩余的概率差距
                # 可以更积极一点，比如尝试弥补剩余差距的10%到30%
                target_prob_gain_for_this_step = max(min_meaningful_prob_gain_per_step, (effective_target_prob - prob_at_step_start) * 0.25)
                target_prob_gain_for_this_step = min(target_prob_gain_for_this_step, (effective_target_prob - prob_at_step_start) ) # 不要超过总需求


                needed_feature_value_change = target_prob_gain_for_this_step / sensitivity
                
                # 定义此特征的“理论最大”调整边界 (基于原始值)
                lower_bound_overall = original_val - abs(original_val * max_abs_change_ratio_overall) if original_val != 0 else -min_absolute_change_for_zero_original_overall
                upper_bound_overall = original_val + abs(original_val * max_abs_change_ratio_overall) if original_val != 0 else min_absolute_change_for_zero_original_overall

                # 计算单步调整量，并确保调整后的值不超过全局边界
                potential_new_value_after_step = current_val_before_this_step + needed_feature_value_change
                clipped_new_value_this_step = np.clip(potential_new_value_after_step, lower_bound_overall, upper_bound_overall)
                actual_feature_change_this_step = clipped_new_value_this_step - current_val_before_this_step
                
                if abs(actual_feature_change_this_step) < absolute_min_feature_change_value_step:
                    # app.logger.debug(f"特征 {feature_name}: 计算的调整量过小 ({actual_feature_change_this_step:.3e}), 跳过。")
                    continue
                
                current_adjusted_values[idx] += actual_feature_change_this_step # 应用调整
                made_adjustment_in_feature_loop = True
                made_any_adjustment_in_this_iteration = True
                
                prob_after_this_step = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
                actual_prob_gain_this_step = prob_after_this_step - prob_at_step_start

                display_gain = 0.0
                if not (math.isnan(actual_prob_gain_this_step) or math.isinf(actual_prob_gain_this_step)):
                    display_gain = actual_prob_gain_this_step
                else:
                     app.logger.warning(f"特征 {feature_name} 的预期贡献计算为NaN/inf，显示为0。Sensitivity: {sensitivity}, Change: {actual_feature_change_this_step}")

                # 更新累积调整信息（相对于最原始的值）
                cumulative_adjustments_dict[feature_name] = {
                    'current_value': float(original_values_np[idx]), # 样本的原始值
                    'adjustment': float(current_adjusted_values[idx] - original_values_np[idx]), # 从原始值算起的总调整量
                    'new_value': float(current_adjusted_values[idx]), # 调整后的新值
                    'expected_gain_this_step': display_gain # 这是这一小步的实际概率增益
                }
            
            if not made_adjustment_in_feature_loop and feature_loop_count > 0: # 如果一轮特征遍历没有任何调整
                app.logger.info(f"迭代 {iteration_num+1}, 内循环 {feature_loop_count+1}: 未做任何调整，可能已达局部最优或调整极限。")
                break # 跳出内层特征循环
        
        if not made_any_adjustment_in_this_iteration: # 如果一轮大迭代没有任何有效调整
            app.logger.info(f"迭代 {iteration_num+1}: 未做任何调整，终止迭代。")
            break 
            
    final_prob_ok_after_all = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
    
    message = "" # 初始化消息
    if final_prob_ok_after_all >= effective_target_prob:
        message = f"调整建议已生成。调整后样本预测合格概率为: {final_prob_ok_after_all:.4f} (目标: ≥{target_ok_prob:.3f})。"
        if not cumulative_adjustments_dict and initial_is_ng : 
             # 如果最初是NG，但没做任何调整就已达标（例如，initial_prob_ok本身就高于effective_target_prob）
             message = f"样本虽初判为NG（初始概率 {initial_prob_ok:.4f}），但其概率已满足或超过调整目标 {effective_target_prob:.4f}，无需特定参数调整。"
    else: # 未达到目标
        message = f"已尝试在极大范围内调整特征（共 {iteration_num+1} 轮迭代）。"
        if cumulative_adjustments_dict: # 如果做了一些调整
             message += f"调整后样本预测合格概率为: {final_prob_ok_after_all:.4f}，但仍未达到目标 {target_ok_prob:.3f}。"
        else: # 如果没有任何调整被记录（例如所有特征都不敏感），但仍未达标
             message += f"未能找到任何有效的调整组合使样本达到合格标准（当前实际概率 {initial_prob_ok:.4f}，目标 {target_ok_prob:.3f}）。"
        message += " 这可能表示：1. 模型对此特定NG样本的判定非常“顽固”（即其特征值落入了一个很难通过调整离开的决策区域）。2. 所有特征的调整对于提升合格概率的效果都非常有限。建议：请人工复核此样本的实际情况，或考虑在训练数据中补充更多此类“边界”样本后重新训练模型，以增强模型对此类情况的辨别和调整指导能力。"

    return cumulative_adjustments_dict, float(final_prob_ok_after_all), message

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ... (与上一版 v8 GET部分完全一致)
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
        # ... (与上一版 v8 POST部分完全一致)
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
            sample_weights = compute_sample_weight(class_weight={0:2.0, 1:1}, y=y) # NG样本权重是OK的2倍
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
    # ... (与上一版 v8 predict部分完全一致)
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
            data=df_input.iloc[0].values.astype(float), feature_names=features # 确保是英文特征名
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
    # ... (与上一版v8 adjust_single部分一致，但调用新的 calculate_adjustment_robust_iterative)
    global model_cache
    if 'clf' not in model_cache: return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']; features = model_cache['features']
        threshold = model_cache['metrics']['threshold'] # 这是判定的原始阈值
        json_data = request.get_json()
        if not json_data: return jsonify({'error': '请求体为空或不是有效的JSON。'}), 400
        input_data_dict = json_data.get('input_data')
        shap_values_list = json_data.get('shap_values')
        initial_is_ng = json_data.get('initial_is_ng_for_adjustment', True)
        if not input_data_dict or not isinstance(input_data_dict, dict): return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features): return jsonify({'error': '缺少或无效的 shap_values。'}), 400
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float)
        shap_values_np_array = np.array(shap_values_list, dtype=float)
        
        # 调用新的、更鲁棒的迭代调整函数
        # 注意：target_ok_prob 参数使用的是模型的原始判定阈值
        adjustments, final_prob_after_adjustment, message = calculate_adjustment_robust_iterative(
            clf, current_values_np_array, shap_values_np_array, threshold, features, initial_is_ng
        )
        return jsonify({
            'adjustments': adjustments, # 字典，键是英文特征名
            'final_prob_after_adjustment': float(final_prob_after_adjustment),
            'message': message # 返回详细的执行消息
        })
    except Exception as e:
        app.logger.error(f"优化建议接口 (/adjust_single) 出错: {e}", exc_info=True)
        return jsonify({'error': f'优化建议过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # 生产环境通常 debug=False
