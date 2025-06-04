import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 使用Matplotlib默认英文字体，确保兼容性
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24) # 用于session等
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# 字段英文名到中文名的映射，仅用于HTML模板中的文本显示，图表将使用英文
FIELD_LABELS = {
    "F_cut_act": "刀头实际压力",
    "v_cut_act": "切割实际速度",
    "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度",
    "F_wheel_act": "磨轮压紧力",
    "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}
model_cache = {} # 用于存储训练好的模型和相关数据

def fig_to_base64(fig):
    """将Matplotlib图像对象转换为Base64编码的字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    """生成SHAP Waterfall图，使用英文标签"""
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14) # 英文标题
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    """生成特征重要性图，使用英文标签"""
    booster = clf.get_booster()
    # 确保booster的feature_names设置正确，以便get_score能正确返回基于名称的重要性
    # 如果模型训练时使用的是DataFrame，XGBoost通常会自动处理。
    # 如果不是，或者get_score返回的是f0, f1...，则需要手动映射。
    # booster.feature_names = feature_names_original_english # 一般不需要显式设置，除非遇到问题

    importance_scores = booster.get_score(importance_type='weight') # 获取特征重要性
    if not importance_scores:
        app.logger.warning("无法从XGBoost booster获取特征重要性分数。")
        # 尝试使用Scikit-learn接口的feature_importances_ (如果clf是XGBClassifier包装器)
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else:
            return None # 确实无法获取

    # 检查importance_scores的键是否是f0, f1...形式，如果是，尝试映射
    mapped_importance = {}
    if all(k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        app.logger.info("特征重要性键为f0,f1...形式，尝试映射到原始特征名。")
        # 假设feature_names_original_english的顺序与模型内部特征顺序一致
        for i, original_name in enumerate(feature_names_original_english):
            f_key = f"f{i}"
            if f_key in importance_scores:
                mapped_importance[original_name] = importance_scores[f_key]
        if not mapped_importance and importance_scores: # 如果映射后为空但原始分数存在
             app.logger.warning("映射f0,f1...到原始特征名失败，图表可能使用f0,f1...标签。")
             mapped_importance = importance_scores # 回退到使用f0, f1...
    else: # 假设键已经是原始特征名
        mapped_importance = importance_scores
        
    if not mapped_importance: return None # 如果最终还是没有重要性分数

    sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10) # 最多显示10个特征
    top_features_data = sorted_importance[:num_features_to_display]
    
    feature_labels_for_plot_english = [item[0] for item in top_features_data] # 特征名（英文）
    scores_for_plot = [float(item[1]) for item in top_features_data] # 分数
    
    fig, ax = plt.subplots(figsize=(10, 8)) # 调整图表大小以更好显示标签
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot_english, fontsize=9) # 设置Y轴为英文标签
    ax.invert_yaxis()  # 重要性高的在上面
    ax.set_xlabel('Importance Score (Weight)', fontsize=12) # X轴英文标签
    ax.set_title('Feature Importance Ranking', fontsize=16) # 图表英文标题
    plt.tight_layout() # 自动调整布局防止标签重叠
    return fig_to_base64(fig)

def find_best_threshold_f1(clf, X, y):
    """使用F1分数最大化原则自动选择最优分类阈值"""
    # ... (此函数与上一版本v5一致，逻辑已较优)
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

def calculate_precise_adjustment_aggressive(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names, initial_is_ng):
    """更积极的智能优化建议算法，确保为NG样本提供调整方案"""
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()
    
    initial_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]

    # 如果样本最初就不是NG，并且其初始概率已经合格，则无需调整
    if not initial_is_ng and initial_prob_ok >= threshold_ok_prob:
        return {}, float(initial_prob_ok), "Sample is already predicted as OK and meets/exceeds threshold."

    # --- 调整参数 ---
    max_iterations = 5  # 增加迭代次数
    max_features_to_try_adjusting_in_total = len(feature_names) # 尝试调整所有特征
    max_abs_change_ratio = 0.60 # 单个特征值允许从其原始值变化的最大相对比例 (例如60%)
    min_meaningful_adjustment_abs = 1e-5 # 忽略绝对值小于此的特征调整量
    prob_consolidation_target_if_ok = 0.03 # 如果已达标但想巩固，目标再提升这么多概率
    target_prob_for_ng = threshold_ok_prob + 0.01 # 对于NG样本，目标是略高于阈值

    # --- 迭代调整初始化 ---
    cumulative_adjustments_dict = {} # 存储每个特征的累积调整信息 {英文特征名: {'current_value', 'adjustment', 'new_value', 'expected_gain_this_step'}}
    adjusted_values_iter = current_values_np.copy() # 这个数组的值会在迭代中被修改
    
    # 按SHAP值绝对大小对特征排序，优先调整影响大的
    sorted_shap_indices = sorted(range(len(feature_names)), key=lambda k: -abs(shap_values_np[k]))

    for iteration in range(max_iterations):
        prob_before_this_iteration = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
        
        # 定义本轮迭代的目标概率
        current_target_prob = target_prob_for_ng
        if prob_before_this_iteration >= threshold_ok_prob and initial_is_ng: # 如果是NG变OK，且已达标
            current_target_prob = threshold_ok_prob + prob_consolidation_target_if_ok # 设定巩固目标

        # 如果已达到最终目标（对于NG样本，即 target_prob_for_ng），则停止
        if prob_before_this_iteration >= current_target_prob :
            break 
        
        made_change_in_this_iteration = False
        features_adjusted_this_iteration_count = 0
        
        for original_idx in sorted_shap_indices:
            if features_adjusted_this_iteration_count >= 3 and iteration > 0 : # 每轮后续迭代中，重点调整前3个
                 break
            if len(cumulative_adjustments_dict) >= max_features_to_try_adjusting_in_total and feature_names[original_idx] not in cumulative_adjustments_dict:
                continue

            feature_name = feature_names[original_idx]
            
            # 计算当前状态下，此特征的敏感度
            prob_at_feature_eval_start = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1] # 以最新的概率为基准
            delta = 0.001 
            temp_for_sensitivity_iter = adjusted_values_iter.copy()
            temp_for_sensitivity_iter[original_idx] += delta
            prob_after_delta_iter = clf.predict_proba(temp_for_sensitivity_iter.reshape(1, -1))[0, 1]
            sensitivity = (prob_after_delta_iter - prob_at_feature_eval_start) / delta

            if abs(sensitivity) < 1e-8: # 如果特征非常不敏感，跳过
                continue

            # 计算当前还需要提升多少概率才能达到本轮目标
            effective_required_boost_for_step = float(current_target_prob - prob_at_feature_eval_start)
            
            if effective_required_boost_for_step <= 0: # 如果已达到本轮迭代目标
                continue

            needed_feature_value_change_step = effective_required_boost_for_step / sensitivity
            
            original_feature_val = current_values_np[original_idx] 
            max_change_val_abs = abs(original_feature_val * max_abs_change_ratio) if original_feature_val != 0 else 0.30 # 如果原始值为0，允许调整0.3
            
            # 限制单步调整量
            actual_feature_change_this_step = float(np.clip(needed_feature_value_change_step, -max_change_val_abs, max_change_val_abs))
            
            # 检查调整后的新值是否会超出基于原始值的总体调整限制
            current_total_adjustment_on_feature = (adjusted_values_iter[original_idx] - original_feature_val)
            if current_total_adjustment_on_feature + actual_feature_change_this_step > max_change_val_abs:
                actual_feature_change_this_step = max_change_val_abs - current_total_adjustment_on_feature
            elif current_total_adjustment_on_feature + actual_feature_change_this_step < -max_change_val_abs:
                actual_feature_change_this_step = -max_change_val_abs - current_total_adjustment_on_feature
            
            if abs(actual_feature_change_this_step) < min_meaningful_adjustment_abs:
                continue

            # 应用这一步的调整
            adjusted_values_iter[original_idx] += actual_feature_change_this_step
            expected_gain_this_step = float(sensitivity * actual_feature_change_this_step)

            # 更新累积调整信息
            cumulative_adjustments_dict[feature_name] = {
                'current_value': float(original_feature_val),
                'adjustment': float(adjusted_values_iter[original_idx] - original_feature_val),
                'new_value': float(adjusted_values_iter[original_idx]),
                'expected_gain_this_step': expected_gain_this_step # 这个是单步的预估，非累积
            }
            made_change_in_this_iteration = True
            features_adjusted_this_iteration_count +=1
        
        if not made_change_in_this_iteration and iteration > 0 :
            break # 如果一轮迭代下来没有任何特征能被有效调整
            
    final_prob_after_all_adjustments = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
    
    message = None
    if not cumulative_adjustments_dict and initial_is_ng:
        if initial_prob_ok >= threshold_ok_prob:
             message = "Sample was initially NG, but its current probability already meets/exceeds threshold. No specific adjustments proposed."
        elif abs(initial_prob_ok - threshold_ok_prob) < 0.03:
            message = "Sample is very close to the OK threshold. The algorithm could not find further significant improvements with current constraints."
        else:
            message = "Could not compute effective adjustments. Features might be insensitive, at adjustment limits, or model is firm on this NG sample. Consider reviewing data or model if adjustments are critical."
    elif cumulative_adjustments_dict and initial_is_ng:
        if final_prob_after_all_adjustments < threshold_ok_prob:
            message = f"Adjustments suggested. Final predicted OK probability: {final_prob_after_all_adjustments:.3f}. This may still be below the threshold of {threshold_ok_prob:.3f}. More iterations or different feature interactions might be needed."
        else:
            message = f"Adjustments suggested. Final predicted OK probability: {final_prob_after_all_adjustments:.3f} (meets/exceeds threshold of {threshold_ok_prob:.3f})."

    return cumulative_adjustments_dict, float(final_prob_after_all_adjustments), message

# --- Routes (与上一版v5一致，确保所有从模型或NumPy获取的数值在放入JSON响应前都转换为Python原生类型) ---
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
        shap_explanation_obj = explainer(df_input) # SHAP values for the input
        shap_values_for_output = shap_explanation_obj.values[0] # This should be a 1D array of SHAP values
        
        base_val_for_waterfall = explainer.expected_value
        if isinstance(base_val_for_waterfall, (np.ndarray, list)):
             base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
        base_val_for_waterfall = float(base_val_for_waterfall)

        # Create SHAP Explanation object with English feature names for the plot
        shap_explanation_for_plot = shap.Explanation(
            values=shap_values_for_output.astype(float), 
            base_values=base_val_for_waterfall,
            data=df_input.iloc[0].values.astype(float), # The actual feature values
            feature_names=features # Original English feature names
        )
        waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_plot)
        
        response = {
            'prob': float(round(prob_ok, 3)), 'threshold': float(round(threshold, 3)), 'is_ng': is_ng,
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output], # SHAP values for adjustment
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
        
        adjustments, final_prob_after_adjustment, message = calculate_precise_adjustment_aggressive(
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
