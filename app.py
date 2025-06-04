import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 必须在导入 pyplot 之前
import matplotlib.pyplot as plt

# 尝试全局设置matplotlib中文字体
# 注意：这些字体需要在服务器环境中存在才能生效
# 如果服务器没有这些字体，Matplotlib会回退到默认字体（通常是DejaVu Sans，不支持中文）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'] 
    plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题
except Exception as e:
    logging.warning(f"设置中文字体时出错: {e}。图表中的中文可能无法正确显示，将使用默认英文字体。")

import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# 字段英文名到中文名的映射，用于图表和前端显示
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
    """将Matplotlib图像对象转换为Base64编码的字符串，用于在HTML中显示"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') # 保存图像到内存中的缓冲区
    plt.close(fig) # 关闭图像，释放内存
    buf.seek(0) # 将缓冲区指针移到开头
    return base64.b64encode(buf.getvalue()).decode('utf-8') # 编码并转换为字符串

def generate_shap_waterfall_base64(shap_explanation_object):
    """生成SHAP Waterfall图的Base64编码字符串"""
    # 尝试将shap_explanation_object中的英文特征名替换为中文名用于绘图
    display_feature_names = [FIELD_LABELS.get(name, name) for name in shap_explanation_object.feature_names]
    
    # 创建一个新的Explanation对象用于绘图，使用中文特征名
    # SHAP绘图函数通常接受feature_names参数，或者直接从Explanation对象读取
    try:
        temp_explanation = shap.Explanation(
            values=shap_explanation_object.values,
            base_values=shap_explanation_object.base_values,
            data=shap_explanation_object.data,
            feature_names=display_feature_names # 使用中文名
        )
        fig = plt.figure(figsize=(10, 7)) # 调整图像大小以更好容纳中文标签
        shap.plots.waterfall(temp_explanation, show=False, max_display=10)
        plt.title("SHAP Waterfall图 (各特征对“合格”概率的贡献)", fontsize=14) # 中文标题
    except Exception as e_shap_plot:
        app.logger.warning(f"使用中文标签绘制SHAP Waterfall图失败 ({e_shap_plot}), 尝试使用英文标签...")
        # 如果中文标签绘图失败，回退到使用原始英文特征名
        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap_explanation_object, show=False, max_display=10)
        plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14) # 英文标题

    plt.tight_layout() # 自动调整布局
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    """生成特征重要性图的Base64编码字符串，尝试使用中文标签"""
    booster = clf.get_booster()
    # booster.feature_names = feature_names_original_english # 通常在fit时已设置，或XGBoost内部处理

    importance_scores = booster.get_score(importance_type='weight') # 获取特征重要性分数
    if not importance_scores:
        app.logger.warning("无法从模型获取特征重要性分数。")
        if hasattr(clf, 'feature_importances_'): # 尝试scikit-learn接口的
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else:
            return None # 确实无法获取

    # 处理XGBoost内部可能使用f0, f1...作为键的情况
    mapped_importance = {}
    if all(k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        for i, f_name_original in enumerate(feature_names_original_english):
            internal_f_key = f"f{i}"
            if internal_f_key in importance_scores:
                mapped_importance[f_name_original] = importance_scores[internal_f_key]
        if not mapped_importance and importance_scores: # 如果映射失败但有分数
             app.logger.warning("特征重要性键为f0,f1...形式但无法映射到原始特征名。图表可能使用f0,f1...标签。")
             mapped_importance = importance_scores # 回退到使用f0, f1...
    else: # 假设键已经是原始特征名
        mapped_importance = importance_scores
        
    if not mapped_importance: return None # 如果最终还是没有重要性分数

    sorted_importance = sorted(mapped_importance.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10) # 最多显示10个特征
    top_features_english_keys = [item[0] for item in sorted_importance[:num_features_to_display]]
    scores_for_plot = [float(item[1]) for item in sorted_importance[:num_features_to_display]]
    
    # 将英文键转换为中文标签用于绘图
    feature_labels_for_plot_chinese = [FIELD_LABELS.get(key, key) for key in top_features_english_keys]
    
    fig, ax = plt.subplots(figsize=(10, 8)) # 调整图表大小
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot_chinese, fontsize=9) # 设置Y轴为中文标签
    ax.invert_yaxis()  # 重要性高的在上面
    ax.set_xlabel('重要性分数 (Weight)', fontsize=12) # X轴中文标签
    ax.set_title('模型特征重要性排序', fontsize=16) # 图表中文标题
    plt.tight_layout() # 自动调整布局防止标签重叠
    return fig_to_base64(fig)

def find_best_threshold_f1(clf, X, y):
    """使用F1分数最大化原则自动选择最优分类阈值"""
    probs_ok = clf.predict_proba(X)[:, 1] # 获取所有样本预测为OK的概率
    best_f1_macro, best_thresh = 0.0, 0.5
    best_metrics_at_thresh = {}

    for t in np.arange(0.01, 1.0, 0.01): # 遍历0.01到0.99的阈值
        y_pred = (probs_ok >= t).astype(int) # 根据当前阈值t得到预测类别
        # 计算宏平均F1分数，它平等对待每个类别，适合类别不平衡或我们关心所有类别性能的情况
        f1_macro_current = f1_score(y, y_pred, average='macro', zero_division=0)
        
        if f1_macro_current > best_f1_macro: # 如果当前F1分数更高
            best_f1_macro = f1_macro_current
            best_thresh = t
            # 记录此阈值下的所有相关指标
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
            
    if not best_metrics_at_thresh: # 如果没有找到任何F1>0的阈值（例如数据极端或模型非常差）
        app.logger.warning(f"未能通过F1分数找到优化阈值，将使用默认0.5或最后计算的阈值。")
        # 提供一个默认的回退指标集
        dummy_preds = (probs_ok >= 0.5).astype(int)
        best_metrics_at_thresh = {
            'accuracy': accuracy_score(y, dummy_preds), 'recall_ok': recall_score(y, dummy_preds, pos_label=1, zero_division=0),
            'recall_ng': recall_score(y, dummy_preds, pos_label=0, zero_division=0), 'precision_ok': precision_score(y, dummy_preds, pos_label=1, zero_division=0),
            'precision_ng': precision_score(y, dummy_preds, pos_label=0, zero_division=0), 'f1_ok': f1_score(y, dummy_preds, pos_label=1, zero_division=0),
            'f1_ng': f1_score(y, dummy_preds, pos_label=0, zero_division=0), 'threshold': 0.5
        }
        best_thresh = 0.5

    final_threshold = float(best_metrics_at_thresh.get('threshold', best_thresh))
    # 确保所有指标值都是Python float类型，而不是NumPy float
    final_metrics = {k: float(v) for k, v in best_metrics_at_thresh.items()}
    final_metrics['threshold'] = final_threshold # 再次确保阈值正确设置
    
    return final_threshold, final_metrics

def calculate_precise_adjustment_enhanced(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names, initial_is_ng):
    """增强的智能优化建议算法，更积极地为NG样本提供调整方案"""
    adjustments = {} # 存储最终的调整建议 {英文特征名: {调整细节}}
    current_values_np = np.array(current_values_array, dtype=float).flatten() # 样本的原始特征值
    shap_values_np = np.array(shap_values_array, dtype=float).flatten() # 对应的SHAP值
    
    initial_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1] # 样本的初始OK概率

    # 如果样本最初就不是NG，并且其初始概率已经合格，则无需调整
    if not initial_is_ng and initial_prob_ok >= threshold_ok_prob:
        return adjustments, float(initial_prob_ok), "样本当前已判定为合格且概率达到/超过阈值。"

    # --- 调整参数 ---
    max_iterations = 3  # 最多进行几轮完整特征遍历调整
    max_features_to_adjust_total = 5 # 在所有迭代中，最多调整多少个不同的特征
    max_abs_change_ratio = 0.50 # 单个特征值允许从其原始值变化的最大相对比例 (例如50%)
    min_meaningful_adjustment_abs = 1e-4 # 忽略绝对值小于此的特征调整量
    prob_consolidation_target = 0.02 # 如果已达标但想巩固，目标再提升这么多概率
    max_steps_per_feature_per_iteration = 1 # 每轮迭代中，每个特征最多调整一次（避免震荡）

    # --- 迭代调整初始化 ---
    cumulative_adjustments_dict = {} # 存储每个特征的累积调整信息
    adjusted_values_iter = current_values_np.copy() # 这个数组的值会在迭代中被修改
    
    # 按SHAP值绝对大小对特征排序，优先调整影响大的
    sorted_shap_indices = sorted(range(len(feature_names)), key=lambda k: -abs(shap_values_np[k]))

    for iteration in range(max_iterations):
        prob_before_this_iteration = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
        
        # 检查是否已达到目标，如果已达标且是巩固阶段，或者已调整足够多特征，则可停止
        if prob_before_this_iteration >= threshold_ok_prob:
            if not initial_is_ng: break # 本来就OK，现在还OK，停止
            if prob_before_this_iteration >= threshold_ok_prob + prob_consolidation_target: break # NG变OK且已巩固，停止
        
        made_change_in_this_iteration = False
        features_adjusted_this_iteration = 0
        
        for original_idx in sorted_shap_indices: # 遍历所有特征（按SHAP重要性）
            # 控制每轮调整的特征数量 和 总调整特征数量
            if features_adjusted_this_iteration >= 3: break # 每轮最多尝试调整3个特征
            if len(cumulative_adjustments_dict) >= max_features_to_adjust_total and feature_names[original_idx] not in cumulative_adjustments_dict:
                continue # 已调整足够多的不同特征，跳过新的

            feature_name = feature_names[original_idx] # 当前考虑的特征（英文名）
            
            # 计算当前状态下，此特征的敏感度
            current_val_for_sensitivity_calc = adjusted_values_iter[original_idx]
            delta = 0.001 
            temp_for_sensitivity_iter = adjusted_values_iter.copy()
            temp_for_sensitivity_iter[original_idx] += delta
            prob_after_delta_iter = clf.predict_proba(temp_for_sensitivity_iter.reshape(1, -1))[0, 1]
            sensitivity = (prob_after_delta_iter - prob_before_this_iteration) / delta

            if abs(sensitivity) < 1e-7: # 如果特征不敏感，跳过
                continue

            # 计算当前还需要提升多少概率
            effective_required_boost = float(threshold_ok_prob - prob_before_this_iteration)
            if initial_is_ng and effective_required_boost <= 0: # 如果最初是NG，但现在已达标
                effective_required_boost = prob_consolidation_target # 设定一个小的巩固提升目标

            if effective_required_boost <= 0: # 如果不需要提升了（且不是NG巩固情况）
                continue

            # 根据敏感度和所需提升，计算理论上该特征需要变化的量
            needed_feature_value_change_step = effective_required_boost / sensitivity
            
            # 计算该特征允许的最大调整量（基于其原始值）
            original_feature_val = current_values_np[original_idx] 
            max_change_val_abs = abs(original_feature_val * max_abs_change_ratio) if original_feature_val != 0 else 0.25 # 如果原始值为0，允许调整0.25
            
            # 计算此步骤实际能调整的量，要考虑已累积的调整量
            # current_total_change_on_this_feature = adjusted_values_iter[original_idx] - original_feature_val
            # max_further_positive_change = max_change_val_abs - current_total_change_on_this_feature
            # max_further_negative_change = -max_change_val_abs - current_total_change_on_this_feature
            
            # 简化：直接限制单步调整量，并在应用后检查是否超出原始值的总体限制
            # 这里的 actual_feature_change_this_step 是指 *这一步* 的调整量
            actual_feature_change_this_step = float(np.clip(needed_feature_value_change_step, -max_change_val_abs, max_change_val_abs))

            # 检查调整后的新值是否超出了基于原始值的总体调整限制
            potential_new_value = adjusted_values_iter[original_idx] + actual_feature_change_this_step
            if potential_new_value > original_feature_val + max_change_val_abs:
                actual_feature_change_this_step = (original_feature_val + max_change_val_abs) - adjusted_values_iter[original_idx]
            elif potential_new_value < original_feature_val - max_change_val_abs:
                actual_feature_change_this_step = (original_feature_val - max_change_val_abs) - adjusted_values_iter[original_idx]


            if abs(actual_feature_change_this_step) < min_meaningful_adjustment_abs: # 如果这一步的调整量太小，忽略
                continue

            # 应用这一步的调整
            adjusted_values_iter[original_idx] += actual_feature_change_this_step
            
            # 更新累积调整信息
            cumulative_adjustments_dict[feature_name] = {
                'current_value': float(original_feature_val), # 始终是样本的原始值
                'adjustment': float(adjusted_values_iter[original_idx] - original_feature_val), # 从原始值算起的总调整量
                'new_value': float(adjusted_values_iter[original_idx]), # 调整后的新值
                'expected_gain_this_step': float(sensitivity * actual_feature_change_this_step) # 这一小步预期的概率增益
            }
            made_change_in_this_iteration = True
            features_adjusted_this_iteration +=1
        
        if not made_change_in_this_iteration and iteration > 0 : # 如果一轮迭代下来没有任何特征能被有效调整（在第一轮之后）
            break # 可能已经达到调整极限或所有特征都不敏感了
            
    final_prob_after_all_adjustments = clf.predict_proba(adjusted_values_iter.reshape(1, -1))[0, 1]
    
    message = None
    if not cumulative_adjustments_dict and initial_is_ng:
        if initial_prob_ok >= threshold_ok_prob:
             message = "样本虽初判为NG，但其当前概率已达合格标准，无需调整。"
        elif abs(initial_prob_ok - threshold_ok_prob) < 0.03: # 初始概率就非常接近阈值
            message = "样本非常接近合格标准。当前算法未能找到有效的微小调整来显著提升概率，或特征对模型输出不敏感。"
        else:
            message = "未能计算出有效的调整建议。可能原因：特征对模型输出不敏感、已达调整上限、或模型对此类NG样本的判定非常固定。"
    elif cumulative_adjustments_dict and initial_is_ng and final_prob_after_all_adjustments < threshold_ok_prob:
        message = f"已给出调整建议。调整后预测OK概率为 {final_prob_after_all_adjustments:.3f}，可能仍低于阈值 {threshold_ok_prob:.3f}。可能需要更大胆的调整或检查数据/模型。"
    elif cumulative_adjustments_dict and initial_is_ng and final_prob_after_all_adjustments >= threshold_ok_prob:
        message = f"已给出调整建议。调整后预测OK概率为 {final_prob_after_all_adjustments:.3f}，已达到或超过合格阈值。"

    return cumulative_adjustments_dict, float(final_prob_after_all_adjustments), message


# --- Routes (与上一版逻辑基本一致, 确保所有从模型或NumPy获取的数值在放入JSON响应前都转换为Python原生类型) ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD': return make_response('', 200)
    if request.method == 'GET':
        # ... (与上一版v4 GET部分一致)
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
        # ... (与上一版v4 POST部分一致)
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
            sample_weights = compute_sample_weight(class_weight={0:2.0, 1:1}, y=y) # 权重可以根据实际情况微调
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
    # ... (与上一版v4 predict部分一致，确保传递 initial_is_ng_for_adjustment)
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
        
        # 为所有样本生成Waterfall图
        base_val_for_waterfall = explainer.expected_value
        if isinstance(base_val_for_waterfall, (np.ndarray, list)):
             base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
        base_val_for_waterfall = float(base_val_for_waterfall)
        shap_explanation_for_waterfall = shap.Explanation(
            values=shap_values_for_output.astype(float), base_values=base_val_for_waterfall,
            data=df_input.iloc[0].values.astype(float), feature_names=features # 使用英文名
        )
        waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
        
        response = {
            'prob': float(round(prob_ok, 3)), 'threshold': float(round(threshold, 3)), 'is_ng': is_ng,
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output],
            'metrics': model_cache['metrics'], 'waterfall': waterfall_plot_base64,
            'input_data': input_data_dict, 'initial_is_ng_for_adjustment': is_ng # 传递此状态
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500

@app.route('/adjust_single', methods=['POST'])
def adjust_single():
    # ... (与上一版v4 adjust_single部分一致，确保调用增强的 calculate_precise_adjustment_enhanced)
    global model_cache
    if 'clf' not in model_cache: return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']; features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        json_data = request.get_json()
        if not json_data: return jsonify({'error': '请求体为空或不是有效的JSON。'}), 400
        input_data_dict = json_data.get('input_data')
        shap_values_list = json_data.get('shap_values')
        initial_is_ng = json_data.get('initial_is_ng_for_adjustment', True) # 获取此状态
        if not input_data_dict or not isinstance(input_data_dict, dict): return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features): return jsonify({'error': '缺少或无效的 shap_values。'}), 400
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float)
        shap_values_np_array = np.array(shap_values_list, dtype=float)
        
        # 调用增强的调整函数
        adjustments, final_prob_after_adjustment, message = calculate_precise_adjustment_enhanced(
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
