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

def calculate_adjustment_guaranteed_final(clf, current_values_array, shap_values_array, target_ok_prob, feature_names, initial_is_ng):
    """
    “理论可行”的调整算法：迭代调整，直至达到目标概率，允许极大调整幅度。
    目标：对于 NG 样本，必须给出一套调整方案。
    """
    original_values_np = np.array(current_values_array, dtype=float).flatten()
    current_adjusted_values = original_values_np.copy() # 当前迭代中的特征值
    
    initial_prob_ok = clf.predict_proba(original_values_np.reshape(1, -1))[0, 1]

    # 如果样本最初就不是NG，并且其初始概率已经合格，则无需调整
    if not initial_is_ng and initial_prob_ok >= target_ok_prob:
        return {}, float(initial_prob_ok), "样本当前已为合格状态且满足目标概率，无需调整。"

    # --- 调整参数 ---
    # 对于NG样本，目标是略高于阈值，例如高出0.01，以确保稳定合格
    effective_target_prob = target_ok_prob + 0.01 if initial_is_ng else target_ok_prob
    effective_target_prob = min(effective_target_prob, 0.999) # 概率不超过0.999

    max_iterations_total = 100  # 总迭代次数上限，防止死循环
    max_feature_adjustment_ratio = 5.0 # 允许特征值变化为其原始值的 +/- 500%
    min_absolute_change_for_zero_original = 100.0 # 如果原始值为0，允许的最大绝对变化
    min_meaningful_prob_change_per_step = 0.0001 # 每一步调整期望的最小概率提升
    max_features_to_consider_per_iteration = 5 # 每轮迭代中重点考虑前几个影响最大的特征

    cumulative_adjustments_dict = {} # 最终返回的调整方案

    for iteration in range(max_iterations_total):
        current_prob_iter = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]

        if current_prob_iter >= effective_target_prob:
            break # 已达到目标

        prob_gap_to_target = effective_target_prob - current_prob_iter
        if prob_gap_to_target <= 0: # 以防万一
            break
        
        # 动态计算当前状态下各特征的“伪敏感度”（基于初始SHAP值和当前调整方向）
        # SHAP值大的特征，如果调整方向正确，则优先调整
        # (feature_idx, shap_value, current_value, original_value)
        feature_potentials = []
        for i in range(len(feature_names)):
            shap_val = shap_values_array[i]
            # 调整方向：如果SHAP > 0, 增加特征值有利；如果SHAP < 0, 减少特征值有利
            # 调整潜力：SHAP值越大，潜力越大
            feature_potentials.append((i, shap_val, current_adjusted_values[i], original_values_np[i]))
        
        # 优先调整SHAP绝对值大，且尚未达到“极端”边界的特征
        feature_potentials.sort(key=lambda x: -abs(x[1])) 

        made_adjustment_this_iteration = False
        
        for i_pot in range(min(max_features_to_consider_per_iteration, len(feature_potentials))):
            idx, shap_val, current_val, original_val = feature_potentials[i_pot]
            feature_name = feature_names[idx]

            # 定义此特征的极宽松边界
            lower_bound = original_val - abs(original_val * max_feature_adjustment_ratio) if original_val != 0 else -min_absolute_change_for_zero_original
            upper_bound = original_val + abs(original_val * max_feature_adjustment_ratio) if original_val != 0 else min_absolute_change_for_zero_original

            # 确定理想调整方向
            adjustment_direction = 1.0 if shap_val > 0 else -1.0 # 如果SHAP>0想增加，SHAP<0想减少
            
            # 尝试一个“大胆”的调整步长，例如该特征原始值的10% 或一个固定值
            step_size = abs(original_val * 0.10) if original_val != 0 else 1.0
            step_size = max(step_size, 0.01) # 保证最小步长

            potential_change = adjustment_direction * step_size
            
            # 确保调整后的值在极宽松边界内
            new_val_candidate = current_val + potential_change
            new_val_clipped = np.clip(new_val_candidate, lower_bound, upper_bound)
            
            actual_change_this_step = new_val_clipped - current_val

            if abs(actual_change_this_step) < 1e-5: # 调整量过小或已达边界
                continue

            # 模拟应用此调整，看概率变化
            temp_adjusted_values = current_adjusted_values.copy()
            temp_adjusted_values[idx] = new_val_clipped
            prob_after_step = clf.predict_proba(temp_adjusted_values.reshape(1, -1))[0, 1]
            
            prob_improvement_this_step = prob_after_step - current_prob_iter

            # 如果这一步确实带来了概率提升（或至少没有显著降低且是朝正确方向努力）
            if prob_improvement_this_step > -1e-4 : # 允许微小的负向波动
                current_adjusted_values[idx] = new_val_clipped # 正式应用调整
                made_adjustment_this_iteration = True
                
                # 更新累积调整信息
                cumulative_adjustments_dict[feature_name] = {
                    'current_value': float(original_val),
                    'adjustment': float(current_adjusted_values[idx] - original_val),
                    'new_value': float(current_adjusted_values[idx]),
                    'expected_gain_this_step': "迭代调整中" 
                }
                current_prob_iter = prob_after_step # 更新当前概率，为下一个特征调整做准备
                if current_prob_iter >= effective_target_prob: break # 已达标，跳出内层循环

        if not made_adjustment_this_iteration or current_prob_iter >= effective_target_prob:
            break # 如果一轮没调整或已达标，跳出外层循环
            
    final_prob_ok = clf.predict_proba(current_adjusted_values.reshape(1, -1))[0, 1]
    
    message = ""
    if final_prob_ok >= effective_target_prob:
        message = f"调整建议已生成。调整后样本预测合格概率为: {final_prob_ok:.3f} (目标: ≥{target_ok_prob:.3f})。"
        if not cumulative_adjustments_dict and initial_is_ng : # 如果没做任何调整就达标了
             message = f"样本虽初判为NG（初始概率{initial_prob_ok:.3f}），但其概率已满足或超过目标 {target_ok_prob:.3f}，无需调整。"
    else:
        message = f"已尝试在极大范围内调整特征（最多{max_iterations_total}轮迭代，特征值可变动达原始值的+/-{max_feature_adjustment_ratio*100}%）。"
        if cumulative_adjustments_dict:
             message += f"调整后样本预测合格概率为: {final_prob_ok:.3f}，仍未达到目标 {target_ok_prob:.3f}。"
        else:
             message += f"未能找到任何有效的调整组合使样本达到合格标准（当前概率{initial_prob_ok:.3f}，目标{target_ok_prob:.3f}）。"
        message += " 这可能表示模型对此特定NG样本的判定非常顽固，或所有特征的调整都无法有效提升其合格概率。建议人工复核此样本，或评估模型对此类样本的泛化能力。"

    return cumulative_adjustments_dict, float(final_prob_ok), message

# --- Routes (与上一版v6一致，除了调用新的调整函数 calculate_adjustment_guaranteed_final) ---
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
    # ... (与上一版v6 predict部分完全一致)
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
    # ... (与上一版v6 adjust_single部分一致，但调用新的 calculate_adjustment_guaranteed_final)
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
        
        adjustments, final_prob_after_adjustment, message = calculate_adjustment_guaranteed_final(
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
