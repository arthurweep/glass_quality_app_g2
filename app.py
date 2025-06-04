import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 使用Matplotlib默认英文字体，不再尝试设置中文，以保证兼容性
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# 字段中文名映射，主要用于HTML模板中的文本显示
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
    fig = plt.figure(figsize=(10, 7)) # 调整大小以更好显示英文标签
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14) # 英文标题
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    booster = clf.get_booster()
    booster.feature_names = feature_names_original_english
    importance_scores = booster.get_score(importance_type='weight')
    if not importance_scores:
        app.logger.warning("无法获取特征重要性分数。")
        return None
    sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(len(sorted_importance), 10)
    top_features = sorted_importance[:num_features_to_display]
    feature_labels_for_plot_english = [f[0] for f in top_features] 
    scores_for_plot = [float(f[1]) for f in top_features]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot_english) # 英文Y轴标签
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score (Weight)', fontsize=12) # 英文标签
    ax.set_title('Feature Importance Ranking', fontsize=16) # 英文标题
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
    if not best_metrics_at_thresh: # 如果循环没有找到任何F1>0的阈值（极端情况）
        app.logger.warning(f"未能通过F1分数找到优化阈值，将使用默认0.5或最后计算的阈值。")
        # 至少返回一个包含所有键的字典
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
    final_metrics['threshold'] = final_threshold # 确保最终阈值是这个
    return final_threshold, final_metrics

def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names, initial_is_ng):
    adjustments = {}
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()
    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    
    # 即使样本最初是NG，如果当前概率（可能是由于浮点运算或之前的微小调整）已经略高于阈值，
    # 我们仍然希望算法给出“巩固”或“进一步优化”的建议，而不是直接说“已合格”。
    # required_boost 现在可以是负数，表示当前已超过阈值多少。
    required_boost = float(threshold_ok_prob - current_prob_ok)
    
    # 只有当样本最初就不是NG，并且当前概率也合格时，才不给建议。
    if not initial_is_ng and current_prob_ok >= threshold_ok_prob:
        return adjustments, float(current_prob_ok), "当前已合格，无需调整。"

    # 如果最初是NG，但现在已非常接近或略超阈值 (required_boost非常小或负)，我们仍然要尝试。
    # 目标是让required_boost变为0或更小（即概率达到或超过阈值）
    
    sorted_features_by_shap = sorted(enumerate(shap_values_np), key=lambda x: -abs(x[1]))
    adjusted_values_for_final_check = current_values_np.copy()
    adjustment_made_count = 0 # 计数实际发生的调整

    for idx, shap_val_for_feature in sorted_features_by_shap:
        # 如果已经做了足够调整使得required_boost变为负数（即远超阈值），可以停止
        if required_boost <= -0.02 and adjustment_made_count > 0 : # -0.02 意味着超过阈值2%
             break
        if adjustment_made_count >= 3 and required_boost <= 0: # 最多调整3个特征如果已达标
            break

        feature_name = feature_names[idx]
        delta = 0.001 
        temp_values_plus_delta = current_values_np.copy()
        temp_values_plus_delta[idx] += delta
        prob_after_delta_change = clf.predict_proba(temp_values_plus_delta.reshape(1, -1))[0, 1]
        # 敏感度计算应基于当前累积调整后的概率，而非原始current_prob_ok
        # 但为了简化，这里仍然用原始current_prob_ok计算一次性敏感度，后续通过迭代弥补
        current_prob_for_sensitivity = clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1]
        prob_after_delta_on_adjusted = clf.predict_proba( (adjusted_values_for_final_check + (np.eye(len(features))[idx] * delta)).reshape(1,-1) )[0,1]
        sensitivity = (prob_after_delta_on_adjusted - current_prob_for_sensitivity) / delta
        
        if abs(sensitivity) < 1e-6: 
            continue
        
        # 目标是让 current_prob_for_sensitivity + (sensitivity * change) >= threshold_ok_prob
        # (sensitivity * change) >= threshold_ok_prob - current_prob_for_sensitivity
        # change >= (threshold_ok_prob - current_prob_for_sensitivity) / sensitivity
        effective_required_boost = threshold_ok_prob - current_prob_for_sensitivity

        if effective_required_boost <= 0 and initial_is_ng: # 如果已达标但最初是NG，做一次巩固性调整
            # 如果shap值是负的（降低OK概率的特征），我们应该减少它
            # 如果shap值是正的（提高OK概率的特征），我们应该增加它
            # 目标是让它更“OK”一点
            # 这里的逻辑是：如果一个特征对“不合格”贡献大（SHAP值推动远离OK），我们就反向调整它
            # 如果SHAP < 0，说明这个特征使样本更NG，尝试增加其值。
            # 如果SHAP > 0，说明这个特征使样本更OK，如果还想巩固，也尝试增加其值（或保持）。
            # 这里简单地尝试推动概率增加0.01
            effective_required_boost = 0.01 


        needed_feature_change = effective_required_boost / sensitivity
        
        max_abs_change_ratio = 0.40 
        current_feature_val = current_values_np[idx] # 用原始值计算调整范围
        max_abs_change_value = abs(current_feature_val * max_abs_change_ratio) if current_feature_val != 0 else 0.20
        
        actual_feature_change = float(np.clip(needed_feature_change, -max_abs_change_value, max_abs_change_value))
        
        if abs(actual_feature_change) < 1e-5:
            continue

        actual_prob_gain_step = float(sensitivity * actual_feature_change)
        
        # 更新累积调整的值
        adjusted_values_for_final_check[idx] += actual_feature_change
        
        adjustments[feature_name] = {
            'current_value': float(current_values_np[idx]),
            'adjustment': actual_feature_change,
            'new_value': float(adjusted_values_for_final_check[idx]),
            'expected_gain': actual_prob_gain_step
        }
        adjustment_made_count +=1
        # 更新 required_boost 是基于最新的整体概率和目标阈值的差
        # 这里简化为只看累积的概率增益，或者在循环外重新计算最终概率
            
    final_prob_after_all_adjustments = float(clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1])
    
    message = None
    if not adjustments and initial_is_ng: # 如果最初是NG但没有生成任何调整
        current_final_prob = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1] # 重新获取当前概率
        if current_final_prob >= threshold_ok_prob:
             message = "样本当前预测概率已达到或超过合格阈值，无需调整。"
        elif abs(current_final_prob - threshold_ok_prob) < 0.02: # 差距小于2%
            message = "当前状态已非常接近合格标准，建议的微小调整可能效果不显著或难以实现。"
        else:
            message = "未能计算出有效的调整建议。可能原因：特征对模型输出不敏感、已达调整上限、或模型对此类样本的判定较为固定。"
            
    return adjustments, final_prob_after_all_adjustments, message

# --- Routes ---
@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ... (与上一版逻辑一致, 确保所有存入model_cache的metrics数值是Python float)
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
            if isinstance(v, (np.float32, np.float64)): final_metrics[k] = float(v)
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
            for col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(X.mean())
            y = pd.to_numeric(df["OK_NG"], errors='coerce').fillna(0).astype(int)
            features = X.columns.tolist()
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                random_state=42, use_label_encoder=False
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
    # ... (与上一版逻辑一致, 但要传递 initial_is_ng 给 calculate_precise_adjustment)
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
        if is_ng: # 只有不合格时才考虑生成waterfall图
            base_val_for_waterfall = explainer.expected_value
            if isinstance(base_val_for_waterfall, (np.ndarray, list)):
                 base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
            base_val_for_waterfall = float(base_val_for_waterfall)
            shap_explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_output.astype(float), base_values=base_val_for_waterfall,
                data=df_input.iloc[0].values.astype(float), feature_names=features
            )
            waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
        response = {
            'prob': float(round(prob_ok, 3)), 'threshold': float(round(threshold, 3)),
            'is_ng': is_ng, 'shap_values': [float(round(v, 4)) for v in shap_values_for_output],
            'metrics': model_cache['metrics'], 'waterfall': waterfall_plot_base64,
            'input_data': input_data_dict, # 英文键的输入数据
            'initial_is_ng_for_adjustment': is_ng # 传递此状态
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500

@app.route('/adjust_single', methods=['POST'])
def adjust_single():
    # ... (与上一版逻辑一致)
    global model_cache
    if 'clf' not in model_cache:
        return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']
        features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        json_data = request.get_json()
        if not json_data: return jsonify({'error': '请求体为空或不是有效的JSON。'}), 400
        input_data_dict = json_data.get('input_data')
        shap_values_list = json_data.get('shap_values')
        initial_is_ng = json_data.get('initial_is_ng_for_adjustment', True) # 获取最初的NG状态

        if not input_data_dict or not isinstance(input_data_dict, dict): return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features): return jsonify({'error': '缺少或无效的 shap_values。'}), 400
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float)
        shap_values_np_array = np.array(shap_values_list, dtype=float)
        adjustments, final_prob_after_adjustment, message = calculate_precise_adjustment(
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
