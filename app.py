import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 必须在导入 pyplot 之前
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig) # 关闭图像，释放内存
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object):
    fig = plt.figure(figsize=(10, 6)) # 可以调整图像大小
    shap.plots.waterfall(shap_explanation_object, show=False, max_display=12) # 最多显示12个特征
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names):
    importance = clf.feature_importances_
    # 将重要性转换为Python float
    importance = [float(imp) for imp in importance]
    indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 7)) # 调整图像大小
    num_features_to_display = min(len(feature_names), 15) # 最多显示15个
    
    ax.bar(range(num_features_to_display), [importance[i] for i in indices[:num_features_to_display]])
    ax.set_title('Feature Importance (XGBoost Built-in)', fontsize=14)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_xticks(range(num_features_to_display))
    ax.set_xticklabels([feature_names[i] for i in indices[:num_features_to_display]], rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_for_ng_recall(clf, X, y, ng_label=0, ok_label=1, min_target_ng_recall=0.95):
    """
    寻找阈值，首要目标是 NG 类别召回率 >= min_target_ng_recall。
    在此基础上，选择能使 OK 类别精确率 (Precision OK) 最高的阈值。
    如果无法达到 min_target_ng_recall，则选择能最大化 NG 召回率的阈值。
    """
    probs_ok = clf.predict_proba(X)[:, ok_label] # OK类别的概率
    
    best_threshold = 0.5 # 默认值
    
    candidate_thresholds_met_target = []

    # 遍历可能的阈值 (判定为OK的概率阈值)
    for threshold_candidate in np.arange(0.01, 1.0, 0.005): # 更细的步长
        y_pred = (probs_ok >= threshold_candidate).astype(int) # 如果概率 >= 阈值，则为OK(1)，否则为NG(0)
        
        recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
        
        if recall_ng_current >= min_target_ng_recall:
            precision_ok_current = precision_score(y, y_pred, pos_label=ok_label, zero_division=0)
            candidate_thresholds_met_target.append((threshold_candidate, precision_ok_current, recall_ng_current))

    if candidate_thresholds_met_target:
        # 从满足NG召回率的阈值中，选择使OK精确率最高的那个
        # 如果OK精确率相同，倾向于选择使得NG召回率也较高的（更严格的阈值通常意味着更低的OK概率阈值）
        candidate_thresholds_met_target.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True) # x[0]取负是为了让阈值本身也参与排序（越高越好）
        best_threshold = candidate_thresholds_met_target[0][0]
        app.logger.info(f"找到优化NG召回率的平衡阈值: {best_threshold:.3f} (NG召回率 >= {min_target_ng_recall}, OK精确率最高)")
    else:
        # 如果没有阈值能满足NG目标召回率，则选择能最大化NG召回率的阈值
        max_ng_recall_overall = -1.0
        best_threshold_for_max_ng_recall = 0.5
        for threshold_candidate in np.arange(0.01, 1.0, 0.005):
            y_pred = (probs_ok >= threshold_candidate).astype(int)
            recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
            if recall_ng_current > max_ng_recall_overall:
                max_ng_recall_overall = recall_ng_current
                best_threshold_for_max_ng_recall = threshold_candidate
            # 如果召回率相同，选择更低的阈值以捕捉更多NG
            elif recall_ng_current == max_ng_recall_overall and threshold_candidate < best_threshold_for_max_ng_recall:
                best_threshold_for_max_ng_recall = threshold_candidate

        best_threshold = best_threshold_for_max_ng_recall
        app.logger.warning(f"未能达到目标NG召回率 {min_target_ng_recall}。选择最大化NG召回率的阈值: {best_threshold:.3f} (此时NG召回率: {max_ng_recall_overall:.2f})")
        
    return float(best_threshold) # 确保返回标准float

def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names):
    adjustments = {}
    current_values_np = np.array(current_values_array, dtype=float).flatten() # 确保是float64
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()

    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    required_boost = max(threshold_ok_prob - current_prob_ok, 0.0) # 确保是float
    
    if required_boost <= 1e-4:
        return adjustments, float(current_prob_ok)

    sorted_features_by_shap = sorted(enumerate(shap_values_np), key=lambda x: -abs(x[1]))
    adjusted_values_for_final_check = current_values_np.copy()

    for idx, shap_val_for_feature in sorted_features_by_shap:
        if required_boost <= 1e-4:
            break

        feature_name = feature_names[idx]
        delta = 0.001 
        temp_values_plus_delta = current_values_np.copy()
        temp_values_plus_delta[idx] += delta
        
        prob_after_delta_change = clf.predict_proba(temp_values_plus_delta.reshape(1, -1))[0, 1]
        sensitivity = (prob_after_delta_change - current_prob_ok) / delta
        
        if abs(sensitivity) < 1e-6: 
            continue
            
        needed_feature_change = required_boost / sensitivity
        
        max_abs_change_ratio = 0.25 # 稍微放宽调整比例
        max_abs_change_value = abs(current_values_np[idx] * max_abs_change_ratio) if current_values_np[idx] != 0 else 0.15
        
        actual_feature_change = np.clip(needed_feature_change, -max_abs_change_value, max_abs_change_value)
        
        if abs(actual_feature_change) < 1e-5:
            continue

        actual_prob_gain = sensitivity * actual_feature_change
        adjusted_values_for_final_check[idx] += actual_feature_change
        
        adjustments[feature_name] = {
            'current_value': float(current_values_np[idx]),
            'adjustment': float(actual_feature_change),
            'new_value': float(adjusted_values_for_final_check[idx]),
            'expected_gain': float(actual_prob_gain) 
        }
        required_boost -= actual_prob_gain
            
    final_prob_after_all_adjustments = clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1]
    return adjustments, float(final_prob_after_all_adjustments)


@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    
    if request.method == 'GET':
        metrics = model_cache.get('metrics', {
            'threshold': 0.5, 'accuracy': 0.0, 'recall_ok': 0.0, 'recall_ng': 0.0, 
            'precision_ok': 0.0, 'precision_ng': 0.0, 'f1_ok': 0.0, 'f1_ng': 0.0,
            'trees': 'N/A', 'depth': 'N/A', 'lr': 'N/A'
        })
        # 确保所有metrics中的数值是Python float
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                metrics[k] = float(v)

        return render_template('index.html',
            show_results=bool(model_cache.get('show_results', False)), # 确保是Python bool
            filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('features', []),
            default_values=model_cache.get('defaults', {}),
            model_metrics=metrics,
            feature_plot=model_cache.get('feature_plot', None),
            error_msg=model_cache.pop('error', None)
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
            
            # 进一步调整XGBoost参数，尝试提高NG召回率
            clf = xgb.XGBClassifier(
                n_estimators=250,     # 增加树的数量
                max_depth=7,          # 适当增加树的深度
                learning_rate=0.03,   # 更小的学习率
                subsample=0.6,        # 每次迭代用60%的数据
                colsample_bytree=0.6, # 构建每棵树时用60%的特征
                gamma=0.2,            # 增加gamma值进行正则化
                min_child_weight=1,   # 默认值，可以尝试增加
                reg_alpha=0.1,        # L1正则化
                reg_lambda=0.1,       # L2正则化
                random_state=42,
                use_label_encoder=False 
            )
            
            # 提高NG类的权重，例如3.0 或 3.5
            sample_weights = compute_sample_weight(class_weight={0:3.0, 1:1}, y=y)
            clf.fit(X, y, sample_weight=sample_weights)
            
            # 使用新的阈值寻找函数，目标NG召回率0.95
            best_threshold = find_best_threshold_for_ng_recall(clf, X, y, ng_label=0, ok_label=1, min_target_ng_recall=0.95)
            
            probs_ok_train = clf.predict_proba(X)[:, 1]
            preds_at_best_thresh_train = (probs_ok_train >= best_threshold).astype(int)

            model_metrics = {
                'trees': int(clf.get_params()['n_estimators']), # 转为Python int
                'depth': int(clf.get_params()['max_depth']),   # 转为Python int
                'lr': float(clf.get_params()['learning_rate']),  # 转为Python float
                'threshold': float(best_threshold),
                'accuracy': float(accuracy_score(y, preds_at_best_thresh_train)),
                'recall_ok': float(recall_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0)),
                'recall_ng': float(recall_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0)),
                'precision_ok': float(precision_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0)),
                'precision_ng': float(precision_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0)),
                'f1_ok': float(f1_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0)),
                'f1_ng': float(f1_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0)),
            }
            
            model_cache.update({
                'show_results': True,
                'features': features,
                'defaults': {k: float(v) for k, v in X.mean().to_dict().items()}, # 确保默认值是float
                'clf': clf,
                'X_train_df': X.copy(),
                'metrics': model_metrics,
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
        # metrics中的threshold已经是python float
        threshold = model_cache['metrics']['threshold'] 
        
        input_data_dict = {}
        for f_name in features:
            val_str = request.form.get(f_name)
            if val_str is None or val_str.strip() == '':
                return jsonify({'error': f'特征 "{f_name}" 的值不能为空。'}), 400
            try:
                input_data_dict[f_name] = float(val_str) # 确保输入是float
            except ValueError:
                 return jsonify({'error': f'特征 "{f_name}" 的输入值 "{val_str}" 不是有效的数字。'}), 400
        
        df_input = pd.DataFrame([input_data_dict], columns=features)
        prob_ok = clf.predict_proba(df_input)[0, 1]
        is_ng = bool(prob_ok < threshold) # 强制转为Python bool
        
        background_data_df = model_cache['X_train_df']
        explainer = shap.Explainer(clf, background_data_df)
        shap_explanation_obj = explainer(df_input)
        shap_values_for_output = shap_explanation_obj.values[0] # 这已经是NumPy array
        
        adjustments = {}
        waterfall_plot_base64 = None
        final_prob_after_adjustment = float(prob_ok) # 确保是Python float

        if is_ng:
            current_values_np_array = df_input.iloc[0].values.astype(float) # 确保是float64 NumPy array
            # shap_values_for_output已经是NumPy array, calculate_precise_adjustment内部会处理类型
            adjustments, final_prob_after_adjustment = calculate_precise_adjustment(
                clf, current_values_np_array, shap_values_for_output, 
                threshold, features
            )
            base_val_for_waterfall = explainer.expected_value
            if isinstance(base_val_for_waterfall, (np.ndarray, list)):
                 base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
            base_val_for_waterfall = float(base_val_for_waterfall) # 确保是Python float

            shap_explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_output.astype(float), # 确保是Python float array
                base_values=base_val_for_waterfall, 
                data=current_values_np_array.astype(float), # 确保是Python float array
                feature_names=features
            )
            waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
        
        response = {
            'prob': float(round(prob_ok, 3)),
            'threshold': float(round(threshold, 3)),
            'is_ng': is_ng, # 已经是Python bool
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output], 
            'adjustments': {k: {ik: float(iv) for ik, iv in v.items()} for k, v in adjustments.items()}, # 确保字典内部值是float
            'metrics': model_cache['metrics'], # 确保 metrics 内部值也是Python原生类型
            'waterfall': waterfall_plot_base64,
            'final_prob_after_adjustment': float(round(final_prob_after_adjustment, 3)) if is_ng and adjustments else None
        }
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
