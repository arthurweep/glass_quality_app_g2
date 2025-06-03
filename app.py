import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score # 添加 precision 和 f1
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object): # 参数改为SHAP Explanation对象
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation_object, show=False, max_display=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names):
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importance)), importance[indices])
    ax.set_title('Feature Importance (Built-in)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    # 只显示前15个重要特征的标签，避免拥挤
    num_features_to_display = min(len(feature_names), 15)
    ax.set_xticks(range(num_features_to_display))
    ax.set_xticklabels([feature_names[i] for i in indices[:num_features_to_display]], rotation=45, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_balanced(clf, X, y, ng_label=0, ok_label=1, target_ng_recall=0.92):
    """
    寻找阈值，目标是 NG 类别召回率 >= target_ng_recall，
    在此基础上，最大化 OK 类别召回率。
    如果无法达到 target_ng_recall，则选择能最大化 NG 类别召回率的阈值。
    """
    probs_ok = clf.predict_proba(X)[:, ok_label] # OK类别的概率
    
    best_threshold = 0.5
    best_ok_recall_at_target_ng = -1.0
    
    candidate_thresholds_met_target = []

    for threshold_candidate in np.arange(0.01, 1.0, 0.01):
        y_pred = (probs_ok >= threshold_candidate).astype(int)
        
        recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
        recall_ok_current = recall_score(y, y_pred, pos_label=ok_label, zero_division=0)
        
        if recall_ng_current >= target_ng_recall:
            candidate_thresholds_met_target.append((threshold_candidate, recall_ok_current))

    if candidate_thresholds_met_target:
        # 从满足NG召回率的阈值中，选择使OK召回率最高的那个
        # 如果OK召回率相同，倾向于选择较高的阈值（更严格判定为OK）
        candidate_thresholds_met_target.sort(key=lambda x: (x[1], x[0]), reverse=True)
        best_threshold = candidate_thresholds_met_target[0][0]
        app.logger.info(f"找到平衡阈值: {best_threshold:.2f} (NG召回率满足目标，OK召回率最大化)")
    else:
        # 如果没有阈值能满足NG目标召回率，则退而求其次，选择能最大化NG召回率的阈值
        max_ng_recall_overall = -1.0
        for threshold_candidate in np.arange(0.01, 1.0, 0.01):
            y_pred = (probs_ok >= threshold_candidate).astype(int)
            recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
            if recall_ng_current > max_ng_recall_overall:
                max_ng_recall_overall = recall_ng_current
                best_threshold = threshold_candidate
        app.logger.warning(f"未能达到目标NG召回率 {target_ng_recall}。选择最大化NG召回率的阈值: {best_threshold:.2f} (此时NG召回率: {max_ng_recall_overall:.2f})")
        
    return best_threshold


def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names):
    # ... (此函数逻辑与上一版本基本一致，确保输入是NumPy Array)
    adjustments = {}
    # 确保 current_values_array 是一个一维 NumPy 数组
    current_values_np = np.array(current_values_array).flatten()
    # 确保 shap_values_array 也是一维 NumPy 数组
    shap_values_np = np.array(shap_values_array).flatten()

    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    required_boost = max(threshold_ok_prob - current_prob_ok, 0)
    
    if required_boost <= 0:
        return adjustments, current_prob_ok # 返回当前概率
    
    sorted_features_by_shap = sorted(enumerate(shap_values_np), key=lambda x: -abs(x[1]))
    
    adjusted_values_for_final_check = current_values_np.copy()

    for idx, shap_val_for_feature in sorted_features_by_shap:
        if required_boost <= 1e-4: # 如果提升需求很小，停止
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
        
        max_abs_change_ratio = 0.20 
        max_abs_change_value = abs(current_values_np[idx] * max_abs_change_ratio) if current_values_np[idx] != 0 else 0.1
        
        actual_feature_change = np.clip(needed_feature_change, -max_abs_change_value, max_abs_change_value)
        
        actual_prob_gain = sensitivity * actual_feature_change
        
        # 更新用于最终检查的调整值数组
        adjusted_values_for_final_check[idx] += actual_feature_change
        
        adjustments[feature_name] = {
            'current_value': float(current_values_np[idx]),
            'adjustment': float(actual_feature_change),
            'new_value': float(current_values_np[idx] + actual_feature_change), # 这里应该是调整后的值
            'expected_gain': float(actual_prob_gain) 
        }
        
        required_boost -= actual_prob_gain
            
    # 使用累积调整后的值重新计算最终概率
    final_prob_after_all_adjustments = clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1]
    return adjustments, final_prob_after_all_adjustments


@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    
    if request.method == 'GET':
        # 提供默认值以避免模板渲染错误
        metrics = model_cache.get('metrics', {
            'threshold': 0.5, 'accuracy': 0, 'recall_ok': 0, 'recall_ng': 0, 
            'precision_ok': 0, 'precision_ng': 0, 'f1_ok': 0, 'f1_ng': 0,
            'trees': 'N/A', 'depth': 'N/A', 'lr': 'N/A'
        })
        return render_template('index.html',
            show_results=model_cache.get('show_results', False),
            filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('features', []),
            default_values=model_cache.get('defaults', {}),
            model_metrics=metrics,
            feature_plot=model_cache.get('feature_plot', None),
            error_msg=model_cache.pop('error', None)
        )
    
    if request.method == 'POST':
        model_cache.clear() # 清空之前的模型和数据
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
            
            X = df.drop("OK_NG", axis=1)
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(X.mean()) # 用均值填充转换失败的NaN
            
            y = pd.to_numeric(df["OK_NG"], errors='coerce').fillna(0).astype(int)
            
            features = X.columns.tolist()
            
            # 调整XGBoost参数以期提高NG召回率
            clf = xgb.XGBClassifier(
                n_estimators=200,     # 增加树的数量
                max_depth=6,          # 增加树的深度
                learning_rate=0.05,   # 减小学习率
                subsample=0.7,        # 每次迭代用70%的数据
                colsample_bytree=0.7, # 构建每棵树时用70%的特征
                gamma=0.1,            # 增加gamma值进行正则化
                # scale_pos_weight 可以根据实际NG/OK比例动态调整，例如 (sum(y==0)/sum(y==1))
                # 但这里我们先用 compute_sample_weight
                random_state=42,
                use_label_encoder=False 
            )
            
            # 调整类别权重，更侧重NG类 (label 0)
            # 例如，如果NG是少数类，给它更高的权重
            # 假设NG是0, OK是1。我们要提高NG(0)的召回率，所以给0类更高的权重
            sample_weights = compute_sample_weight(class_weight={0:2.5, 1:1}, y=y) # 给NG类2.5倍权重
            
            clf.fit(X, y, sample_weight=sample_weights)
            
            # 使用新的阈值寻找函数，目标NG召回率0.92
            best_threshold = find_best_threshold_balanced(clf, X, y, ng_label=0, ok_label=1, target_ng_recall=0.92)
            
            # 在最优（平衡后）阈值下评估模型
            probs_ok_train = clf.predict_proba(X)[:, 1]
            preds_at_best_thresh_train = (probs_ok_train >= best_threshold).astype(int)

            model_metrics = {
                'trees': clf.get_params()['n_estimators'],
                'depth': clf.get_params()['max_depth'],
                'lr': clf.get_params()['learning_rate'],
                'threshold': best_threshold,
                'accuracy': accuracy_score(y, preds_at_best_thresh_train),
                'recall_ok': recall_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0),
                'recall_ng': recall_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0),
                'precision_ok': precision_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0),
                'precision_ng': precision_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0),
                'f1_ok': f1_score(y, preds_at_best_thresh_train, pos_label=1, zero_division=0),
                'f1_ng': f1_score(y, preds_at_best_thresh_train, pos_label=0, zero_division=0),
            }
            
            model_cache.update({
                'show_results': True,
                'features': features,
                'defaults': X.mean().to_dict(), # 使用处理后的X计算均值
                'clf': clf,
                'X_train_df': X.copy(), # 存储DataFrame格式的X_train用于SHAP背景
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
        threshold = model_cache['metrics']['threshold']
        
        input_data_dict = {}
        for f_name in features:
            val_str = request.form.get(f_name)
            if val_str is None or val_str.strip() == '':
                return jsonify({'error': f'特征 "{f_name}" 的值不能为空。'}), 400
            try:
                input_data_dict[f_name] = float(val_str)
            except ValueError:
                 return jsonify({'error': f'特征 "{f_name}" 的输入值 "{val_str}" 不是有效的数字。'}), 400
        
        # 确保输入是 DataFrame (1行，n列)
        df_input = pd.DataFrame([input_data_dict], columns=features)
        
        prob_ok = clf.predict_proba(df_input)[0, 1] # 预测OK的概率
        is_ng = bool(prob_ok < threshold)
        
        # SHAP分析
        background_data_df = model_cache['X_train_df']
        explainer = shap.Explainer(clf, background_data_df)
        
        shap_explanation_obj = explainer(df_input) # 获取SHAP Explanation对象
        shap_values_for_output = shap_explanation_obj.values[0] # 单个样本的SHAP值数组
        
        adjustments = {}
        waterfall_plot_base64 = None
        final_prob_after_adjustment = prob_ok # 默认是当前概率

        if is_ng:
            # 将df_input转为numpy array给调整函数
            current_values_np_array = df_input.iloc[0].values 
            adjustments, final_prob_after_adjustment = calculate_precise_adjustment(
                clf, current_values_np_array, shap_values_for_output, 
                threshold, features
            )
            # 为waterfall图创建SHAP Explanation对象
            # explainer.expected_value 对于二分类通常是单个值或者[P(class0), P(class1)]，取对应OK类别的
            base_val_for_waterfall = explainer.expected_value
            if isinstance(base_val_for_waterfall, (np.ndarray, list)) and len(base_val_for_waterfall) == 2:
                 base_val_for_waterfall = base_val_for_waterfall[1] # OK类别的基准值

            shap_explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_output, 
                base_values=base_val_for_waterfall, 
                data=current_values_np_array, # 使用 numpy array
                feature_names=features
            )
            waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
        
        response = {
            'prob': round(prob_ok, 3),
            'threshold': round(threshold, 2),
            'is_ng': is_ng,
            'shap_values': [round(float(v), 4) for v in shap_values_for_output], 
            'adjustments': adjustments, 
            'metrics': model_cache['metrics'], # 包含模型参数和训练表现
            'waterfall': waterfall_plot_base64,
            'final_prob_after_adjustment': round(final_prob_after_adjustment, 3) if is_ng else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

