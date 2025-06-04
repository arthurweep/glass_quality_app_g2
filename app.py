import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 必须在导入 pyplot 之前
import matplotlib.pyplot as plt

# 全局设置matplotlib中文字体
# Render服务器可能没有这些字体，需要确保服务器环境支持或提供字体文件
# 本地测试时，确保你的系统有SimHei或Microsoft YaHei
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'] # DejaVu Sans作为后备
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
except Exception as e:
    logging.warning(f"设置中文字体时出错: {e}. 图表中的中文可能无法正确显示。")

import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24) # 用于session等，保持不变
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# 字段中文名映射，用于图表和前端显示
FIELD_LABELS = {
    "F_cut_act": "刀头实际压力",
    "v_cut_act": "切割实际速度",
    "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度",
    "F_wheel_act": "磨轮压紧力",
    "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}

model_cache = {} # 用于存储模型和相关数据，保持不变

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object):
    # 更新shap_explanation_object中的feature_names为中文
    original_feature_names = shap_explanation_object.feature_names
    display_feature_names = [FIELD_LABELS.get(name, name) for name in original_feature_names]
    
    # 创建一个新的Explanation对象，但只替换feature_names用于显示
    # 注意：shap库本身可能不直接支持在Explanation对象中修改feature_names后直接绘图
    # 因此，我们可能需要在绘图时手动传递display_names，或者修改绘图函数的内部逻辑
    # 一个更简单的方法是，如果shap.plots.waterfall支持`feature_names`参数，就用它
    
    temp_explanation = shap.Explanation(
        values=shap_explanation_object.values,
        base_values=shap_explanation_object.base_values,
        data=shap_explanation_object.data,
        feature_names=display_feature_names # 使用中文名
    )

    fig = plt.figure(figsize=(10, 7)) # 调整图像大小以容纳中文标签
    try:
        # 尝试直接使用修改后的Explanation对象
        shap.plots.waterfall(temp_explanation, show=False, max_display=10)
    except Exception:
        # 如果失败，退回使用原始英文名（但仍然会有中文字体问题）
        shap.plots.waterfall(shap_explanation_object, show=False, max_display=10)
        app.logger.warning("SHAP Waterfall图中文标签可能设置失败，回退到英文名。")

    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    # 使用XGBoost内置的plot_importance，并尝试设置中文标签
    booster = clf.get_booster()
    # 确保booster有正确的英文特征名，这对于get_score很重要
    booster.feature_names = feature_names_original_english
    
    # 获取重要性分数
    importance_scores = booster.get_score(importance_type='weight') #或者 'gain', 'cover'
    
    if not importance_scores:
        app.logger.warning("无法获取特征重要性分数。")
        return None

    # 将特征名和分数组织起来，并按分数排序
    sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 只取前N个特征进行显示，例如前10个
    num_features_to_display = min(len(sorted_importance), 10)
    top_features = sorted_importance[:num_features_to_display]
    
    feature_labels_for_plot = [FIELD_LABELS.get(f[0], f[0]) for f in top_features] # 中文标签
    scores_for_plot = [f[1] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 7)) # 调整图表大小
    ax.barh(range(len(scores_for_plot)), scores_for_plot, align='center')
    ax.set_yticks(range(len(scores_for_plot)))
    ax.set_yticklabels(feature_labels_for_plot) # 设置中文Y轴标签
    ax.invert_yaxis()  # 重要性高的在上面
    ax.set_xlabel('重要性分数 (Weight)')
    ax.set_title('模型特征重要性排序', fontsize=16)
    plt.tight_layout() # 自动调整布局防止标签重叠
    return fig_to_base64(fig)


def find_best_threshold_f1(clf, X, y): # 使用F1最大化选择阈值
    probs_ok = clf.predict_proba(X)[:, 1]
    best_f1_macro, best_thresh = 0.0, 0.5 # 初始化为0.0
    best_metrics_at_thresh = {}

    for t in np.arange(0.01, 1.0, 0.01): # 遍历阈值
        y_pred = (probs_ok >= t).astype(int)
        f1_macro_current = f1_score(y, y_pred, average='macro', zero_division=0) # 使用宏平均F1
        
        if f1_macro_current > best_f1_macro:
            best_f1_macro = f1_macro_current
            best_thresh = t
            # 记录此阈值下的所有指标
            best_metrics_at_thresh = {
                'accuracy': accuracy_score(y, y_pred),
                'recall_ok': recall_score(y, y_pred, pos_label=1, zero_division=0),
                'recall_ng': recall_score(y, y_pred, pos_label=0, zero_division=0),
                'precision_ok': precision_score(y, y_pred, pos_label=1, zero_division=0),
                'precision_ng': precision_score(y, y_pred, pos_label=0, zero_division=0),
                'f1_ok': f1_score(y, y_pred, pos_label=1, zero_division=0),
                'f1_ng': f1_score(y, y_pred, pos_label=0, zero_division=0),
                'threshold': t # 记录当前阈值
            }
    app.logger.info(f"通过F1-score找到最优阈值: {best_thresh:.3f} (Macro F1: {best_f1_macro:.3f})")
    # 确保返回的是Python float 和 包含Python float的字典
    final_threshold = float(best_metrics_at_thresh.get('threshold', best_thresh)) # 如果没找到，用最后一次迭代的best_thresh
    final_metrics = {k: float(v) for k, v in best_metrics_at_thresh.items()}
    final_metrics['threshold'] = final_threshold # 确保最终阈值也被加入
    
    return final_threshold, final_metrics


def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names):
    # ... (此函数与上一版本逻辑一致，确保所有返回的数值是Python float)
    adjustments = {}
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()
    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    required_boost = float(max(threshold_ok_prob - current_prob_ok, 0.0))
    if required_boost <= 1e-4: # 如果已经很接近或超过阈值
        return adjustments, float(current_prob_ok) # 返回当前概率
    
    sorted_features_by_shap = sorted(enumerate(shap_values_np), key=lambda x: -abs(x[1]))
    adjusted_values_for_final_check = current_values_np.copy()

    for idx, shap_val_for_feature in sorted_features_by_shap:
        if required_boost <= 1e-4: 
            break
        feature_name = feature_names[idx] # 原始英文名
        delta = 0.001 
        temp_values_plus_delta = current_values_np.copy()
        temp_values_plus_delta[idx] += delta
        prob_after_delta_change = clf.predict_proba(temp_values_plus_delta.reshape(1, -1))[0, 1]
        sensitivity = (prob_after_delta_change - current_prob_ok) / delta
        if abs(sensitivity) < 1e-6: 
            continue
        needed_feature_change = required_boost / sensitivity
        max_abs_change_ratio = 0.40 # 允许最大调整40%
        current_feature_val = current_values_np[idx]
        max_abs_change_value = abs(current_feature_val * max_abs_change_ratio) if current_feature_val != 0 else 0.20 # 如果当前值为0，允许调整0.2
        
        actual_feature_change = float(np.clip(needed_feature_change, -max_abs_change_value, max_abs_change_value))
        
        if abs(actual_feature_change) < 1e-5: # 如果调整量过小，忽略
            continue

        actual_prob_gain = float(sensitivity * actual_feature_change)
        adjusted_values_for_final_check[idx] += actual_feature_change
        
        adjustments[feature_name] = { # key 使用原始英文名
            'current_value': float(current_values_np[idx]),
            'adjustment': actual_feature_change,
            'new_value': float(adjusted_values_for_final_check[idx]),
            'expected_gain': actual_prob_gain 
        }
        required_boost -= actual_prob_gain
            
    final_prob_after_all_adjustments = float(clf.predict_proba(adjusted_values_for_final_check.reshape(1, -1))[0, 1])
    return adjustments, final_prob_after_all_adjustments


@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    
    if request.method == 'GET':
        # 提供更健壮的默认值
        metrics = model_cache.get('metrics', {})
        default_metrics = {
            'threshold': 0.5, 'accuracy': 0.0, 'recall_ok': 0.0, 'recall_ng': 0.0, 
            'precision_ok': 0.0, 'precision_ng': 0.0, 'f1_ok': 0.0, 'f1_ng': 0.0,
            'trees': 'N/A', 'depth': 'N/A', 'lr': 'N/A'
        }
        # 合并默认值和缓存值，确保所有键存在
        final_metrics = {**default_metrics, **metrics}

        for k, v in final_metrics.items(): # 确保所有数值是Python float
            if isinstance(v, (np.float32, np.float64)):
                final_metrics[k] = float(v)

        return render_template('index.html',
            show_results=bool(model_cache.get('show_results', False)),
            filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('features', []), # 这是原始英文名列表
            default_values=model_cache.get('defaults', {}),
            model_metrics=final_metrics,
            feature_plot=model_cache.get('feature_plot', None),
            error_msg=model_cache.pop('error', None),
            field_labels=FIELD_LABELS # 传递字段中文名映射给模板
        )
    
    if request.method == 'POST':
        # ... (POST逻辑与上一版基本一致, 确保所有存入model_cache的metrics数值是Python float)
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
            features = X.columns.tolist() # 这是原始英文特征名列表
            
            # 调整XGBoost参数以平衡Recall和Precision
            clf = xgb.XGBClassifier(
                n_estimators=150,     # 适中的树数量
                max_depth=5,          # 适中的深度
                learning_rate=0.1,   # 稍大的学习率配合适中树数量
                subsample=0.8,        
                colsample_bytree=0.8, 
                gamma=0.1,            # 轻微正则化
                random_state=42,
                use_label_encoder=False 
            )
            # 调整类别权重，如果NG仍然过高，可以减小这里的权重，例如{0:1.5, 1:1}
            sample_weights = compute_sample_weight(class_weight={0:2.0, 1:1}, y=y) 
            clf.fit(X, y, sample_weight=sample_weights)
            
            best_threshold, calculated_metrics = find_best_threshold_f1(clf, X, y)
            
            # 整合模型参数和计算出的指标
            model_params = clf.get_params()
            final_model_metrics = {
                'trees': int(model_params['n_estimators']),
                'depth': int(model_params['max_depth']),
                'lr': float(model_params['learning_rate']),
                **calculated_metrics # 合并find_best_threshold_f1返回的指标字典
            }
            
            model_cache.update({
                'show_results': True,
                'features': features, # 存储原始英文特征名
                'defaults': {k: float(v) for k, v in X.mean().to_dict().items()},
                'clf': clf,
                'X_train_df': X.copy(),
                'metrics': final_model_metrics, # 存储包含所有Python float的指标
                'feature_plot': generate_feature_importance_plot(clf, features) # 传递原始英文名
            })
            
        except Exception as e:
            model_cache['error'] = f"处理文件时出错: {str(e)}"
            app.logger.error(f"文件处理或模型训练出错: {e}", exc_info=True)
        return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    # ... (与上一版逻辑一致，确保所有返回JSON的数值是Python float)
    global model_cache
    if 'clf' not in model_cache:
        return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']
        features = model_cache['features'] # 原始英文名
        threshold = model_cache['metrics']['threshold']
        
        input_data_dict = {} # key是原始英文名
        for f_name in features:
            val_str = request.form.get(f_name)
            if val_str is None or val_str.strip() == '':
                return jsonify({'error': f'特征 "{FIELD_LABELS.get(f_name, f_name)}" 的值不能为空。'}), 400
            try:
                input_data_dict[f_name] = float(val_str)
            except ValueError:
                 return jsonify({'error': f'特征 "{FIELD_LABELS.get(f_name, f_name)}" 的输入值 "{val_str}" 不是有效的数字。'}), 400
        
        df_input = pd.DataFrame([input_data_dict], columns=features)
        prob_ok = clf.predict_proba(df_input)[0, 1]
        is_ng = bool(prob_ok < threshold)
        
        background_data_df = model_cache['X_train_df'] # DataFrame with original English names
        explainer = shap.Explainer(clf, background_data_df)
        shap_explanation_obj = explainer(df_input) # df_input also uses original English names
        shap_values_for_output = shap_explanation_obj.values[0]
        
        waterfall_plot_base64 = None
        if is_ng:
            base_val_for_waterfall = explainer.expected_value
            if isinstance(base_val_for_waterfall, (np.ndarray, list)):
                 base_val_for_waterfall = base_val_for_waterfall[1] if len(base_val_for_waterfall) == 2 else base_val_for_waterfall[0]
            base_val_for_waterfall = float(base_val_for_waterfall)

            shap_explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_output.astype(float), 
                base_values=base_val_for_waterfall, 
                data=df_input.iloc[0].values.astype(float), # NumPy array of data
                feature_names=features # Original English feature names for SHAP object
            )
            waterfall_plot_base64 = generate_shap_waterfall_base64(shap_explanation_for_waterfall)
        
        response = {
            'prob': float(round(prob_ok, 3)),
            'threshold': float(round(threshold, 3)),
            'is_ng': is_ng,
            'shap_values': [float(round(v, 4)) for v in shap_values_for_output], 
            'metrics': model_cache['metrics'], 
            'waterfall': waterfall_plot_base64,
            'input_data': input_data_dict # 英文键的输入数据
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"预测接口 (/predict) 出错: {e}", exc_info=True)
        return jsonify({'error': f'预测过程中发生内部错误: {str(e)}'}), 500


@app.route('/adjust_single', methods=['POST'])
def adjust_single():
    # ... (与上一版逻辑一致, calculate_precise_adjustment返回的adjustments键是英文名)
    global model_cache
    if 'clf' not in model_cache:
        return jsonify({'error': '请先上传并训练模型。'}), 400
    try:
        clf = model_cache['clf']
        features = model_cache['features'] # 原始英文名
        threshold = model_cache['metrics']['threshold']
        
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': '请求体为空或不是有效的JSON。'}), 400
            
        input_data_dict = json_data.get('input_data') # 英文键
        shap_values_list = json_data.get('shap_values') # shap值数组

        if not input_data_dict or not isinstance(input_data_dict, dict):
            return jsonify({'error': '缺少或无效的 input_data。'}), 400
        if not shap_values_list or not isinstance(shap_values_list, list) or len(shap_values_list) != len(features):
            return jsonify({'error': '缺少或无效的 shap_values。'}), 400
            
        current_values_np_array = np.array([input_data_dict[f] for f in features], dtype=float) # 确保顺序和类型
        shap_values_np_array = np.array(shap_values_list, dtype=float)
        
        adjustments, final_prob_after_adjustment = calculate_precise_adjustment(
            clf, current_values_np_array, shap_values_np_array, threshold, features
        )
        # adjustments 字典的键是原始英文特征名，其内部的值已经确保是Python float
        
        return jsonify({
            'adjustments': adjustments, # 返回的键是原始特征名
            'final_prob_after_adjustment': float(final_prob_after_adjustment)
        })
    except Exception as e:
        app.logger.error(f"优化建议接口 (/adjust_single) 出错: {e}", exc_info=True)
        return jsonify({'error': f'优化建议过程中发生内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
