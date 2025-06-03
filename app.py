import base64
import io
import os
import logging

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response # 添加 make_response
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

app = Flask(__name__)
app.secret_key = os.urandom(24) 

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO) 

model_cache = {}

try:
    matplotlib.rcParams['font.family'] = 'sans-serif' 
    matplotlib.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    app.logger.warning(f"Matplotlib 字体设置警告: {e}")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_values_single_instance):
    fig = plt.figure()
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=15)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold(clf, X, y, pos_label=1, min_recall=0.95):
    probs = clf.predict_proba(X)[:, pos_label] 
    thresholds = np.arange(0.01, 1.0, 0.01) 
    
    candidate_thresholds_satisfying_min_recall = []
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        current_recall = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
        if current_recall >= min_recall:
            candidate_thresholds_satisfying_min_recall.append(thresh)
            
    if candidate_thresholds_satisfying_min_recall:
        best_threshold = max(candidate_thresholds_satisfying_min_recall)
        app.logger.info(f"找到满足最小召回率 {min_recall} 的最佳阈值: {best_threshold:.2f}")
    else:
        max_found_recall = -1.0
        best_threshold_for_max_recall = 0.5 
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            current_recall = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
            if current_recall > max_found_recall:
                max_found_recall = current_recall
                best_threshold_for_max_recall = thresh
        best_threshold = best_threshold_for_max_recall
        app.logger.warning(f"未找到满足最小召回率 {min_recall} 的阈值。选择最大化召回率的阈值: {best_threshold:.2f} (此时召回率为: {max_found_recall:.2f})")
        
    return best_threshold

def bayes_opt_adjustment(clf, x_original, feature_names, target_probability_threshold=0.5):
    adjustment_range = 0.2 
    dims = [Real(-adjustment_range, adjustment_range, name=fn) for fn in feature_names]

    @use_named_args(dims)
    def objective_function(**kwargs_adjustments):
        x_adjusted = x_original.copy()
        total_adjustment_magnitude = 0
        for i, feature_name in enumerate(feature_names):
            adjustment = kwargs_adjustments[feature_name]
            x_adjusted[i] += adjustment
            total_adjustment_magnitude += abs(adjustment)
        
        df_adjusted = pd.DataFrame([x_adjusted], columns=feature_names)
        predicted_probability_ok = clf.predict_proba(df_adjusted)[0, 1] 
        
        if predicted_probability_ok >= target_probability_threshold:
            return total_adjustment_magnitude 
        else:
            probability_gap_penalty = (target_probability_threshold - predicted_probability_ok) * 10 
            return total_adjustment_magnitude + probability_gap_penalty

    optimization_result = gp_minimize(objective_function, dims, acq_func='EI',
                                      n_calls=50, n_random_starts=10, random_state=42)
    
    best_found_adjustments = optimization_result.x
    
    x_final_adjusted = x_original.copy()
    for i, adj in enumerate(best_found_adjustments):
        x_final_adjusted[i] += adj
    df_final_adjusted = pd.DataFrame([x_final_adjusted], columns=feature_names)
    final_adjusted_probability_ok = clf.predict_proba(df_final_adjusted)[0, 1]
    
    return best_found_adjustments, final_adjusted_probability_ok

@app.route('/', methods=['GET', 'POST', 'HEAD']) # 添加 HEAD 方法
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由, 方法: {request.method}")

    if request.method == 'HEAD': # 处理健康检查的 HEAD 请求
        response = make_response()
        response.status_code = 200
        return response

    if request.method == 'GET':
        for key in ['show_single_pred_results', 'single_pred_input_data_html', 'single_pred_prob',
                    'single_pred_label', 'single_pred_shap_table_html', 'base64_waterfall_plot',
                    'single_pred_error_to_display', 'bayes_suggestion_to_display']: 
            model_cache.pop(key, None)
        
        template_vars = {
            'show_results': model_cache.get('show_results_from_upload', False),
            'filename': model_cache.get('filename'),
            'form_inputs': model_cache.get('form_inputs'),
            'base64_perf_plot': model_cache.get('base64_perf_plot'),
            'metrics_df_html': model_cache.get('metrics_df_html'),
            'recommended_threshold_text': model_cache.get('recommended_threshold_text'),
            'best_recommendation_html': model_cache.get('best_recommendation_html'),
            'base64_global_shap_plot': model_cache.get('base64_global_shap_plot'),
            'default_values': model_cache.get('default_values'),
            'error_message': model_cache.pop('error_message_to_display', None), 
            'current_best_threshold': model_cache.get('best_thresh') 
        }
        return render_template('index.html', **template_vars)

    if request.method == 'POST': 
        # ... (POST部分的其余代码与您上一版本完全相同，此处省略以保持简洁) ...
        # ... (确保所有中文日志、错误消息、计算逻辑都正确) ...
        app.logger.info("POST 请求到 '/', 开始处理上传文件...")
        model_cache.clear() 

        if 'file' not in request.files:
            model_cache['error_message_to_display'] = "请求中没有文件部分。"
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            model_cache['error_message_to_display'] = "未选择文件。"
            return redirect(url_for('index'))

        if file and file.filename.endswith('.csv'):
            try:
                filename = file.filename
                app.logger.info(f"成功接收到文件: {filename}")
                df = pd.read_csv(file)
                model_cache['filename'] = filename

                if "OK_NG" not in df.columns:
                    model_cache['error_message_to_display'] = "上传的 CSV 文件必须包含 'OK_NG' 列。"
                    return redirect(url_for('index'))
                
                X = df.drop("OK_NG", axis=1).copy()
                for col in X.columns: 
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                if X.isnull().values.any():
                    app.logger.warning(f"文件 {filename} 中的特征列在转换为数值型后包含NaN值。将使用均值填充。")
                    X = X.fillna(X.mean())

                y_numeric_from_csv = df["OK_NG"].copy()
                if not pd.api.types.is_numeric_dtype(y_numeric_from_csv) or not y_numeric_from_csv.isin([0, 1]).all():
                    try:
                        y_numeric_from_csv = pd.to_numeric(y_numeric_from_csv, errors='raise')
                        if not y_numeric_from_csv.isin([0, 1]).all():
                            raise ValueError("转换后仍包含非0或1的值")
                    except (ValueError, TypeError) as e_val:
                        invalid_values_original = df["OK_NG"][~df["OK_NG"].astype(str).isin(['0', '1'])].unique()
                        error_msg_part = f"无效值示例: {', '.join(map(str, invalid_values_original[:3]))}{'...' if len(invalid_values_original) > 3 else ''}"
                        model_cache['error_message_to_display'] = f"列 'OK_NG' 必须只包含数字 0 (不合格) 或 1 (合格)。{error_msg_part}"
                        return redirect(url_for('index'))
                
                y_numeric = y_numeric_from_csv.astype(int)
                ok_label_numeric_val = 1
                ng_label_numeric_val = 0
                
                model_cache['feature_names'] = list(X.columns)
                model_cache['X_train_df_for_explainer'] = X.copy() 

                weights = compute_sample_weight(class_weight={ng_label_numeric_val: 1.0, ok_label_numeric_val: 2.0}, y=y_numeric)

                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False)
                clf.fit(X, y_numeric, sample_weight=weights)
                model_cache['clf'] = clf
                app.logger.info("XGBoost 模型训练完成。")

                best_threshold_val = find_best_threshold(clf, X, y_numeric, pos_label=ok_label_numeric_val, min_recall=0.95)
                model_cache['best_thresh'] = best_threshold_val
                
                recommended_threshold_text = f"ℹ️ 当前自动选择的最优分类阈值为: {best_threshold_val:.2f} (优先确保 OK 召回率 ≥ 0.95)"
                model_cache['recommended_threshold_text'] = recommended_threshold_text
                model_cache['best_recommendation_html'] = f"<p>当前自动选择的分类阈值为 <strong>{best_threshold_val:.2f}</strong>。此阈值优先保证OK类别召回率不低于95%，若无法达到则选择最大化OK召回率的阈值。</p>"
                app.logger.info(f"自动最优阈值计算完成: {best_threshold_val:.2f}")

                probs = clf.predict_proba(X)[:, ok_label_numeric_val]
                preds_at_best_thresh = (probs >= best_threshold_val).astype(int)
                report_at_best_thresh = classification_report(y_numeric, preds_at_best_thresh, output_dict=True, zero_division=0,
                                                               labels=[ng_label_numeric_val, ok_label_numeric_val],
                                                               target_names=['NG_Class_0', 'OK_Class_1'])
                
                metrics_list_for_display = [{
                    "threshold": f"{best_threshold_val:.2f} (自动最优)", 
                    "accuracy": report_at_best_thresh["accuracy"],
                    "NG_recall": report_at_best_thresh["NG_Class_0"].get("recall", 0),
                    "OK_recall": report_at_best_thresh["OK_Class_1"].get("recall", 0),
                    "OK_precision": report_at_best_thresh["OK_Class_1"].get("precision", 0),
                    "f1_score_weighted": f1_score(y_numeric, preds_at_best_thresh, average='weighted', zero_division=0)
                }]
                metrics_df_for_display = pd.DataFrame(metrics_list_for_display)
                model_cache['metrics_df_html'] = metrics_df_for_display.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.2f}'.format)

                fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                scan_thresholds = np.arange(0.01, 1.0, 0.01)
                ng_recalls_scan, ok_recalls_scan, f1_scores_scan = [], [], []

                for t_scan in scan_thresholds:
                    preds_scan = (probs >= t_scan).astype(int)
                    report_scan = classification_report(y_numeric, preds_scan, output_dict=True, zero_division=0,
                                                        labels=[ng_label_numeric_val, ok_label_numeric_val],
                                                        target_names=['NG_Class_0', 'OK_Class_1'])
                    ng_recalls_scan.append(report_scan["NG_Class_0"].get("recall", 0))
                    ok_recalls_scan.append(report_scan["OK_Class_1"].get("recall", 0))
                    f1_scores_scan.append(f1_score(y_numeric, preds_scan, average='weighted', zero_division=0))
                
                ax_perf.plot(scan_thresholds, ng_recalls_scan, label="NG (Class 0) Recall", color="red")
                ax_perf.plot(scan_thresholds, ok_recalls_scan, label="OK (Class 1) Recall", color="green")
                ax_perf.plot(scan_thresholds, f1_scores_scan, label="F1 Score (Weighted)", color="blue")
                ax_perf.axvline(x=best_threshold_val, color="purple", linestyle="--", label=f"Selected Best Threshold: {best_threshold_val:.2f}") 
                
                ax_perf.set_xlabel("Classification Threshold") 
                ax_perf.set_ylabel("Metric Value") 
                ax_perf.set_title("Model Performance vs. Threshold (Best Threshold Applied)") 
                ax_perf.legend(loc='best')
                ax_perf.grid(True)
                plt.tight_layout()
                model_cache['base64_perf_plot'] = fig_to_base64(fig_perf)

                explainer_global = shap.Explainer(clf, X)
                shap_values_all = explainer_global(X)
                fig_global_shap, _ = plt.subplots()
                shap.plots.bar(shap_values_all, show=False, max_display=15)
                plt.tight_layout()
                model_cache['base64_global_shap_plot'] = fig_to_base64(fig_global_shap)
                app.logger.info("性能图和全局 SHAP 图生成完毕。")
                
                model_cache['form_inputs'] = model_cache['feature_names']
                model_cache['default_values'] = X.mean().to_dict()
                model_cache['show_results_from_upload'] = True 

                return redirect(url_for('index')) 

            except Exception as e:
                app.logger.error(f"处理文件 {filename} 时发生严重错误: {e}", exc_info=True)
                model_cache.clear() 
                model_cache['error_message_to_display'] = f"处理文件时发生严重错误: {str(e)}" 
                return redirect(url_for('index'))
        else: 
             app.logger.error("POST 请求错误: 文件类型无效, 非CSV文件。")
             model_cache.clear()
             model_cache['error_message_to_display'] = "文件类型无效。请上传 CSV 文件。" 
             return redirect(url_for('index'))
    
    # 如果不是 'GET', 'POST', 'HEAD' 中的任何一种，则重定向到 index
    app.logger.warning(f"接收到未明确处理的请求方法 '{request.method}' 在 '/' 路由, 重定向到主页。")
    model_cache.clear() # 对于未知请求也清理缓存
    return redirect(url_for('index'))


# --- AJAX 单样本预测路由 ---
@app.route('/predict_single_ajax', methods=['POST'])
def predict_single_ajax():
    global model_cache
    app.logger.info("AJAX POST 请求到 '/predict_single_ajax', 开始单一样本预测...")

    if 'clf' not in model_cache or 'best_thresh' not in model_cache:
        app.logger.warning("'/predict_single_ajax' 错误: 模型或最优阈值未在缓存中找到。")
        return jsonify({
            "success": False,
            "error": "请先上传并成功处理一个CSV文件, 然后再进行单一样本预测。" 
        }), 400 

    try:
        clf = model_cache['clf']
        current_threshold = model_cache['best_thresh'] 
        X_background = model_cache.get('X_train_df_for_explainer')
        feature_names = model_cache.get('feature_names')
        ok_label_numeric_val = 1
        ng_label_numeric_val = 0

        if X_background is None or feature_names is None:
            app.logger.error("'/predict_single_ajax' 错误: 缓存中缺少 X_background 或 feature_names。")
            return jsonify({
                "success": False,
                "error": "内部错误：缺少必要的训练数据信息。" 
            }), 500

        input_data_dict = {}
        form_data = request.form 
        for f_name in feature_names:
            form_value = form_data.get(f_name)
            if form_value is None or form_value.strip() == "":
                 app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值为空。")
                 return jsonify({
                     "success": False,
                     "error": f"特征 '{f_name}' 的值不能为空。", 
                     "field": f_name 
                 }), 400
            try:
                input_data_dict[f_name] = float(form_value)
            except ValueError:
                app.logger.warning(f"单样本预测表单输入错误: 特征 '{f_name}' 的值 '{form_value}' 不是有效数字。")
                return jsonify({
                    "success": False,
                    "error": f"特征 '{f_name}' 的输入无效, 请输入一个数字。当前值为: '{form_value}'", 
                    "field": f_name
                }), 400
        
        df_input = pd.DataFrame([input_data_dict], columns=feature_names)
        app.logger.info(f"单样本预测输入数据 (AJAX): {df_input.to_dict(orient='records')}")

        prob_ok_array = clf.predict_proba(df_input)[:, ok_label_numeric_val]
        prob_ok_scalar = prob_ok_array[0] if prob_ok_array.ndim > 0 else prob_ok_array
        predicted_numeric_label = ok_label_numeric_val if prob_ok_scalar >= current_threshold else ng_label_numeric_val
        pred_label_text = "✅ 合格 (OK)" if predicted_numeric_label == ok_label_numeric_val else "❌ 不合格 (NG)" 
        app.logger.info(f"单样本预测概率 (OK): {prob_ok_scalar:.3f}, 使用阈值: {current_threshold:.2f}, 预测数值标签: {predicted_numeric_label}, 文本标签: {pred_label_text}")

        base64_waterfall_plot_str = None
        single_pred_shap_table_html_str = None
        bayes_suggestion_text = None

        if predicted_numeric_label == ng_label_numeric_val:
            app.logger.info("样本预测为不合格, 开始 SHAP 分析和贝叶斯优化...")
            explainer_instance = shap.Explainer(clf, X_background) 
            shap_values_instance = explainer_instance(df_input) 
            base64_waterfall_plot_str = generate_shap_waterfall_base64(shap_values_instance[0])
            
            shap_values_for_df = shap_values_instance.values[0]
            base_value_for_df = shap_values_instance.base_values[0]
            shap_df_data = pd.DataFrame({
                "Feature": df_input.columns, 
                "SHAP Value": shap_values_for_df, 
            })
            predicted_logit_for_shap_impact = base_value_for_df + np.sum(shap_values_for_df)
            impacts = []
            for _, shap_val_feature in enumerate(shap_values_for_df):
                impact = sigmoid(predicted_logit_for_shap_impact) - sigmoid(predicted_logit_for_shap_impact - shap_val_feature)
                impacts.append(impact)
            shap_df_data["Impact on Probability"] = impacts 
            shap_df_data = shap_df_data.sort_values(by="SHAP Value", key=abs, ascending=False).head(5)
            single_pred_shap_table_html_str = shap_df_data.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.4f}'.format)
            app.logger.info("为不合格样本 (AJAX) 生成了 SHAP 分析。")

            x_original_for_bayes = np.array([input_data_dict[fn] for fn in feature_names])
            best_adjustments, final_adjusted_prob = bayes_opt_adjustment(clf, x_original_for_bayes, feature_names, target_probability_threshold=current_threshold) 
            
            suggestion_lines = []
            for feature_name, adjustment_value in zip(feature_names, best_adjustments):
                if abs(adjustment_value) > 1e-4: 
                    suggestion_lines.append(f"  - 将特征 '{feature_name}' 调整 {adjustment_value:+.4f}")
            
            if suggestion_lines:
                bayes_suggestion_text = "要使此样本合格 (预测概率 ≥ {:.2f}), 建议进行以下最小调整 (或类似组合):\n".format(current_threshold)
                bayes_suggestion_text += "\n".join(suggestion_lines)
                bayes_suggestion_text += f"\n调整后，预测的合格概率约为: {final_adjusted_prob:.3f}"
            else:
                bayes_suggestion_text = "贝叶斯优化未能找到有效的调整建议，使样本合格。"
            app.logger.info(f"贝叶斯优化建议生成完毕。调整后概率: {final_adjusted_prob:.3f}")
        else:
            app.logger.info("合格样本 (AJAX), 不进行 SHAP 分析和贝叶斯优化。")

        model_cache['default_values'] = df_input.iloc[0].to_dict()

        return jsonify({
            "success": True,
            "input_data_html": pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format),
            "prob_ok": f"{prob_ok_scalar:.3f}",
            "label": pred_label_text,
            "threshold_used": f"{current_threshold:.2f}", 
            "shap_table_html": single_pred_shap_table_html_str,
            "shap_waterfall_plot_base64": base64_waterfall_plot_str,
            "bayes_suggestion": bayes_suggestion_text, 
            "is_ng": predicted_numeric_label == ng_label_numeric_val 
        })

    except Exception as e:
        app.logger.error(f"'/predict_single_ajax' 发生严重错误: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"单次预测过程中发生严重错误: {str(e)}" 
        }), 500

# --- 为 `python app.py` 启动添加主程序入口 ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) # Render 会设置 PORT 环境变量
    # host='0.0.0.0' 确保外部可以访问
    # debug=False 用于生产/部署环境
    app.logger.info(f"应用启动中 (通过 python app.py), 监听地址 0.0.0.0, 端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
