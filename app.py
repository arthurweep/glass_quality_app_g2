import base64
import io
import os
import logging
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

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
    candidate_thresholds = []
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        recall_ok = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
        if recall_ok >= min_recall:
            candidate_thresholds.append(thresh)
    if candidate_thresholds:
        best_threshold = max(candidate_thresholds)
        app.logger.info(f"找到满足最小召回率 {min_recall} 的最佳阈值: {best_threshold:.2f}")
    else:
        max_recall = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            recall_ok = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
            if recall_ok > max_recall:
                max_recall = recall_ok
                best_threshold = thresh
        app.logger.warning(f"未找到满足最小召回率 {min_recall} 的阈值。选择最大化召回率的阈值: {best_threshold:.2f}")
    return best_threshold

def simple_optimization_suggestion(clf, x_original, feature_names, shap_values, target_threshold=0.5):
    """
    基于SHAP值的快速优化建议 - 替代复杂的贝叶斯优化
    """
    app.logger.info("开始快速优化建议计算...")
    start_time = time.time()
    
    # 基于SHAP值确定调整方向和幅度
    suggestions = []
    x_adjusted = x_original.copy()
    
    # 按SHAP值绝对值排序，优先调整影响最大的特征
    shap_importance = [(abs(val), i, val) for i, val in enumerate(shap_values)]
    shap_importance.sort(reverse=True)
    
    total_adjustment_budget = 0.3  # 总调整预算
    
    for abs_shap, feature_idx, shap_val in shap_importance[:3]:  # 只调整前3个最重要的特征
        feature_name = feature_names[feature_idx]
        
        # 如果SHAP值为负（降低合格概率），我们需要反向调整
        if shap_val < 0:
            # 建议增加该特征值
            adjustment = min(0.1, total_adjustment_budget * 0.4)
            x_adjusted[feature_idx] += adjustment
            suggestions.append(f"将 '{feature_name}' 增加 {adjustment:.3f}")
        else:
            # SHAP值为正但仍不够，可能需要进一步增加
            adjustment = min(0.05, total_adjustment_budget * 0.2)
            x_adjusted[feature_idx] += adjustment
            suggestions.append(f"将 '{feature_name}' 微调 +{adjustment:.3f}")
        
        total_adjustment_budget -= abs(adjustment)
        if total_adjustment_budget <= 0:
            break
    
    # 计算调整后的预测概率
    df_adjusted = pd.DataFrame([x_adjusted], columns=feature_names)
    final_prob = clf.predict_proba(df_adjusted)[0, 1]
    
    end_time = time.time()
    app.logger.info(f"快速优化建议计算完成，耗时: {end_time - start_time:.2f} 秒")
    
    return suggestions, final_prob

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由, 方法: {request.method}")

    if request.method == 'HEAD':
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
                    app.logger.warning(f"文件 {filename} 中的特征列包含NaN，使用均值填充。")
                    X = X.fillna(X.mean())
                y = df["OK_NG"].copy()
                if not pd.api.types.is_numeric_dtype(y) or not y.isin([0, 1]).all():
                    y = pd.to_numeric(y, errors='raise')
                    if not y.isin([0, 1]).all():
                        model_cache['error_message_to_display'] = "列 'OK_NG' 必须只包含数字 0 或 1。"
                        return redirect(url_for('index'))
                y = y.astype(int)
                ok_label = 1
                ng_label = 0
                model_cache['feature_names'] = list(X.columns)
                model_cache['X_train_df_for_explainer'] = X.copy()
                weights = compute_sample_weight(class_weight={ng_label: 1.0, ok_label: 2.0}, y=y)
                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        objective="binary:logistic", eval_metric="logloss",
                                        random_state=42, use_label_encoder=False)
                clf.fit(X, y, sample_weight=weights)
                model_cache['clf'] = clf
                app.logger.info("XGBoost 模型训练完成。")
                best_threshold = find_best_threshold(clf, X, y, pos_label=ok_label, min_recall=0.95)
                model_cache['best_thresh'] = best_threshold
                model_cache['recommended_threshold_text'] = f"ℹ️ 当前最优阈值: {best_threshold:.2f} (OK召回率≥0.95优先)"
                model_cache['best_recommendation_html'] = f"<p>当前自动选择的分类阈值为 <strong>{best_threshold:.2f}</strong>。</p>"
                probs = clf.predict_proba(X)[:, ok_label]
                preds = (probs >= best_threshold).astype(int)
                report = classification_report(y, preds, output_dict=True, zero_division=0,
                                               labels=[ng_label, ok_label],
                                               target_names=['NG_Class_0', 'OK_Class_1'])
                metrics = [{
                    "threshold": best_threshold,
                    "accuracy": report["accuracy"],
                    "NG_recall": report["NG_Class_0"]["recall"],
                    "OK_recall": report["OK_Class_1"]["recall"],
                    "OK_precision": report["OK_Class_1"]["precision"],
                    "f1_score_weighted": f1_score(y, preds, average='weighted', zero_division=0)
                }]
                metrics_df = pd.DataFrame(metrics)
                model_cache['metrics_df_html'] = metrics_df.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.2f}'.format)
                scan_thresholds = np.arange(0.01, 1.0, 0.01)
                ng_recalls, ok_recalls, f1_scores = [], [], []
                for t in scan_thresholds:
                    p = (probs >= t).astype(int)
                    r = classification_report(y, p, output_dict=True, zero_division=0,
                                              labels=[ng_label, ok_label],
                                              target_names=['NG_Class_0', 'OK_Class_1'])
                    ng_recalls.append(r["NG_Class_0"]["recall"])
                    ok_recalls.append(r["OK_Class_1"]["recall"])
                    f1_scores.append(f1_score(y, p, average='weighted', zero_division=0))
                fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
                ax_perf.plot(scan_thresholds, ng_recalls, label="NG (Class 0) Recall", color="red")
                ax_perf.plot(scan_thresholds, ok_recalls, label="OK (Class 1) Recall", color="green")
                ax_perf.plot(scan_thresholds, f1_scores, label="F1 Score (Weighted)", color="blue")
                ax_perf.axvline(x=best_threshold, color="purple", linestyle="--", label=f"Best Threshold: {best_threshold:.2f}")
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
            model_cache.clear()
            model_cache['error_message_to_display'] = "文件类型无效。请上传 CSV 文件。"
            return redirect(url_for('index'))
    
    app.logger.warning(f"接收到未明确处理的请求方法 '{request.method}' 在 '/' 路由, 重定向到主页。")
    model_cache.clear()
    return redirect(url_for('index'))

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
        threshold = model_cache['best_thresh']
        X_background = model_cache.get('X_train_df_for_explainer')
        feature_names = model_cache.get('feature_names')
        ok_label = 1
        ng_label = 0

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

        prob_ok_array = clf.predict_proba(df_input)[:, ok_label]
        prob_ok_scalar = prob_ok_array[0] if prob_ok_array.ndim > 0 else prob_ok_array
        predicted_numeric_label = ok_label if prob_ok_scalar >= threshold else ng_label
        pred_label_text = "✅ 合格 (OK)" if predicted_numeric_label == ok_label else "❌ 不合格 (NG)"
        app.logger.info(f"单样本预测概率 (OK): {prob_ok_scalar:.3f}, 使用阈值: {threshold:.2f}, 预测数值标签: {predicted_numeric_label}, 文本标签: {pred_label_text}")

        base64_waterfall_plot_str = None
        single_pred_shap_table_html_str = None
        optimization_suggestion_text = None

        if predicted_numeric_label == ng_label:
            app.logger.info("样本预测为不合格, 开始 SHAP 分析和快速优化建议...")
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

            # 使用快速优化建议替代贝叶斯优化
            x_original_for_optimization = np.array([input_data_dict[fn] for fn in feature_names])
            suggestions, final_optimized_prob = simple_optimization_suggestion(
                clf, x_original_for_optimization, feature_names, shap_values_for_df, target_threshold=threshold
            )

            if suggestions:
                optimization_suggestion_text = "快速优化建议 (基于SHAP分析):\n"
                optimization_suggestion_text += "\n".join(suggestions)
                optimization_suggestion_text += f"\n\n预期调整后合格概率: {final_optimized_prob:.3f}"
                if final_optimized_prob >= threshold:
                    optimization_suggestion_text += f" (≥ 阈值 {threshold:.2f}, 预期合格)"
                else:
                    optimization_suggestion_text += f" (< 阈值 {threshold:.2f}, 可能仍需进一步调整)"
            else:
                optimization_suggestion_text = "暂无明确的优化建议。建议检查关键特征参数。"

            app.logger.info(f"快速优化建议生成完毕。预期优化后概率: {final_optimized_prob:.3f}")
        else:
            app.logger.info("合格样本 (AJAX), 不进行 SHAP 分析和优化建议。")

        model_cache['default_values'] = df_input.iloc[0].to_dict()

        return jsonify({
            "success": True,
            "input_data_html": pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format),
            "prob_ok": f"{prob_ok_scalar:.3f}",
            "label": pred_label_text,
            "threshold_used": f"{threshold:.2f}",
            "shap_table_html": single_pred_shap_table_html_str,
            "shap_waterfall_plot_base64": base64_waterfall_plot_str,
            "bayes_suggestion": optimization_suggestion_text,
            "is_ng": predicted_numeric_label == ng_label
        })

    except Exception as e:
        app.logger.error(f"'/predict_single_ajax' 发生严重错误: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"单次预测过程中发生严重错误: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"应用启动中 (通过 python app.py), 监听地址 0.0.0.0, 端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
