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
from flask import Flask, render_template, request, redirect, url_for, jsonify
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
    app.logger.warning(f"Matplotlib font config warning: {e}")

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
    else:
        max_recall = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            recall_ok = recall_score(y, y_pred, pos_label=pos_label, zero_division=0)
            if recall_ok > max_recall:
                max_recall = recall_ok
                best_threshold = thresh
    return best_threshold

def bayes_opt_adjustment(clf, x_original, feature_names, target_probability_threshold=0.5):
    dims = [Real(-0.5, 0.5, name=fn) for fn in feature_names]
    @use_named_args(dims)
    def objective_function(**kwargs_adjustments):
        x_adj = x_original.copy()
        total_adjustment_magnitude = 0
        for i, feature_name in enumerate(feature_names):
            adjustment = kwargs_adjustments[feature_name]
            x_adj[i] += adjustment
            total_adjustment_magnitude += abs(adjustment)
        df_adj = pd.DataFrame([x_adj], columns=feature_names)
        predicted_probability_ok = clf.predict_proba(df_adj)[0, 1]
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

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_cache
    app.logger.info(f"访问 '/' 路由, 方法: {request.method}")
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
    app.logger.warning("接收到未知类型的请求或意外的流程, 重定向到主页。")
    model_cache.clear()
    return redirect(url_for('index'))

@app.route('/predict_single_ajax', methods=['POST'])
def predict_single_ajax():
    global model_cache
    if 'clf' not in model_cache or 'best_thresh' not in model_cache:
        return jsonify({"success": False, "error": "请先上传并处理一个CSV文件, 然后再进行单一样本预测。"}), 400
    try:
        clf = model_cache['clf']
        threshold = model_cache['best_thresh']
        X_background = model_cache.get('X_train_df_for_explainer')
        feature_names = model_cache.get('feature_names')
        ok_label = 1
        ng_label = 0
        input_data_dict = {}
        form_data = request.form
        for fn in feature_names:
            val = form_data.get(fn)
            if val is None or val.strip() == '':
                return jsonify({"success": False, "error": f"特征 '{fn}' 的值不能为空。", "field": fn}), 400
            try:
                input_data_dict[fn] = float(val)
            except ValueError:
                return jsonify({"success": False, "error": f"特征 '{fn}' 的输入无效，请输入数字。当前值: '{val}'", "field": fn}), 400
        df_input = pd.DataFrame([input_data_dict], columns=feature_names)
        prob_ok = clf.predict_proba(df_input)[:, ok_label][0]
        pred_label = "✅ 合格 (OK)" if prob_ok >= threshold else "❌ 不合格 (NG)"
        base64_waterfall = None
        shap_table_html = None
        bayes_suggestion = None
        if prob_ok < threshold:
            explainer = shap.Explainer(clf, X_background)
            shap_values = explainer(df_input)
            base64_waterfall = generate_shap_waterfall_base64(shap_values[0])
            shap_vals = shap_values.values[0]
            base_val = shap_values.base_values[0]
            shap_df = pd.DataFrame({
                "特征": feature_names,
                "SHAP值": shap_vals
            })
            predicted_logit = base_val + np.sum(shap_vals)
            impacts = [sigmoid(predicted_logit) - sigmoid(predicted_logit - v) for v in shap_vals]
            shap_df["概率影响"] = impacts
            shap_df = shap_df.sort_values(by="SHAP值", key=abs, ascending=False).head(5)
            shap_table_html = shap_df.to_html(classes='table table-sm table-striped table-hover', index=False, float_format='{:.4f}'.format)
            x_orig = np.array([input_data_dict[fn] for fn in feature_names])
            best_adj, best_prob = bayes_opt_adjustment(clf, x_orig, feature_names, target_probability_threshold=threshold)
            suggestion_lines = []
            for fn, adj in zip(feature_names, best_adj):
                if abs(adj) > 1e-4:
                    suggestion_lines.append(f"将特征 '{fn}' 调整 {adj:+.4f}")
            if suggestion_lines:
                bayes_suggestion = "建议调整:\n" + "\n".join(suggestion_lines) + f"\n以使预测概率达到阈值，优化后概率约为 {best_prob:.3f}"
            else:
                bayes_suggestion = "无法找到有效调整使预测合格。"
        model_cache['default_values'] = df_input.iloc[0].to_dict()
        return jsonify({
            "success": True,
            "input_data_html": pd.DataFrame([input_data_dict]).to_html(classes='table table-sm table-bordered', index=False, float_format='{:.2f}'.format),
            "prob_ok": f"{prob_ok:.3f}",
            "label": pred_label,
            "threshold_used": f"{threshold:.2f}",
            "shap_table_html": shap_table_html,
            "shap_waterfall_plot_base64": base64_waterfall,
            "bayes_suggestion": bayes_suggestion,
            "is_ng": prob_ok < threshold
        })
    except Exception as e:
        app.logger.error(f"'/predict_single_ajax' 发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"单次预测过程中发生错误: {str(e)}"}), 500
