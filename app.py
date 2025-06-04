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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

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

def generate_shap_waterfall_base64(shap_explanation_object):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation_object, show=False, max_display=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names):
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    num_features_to_display = min(len(feature_names), 15)
    ax.bar(range(num_features_to_display), [float(importance[i]) for i in indices[:num_features_to_display]])
    ax.set_title('特征重要性', fontsize=14)
    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('重要性分数', fontsize=12)
    ax.set_xticks(range(num_features_to_display))
    ax.set_xticklabels([FIELD_LABELS.get(feature_names[i], feature_names[i]) for i in indices[:num_features_to_display]], rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def find_best_threshold_for_ng_recall(clf, X, y, ng_label=0, ok_label=1, min_target_ng_recall=0.95):
    probs_ok = clf.predict_proba(X)[:, ok_label]
    best_threshold = 0.5
    candidate_thresholds_met_target = []
    for threshold_candidate in np.arange(0.01, 1.0, 0.005):
        y_pred = (probs_ok >= threshold_candidate).astype(int)
        recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
        if recall_ng_current >= min_target_ng_recall:
            precision_ok_current = precision_score(y, y_pred, pos_label=ok_label, zero_division=0)
            candidate_thresholds_met_target.append((threshold_candidate, precision_ok_current, recall_ng_current))
    if candidate_thresholds_met_target:
        candidate_thresholds_met_target.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
        best_threshold = candidate_thresholds_met_target[0][0]
        app.logger.info(f"找到优化NG召回率的平衡阈值: {best_threshold:.3f} (NG召回率 >= {min_target_ng_recall}, OK精确率最高)")
    else:
        max_ng_recall_overall = -1.0
        best_threshold_for_max_ng_recall = 0.5
        for threshold_candidate in np.arange(0.01, 1.0, 0.005):
            y_pred = (probs_ok >= threshold_candidate).astype(int)
            recall_ng_current = recall_score(y, y_pred, pos_label=ng_label, zero_division=0)
            if recall_ng_current > max_ng_recall_overall:
                max_ng_recall_overall = recall_ng_current
                best_threshold_for_max_ng_recall = threshold_candidate
            elif recall_ng_current == max_ng_recall_overall and threshold_candidate < best_threshold_for_max_ng_recall:
                best_threshold_for_max_ng_recall = threshold_candidate
        best_threshold = best_threshold_for_max_ng_recall
        app.logger.warning(f"未能达到目标NG召回率 {min_target_ng_recall}。选择最大化NG召回率的阈值: {best_threshold:.3f} (此时NG召回率: {max_ng_recall_overall:.2f})")
    return float(best_threshold)

def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names):
    adjustments = {}
    current_values_np = np.array(current_values_array, dtype=float).flatten()
    shap_values_np = np.array(shap_values_array, dtype=float).flatten()
    current_prob_ok = clf.predict_proba(current_values_np.reshape(1, -1))[0, 1]
    required_boost = float(max(threshold_ok_prob - current_prob_ok, 0.0))
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
        max_abs_change_ratio = 0.4
        max_abs_change_value = abs(current_values_np[idx] * max_abs_change_ratio) if current_values_np[idx] !=
