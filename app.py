import base64
import io
import os
import logging
import math 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

FIELD_LABELS = {
    "F_cut_act": "刀头实际压力", "v_cut_act": "切割实际速度", "F_break_peak": "崩边力峰值",
    "v_wheel_act": "磨轮线速度", "F_wheel_act": "磨轮压紧力", "P_cool_act": "冷却水压力",
    "t_glass_meas": "玻璃厚度"
}
model_cache = {}

def fig_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_shap_waterfall_base64(shap_explanation_object_with_english_names):
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation_object_with_english_names, show=False, max_display=10)
    plt.title("SHAP Waterfall Plot (Feature Contributions to OK Probability)", fontsize=14)
    plt.tight_layout(); return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names_original_english):
    # ... (与上一版 v9-v10 逻辑一致) ...
    booster = clf.get_booster()
    importance_scores = booster.get_score(importance_type='weight') 
    if not importance_scores:
        if hasattr(clf, 'feature_importances_') and clf.feature_importances_ is not None:
            importances_sklearn = clf.feature_importances_
            importance_scores = {name: score for name, score in zip(feature_names_original_english, importances_sklearn)}
        else:
            app.logger.warning("无法获取特征重要性分数。")
            return None
    mapped_importance = {}
    if importance_scores and all(isinstance(k, str) and k.startswith('f') and k[1:].isdigit() for k in importance_scores.keys()):
        temp_map = {f"f{i}": name for i, name in enumerate(feature_names_original_english)}
        for f_key, score in importance_scores.items():
            if f_key in temp_map: mapped_importance[temp_map[f_key]] = score
            else: mapped_importance[f_key] = score 
        if not mapped_importance and importance_scores : mapped_importance = importance_scores
    elif importance_scores: mapped_importance = importance_scores
    if not mapped_importance: 
        app.logger.warning("最终未能准备好用于绘图的特征重要性数据。")
        return None
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
    # ... (与上一版 v9-v10 逻辑一致) ...
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

def calculate_adjustment_direct_search_v11(clf, current_values_array, shap_values_array, target_ok_prob_original, feature_names, initial_is_ng):
    """
    直接搜索调整算法 V11：更努力地寻找调整方案，简化版爬山。
    """
    original_values_np = np.array(current_values_array, dtype=float).flatten()
    current_adjusted_values = original_values_np.copy()
    
    initial_prob_ok = clf.predict_proba(original_values_np.reshape(1, -1))[0, 1]
    app.logger.info(f"调整(v11)开始：初始NG={initial_is_ng}, 初始OK概率={initial_prob_ok:.4f}, 原始目标={target_ok_prob_original:.4f}")

    if not initial_is_ng and initial_prob_ok >= target_ok_prob_original:
        return {}, float(initial_prob_ok), "样本当前已合格且满足目标概率，无需调整。"

    effective_target_prob = min(target_ok_prob_original + 0.015, 0.999) # 内部调整目标
    app.logger.info(f"内部调整目标概率设为: {effective_target_prob:.4f}")

    # --- 调整参数 ---
    max_total_iterations = 150  # 增加总迭代轮次
    
    # 特征调整的“极宽松”边界
    max_abs_change_ratio_from_original = 10.0 
    abs_change_limit_for_zero_original = 200.0 

    min_significant_prob_gain_for
