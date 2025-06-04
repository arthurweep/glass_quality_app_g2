# ...前略...
def find_best_threshold_f1(clf, X, y):
    # 用F1-score最大化原则选阈值，兼顾Recall和Precision
    probs_ok = clf.predict_proba(X)[:, 1]
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred = (probs_ok >= t).astype(int)
        f1 = f1_score(y, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh)

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ...略...
    if request.method == 'POST':
        # ...略...
        try:
            # ...数据预处理同前...
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                random_state=42, use_label_encoder=False
            )
            sample_weights = compute_sample_weight(class_weight={0:2, 1:1}, y=y)
            clf.fit(X, y, sample_weight=sample_weights)
            best_threshold = find_best_threshold_f1(clf, X, y)
            # ...其余同前...
        except Exception as e:
            # ...同前...

def calculate_precise_adjustment(clf, current_values_array, shap_values_array, threshold_ok_prob, feature_names):
    # 允许特征最大调整幅度40%，并允许多特征组合调整
    # ...略，核心逻辑同前，只是max_abs_change_ratio=0.4...
    max_abs_change_ratio = 0.4
    # ...其余同前...

@app.route('/adjust_single', methods=['POST'])
def adjust_single():
    # ...同前...
    try:
        # ...同前...
        adjustments, final_prob_after_adjustment = calculate_precise_adjustment(
            clf, current_values_np_array, shap_values_np_array, threshold, features
        )
        # 新增：如果单特征调整无法合格，返回最接近的概率和建议
        if not adjustments:
            # 尝试多特征联合调整（如全都往有利方向调最大幅度）
            # ...可选进阶：实现联合调整逻辑...
            return jsonify({
                'adjustments': {},
                'final_prob_after_adjustment': float(final_prob_after_adjustment),
                'msg': '即使大幅调整，样本也难以合格，建议人工复核或检查模型。'
            })
        return jsonify({
            'adjustments': adjustments,
            'final_prob_after_adjustment': float(final_prob_after_adjustment)
        })
    except Exception as e:
        # ...同前...
