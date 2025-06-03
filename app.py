@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    # ... [其他代码保持不变] ...
    
    if request.method == 'GET':
        return render_template('index.html',
            # ... [原有参数] ...
            model_metrics=model_cache.get('model_metrics', None)  # 新增模型指标参数
        )

@app.route('/predict_single_ajax', methods=['POST'])
def predict_single_ajax():
    # ... [原有代码] ...

    # 新增模型指标计算
    model_metrics = {
        'n_estimators': clf.get_params().get('n_estimators', 'N/A'),
        'max_depth': clf.get_params().get('max_depth', 'N/A'),
        'learning_rate': clf.get_params().get('learning_rate', 'N/A'),
        'train_accuracy': float(accuracy_score(y, clf.predict(X))),
        'train_recall_ok': float(recall_score(y, clf.predict(X), pos_label=1)),
        'train_recall_ng': float(recall_score(y, clf.predict(X), pos_label=0)),
    }
    model_cache['model_metrics'] = model_metrics

    return jsonify({
        # ... [原有返回字段] ...
        "model_metrics": model_metrics
    })
