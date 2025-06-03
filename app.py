import base64
import io
import os
import logging
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from sklearn.metrics import accuracy_score, recall_score
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

def generate_shap_waterfall_base64(shap_values_single_instance):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_single_instance, show=False, max_display=10)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_plot(clf, feature_names):
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importance)), importance[indices])
    ax.set_title('Feature Importance')
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

def calculate_precise_adjustment(clf, current_values, shap_values, threshold, feature_names):
    adjustments = {}
    current_prob = clf.predict_proba(pd.DataFrame([current_values], columns=feature_names))[0, 1]
    required_boost = max(threshold - current_prob, 0)
    
    if required_boost <= 0:
        return adjustments
    
    sorted_features = sorted(enumerate(shap_values), key=lambda x: -abs(x[1]))
    
    for idx, val in sorted_features:
        feature = feature_names[idx]
        
        # 计算特征敏感度
        delta = 0.001
        temp_values = current_values.copy()
        temp_values[idx] += delta
        temp_prob = clf.predict_proba(pd.DataFrame([temp_values], columns=feature_names))[0, 1]
        sensitivity = (temp_prob - current_prob) / delta
        
        if abs(sensitivity) < 1e-6:
            continue
        
        # 计算需要调整的量
        needed_change = required_boost / sensitivity
        max_change = 0.2 * abs(current_values[idx])
        needed_change = np.clip(needed_change, -max_change, max_change)
        
        # 计算预期提升
        expected_gain = sensitivity * needed_change
        
        adjustments[feature] = {
            'current_value': float(current_values[idx]),
            'adjustment': float(needed_change),
            'new_value': float(current_values[idx] + needed_change),
            'expected_gain': float(expected_gain)
        }
        
        required_boost -= expected_gain
        if required_boost <= 0:
            break
    
    return adjustments

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    global model_cache
    if request.method == 'HEAD':
        return make_response('', 200)
    
    if request.method == 'GET':
        return render_template('index.html',
            show_results=model_cache.get('show_results', False),
            filename=model_cache.get('filename', ''),
            form_inputs=model_cache.get('features', []),
            default_values=model_cache.get('defaults', {}),
            model_metrics=model_cache.get('metrics', None),
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
            model_cache['error'] = "文件无效"
            return redirect(url_for('index'))
        
        try:
            df = pd.read_csv(file)
            if "OK_NG" not in df.columns:
                model_cache['error'] = "缺少OK_NG列"
                return redirect(url_for('index'))
            
            X = df.drop("OK_NG", axis=1).fillna(df.mean(numeric_only=True))
            y = pd.to_numeric(df["OK_NG"], errors='coerce').fillna(0).astype(int)
            
            features = X.columns.tolist()
            clf = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            clf.fit(X, y, sample_weight=compute_sample_weight({0:1, 1:2}, y))
            
            # 计算模型指标
            preds = clf.predict(X)
            model_metrics = {
                'trees': clf.n_estimators,
                'depth': clf.max_depth,
                'lr': clf.learning_rate,
                'accuracy': accuracy_score(y, preds),
                'recall_ok': recall_score(y, preds, pos_label=1),
                'recall_ng': recall_score(y, preds, pos_label=0),
                'threshold': np.percentile(clf.predict_proba(X)[:,1], 95)
            }
            
            model_cache.update({
                'show_results': True,
                'filename': file.filename,
                'features': features,
                'defaults': X.mean().to_dict(),
                'clf': clf,
                'X_train': X.values,
                'metrics': model_metrics,
                'feature_plot': generate_feature_importance_plot(clf, features)
            })
            
        except Exception as e:
            model_cache['error'] = f"处理错误: {str(e)}"
        
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    global model_cache
    if 'clf' not in model_cache:
        return jsonify({'error': '请先训练模型'}), 400
    
    try:
        clf = model_cache['clf']
        features = model_cache['features']
        threshold = model_cache['metrics']['threshold']
        
        # 获取输入数据
        input_data = [float(request.form.get(f)) for f in features]
        
        # 基础预测
        prob = clf.predict_proba([input_data])[0][1]
        is_ng = bool(prob < threshold)
        
        # SHAP分析
        explainer = shap.Explainer(clf, model_cache['X_train'])
        shap_values = explainer.shap_values(np.array([input_data]))[0]
        
        # 生成调整建议
        adjustments = calculate_precise_adjustment(
            clf, input_data, shap_values, 
            threshold, features
        )
        
        # 构建响应
        response = {
            'prob': round(prob, 3),
            'threshold': round(threshold, 2),
            'is_ng': is_ng,
            'shap': [round(float(v), 4) for v in shap_values],
            'adjustments': adjustments,
            'metrics': model_cache['metrics']
        }
        
        if is_ng:
            response['waterfall'] = generate_shap_waterfall_base64(
                shap.Explanation(values=shap_values, base_values=explainer.expected_value[1], data=input_data)
            )
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
