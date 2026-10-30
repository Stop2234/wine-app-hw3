"""
Демо-приложение для модели отбора хороших вин.

Ответ на вопрос задания 4.5:
- модель загружается через st.cache_resource, потому что joblib.load возвращает
  изменяемый объект (обученный пайплайн), который должен существовать на сервере
  в одном экземпляре и разделяться между всеми сессиями; cache_resource хранит
  сам объект, не сериализуя его;
- таблицы (test.csv, вероятности) грузятся через st.cache_data, потому что это
  данные: Streamlit умеет их сериализовать и отдавать копии каждой сессии,
  что защищает закэшированный оригинал от случайных изменений;
- если вызвать joblib.load без декоратора, модель будет читаться с диска при
  каждом перезапуске скрипта, то есть при каждом движении слайдера, и
  приложение станет заметно тормозить.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((BASE_DIR / 'config.json').read_text(encoding='utf-8'))


@st.cache_data
def load_test():
    return pd.read_csv(BASE_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test_df = load_test()

feature_cols = [c for c in test_df.columns if c != 'target']
X_test = test_df[feature_cols]
y_test = test_df['target'].to_numpy()
proba_test = test_proba(model, X_test)

st.set_page_config(page_title='Подбор вин', layout='wide')
st.title('Модель отбора хороших вин')

threshold = st.sidebar.slider(
    'Порог отбора', 0.0, 1.0, float(config['threshold']), 0.01,
    key='threshold',
    help='Вино попадает в подборку, если вероятность класса 1 не ниже порога')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab_one:
    st.subheader('Оценка одного вина')
    values = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_cols):
        if col == 'is_red':
            choice = cols[i % 3].radio('Тип вина', ['Белое', 'Красное'],
                                       horizontal=True)
            values[col] = 1.0 if choice == 'Красное' else 0.0
        else:
            values[col] = cols[i % 3].number_input(
                col,
                min_value=float(X_test[col].min()),
                max_value=float(X_test[col].max()),
                value=float(X_test[col].median()))
    X_one = pd.DataFrame([values])[feature_cols]
    p_one = float(model.predict_proba(X_one)[0, 1])
    st.metric('Вероятность, что вино хорошее', f'{p_one:.3f}')
    if p_one >= threshold:
        st.success(f'При пороге {threshold:.2f} вино ПОПАДАЕТ в подборку')
    else:
        st.warning(f'При пороге {threshold:.2f} вино НЕ попадает в подборку')

with tab_metrics:
    st.subheader('Метрики на тестовой выборке при текущем пороге')
    y_pred = (proba_test >= threshold).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Precision', f"{metrics.precision_score(y_test, y_pred, zero_division=0):.3f}")
    c2.metric('Recall', f'{metrics.recall_score(y_test, y_pred):.3f}')
    c3.metric('F1', f'{metrics.f1_score(y_test, y_pred):.3f}')
    c4.metric('Accuracy', f'{metrics.accuracy_score(y_test, y_pred):.3f}')

    col_cm, col_roc, col_pr = st.columns(3)

    with col_cm:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, colorbar=False)
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)

    with col_roc:
        fpr, tpr, thr_roc = metrics.roc_curve(y_test, proba_test)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y_test, proba_test):.3f}")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        i = int(np.argmin(np.abs(thr_roc - threshold)))
        ax.scatter(fpr[i], tpr[i], color='red', zorder=5, label='текущий порог')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC-кривая')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    with col_pr:
        prec, rec, thr_pr = metrics.precision_recall_curve(y_test, proba_test)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(rec, prec, label=f"PR-AUC = {metrics.average_precision_score(y_test, proba_test):.3f}")
        ax.axhline(y_test.mean(), color='gray', linestyle='--', linewidth=1,
                   label='случайная модель')
        i = int(np.argmin(np.abs(thr_pr - threshold)))
        ax.scatter(rec[i], prec[i], color='red', zorder=5, label='текущий порог')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR-кривая')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        'Подсказка для сомелье. Если увеличить порог, в подборку попадут только '
        'вина, в которых модель уверена: подборка станет меньше и чище '
        '(precision вырастет), но часть хороших вин потеряется (recall упадёт). '
        'Если уменьшить порог, подборка станет шире и хороших вин в ней будет '
        'больше, но посредственных тоже прибавится.')

with tab_batch:
    st.subheader('Пакетное предсказание')
    uploaded = st.file_uploader(
        'CSV-файл с признаками вин (без столбца target)', type='csv')
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')
        else:
            missing = [c for c in feature_cols if c not in data.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                probs = model.predict_proba(data[feature_cols])[:, 1]
                result = data.copy()
                result['proba_good'] = probs.round(4)
                result['decision'] = np.where(
                    probs >= threshold, 'в подборку', 'не в подборку')
                st.write(f'Строк в файле: {len(result)}, '
                         f'в подборку при пороге {threshold:.2f}: '
                         f"{int((probs >= threshold).sum())}")
                st.dataframe(result)
                st.download_button(
                    'Скачать результат (CSV)',
                    result.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv')

with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    coef = model.named_steps['logisticregression'].coef_[0]
    coef_s = pd.Series(coef, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in coef_s]
    ax.barh(coef_s.index, coef_s.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('коэффициент (после масштабирования признаков)')
    st.pyplot(fig)
    plt.close(fig)

    top_for = coef_s.tail(3).index[::-1].tolist()
    top_against = coef_s.head(3).index.tolist()
    st.write('Сильнее всего за хорошее вино голосуют: ' + ', '.join(top_for) + '.')
    st.write('Сильнее всего против: ' + ', '.join(top_against) + '.')
