"""Métricas y evaluación de los modelos.

Uso desde un notebook (con la raíz del repo en sys.path):
    from src import evaluation as E
    fila, prob = E.evaluar_clf("Modelo 1", modelo, tr, te, cols)     # test 80/20, umbral elegido por CV en train
    pr, roc = E.cv_repetida(modelo, df[cols], df["ES_FATAL"])        # PR-AUC y ROC-AUC en CV 5×5
    print(E.comparacion_pareada(pr_modelo, pr_baseline))
"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_poisson_deviance, mean_squared_error, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate

SEED = 42
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)                          # selección dentro de train
CV_REPETIDA = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)  # comparación final


def umbral_f1(modelo, X, y, cv=CV):
    """Umbral que maximiza el F1 en predicciones fuera de muestra (CV) de train; el test no interviene.

    Hace falta porque cada modelo produce probabilidades en otra escala (p. ej. con pesos de clase),
    así que un umbral fijo de 0.5 no los compara en igualdad de condiciones.
    """
    oof = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y, oof)
    return t[np.argmax((2 * p * r / (p + r + 1e-12))[:-1])]


def metricas_clf(y, prob, umbral):
    """Métricas de la Tarea A. PR-AUC es la principal (clase fatal ≈ 12 %; su valor al azar es la prevalencia)."""
    pred = (prob >= umbral).astype(int)
    return dict(roc_auc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob), f1=f1_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                balanced_acc=balanced_accuracy_score(y, pred), prevalencia_test=float(np.mean(y)))


def evaluar_clf(nombre, modelo, tr, te, cols, y="ES_FATAL", umbral=None, split="aleatorio 80/20"):
    """Entrena en `tr`, evalúa en `te`. Si no se da `umbral`, se elige con umbral_f1 sobre `tr`."""
    umbral = umbral_f1(modelo, tr[cols], tr[y]) if umbral is None else umbral
    prob = modelo.fit(tr[cols], tr[y]).predict_proba(te[cols])[:, 1]
    fila = dict(tarea="A_clasificacion", modelo=nombre, split=split, umbral=round(float(umbral), 3))
    return fila | metricas_clf(te[y], prob, umbral), prob


def evaluar_conteo(nombre, modelo, tr, te, cols, y="NUM_FALLECIDOS"):
    """Tarea B: MAE, RMSE y devianza de Poisson del número de fallecidos por accidente."""
    pred = np.clip(modelo.fit(tr[cols], tr[y]).predict(te[cols]), 1e-6, None)
    return dict(tarea="B_regresion_conteo", modelo=nombre, split="aleatorio 80/20",
                mae=mean_absolute_error(te[y], pred), rmse=np.sqrt(mean_squared_error(te[y], pred)),
                poisson_deviance=mean_poisson_deviance(te[y], pred)), pred


def cv_repetida(modelo, X, y, cv=CV_REPETIDA):
    """PR-AUC y ROC-AUC en cada una de las 25 particiones. Con el mismo `cv`, todos los modelos ven las mismas."""
    r = cross_validate(modelo, X, y, cv=cv, scoring=["average_precision", "roc_auc"], n_jobs=-1)
    return r["test_average_precision"], r["test_roc_auc"]


def comparacion_pareada(a, b):
    """Diferencia a − b partición a partición, con IC 95 % aproximado.

    ponytail: IC normal sobre particiones que comparten datos, así que es optimista; si se necesita una
    prueba formal, usar la corrección de Nadeau y Bengio (2003).
    """
    d = np.asarray(a) - np.asarray(b)
    h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.3f} (IC95% aprox. {d.mean() - h:+.3f} a {d.mean() + h:+.3f}), mejor en {(d > 0).sum()} de {len(d)} particiones"


def fila_cv(nombre, pr, roc, prevalencia):
    """Fila de results/metrics.csv con el resumen de la CV repetida."""
    return dict(tarea="A_clasificacion", modelo=nombre, split="CV 5x5 (dataset completo)", roc_auc=roc.mean(),
                pr_auc=pr.mean(), pr_auc_sd=pr.std(), prevalencia_test=prevalencia)


if __name__ == "__main__":  # chequeo rápido de las métricas y del umbral
    y = np.array([0, 0, 0, 1, 1]); prob = np.array([.1, .2, .3, .8, .9])
    m = metricas_clf(y, prob, 0.5)
    assert m["pr_auc"] == 1 and m["f1"] == 1 and m["balanced_acc"] == 1
    assert comparacion_pareada([.2, .3], [.1, .1]).startswith("+0.150")
    print("OK evaluation")
