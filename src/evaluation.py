"""Métricas y esquemas de validación de las dos tareas.

Uso desde un notebook (con la raíz del repo en sys.path):
    from src import evaluation as E
    folds = E.cv_temporal(panel["FECHA"])                          # Tarea 1: origen móvil por fechas
    r = E.cv_puntajes(modelo, X, y, folds, E.metricas_conteo)      # métricas por pliegue
    r = E.cv_puntajes(modelo, X, y, E.CV_REPETIDA.split(X, y), E.metricas_multiclase, proba=True)   # Tarea 2
    print(E.comparacion_pareada(r_a["pr_auc_fatal"], r_b["pr_auc_fatal"]))
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, d2_tweedie_score,
                             f1_score, log_loss, mean_absolute_error, mean_poisson_deviance, mean_squared_error,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

SEED = 42
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)                          # tuning dentro de train
CV_REPETIDA = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)  # comparación final (Tarea 2)


def cv_temporal(fechas, n_splits=5, min_train_dias=180):
    """Validación con origen móvil (Tarea 1): los días posteriores a los primeros `min_train_dias` se parten en
    `n_splits` bloques consecutivos; cada pliegue entrena con TODOS los días anteriores a su bloque y evalúa en
    él. Se parte por FECHAS, no por filas, para que un mismo día (los 24 departamentos) no quede repartido entre
    train y test. El mínimo de 180 días evita pliegues que entrenan con 2-3 meses y evalúan en plena cuarentena."""
    dias = np.sort(fechas.unique())
    for bloque in np.array_split(dias[min_train_dias:], n_splits):
        yield np.flatnonzero(fechas < bloque[0]), np.flatnonzero(fechas.isin(bloque))


def metricas_conteo(y, pred):
    """Tarea 1 · N_ACC. La principal es la devianza de Poisson (y su D², la fracción de devianza explicada
    frente a predecir la media): el MAE premia predecir 0, que es el valor más común (66 % de los días)."""
    pred = np.clip(pred, 1e-6, None)
    return dict(poisson_dev=mean_poisson_deviance(y, pred), d2_poisson=d2_tweedie_score(y, pred, power=1),
                mae=mean_absolute_error(y, pred), rmse=np.sqrt(mean_squared_error(y, pred)))


def metricas_binaria(y, prob):
    """Tarea 1 · HAY_FATAL (≈ 6 % de los días): PR-AUC (al azar = prevalencia), ROC-AUC y Brier (calibración)."""
    return dict(pr_auc=average_precision_score(y, prob), roc_auc=roc_auc_score(y, prob),
                brier=brier_score_loss(y, prob), prevalencia=float(np.mean(y)))


def metricas_multiclase(y, proba):
    """Tarea 2 · GRAVEDAD (0 sin víctimas, 1 heridos, 2 fatal). Macro-F1 y balanced accuracy pesan igual las
    tres clases; la PR-AUC de la clase fatal (one-vs-rest) se compara con la del Modelo 2 del avance (0.266)."""
    y = np.asarray(y)
    pred = proba.argmax(axis=1)
    return dict(macro_f1=f1_score(y, pred, average="macro"), balanced_acc=balanced_accuracy_score(y, pred),
                pr_auc_fatal=average_precision_score(y == 2, proba[:, 2]), log_loss=log_loss(y, proba, labels=[0, 1, 2]))


def predecir(modelo, X, proba):
    if not proba:
        return modelo.predict(X)
    p = modelo.predict_proba(X)
    return p[:, 1] if p.shape[1] == 2 else p


def cv_puntajes(modelo, X, y, folds, metricas, proba=False):
    """Entrena una copia de `modelo` en cada pliegue y devuelve un DataFrame con las métricas por pliegue.
    Con los mismos `folds`, todos los modelos se evalúan en las mismas particiones (comparación pareada)."""
    filas = []
    for tr, te in folds:
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        filas.append(metricas(y.iloc[te], predecir(m, X.iloc[te], proba)))
    return pd.DataFrame(filas)


def comparacion_pareada(a, b, mayor_es_mejor=True):
    """Diferencia a − b pliegue a pliegue, con IC 95 % aproximado.

    ponytail: IC normal sobre pliegues que comparten datos, así que es optimista; si se necesita una prueba
    formal, usar la corrección de Nadeau y Bengio (2003).
    """
    d = np.asarray(a) - np.asarray(b)
    h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    gana = (d > 0) if mayor_es_mejor else (d < 0)
    return f"{d.mean():+.3f} (IC95% aprox. {d.mean() - h:+.3f} a {d.mean() + h:+.3f}), mejor en {gana.sum()} de {len(d)} pliegues"


if __name__ == "__main__":  # chequeo rápido de las métricas y de la partición temporal
    f = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]))
    for tr, te in cv_temporal(f, n_splits=2, min_train_dias=1):
        assert f.iloc[tr].max() < f.iloc[te].min(), "fuga temporal"
    proba = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    m = metricas_multiclase([0, 1, 2], proba)
    assert m["macro_f1"] == 1 and m["pr_auc_fatal"] == 1
    assert metricas_conteo(np.array([1, 2]), np.array([1, 2]))["d2_poisson"] == 1
    assert comparacion_pareada([.2, .3], [.1, .1]).startswith("+0.150")
    print("OK evaluation")
