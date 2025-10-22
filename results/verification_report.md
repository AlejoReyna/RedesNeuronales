## Verificación y validación del proyecto HornoPrediction

- Fecha: 2025-10-22
- Autor de la verificación: Automatizada (GPT)
- ok_to_submit: false

### Resumen ejecutivo
- Resultado del pipeline: OK (entrenamiento, evaluación y reporte generados).
- Métricas: 100% en accuracy/precision/recall/F1 en test temporal 70/30, split aleatorio 70/30 y en 5-fold CV. Muy inusual.
- Fuga de datos: No se detectaron correlaciones perfectas (|corr| ≥ 0.999), columnas idénticas al target ni variables constantes. Sin evidencia directa de leakage.
- Riesgo principal: Métricas perfectas y dependencia extrema de las 5 features más correlacionadas. Al eliminarlas, la accuracy cae a ~0.56. Esto sugiere separabilidad casi trivial del dataset (p.ej., datos sintéticos con señales fuertes) o diseño que facilita el aprendizaje perfecto.
- Recomendación: NO entregar sin revisión adicional. Ver acciones sugeridas abajo.

---

### 1) Comprobaciones básicas del repo
- Archivos clave presentes:
  - data/Variables_Horno.csv
  - src/preprocess.py, src/train_cnn.py, src/evaluate.py, src/report.py
  - main.py, requirements.txt
- Dependencias: instaladas en venv sin errores.
- CLI: `python main.py --help` ejecuta y lista opciones correctamente.

Comandos ejecutados:
```bash
source venv/bin/activate && pip install -r requirements.txt
python main.py --help
```

---

### 2) Ejecución del pipeline reproducible
Comando usado:
```bash
source venv/bin/activate && python main.py --all
```
Archivos generados/verificados:
- models/cnn_model.h5 (OK)
- models/cnn_model_best.h5 (OK)
- models/scaler.pkl (OK)
- results/metrics.json (OK)
- results/training_curves.png (OK)
- results/confusion_matrix.png (OK)
- results/report.pdf (OK)
- Nota: `models/selected_features.pkl` no se generó porque no se usó top_k.

Salida relevante (resumida):
- Train 700 / Test 300 (split temporal 70/30)
- Métricas test: accuracy=1.0, precision=1.0, recall=1.0, f1=1.0

---

### 3) Pruebas de sanidad del modelo (data leakage)
Script ejecutado: `python data_leakage_analysis.py`
Artefactos:
- results/correlation_heatmap.png
- results/feature_target_scatter.png

Hallazgos:
- Top correlaciones con POWER_ON (|corr|): feature_1=0.968, feature_4=0.967, feature_0=0.967, feature_3=0.966, feature_2=0.963.
- No hubo |corr| ≥ 0.999 con el target.
- 0 columnas idénticas al target, 0 columnas constantes.
- Sin predictores perfectamente deterministas detectados.

Interpretación: No hay fuga obvia. Aun así, la combinación de correlaciones muy altas y métricas perfectas amerita revisión.

Comando:
```bash
source venv/bin/activate && python data_leakage_analysis.py
```

---

### 4) Validación alternativa de robustez
Script ejecutado: `python robustness_tests.py`
Artefactos:
- results/class_distribution.png
- results/shuffled_split_curves.png
- results/kfold_cv_results.png

Resultados:
- Split temporal 70/30: (ya reportado) 100% en todas las métricas.
- Split aleatorio 70/30 (estratificado): accuracy=1.0000, F1=1.0000.
- 5-fold CV: accuracy=1.0000 ± 0.0000, F1=1.0000 ± 0.0000.
- Sin top-5 features más correlacionadas: accuracy≈0.5633, F1≈0.6813.

Conclusión: El desempeño perfecto persiste incluso con validaciones alternativas; al remover las 5 features más correlacionadas, cae drásticamente. Esto indica que el problema puede ser demasiado fácil con dichas variables o que existen señales extremadamente deterministas en ellas.

Comando:
```bash
source venv/bin/activate && python robustness_tests.py
```

---

### 5) Balance de clases / distribución
- POWER_ON=1: 681 (68.1%)
- POWER_ON=0: 319 (31.9%)
- Balance ratio: 0.468 (desequilibrio moderado)
Sugerencias: reportar precision/recall, F1 y AUC; considerar `class_weight` o re-muestreo si se observa sesgo en producción (aunque aquí las métricas son perfectas).

---

### 6) Sanity checks en resultados
- Verificación independiente de métricas: se recalcularon a partir del modelo y split identico; coinciden exactamente con results/metrics.json.
- Curvas de entrenamiento: train y val llegan a 1.0, con val_loss ~1e-5. No hay brecha overfitting clásica; más bien desempeño perfecto y estable, inusual.
- Bandera de sospecha: Sí, por métricas perfectas en múltiples esquemas de validación y dependencias muy altas de top-5 features.

Comando de verificación de métricas:
```bash
source venv/bin/activate && python - << 'PY'
# Reproduce split y recalcula métricas; coincide con results/metrics.json
PY
```

---

### 7) Problemas detectados y pasos de remediación
Problemas:
- Métricas perfectas en múltiples esquemas de validación (temporal, aleatorio, 5-fold).
- Fuerte dependencia de top-5 features; al removerlas, el rendimiento se desploma (~0.56 accuracy).

Evidencia:
- results/metrics.json con 1.0 en todas las métricas.
- Salida de `robustness_tests.py` mostrando 1.0 sostenido y caída sin top-5.
- Correlaciones muy altas (≈0.96-0.97) aunque no perfectas.

Acciones sugeridas:
- Auditar origen de datos: confirmar que ninguna feature codifica directa o indirectamente el target (p.ej., transformaciones, etiquetas desplazadas, artefactos de generación).
- Evaluar un baseline simple (p.ej., regresión logística o árbol) y comparar; si también es ~100%, confirmar si el problema es trivial por diseño.
- Añadir validaciones fuera de muestra verdaderamente independientes (otro período temporal o planta) para descartar sobreajuste a distribución específica.
- Prueba de ablación sistemática: remover o limitar las 5 features más correlacionadas y analizar impacto; considerar regularización/constraints o ingeniería de features más realistas.
- Si procede, documentar que el dataset es linealmente separable (o casi) y que el desempeño perfecto es esperado; de lo contrario, rediseñar features/datos.

---

### 8) Artículos y logs generados
- results/report.pdf (del pipeline)
- results/metrics.json
- results/training_curves.png
- results/confusion_matrix.png
- results/correlation_heatmap.png
- results/feature_target_scatter.png
- results/class_distribution.png
- results/shuffled_split_curves.png
- results/kfold_cv_results.png
- training.log, evaluation.log
- scripts auxiliares: data_leakage_analysis.py, robustness_tests.py

---

### 9) Comandos ejecutados (lista)
```bash
# Instalación y ayuda CLI
source venv/bin/activate && pip install -r requirements.txt
python main.py --help

# Pipeline completo
python main.py --all

# Análisis de fuga de datos
python data_leakage_analysis.py

# Robustez (split aleatorio, k-fold, ablación top-5)
python robustness_tests.py

# Verificación independiente de métricas
python - << 'PY'
# (script inline que reprecarga datos, modelo y recalcula métricas)
PY
```

---

### Veredicto final
- ok_to_submit: false
- Motivo: Métricas perfectas y señales extremadamente predictivas en top-5 features sin evidencia clara de leakage, pero con riesgo alto de que el problema sea trivial o que existan artefactos en el dataset. Se recomienda auditoría y validaciones adicionales fuera de muestra antes de entrega.
