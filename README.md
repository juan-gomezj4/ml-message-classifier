# Proyecto MLOps - Clasificación de Mensajes en Batch

La solución refleja un enfoque práctico -mantenible, reproducible y escalable- para implementar sistemas de ML en producción, aplicando buenas prácticas de MLOps desde la experimentación hasta el despliegue.

El proyecto consiste en un sistema batch que automatiza la clasificación de mensajes enviados por usuarios a un canal de atención, simulados mediante reseñas del **Yelp Open Dataset**. Se diseñó una arquitectura modular basada en el enfoque **FTI (Feature - Training - Inference)**, con énfasis en mantenibilidad, reproducibilidad y escalabilidad.

La implementación considera el ciclo completo: desde la ingesta de datos y procesamiento, hasta el entrenamiento, predicción y activación de reentrenamiento bajo condiciones controladas.

Este repositorio contiene todo el código, configuraciones y artefactos necesarios para replicar la solución.

## **Objetivo**

El objetivo técnico fue desarrollar un sistema de clasificación de mensajes en batch, alineado con principios de MLOps y capaz de escalar hacia entornos productivos.

***La prioridad no estuvo en optimizar cada línea de código***, sino en definir una **arquitectura clara**, una **lógica desacoplada entre etapas** y un flujo robusto de punta a punta. Se diseñó una solución que refleja cómo se debe estructurar un sistema de ML real, más allá de un simple modelo funcional.

El proyecto priorizó:

- **Diseñar pipelines desacoplados**, compatibles con ejecución secuencial o por orquestador.
- **Definir pasos explícitos por etapa** (extracción, validación, agregación, etc.), facilitando la trazabilidad del flujo.
- **Aplicar validaciones automáticas** de datos y modelos.
- **Controlar dependencias y calidad del código** con herramientas reproducibles (`uv`, `ruff`, `mypy`, `pre-commit`).
- **Simular condiciones realistas de producción**, por ejem. reentrenamiento automático por detección de drift.

Este enfoque permitió construir un sistema sólido y extensible, listo para ser escalado o refactorizado sin comprometer su arquitectura base.


