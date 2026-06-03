====================================================================
SISTEMA MULTIAGENTE DE ASISTENCIA CULINARIA Y NUTRICIONAL
MEDIANTE VISIÓN ARTIFICIAL Y RAG
====================================================================

Autor: Pablo Chica González
Titulación: Grado en Ingeniería Informática (Universidad de Málaga)

DESCRIPCIÓN DEL PROYECTO
Este proyecto, también conocido como "Chef Personal", es un sistema inteligente diseñado para automatizar la gestión de la alimentación en el hogar. Conecta los ingredientes físicos disponibles en la cocina del usuario con una planificación nutricional adaptada y saludable, eliminando la tediosa introducción manual de datos.

CARACTERÍSTICAS PRINCIPALES
- Identificación Visual Inteligente: Análisis de fotografías del entorno físico (nevera o despensa) para segmentar y detectar automáticamente los alimentos. Se apoya en una arquitectura de visión artificial híbrida utilizando modelos como YOLO y la herramienta SAHI para la fragmentación de imágenes en la detección.
- Motor RAG (Retrieval-Augmented Generation): Utiliza una base de datos vectorial para recuperar recetas factibles y empíricamente validadas, evitando que la IA genere propuestas imposibles o perfiles nutricionales inexactos.
- Orquestación Multiagente: Implementado con LangGraph, simula un proceso de deliberación entre roles especializados (Chef, Nutricionista y Mediador). Aísla el cálculo determinista de macronutrientes del razonamiento creativo para ofrecer menús matemáticamente exactos.
- Seguimiento Nutricional: Panel de control completo que permite el seguimiento de calorías, macronutrientes, peso y actividad física.

ARQUITECTURA TÉCNICA
- Frontend: Single Page Application (SPA) responsiva desarrollada con React, TypeScript y Vite.
- Backend: API REST construida con FastAPI en Python.
- Bases de Datos: PostgreSQL para datos relacionales, ChromaDB para el motor RAG vectorial y SQLite para índices locales rápidos.
- Procesamiento Asíncrono: Uso de Celery y Redis para delegar la carga computacional pesada (visión artificial e IA generativa) y mantener la fluidez del sistema.