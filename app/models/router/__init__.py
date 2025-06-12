#!/usr/bin/env python3
"""
Router Package - Sistema de Enrutamiento Inteligente
===================================================

Este paquete contiene el sistema de router inteligente para modelos
de IA médica con capacidades avanzadas de ensemble y análisis.

Componentes principales:
- IntelligentMedicalRouter: Router principal
- AdvancedMedicalAIManager: Manager integrado para API
- Adaptadores para cada modelo específico
- Sistema de análisis de imágenes médicas

Uso:
    from .router.intelligent_router import AdvancedMedicalAIManager
    
    # Crear manager avanzado
    ai_system = AdvancedMedicalAIManager(device="auto")
    
    # Inicializar todos los modelos
    success = ai_system.load_model()
    
    # Realizar análisis ensemble
    result = ai_system.predict(image, use_ensemble=True)
"""

# Importaciones principales para fácil acceso
from .intelligent_router import (
    IntelligentMedicalRouter,
    AdvancedMedicalAIManager,
    MedicalRouterFactory,
    ImageType,
    StudyType,
    ModelCapability,
    ConfidenceLevel
)

# Metadata del paquete
__version__ = "1.0.0"
__author__ = "Radiology AI Team"
__description__ = "Sistema de router inteligente para modelos de IA médica"

# Exportar componentes principales
__all__ = [
    "IntelligentMedicalRouter",
    "AdvancedMedicalAIManager", 
    "MedicalRouterFactory",
    "ImageType",
    "StudyType",
    "ModelCapability",
    "ConfidenceLevel"
]