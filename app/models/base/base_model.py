"""
Base Model para Sistema Multi-Extremidades
==========================================

Interfaz común que deben implementar todos los modelos de análisis radiológico.
Basado en el diseño exitoso de TorchXRayVision pero extendido para múltiples extremidades.

Compatibilidad:
- Hereda el estilo de AIModelManager actual
- Mantiene la misma interfaz para predicciones
- Extiende capacidades para nuevas extremidades
- Permite ensemble inteligente entre modelos

Autor: Radiology AI Team  
Basado en: TorchXRayVision implementation
Versión: 1.0.0
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import time
from enum import Enum
import cv2
from PIL import Image

# Importar configuración de la aplicación
from ...core.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMERACIONES Y TIPOS
# =============================================================================

class ModelType(Enum):
    """Tipos de modelos soportados"""
    UNIVERSAL = "universal"          # Detección general (MURA)
    CHEST = "chest"                 # Radiografías de tórax (TorchXRayVision)
    PEDIATRIC = "pediatric"         # Modelos pediátricos (BoneAge)
    SPINE = "spine"                 # Columna vertebral
    HIP = "hip"                     # Cadera y pelvis
    KNEE = "knee"                   # Rodilla
    SHOULDER = "shoulder"           # Hombro
    ANKLE_FOOT = "ankle_foot"       # Tobillo y pie
    ELBOW_FOREARM = "elbow_forearm" # Codo y antebrazo
    PELVIS = "pelvis"               # Pelvis y trauma
    PATHOLOGY = "pathology"         # Patología general/oncológica

class ModelStatus(Enum):
    """Estados posibles del modelo"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    DEPRECATED = "deprecated"

class ProcessingQuality(Enum):
    """Calidades de procesamiento"""
    FAST = "fast"           # Procesamiento rápido, menor precisión
    BALANCED = "balanced"   # Balance velocidad-precisión
    ACCURATE = "accurate"   # Máxima precisión, más lento

# =============================================================================
# CLASES DE DATOS PARA RESULTADOS
# =============================================================================

class PredictionResult:
    """Resultado estructurado de una predicción"""
    
    def __init__(self, 
                 predictions: Dict[str, float],
                 model_id: str,
                 model_type: ModelType,
                 processing_time: float,
                 confidence_level: str = "unknown",
                 metadata: Optional[Dict] = None):
        
        self.predictions = predictions
        self.model_id = model_id
        self.model_type = model_type
        self.processing_time = processing_time
        self.confidence_level = confidence_level
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def get_top_predictions(self, n: int = 3) -> List[Tuple[str, float]]:
        """Obtener las n predicciones con mayor confianza"""
        sorted_predictions = sorted(
            self.predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_predictions[:n]
    
    def get_high_confidence_findings(self, threshold: float = 0.7) -> Dict[str, float]:
        """Obtener hallazgos con alta confianza"""
        return {
            pathology: confidence 
            for pathology, confidence in self.predictions.items()
            if confidence >= threshold
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir resultado a diccionario"""
        return {
            "predictions": self.predictions,
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "processing_time": self.processing_time,
            "confidence_level": self.confidence_level,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "top_findings": self.get_top_predictions(3),
            "high_confidence_findings": self.get_high_confidence_findings()
        }

class ModelInfo:
    """Información detallada del modelo"""
    
    def __init__(self,
                 model_id: str,
                 name: str,
                 version: str,
                 model_type: ModelType,
                 architecture: str,
                 extremities_covered: List[str],
                 pathologies_detected: List[str],
                 status: ModelStatus,
                 device: str,
                 **kwargs):
        
        self.model_id = model_id
        self.name = name
        self.version = version
        self.model_type = model_type
        self.architecture = architecture
        self.extremities_covered = extremities_covered
        self.pathologies_detected = pathologies_detected
        self.status = status
        self.device = device
        
        # Información adicional opcional
        self.training_data = kwargs.get("training_data", "Unknown")
        self.validation_status = kwargs.get("validation_status", "Unknown")
        self.input_resolution = kwargs.get("input_resolution", "Variable")
        self.memory_requirements = kwargs.get("memory_requirements", "Unknown")
        self.inference_time = kwargs.get("inference_time", "Unknown")
        self.capabilities = kwargs.get("capabilities", [])
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertir información a diccionario (compatible con ai_model.py actual)"""
        return {
            "status": "Cargado y funcional" if self.status == ModelStatus.LOADED else self.status.value,
            "model_type": self.name,
            "model_architecture": self.architecture,
            "device": self.device,
            "pathologies_supported": self.pathologies_detected,
            "num_pathologies": len(self.pathologies_detected),
            "extremities_covered": self.extremities_covered,
            "input_resolution": self.input_resolution,
            "training_data": self.training_data,
            "validation_status": self.validation_status,
            "capabilities": self.capabilities,
            "model_id": self.model_id,
            "version": self.version,
            "model_type_category": self.model_type.value
        }

# =============================================================================
# CLASE BASE ABSTRACTA PARA TODOS LOS MODELOS
# =============================================================================

class BaseRadiologyModel(ABC):
    """
    Clase base abstracta que deben implementar todos los modelos de análisis radiológico.
    
    Esta clase define la interfaz común para:
    - TorchXRayVision (tórax) - ya implementado
    - Stanford MURA (universal)
    - BoneAge (pediatría)
    - Hip Fracture, Spine, Knee, etc.
    
    Diseño inspirado en el exitoso TorchXRayVisionModel actual.
    """
    
    def __init__(self, 
                 model_id: str,
                 model_type: ModelType,
                 device: str = "auto"):
        """
        Inicializar modelo base.
        
        Args:
            model_id: Identificador único del modelo
            model_type: Tipo de modelo (universal, chest, pediatric, etc.)
            device: Dispositivo de computación ('auto', 'cpu', 'cuda')
        """
        self.model_id = model_id
        self.model_type = model_type
        
        # Configurar dispositivo (igual que en tu ai_model.py actual)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Estado del modelo
        self.status = ModelStatus.NOT_LOADED
        self.model = None
        self.transform = None
        
        # Metadatos (serán definidos por cada modelo específico)
        self.pathologies = []
        self.extremities_covered = []
        self.model_info = None
        
        # Métricas de rendimiento
        self.last_inference_time = 0.0
        self.total_inferences = 0
        self.average_inference_time = 0.0
        
        logger.info(f"Modelo base '{model_id}' inicializado - Dispositivo: {self.device}")
    
    # =========================================================================
    # MÉTODOS ABSTRACTOS - DEBEN SER IMPLEMENTADOS POR CADA MODELO
    # =========================================================================
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        Cargar el modelo específico desde archivo o descarga.
        
        Cada modelo implementa su propia lógica de carga:
        - TorchXRayVision: xrv.models.DenseNet(weights="densenet121-res224-all")
        - MURA: torch.load("mura_model.pth")
        - BoneAge: cargar desde checkpoint específico
        etc.
        
        Returns:
            bool: True si el modelo se cargó exitosamente
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Realizar predicción en una imagen radiológica.
        
        Cada modelo implementa su propia lógica de predicción:
        - TorchXRayVision: 14 patologías de tórax
        - MURA: Fracturas en 9 extremidades
        - BoneAge: Edad ósea en años
        etc.
        
        Args:
            image: Array numpy de la imagen radiográfica
            
        Returns:
            Dict[str, float]: Predicciones específicas del modelo
        """
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesar imagen para el modelo específico.
        
        Cada modelo tiene requisitos diferentes:
        - TorchXRayVision: 224x224, normalización específica
        - MURA: puede requerir diferentes transformaciones
        - BoneAge: enfoque en región de mano-muñeca
        
        Args:
            image: Array numpy de la imagen original
            
        Returns:
            torch.Tensor: Imagen preprocesada para el modelo
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Obtener información detallada del modelo.
        
        Cada modelo retorna su información específica:
        - Patologías que detecta
        - Extremidades que cubre
        - Arquitectura utilizada
        - Estado de validación clínica
        
        Returns:
            ModelInfo: Información estructurada del modelo
        """
        pass
    
    # =========================================================================
    # MÉTODOS CONCRETOS - FUNCIONALIDAD COMÚN PARA TODOS LOS MODELOS
    # =========================================================================
    
    def predict_with_timing(self, image: np.ndarray) -> PredictionResult:
        """
        Realizar predicción con medición de tiempo (funcionalidad común).
        
        Args:
            image: Array numpy de la imagen
            
        Returns:
            PredictionResult: Resultado estructurado con timing
        """
        if self.status != ModelStatus.LOADED:
            raise RuntimeError(f"Modelo '{self.model_id}' no está cargado")
        
        start_time = time.time()
        
        try:
            # Llamar al método predict específico del modelo
            predictions = self.predict(image)
            
            # Calcular tiempo de procesamiento
            processing_time = time.time() - start_time
            self.last_inference_time = processing_time
            
            # Actualizar estadísticas
            self.total_inferences += 1
            self.average_inference_time = (
                (self.average_inference_time * (self.total_inferences - 1) + processing_time) 
                / self.total_inferences
            )
            
            # Determinar nivel de confianza
            confidence_level = self._determine_confidence_level(predictions)
            
            # Crear resultado estructurado
            result = PredictionResult(
                predictions=predictions,
                model_id=self.model_id,
                model_type=self.model_type,
                processing_time=processing_time,
                confidence_level=confidence_level,
                metadata={
                    "total_inferences": self.total_inferences,
                    "average_inference_time": self.average_inference_time,
                    "device": str(self.device)
                }
            )
            
            logger.debug(f"Predicción completada - {self.model_id} en {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error en predicción de {self.model_id}: {str(e)}")
            
            # Retornar predicciones seguras en caso de error
            safe_predictions = self._generate_safe_predictions()
            
            return PredictionResult(
                predictions=safe_predictions,
                model_id=self.model_id,
                model_type=self.model_type,
                processing_time=processing_time,
                confidence_level="error",
                metadata={"error": str(e)}
            )
    
    def _determine_confidence_level(self, predictions: Dict[str, float]) -> str:
        """Determinar nivel de confianza general de las predicciones"""
        if not predictions:
            return "low"
        
        max_confidence = max(predictions.values())
        avg_confidence = sum(predictions.values()) / len(predictions)
        
        if max_confidence > 0.8 and avg_confidence > 0.3:
            return "high"
        elif max_confidence > 0.6 or avg_confidence > 0.2:
            return "medium"
        else:
            return "low"
    
    def _generate_safe_predictions(self) -> Dict[str, float]:
        """
        Generar predicciones seguras en caso de error.
        
        Cada modelo puede sobrescribir esto para predicciones específicas.
        Por defecto, retorna valores conservadores.
        """
        if not self.pathologies:
            return {"unknown_pathology": 0.05}
        
        # Predicciones conservadoras uniformes
        safe_predictions = {}
        for pathology in self.pathologies:
            safe_predictions[pathology] = 0.05  # 5% conservador
        
        return safe_predictions
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validar que la imagen es apropiada para este modelo.
        
        Args:
            image: Array numpy de la imagen
            
        Returns:
            Tuple[bool, str]: (es_válida, mensaje_error)
        """
        try:
            # Validaciones básicas
            if image is None or image.size == 0:
                return False, "Imagen vacía o nula"
            
            if len(image.shape) not in [2, 3]:
                return False, f"Dimensiones de imagen inválidas: {image.shape}"
            
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False, f"Número de canales inválido: {image.shape[2]}"
            
            # Validar rango de valores
            if image.max() <= image.min():
                return False, "Imagen sin contraste (valores uniformes)"
            
            # Validar tamaño mínimo
            min_size = 64  # Tamaño mínimo razonable
            if min(image.shape[:2]) < min_size:
                return False, f"Imagen demasiado pequeña: {image.shape[:2]}"
            
            return True, "Imagen válida"
            
        except Exception as e:
            return False, f"Error validando imagen: {str(e)}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento del modelo"""
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "total_inferences": self.total_inferences,
            "last_inference_time": self.last_inference_time,
            "average_inference_time": self.average_inference_time,
            "device": str(self.device),
            "pathologies_count": len(self.pathologies),
            "extremities_count": len(self.extremities_covered)
        }
    
    def is_compatible_with(self, other_model: 'BaseRadiologyModel') -> bool:
        """
        Verificar si este modelo es compatible con otro para ensemble.
        
        Args:
            other_model: Otro modelo para comparar
            
        Returns:
            bool: True si son compatibles para ensemble
        """
        if not isinstance(other_model, BaseRadiologyModel):
            return False
        
        # Criterios de compatibilidad
        same_device = self.device == other_model.device
        overlapping_pathologies = bool(
            set(self.pathologies) & set(other_model.pathologies)
        )
        overlapping_extremities = bool(
            set(self.extremities_covered) & set(other_model.extremities_covered)
        )
        
        # Compatible si comparten dispositivo y tienen solapamiento
        return same_device and (overlapping_pathologies or overlapping_extremities)
    
    def reset_stats(self) -> None:
        """Reiniciar estadísticas de rendimiento"""
        self.total_inferences = 0
        self.average_inference_time = 0.0
        self.last_inference_time = 0.0
        logger.info(f"Estadísticas de {self.model_id} reiniciadas")
    
    def __str__(self) -> str:
        """Representación string del modelo"""
        return f"{self.model_id} ({self.model_type.value}) - Status: {self.status.value}"
    
    def __repr__(self) -> str:
        """Representación detallada del modelo"""
        return (f"BaseRadiologyModel(id='{self.model_id}', "
                f"type={self.model_type.value}, "
                f"status={self.status.value}, "
                f"device='{self.device}')")


# =============================================================================
# ADAPTADOR PARA COMPATIBILIDAD CON AI_MODEL.PY ACTUAL
# =============================================================================

class LegacyModelAdapter(BaseRadiologyModel):
    """
    Adaptador para hacer que el TorchXRayVisionModel actual 
    sea compatible con la nueva interfaz BaseRadiologyModel.
    
    Esto permite migración gradual sin romper el código existente.
    """
    
    def __init__(self, legacy_model_instance):
        """
        Inicializar adaptador con instancia del modelo legacy.
        
        Args:
            legacy_model_instance: Instancia de TorchXRayVisionModel actual
        """
        super().__init__(
            model_id="torchxrayvision_legacy",
            model_type=ModelType.CHEST,
            device=str(legacy_model_instance.device)
        )
        
        self.legacy_model = legacy_model_instance
        self.pathologies = legacy_model_instance.pathologies
        self.extremities_covered = ["chest", "thorax", "lungs"]
        
        # Sincronizar estado
        if legacy_model_instance.model is not None:
            self.status = ModelStatus.LOADED
        
        logger.info("Adaptador legacy creado para TorchXRayVision")
    
    def load_model(self) -> bool:
        """Cargar usando el método legacy"""
        success = self.legacy_model.load_model()
        self.status = ModelStatus.LOADED if success else ModelStatus.ERROR
        return success
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Predicción usando el método legacy"""
        return self.legacy_model.predict(image)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesamiento usando el método legacy"""
        return self.legacy_model.preprocess_image(image)
    
    def get_model_info(self) -> ModelInfo:
        """Obtener información del modelo legacy en formato nuevo"""
        legacy_info = self.legacy_model.get_model_info()
        
        return ModelInfo(
            model_id=self.model_id,
            name=legacy_info.get("model_type", "TorchXRayVision"),
            version="legacy",
            model_type=self.model_type,
            architecture=legacy_info.get("model_architecture", "DenseNet-121"),
            extremities_covered=self.extremities_covered,
            pathologies_detected=self.pathologies,
            status=self.status,
            device=str(self.device),
            training_data=legacy_info.get("training_data", "Unknown"),
            validation_status=legacy_info.get("validation_status", "Unknown"),
            input_resolution=legacy_info.get("input_resolution", "224x224"),
            capabilities=legacy_info.get("capabilities", [])
        )


# =============================================================================
# UTILIDADES PARA COMPATIBILIDAD
# =============================================================================

def create_legacy_adapter(ai_model_manager_instance) -> LegacyModelAdapter:
    """
    Crear adaptador para el AIModelManager actual.
    
    Esto permite usar tu implementación actual de TorchXRayVision
    con la nueva interfaz BaseRadiologyModel.
    
    Args:
        ai_model_manager_instance: Instancia de tu AIModelManager actual
        
    Returns:
        LegacyModelAdapter: Adaptador compatible con nueva interfaz
    """
    return LegacyModelAdapter(ai_model_manager_instance.model)

def validate_model_implementation(model: BaseRadiologyModel) -> List[str]:
    """
    Validar que un modelo implementa correctamente la interfaz base.
    
    Args:
        model: Instancia del modelo a validar
        
    Returns:
        List[str]: Lista de problemas encontrados (vacía si todo está bien)
    """
    issues = []
    
    # Verificar que es una instancia válida
    if not isinstance(model, BaseRadiologyModel):
        issues.append("Modelo no hereda de BaseRadiologyModel")
        return issues
    
    # Verificar atributos requeridos
    required_attrs = ["model_id", "model_type", "device", "status", "pathologies"]
    for attr in required_attrs:
        if not hasattr(model, attr):
            issues.append(f"Atributo requerido '{attr}' no encontrado")
    
    # Verificar métodos abstractos (si el modelo está "cargado")
    if model.status == ModelStatus.LOADED:
        try:
            # Test básico de funcionalidad
            dummy_image = np.random.rand(224, 224, 3).astype(np.uint8)
            is_valid, msg = model.validate_image(dummy_image)
            if not is_valid:
                issues.append(f"Validación de imagen falló: {msg}")
        except Exception as e:
            issues.append(f"Error en validación básica: {str(e)}")
    
    return issues


# =============================================================================
# EJEMPLO DE USO Y TESTING
# =============================================================================

if __name__ == "__main__":
    # Este ejemplo muestra cómo usar la nueva interfaz base
    print("=== BASE RADIOLOGICAL MODEL INTERFACE ===")
    
    # Ejemplo de implementación (esto sería implementado por cada modelo específico)
    class DummyModel(BaseRadiologyModel):
        def __init__(self):
            super().__init__("dummy_test", ModelType.UNIVERSAL)
            self.pathologies = ["test_pathology_1", "test_pathology_2"]
            self.extremities_covered = ["test_extremity"]
        
        def load_model(self) -> bool:
            self.status = ModelStatus.LOADED
            return True
        
        def predict(self, image: np.ndarray) -> Dict[str, float]:
            return {"test_pathology_1": 0.3, "test_pathology_2": 0.1}
        
        def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
            return torch.randn(1, 224, 224)
        
        def get_model_info(self) -> ModelInfo:
            return ModelInfo(
                model_id=self.model_id,
                name="Dummy Test Model",
                version="1.0.0",
                model_type=self.model_type,
                architecture="Test Architecture",
                extremities_covered=self.extremities_covered,
                pathologies_detected=self.pathologies,
                status=self.status,
                device=str(self.device)
            )
    
    # Test de la interfaz
    dummy = DummyModel()
    print(f"Modelo creado: {dummy}")
    
    dummy.load_model()
    print(f"Estado después de cargar: {dummy.status}")
    
    # Test de predicción
    test_image = np.random.rand(224, 224, 3).astype(np.uint8)
    result = dummy.predict_with_timing(test_image)
    print(f"Resultado de predicción: {result.predictions}")
    print(f"Tiempo de procesamiento: {result.processing_time:.3f}s")
    
    # Test de información del modelo
    info = dummy.get_model_info()
    print(f"Información del modelo: {info.to_dict()}")
    
    print("\n¡Interfaz base funcional!")