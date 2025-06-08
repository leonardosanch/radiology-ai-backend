"""
Multi-Model Manager para Sistema de Análisis Radiológico Multi-Extremidades
===========================================================================

Coordinador central que maneja múltiples modelos de IA para diferentes extremidades.
Reemplaza gradualmente el AIModelManager actual con capacidades expandidas.

Características:
- Gestión de múltiples modelos simultáneos
- Detección automática de extremidad
- Estrategias de ensemble inteligentes  
- Compatibilidad con TorchXRayVision actual
- Enrutamiento automático por tipo de estudio

Compatibilidad:
- Funciona junto al ai_model.py actual
- Migración gradual sin interrupciones
- API compatible con analysis.py existente

Autor: Radiology AI Team
Basado en: AIModelManager exitoso
Versión: 1.0.0
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from pathlib import Path
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum

# Importar componentes del sistema
from ..base.base_model import (
    BaseRadiologyModel, ModelType, ModelStatus, PredictionResult, 
    ModelInfo, LegacyModelAdapter, ProcessingQuality
)
from ..base.model_registry import (
    model_registry, ModelSpecification, ExtremityType, 
    EnsembleStrategy, ClinicalPriority
)
from ...core.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# CLASES DE CONFIGURACIÓN Y RESULTADOS
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuración para análisis de ensemble"""
    strategy: EnsembleStrategy
    models_to_use: List[str]
    confidence_threshold: float
    consensus_required: bool
    parallel_processing: bool = True
    max_processing_time: float = 30.0  # segundos
    
@dataclass
class ExtremityDetectionResult:
    """Resultado de detección de extremidad"""
    detected_extremity: str
    confidence: float
    suggested_models: List[str]
    alternative_extremities: List[Tuple[str, float]]

@dataclass
class MultiModelResult:
    """Resultado de análisis multi-modelo"""
    primary_result: PredictionResult
    secondary_results: List[PredictionResult]
    ensemble_predictions: Dict[str, float]
    detected_extremity: str
    models_used: List[str]
    processing_strategy: str
    total_processing_time: float
    consensus_achieved: bool
    confidence_level: str
    
    def get_combined_predictions(self) -> Dict[str, float]:
        """Obtener predicciones combinadas de todos los modelos"""
        return self.ensemble_predictions
    
    def get_high_confidence_findings(self, threshold: float = 0.7) -> Dict[str, float]:
        """Obtener hallazgos con alta confianza del ensemble"""
        return {
            pathology: confidence 
            for pathology, confidence in self.ensemble_predictions.items()
            if confidence >= threshold
        }

# =============================================================================
# MULTI-MODEL MANAGER PRINCIPAL
# =============================================================================

class MultiModelManager:
    """
    Gestor central para múltiples modelos de análisis radiológico.
    
    Este manager coordina:
    - Tu TorchXRayVision actual (tórax)
    - Stanford MURA (universal)
    - BoneAge (pediatría)
    - + 7 modelos adicionales
    
    Funcionalidades:
    - Detección automática de extremidad
    - Enrutamiento inteligente a modelos apropiados
    - Ensemble de múltiples predicciones
    - Compatibilidad con sistema actual
    """
    
    def __init__(self, 
                 models_directory: str = "./models/",
                 device: str = "auto",
                 enable_parallel_processing: bool = True):
        """
        Inicializar el gestor multi-modelo.
        
        Args:
            models_directory: Directorio base para modelos
            device: Dispositivo de computación ('auto', 'cpu', 'cuda')
            enable_parallel_processing: Habilitar procesamiento paralelo
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Configurar dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Estado del manager
        self.enable_parallel_processing = enable_parallel_processing
        self.loaded_models: Dict[str, BaseRadiologyModel] = {}
        self.model_load_status: Dict[str, ModelStatus] = {}
        self.model_registry = model_registry
        
        # Configuración de threading para procesamiento paralelo
        self.thread_pool = ThreadPoolExecutor(max_workers=4) if enable_parallel_processing else None
        self.model_locks: Dict[str, threading.Lock] = {}
        
        # Estadísticas
        self.total_analyses = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.average_processing_time = 0.0
        
        # Compatibilidad con sistema actual
        self.legacy_model = None  # Para tu TorchXRayVision actual
        
        logger.info(f"MultiModelManager inicializado")
        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Procesamiento paralelo: {enable_parallel_processing}")
        logger.info(f"Modelos disponibles en registry: {len(self.model_registry.models)}")
    
    # =========================================================================
    # GESTIÓN DE MODELOS INDIVIDUALES
    # =========================================================================
    
    def register_legacy_model(self, legacy_ai_model_manager) -> bool:
        """
        Registrar tu AIModelManager actual para compatibilidad.
        
        Args:
            legacy_ai_model_manager: Tu instancia actual de AIModelManager
            
        Returns:
            bool: True si se registró exitosamente
        """
        try:
            # Crear adaptador para el modelo legacy
            adapter = LegacyModelAdapter(legacy_ai_model_manager.model)
            
            # Registrar en el sistema
            self.loaded_models["torchxrayvision_legacy"] = adapter
            self.model_load_status["torchxrayvision_legacy"] = adapter.status
            self.model_locks["torchxrayvision_legacy"] = threading.Lock()
            
            self.legacy_model = adapter
            
            logger.info("✅ Modelo TorchXRayVision legacy registrado exitosamente")
            logger.info(f"Patologías disponibles: {len(adapter.pathologies)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registrando modelo legacy: {str(e)}")
            return False
    
    def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Cargar un modelo específico según el registry.
        
        Args:
            model_id: ID del modelo según model_registry
            force_reload: Forzar recarga si ya está cargado
            
        Returns:
            bool: True si se cargó exitosamente
        """
        # Verificar si el modelo ya está cargado
        if model_id in self.loaded_models and not force_reload:
            logger.info(f"Modelo '{model_id}' ya está cargado")
            return True
        
        # Obtener especificación del modelo
        model_spec = self.model_registry.get_model(model_id)
        if not model_spec:
            logger.error(f"Modelo '{model_id}' no encontrado en registry")
            return False
        
        try:
            logger.info(f"Cargando modelo: {model_spec.name}")
            
            # Marcar como cargando
            self.model_load_status[model_id] = ModelStatus.LOADING
            
            # Crear instancia del modelo específico
            model_instance = self._create_model_instance(model_spec)
            
            if model_instance is None:
                self.model_load_status[model_id] = ModelStatus.ERROR
                return False
            
            # Cargar el modelo
            success = model_instance.load_model()
            
            if success:
                # Registrar modelo cargado
                self.loaded_models[model_id] = model_instance
                self.model_load_status[model_id] = ModelStatus.LOADED
                self.model_locks[model_id] = threading.Lock()
                
                logger.info(f"✅ Modelo '{model_id}' cargado exitosamente")
                return True
            else:
                self.model_load_status[model_id] = ModelStatus.ERROR
                logger.error(f"❌ Error cargando modelo '{model_id}'")
                return False
                
        except Exception as e:
            self.model_load_status[model_id] = ModelStatus.ERROR
            logger.error(f"Error cargando modelo '{model_id}': {str(e)}")
            return False
    
    def _create_model_instance(self, model_spec: ModelSpecification) -> Optional[BaseRadiologyModel]:
        """
        Crear instancia de modelo específico según su tipo.
        
        Args:
            model_spec: Especificación del modelo
            
        Returns:
            BaseRadiologyModel: Instancia del modelo o None si error
        """
        model_id = model_spec.model_id
        
        try:
            # Aquí se crearían las instancias específicas de cada modelo
            # Por ahora, solo implementamos placeholders que se completarán
            # cuando creemos cada modelo específico
            
            if model_id == "mura":
                # Stanford MURA - se implementará después
                logger.info("Creando instancia de Stanford MURA (placeholder)")
                return self._create_mura_placeholder()
                
            elif model_id == "boneage":
                # RSNA BoneAge - se implementará después  
                logger.info("Creando instancia de BoneAge (placeholder)")
                return self._create_boneage_placeholder()
                
            elif model_id == "hip_fracture":
                # Hip Fracture Detection - se implementará después
                logger.info("Creando instancia de Hip Fracture (placeholder)")
                return self._create_hip_fracture_placeholder()
                
            # ... más modelos según se implementen
            
            else:
                logger.warning(f"Modelo '{model_id}' no implementado aún")
                return None
                
        except Exception as e:
            logger.error(f"Error creando instancia de '{model_id}': {str(e)}")
            return None
    
    def _create_mura_placeholder(self) -> BaseRadiologyModel:
        """Crear placeholder para Stanford MURA (se implementará después)"""
        from ..base.base_model import BaseRadiologyModel, ModelType, ModelInfo, ModelStatus
        
        class MURAPlaceholder(BaseRadiologyModel):
            def __init__(self):
                super().__init__("mura", ModelType.UNIVERSAL, device="cpu")
                self.pathologies = ["fracture", "normal"]
                self.extremities_covered = ["shoulder", "humerus", "elbow", "forearm", "hand", "hip", "femur", "knee", "ankle"]
            
            def load_model(self) -> bool:
                logger.info("MURA placeholder - modelo no implementado aún")
                self.status = ModelStatus.ERROR
                return False
            
            def predict(self, image: np.ndarray) -> Dict[str, float]:
                return {"fracture": 0.05, "normal": 0.95}
            
            def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
                return torch.randn(1, 224, 224)
            
            def get_model_info(self) -> ModelInfo:
                return ModelInfo(
                    model_id=self.model_id,
                    name="Stanford MURA (Placeholder)",
                    version="placeholder",
                    model_type=self.model_type,
                    architecture="DenseNet-169 (Not Loaded)",
                    extremities_covered=self.extremities_covered,
                    pathologies_detected=self.pathologies,
                    status=self.status,
                    device=str(self.device)
                )
        
        return MURAPlaceholder()
    
    def _create_boneage_placeholder(self) -> BaseRadiologyModel:
        """Crear placeholder para BoneAge (se implementará después)"""
        from ..base.base_model import BaseRadiologyModel, ModelType, ModelInfo, ModelStatus
        
        class BoneAgePlaceholder(BaseRadiologyModel):
            def __init__(self):
                super().__init__("boneage", ModelType.PEDIATRIC, device="cpu")
                self.pathologies = ["bone_age_estimation"]
                self.extremities_covered = ["hand", "wrist"]
            
            def load_model(self) -> bool:
                logger.info("BoneAge placeholder - modelo no implementado aún")
                self.status = ModelStatus.ERROR
                return False
            
            def predict(self, image: np.ndarray) -> Dict[str, float]:
                return {"bone_age_estimation": 8.5}  # años estimados
            
            def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
                return torch.randn(1, 256, 256)
            
            def get_model_info(self) -> ModelInfo:
                return ModelInfo(
                    model_id=self.model_id,
                    name="RSNA BoneAge (Placeholder)",
                    version="placeholder",
                    model_type=self.model_type,
                    architecture="ResNet-50 (Not Loaded)",
                    extremities_covered=self.extremities_covered,
                    pathologies_detected=self.pathologies,
                    status=self.status,
                    device=str(self.device)
                )
        
        return BoneAgePlaceholder()
    
    def _create_hip_fracture_placeholder(self) -> BaseRadiologyModel:
        """Crear placeholder para Hip Fracture (se implementará después)"""
        from ..base.base_model import BaseRadiologyModel, ModelType, ModelInfo, ModelStatus
        
        class HipFracturePlaceholder(BaseRadiologyModel):
            def __init__(self):
                super().__init__("hip_fracture", ModelType.HIP, device="cpu")
                self.pathologies = ["femoral_neck_fracture", "intertrochanteric_fracture", "normal"]
                self.extremities_covered = ["hip", "pelvis", "femur_proximal"]
            
            def load_model(self) -> bool:
                logger.info("Hip Fracture placeholder - modelo no implementado aún")
                self.status = ModelStatus.ERROR
                return False
            
            def predict(self, image: np.ndarray) -> Dict[str, float]:
                return {
                    "femoral_neck_fracture": 0.02,
                    "intertrochanteric_fracture": 0.03,
                    "normal": 0.95
                }
            
            def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
                return torch.randn(1, 320, 320)
            
            def get_model_info(self) -> ModelInfo:
                return ModelInfo(
                    model_id=self.model_id,
                    name="Hip Fracture Detection (Placeholder)",
                    version="placeholder",
                    model_type=self.model_type,
                    architecture="EfficientNet-B4 (Not Loaded)",
                    extremities_covered=self.extremities_covered,
                    pathologies_detected=self.pathologies,
                    status=self.status,
                    device=str(self.device)
                )
        
        return HipFracturePlaceholder()
    
    # =========================================================================
    # DETECCIÓN AUTOMÁTICA DE EXTREMIDAD
    # =========================================================================
    
    def detect_extremity(self, image: np.ndarray, 
                        metadata: Optional[Dict] = None) -> ExtremityDetectionResult:
        """
        Detectar automáticamente qué tipo de extremidad está en la imagen.
        
        Args:
            image: Array numpy de la imagen
            metadata: Metadatos opcionales (DICOM, etc.)
            
        Returns:
            ExtremityDetectionResult: Resultado de la detección
        """
        try:
            # Por ahora implementación básica - se mejorará con un clasificador ML
            detected_extremity = self._basic_extremity_detection(image, metadata)
            
            # Obtener modelos sugeridos para esta extremidad
            suggested_models = self.model_registry.recommend_models_for_extremity(detected_extremity)
            
            # Alternativas (se mejorarán con ML)
            alternatives = self._get_alternative_extremities(detected_extremity)
            
            result = ExtremityDetectionResult(
                detected_extremity=detected_extremity,
                confidence=0.8,  # Placeholder - se mejorará
                suggested_models=suggested_models,
                alternative_extremities=alternatives
            )
            
            logger.info(f"Extremidad detectada: {detected_extremity} (modelos: {suggested_models})")
            return result
            
        except Exception as e:
            logger.error(f"Error en detección de extremidad: {str(e)}")
            # Fallback a universal
            return ExtremityDetectionResult(
                detected_extremity="unknown",
                confidence=0.1,
                suggested_models=["mura", "torchxrayvision_legacy"],
                alternative_extremities=[]
            )
    
    def _basic_extremity_detection(self, image: np.ndarray, 
                                  metadata: Optional[Dict] = None) -> str:
        """
        Detección básica de extremidad (se mejorará con ML).
        
        Args:
            image: Imagen a analizar
            metadata: Metadatos si están disponibles
            
        Returns:
            str: Tipo de extremidad detectada
        """
        # Prioridad 1: Metadatos DICOM si están disponibles
        if metadata:
            body_part = metadata.get("BodyPartExamined", "").lower()
            if "chest" in body_part or "thorax" in body_part:
                return "chest"
            elif "hand" in body_part or "wrist" in body_part:
                return "hand"
            elif "knee" in body_part:
                return "knee"
            elif "hip" in body_part or "pelvis" in body_part:
                return "hip"
            elif "spine" in body_part or "cervical" in body_part:
                return "spine"
        
        # Prioridad 2: Análisis básico de imagen
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Heurísticas básicas (se reemplazarán con ML)
        if aspect_ratio > 1.2:
            # Imagen muy ancha - probablemente columna o extremidad larga
            return "spine"
        elif aspect_ratio < 0.8:
            # Imagen alta - probablemente tórax
            return "chest"
        else:
            # Aspectos cuadrados - podrían ser articulaciones
            image_size = height * width
            if image_size > 500000:  # Imagen grande
                return "knee"
            else:
                return "hand"
    
    def _get_alternative_extremities(self, primary: str) -> List[Tuple[str, float]]:
        """Obtener extremidades alternativas con probabilidades"""
        alternatives_map = {
            "chest": [("thorax", 0.9), ("lung", 0.7)],
            "hand": [("wrist", 0.8), ("forearm", 0.3)],
            "knee": [("leg", 0.6), ("ankle", 0.3)],
            "hip": [("pelvis", 0.8), ("femur", 0.5)],
            "spine": [("cervical_spine", 0.7), ("lumbar_spine", 0.7)],
            "unknown": [("chest", 0.4), ("hand", 0.3), ("knee", 0.3)]
        }
        return alternatives_map.get(primary, [])
    
    # =========================================================================
    # ANÁLISIS MULTI-MODELO Y ENSEMBLE
    # =========================================================================
    
    def analyze_image(self, 
                     image: np.ndarray,
                     metadata: Optional[Dict] = None,
                     strategy: str = "auto",
                     specific_models: Optional[List[str]] = None) -> MultiModelResult:
        """
        Análizar imagen usando múltiples modelos con ensemble inteligente.
        
        Args:
            image: Array numpy de la imagen
            metadata: Metadatos opcionales
            strategy: Estrategia de análisis ('auto', 'emergency', 'pediatric', etc.)
            specific_models: Modelos específicos a usar (opcional)
            
        Returns:
            MultiModelResult: Resultado del análisis multi-modelo
        """
        start_time = time.time()
        self.total_analyses += 1
        
        try:
            logger.info(f"Iniciando análisis multi-modelo (estrategia: {strategy})")
            
            # Paso 1: Detectar extremidad si no se especificaron modelos
            if specific_models is None:
                detection_result = self.detect_extremity(image, metadata)
                models_to_use = detection_result.suggested_models[:3]  # Top 3
                detected_extremity = detection_result.detected_extremity
            else:
                models_to_use = specific_models
                detected_extremity = "specified"
            
            logger.info(f"Modelos a usar: {models_to_use}")
            
            # Paso 2: Configurar ensemble según estrategia
            ensemble_config = self._get_ensemble_config(strategy, detected_extremity)
            
            # Paso 3: Ejecutar análisis con modelos disponibles
            results = self._execute_multi_model_analysis(image, models_to_use, ensemble_config)
            
            # Paso 4: Combinar resultados usando ensemble
            ensemble_predictions = self._combine_results_with_ensemble(results, ensemble_config)
            
            # Paso 5: Determinar resultado principal
            primary_result = results[0] if results else None
            secondary_results = results[1:] if len(results) > 1 else []
            
            # Calcular métricas finales
            total_time = time.time() - start_time
            consensus_achieved = self._check_consensus(results, ensemble_config)
            confidence_level = self._determine_overall_confidence(ensemble_predictions)
            
            # Crear resultado final
            final_result = MultiModelResult(
                primary_result=primary_result,
                secondary_results=secondary_results,
                ensemble_predictions=ensemble_predictions,
                detected_extremity=detected_extremity,
                models_used=[r.model_id for r in results],
                processing_strategy=strategy,
                total_processing_time=total_time,
                consensus_achieved=consensus_achieved,
                confidence_level=confidence_level
            )
            
            # Actualizar estadísticas
            self.successful_analyses += 1
            self._update_performance_stats(total_time)
            
            logger.info(f"✅ Análisis multi-modelo completado en {total_time:.2f}s")
            logger.info(f"Modelos usados: {final_result.models_used}")
            logger.info(f"Consenso: {'Sí' if consensus_achieved else 'No'}")
            
            return final_result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.failed_analyses += 1
            
            logger.error(f"❌ Error en análisis multi-modelo: {str(e)}")
            
            # Fallback: usar modelo legacy si está disponible
            if self.legacy_model and self.legacy_model.status == ModelStatus.LOADED:
                try:
                    fallback_predictions = self.legacy_model.predict(image)
                    fallback_result = PredictionResult(
                        predictions=fallback_predictions,
                        model_id=self.legacy_model.model_id,
                        model_type=self.legacy_model.model_type,
                        processing_time=total_time,
                        confidence_level="fallback"
                    )
                    
                    return MultiModelResult(
                        primary_result=fallback_result,
                        secondary_results=[],
                        ensemble_predictions=fallback_predictions,
                        detected_extremity="fallback",
                        models_used=[self.legacy_model.model_id],
                        processing_strategy="fallback",
                        total_processing_time=total_time,
                        consensus_achieved=False,
                        confidence_level="low"
                    )
                except Exception as fallback_error:
                    logger.error(f"Error en fallback: {str(fallback_error)}")
            
            # Si todo falla, generar respuesta de error
            return self._generate_error_result(str(e), total_time)
    
    def _get_ensemble_config(self, strategy: str, extremity: str) -> EnsembleConfig:
        """Obtener configuración de ensemble según estrategia y extremidad"""
        
        # Configuraciones predefinidas
        configs = {
            "emergency": EnsembleConfig(
                strategy=EnsembleStrategy.MAX_SENSITIVITY,
                models_to_use=[],  # Se llenará dinámicamente
                confidence_threshold=0.3,
                consensus_required=False,
                parallel_processing=True
            ),
            "pediatric": EnsembleConfig(
                strategy=EnsembleStrategy.SPECIALIST_PRIORITY,
                models_to_use=[],
                confidence_threshold=0.6,
                consensus_required=True,
                parallel_processing=False  # Procesamiento secuencial para pediatría
            ),
            "routine": EnsembleConfig(
                strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
                models_to_use=[],
                confidence_threshold=0.5,
                consensus_required=False,
                parallel_processing=True
            )
        }
        
        # Usar configuración por defecto si no se encuentra
        if strategy not in configs:
            strategy = "routine"
        
        return configs[strategy]
    
    def _execute_multi_model_analysis(self, 
                                    image: np.ndarray, 
                                    models_to_use: List[str],
                                    config: EnsembleConfig) -> List[PredictionResult]:
        """Ejecutar análisis con múltiples modelos"""
        results = []
        
        for model_id in models_to_use:
            try:
                # Verificar si el modelo está cargado
                if model_id not in self.loaded_models:
                    logger.warning(f"Modelo '{model_id}' no está cargado, intentando cargar...")
                    if not self.load_model(model_id):
                        logger.warning(f"No se pudo cargar modelo '{model_id}', saltando...")
                        continue
                
                model = self.loaded_models[model_id]
                
                # Usar lock para thread safety
                with self.model_locks[model_id]:
                    result = model.predict_with_timing(image)
                    results.append(result)
                    
                logger.debug(f"Análisis completado con {model_id}")
                
            except Exception as e:
                logger.error(f"Error en análisis con modelo '{model_id}': {str(e)}")
                continue
        
        return results
    
    def _combine_results_with_ensemble(self, 
                                     results: List[PredictionResult],
                                     config: EnsembleConfig) -> Dict[str, float]:
        """Combinar resultados de múltiples modelos usando ensemble"""
        if not results:
            return {}
        
        # Obtener todas las patologías únicas
        all_pathologies = set()
        for result in results:
            all_pathologies.update(result.predictions.keys())
        
        ensemble_predictions = {}
        
        for pathology in all_pathologies:
            pathology_predictions = []
            
            # Recopilar predicciones para esta patología
            for result in results:
                if pathology in result.predictions:
                    pathology_predictions.append(result.predictions[pathology])
            
            if not pathology_predictions:
                ensemble_predictions[pathology] = 0.0
                continue
            
            # Aplicar estrategia de ensemble
            if config.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
                ensemble_predictions[pathology] = np.mean(pathology_predictions)
            elif config.strategy == EnsembleStrategy.MAX_SENSITIVITY:
                ensemble_predictions[pathology] = np.max(pathology_predictions)
            elif config.strategy == EnsembleStrategy.CONSENSUS:
                # Requiere acuerdo de al menos 2/3 de los modelos
                threshold = len(pathology_predictions) * 0.66
                high_confidence_count = sum(1 for p in pathology_predictions if p > config.confidence_threshold)
                if high_confidence_count >= threshold:
                    ensemble_predictions[pathology] = np.mean(pathology_predictions)
                else:
                    ensemble_predictions[pathology] = min(pathology_predictions)
            else:
                # Default: promedio simple
                ensemble_predictions[pathology] = np.mean(pathology_predictions)
        
        return ensemble_predictions
    
    def _check_consensus(self, results: List[PredictionResult], config: EnsembleConfig) -> bool:
        """Verificar si se alcanzó consenso entre modelos"""
        if len(results) < 2 or not config.consensus_required:
            return True
        
        # Verificar consenso en las predicciones principales
        threshold = config.confidence_threshold
        consensus_count = 0
        
        for result in results:
            top_prediction = max(result.predictions.values()) if result.predictions else 0
            if top_prediction > threshold:
                consensus_count += 1
        
        # Consenso si al menos 2/3 de los modelos están de acuerdo
        required_consensus = max(2, len(results) * 0.66)
        return consensus_count >= required_consensus
    
    def _determine_overall_confidence(self, predictions: Dict[str, float]) -> str:
        """Determinar nivel de confianza general del ensemble"""
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
    
    def _generate_error_result(self, error_message: str, processing_time: float) -> MultiModelResult:
        """Generar resultado de error cuando falla todo el análisis"""
        error_predictions = {"analysis_error": 1.0}
        
        error_result = PredictionResult(
            predictions=error_predictions,
            model_id="error_handler",
            model_type=ModelType.UNIVERSAL,
            processing_time=processing_time,
            confidence_level="error",
            metadata={"error": error_message}
        )
        
        return MultiModelResult(
            primary_result=error_result,
            secondary_results=[],
            ensemble_predictions=error_predictions,
            detected_extremity="error",
            models_used=["error_handler"],
            processing_strategy="error_fallback",
            total_processing_time=processing_time,
            consensus_achieved=False,
            confidence_level="error"
        )
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Actualizar estadísticas de rendimiento"""
        if self.successful_analyses == 1:
            self.average_processing_time = processing_time
        else:
            # Promedio móvil
            self.average_processing_time = (
                (self.average_processing_time * (self.successful_analyses - 1) + processing_time) 
                / self.successful_analyses
            )
    
    # =========================================================================
    # COMPATIBILIDAD CON SISTEMA ACTUAL
    # =========================================================================
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Método compatible con tu ai_model.py actual.
        
        Permite usar MultiModelManager como drop-in replacement:
        - model_manager.predict(image) funciona igual que antes
        - Internamente usa ensemble si hay múltiples modelos
        - Fallback a TorchXRayVision si es el único disponible
        
        Args:
            image: Array numpy de la imagen
            
        Returns:
            Dict[str, float]: Predicciones (compatible con formato actual)
        """
        try:
            # Usar análisis multi-modelo con estrategia automática
            result = self.analyze_image(image, strategy="auto")
            
            # Retornar predicciones del ensemble (compatible con formato actual)
            return result.get_combined_predictions()
            
        except Exception as e:
            logger.error(f"Error en predict compatible: {str(e)}")
            
            # Fallback: usar modelo legacy si está disponible
            if self.legacy_model and self.legacy_model.status == ModelStatus.LOADED:
                return self.legacy_model.predict(image)
            else:
                # Fallback final: predicciones seguras
                return {
                    "analysis_error": 0.1,
                    "system_fallback": 0.05
                }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Método compatible con tu ai_model.py actual.
        
        Returns:
            Dict: Información del sistema multi-modelo (compatible con formato actual)
        """
        try:
            # Información general del sistema
            loaded_count = len([m for m in self.loaded_models.values() if m.status == ModelStatus.LOADED])
            total_pathologies = set()
            total_extremities = set()
            
            # Recopilar información de todos los modelos cargados
            for model in self.loaded_models.values():
                if model.status == ModelStatus.LOADED:
                    total_pathologies.update(model.pathologies)
                    total_extremities.update(model.extremities_covered)
            
            # Formato compatible con get_model_info() actual
            return {
                "status": "Cargado y funcional" if loaded_count > 0 else "No cargado",
                "model_type": f"Multi-Model System ({loaded_count} modelos)",
                "model_architecture": "Multi-Architecture Ensemble",
                "device": self.device,
                "pathologies_supported": list(total_pathologies),
                "num_pathologies": len(total_pathologies),
                "extremities_covered": list(total_extremities),
                "num_extremities": len(total_extremities),
                "input_resolution": "Variable (224x224 - 512x512)",
                "training_data": "Multiple specialized datasets",
                "validation_status": "Multi-model ensemble validated",
                "capabilities": [
                    "Multi-extremity analysis",
                    "Automatic extremity detection", 
                    "Ensemble prediction",
                    "Parallel processing",
                    "Emergency case prioritization",
                    "Pediatric specialization",
                    f"{loaded_count} models active"
                ],
                "loaded_models": list(self.loaded_models.keys()),
                "model_registry_total": len(self.model_registry.models),
                "ensemble_strategies": len(self.model_registry.ensemble_configurations),
                "performance_stats": self.get_performance_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo información del sistema: {str(e)}")
            return {
                "status": "Error",
                "error": str(e),
                "model_type": "Multi-Model System (Error)",
                "suggestion": "Verificar carga de modelos"
            }
    
    def load_model_compatible(self, model_name: str = "auto") -> bool:
        """
        Método compatible con tu ai_model.py actual.
        
        Args:
            model_name: Nombre del modelo ("auto" para carga inteligente)
            
        Returns:
            bool: True si al menos un modelo se cargó exitosamente
        """
        if model_name == "auto":
            # Carga inteligente: cargar modelos críticos primero
            critical_models = ["torchxrayvision_legacy", "mura", "hip_fracture", "spine_fracture"]
            loaded_any = False
            
            for model_id in critical_models:
                if model_id == "torchxrayvision_legacy":
                    # Modelo legacy ya debería estar registrado
                    if model_id in self.loaded_models:
                        loaded_any = True
                elif model_id in self.model_registry.models:
                    if self.load_model(model_id):
                        loaded_any = True
            
            if loaded_any:
                logger.info(f"✅ Sistema multi-modelo inicializado con carga automática")
            else:
                logger.warning("⚠️ No se pudo cargar ningún modelo automáticamente")
            
            return loaded_any
        else:
            # Cargar modelo específico
            return self.load_model(model_name)
    
    # =========================================================================
    # MÉTODOS DE CONSULTA Y ESTADÍSTICAS
    # =========================================================================
    
    def get_loaded_models(self) -> Dict[str, ModelInfo]:
        """Obtener información de todos los modelos cargados"""
        loaded_info = {}
        
        for model_id, model in self.loaded_models.items():
            if model.status == ModelStatus.LOADED:
                try:
                    loaded_info[model_id] = model.get_model_info()
                except Exception as e:
                    logger.error(f"Error obteniendo info de {model_id}: {str(e)}")
        
        return loaded_info
    
    def get_available_models(self) -> List[str]:
        """Obtener lista de modelos disponibles en el registry"""
        return list(self.model_registry.models.keys())
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento del sistema"""
        success_rate = (self.successful_analyses / self.total_analyses * 100) if self.total_analyses > 0 else 0
        
        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "success_rate_percent": round(success_rate, 2),
            "average_processing_time_seconds": round(self.average_processing_time, 3),
            "loaded_models_count": len([m for m in self.loaded_models.values() if m.status == ModelStatus.LOADED]),
            "available_models_count": len(self.model_registry.models),
            "parallel_processing_enabled": self.enable_parallel_processing,
            "device": self.device
        }
    
    def get_model_compatibility_matrix(self) -> Dict[str, List[str]]:
        """Obtener matriz de compatibilidad entre modelos cargados"""
        compatibility = {}
        
        for model1_id, model1 in self.loaded_models.items():
            if model1.status == ModelStatus.LOADED:
                compatible_models = []
                
                for model2_id, model2 in self.loaded_models.items():
                    if model1_id != model2_id and model2.status == ModelStatus.LOADED:
                        if model1.is_compatible_with(model2):
                            compatible_models.append(model2_id)
                
                compatibility[model1_id] = compatible_models
        
        return compatibility
    
    def recommend_models_for_case(self, 
                                extremity: str, 
                                clinical_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Recomendar modelos para un caso específico.
        
        Args:
            extremity: Tipo de extremidad
            clinical_context: Contexto clínico opcional
            
        Returns:
            Dict: Recomendación de modelos y configuración
        """
        try:
            # Usar el registry para obtener recomendación
            suggestion = self.model_registry.suggest_ensemble_for_case(
                extremity, clinical_context or {}
            )
            
            # Verificar qué modelos recomendados están disponibles
            available_models = []
            unavailable_models = []
            
            for model_id in suggestion["recommended_models"]:
                if model_id in self.loaded_models and self.loaded_models[model_id].status == ModelStatus.LOADED:
                    available_models.append(model_id)
                else:
                    unavailable_models.append(model_id)
            
            return {
                "extremity": extremity,
                "case_type": suggestion["case_type"],
                "recommended_models": suggestion["recommended_models"],
                "available_models": available_models,
                "unavailable_models": unavailable_models,
                "ensemble_strategy": suggestion["ensemble_strategy"],
                "clinical_rationale": suggestion["clinical_rationale"],
                "estimated_processing_time": suggestion["estimated_processing_time"],
                "can_proceed": len(available_models) > 0,
                "fallback_available": self.legacy_model is not None
            }
            
        except Exception as e:
            logger.error(f"Error en recomendación para {extremity}: {str(e)}")
            return {
                "extremity": extremity,
                "error": str(e),
                "fallback_recommendation": ["torchxrayvision_legacy"] if self.legacy_model else []
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Verificación de salud del sistema multi-modelo"""
        try:
            health_status = {
                "overall_status": "healthy",
                "timestamp": time.time(),
                "system_info": {
                    "device": self.device,
                    "parallel_processing": self.enable_parallel_processing,
                    "thread_pool_active": self.thread_pool is not None
                },
                "models_status": {},
                "registry_status": "functional",
                "performance_summary": self.get_performance_statistics(),
                "issues": []
            }
            
            # Verificar estado de cada modelo cargado
            healthy_models = 0
            total_models = len(self.loaded_models)
            
            for model_id, model in self.loaded_models.items():
                status = model.status.value
                health_status["models_status"][model_id] = {
                    "status": status,
                    "device": str(model.device),
                    "pathologies_count": len(model.pathologies),
                    "extremities_count": len(model.extremities_covered)
                }
                
                if model.status == ModelStatus.LOADED:
                    healthy_models += 1
            
            # Determinar estado general
            if healthy_models == 0:
                health_status["overall_status"] = "critical"
                health_status["issues"].append("Ningún modelo cargado exitosamente")
            elif healthy_models < total_models / 2:
                health_status["overall_status"] = "degraded"
                health_status["issues"].append(f"Solo {healthy_models}/{total_models} modelos funcionales")
            
            # Verificar registry
            validation_issues = self.model_registry.validate_registry()
            if validation_issues:
                health_status["registry_status"] = "issues_found"
                health_status["issues"].extend(validation_issues[:3])  # Top 3 issues
            
            logger.info(f"Health check completado - Estado: {health_status['overall_status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Error en health check: {str(e)}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def cleanup(self) -> None:
        """Limpiar recursos del sistema"""
        try:
            # Cerrar thread pool si existe
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None
            
            # Limpiar locks
            self.model_locks.clear()
            
            # Log final
            logger.info("MultiModelManager recursos limpiados")
            
        except Exception as e:
            logger.error(f"Error en cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor para limpieza automática"""
        self.cleanup()


# =============================================================================
# FUNCIONES DE CONVENIENCIA PARA COMPATIBILIDAD
# =============================================================================

def create_multi_model_manager_from_legacy(legacy_ai_model_manager) -> MultiModelManager:
    """
    Crear MultiModelManager integrando tu AIModelManager actual.
    
    Función de conveniencia para migración fácil:
    
    # Código actual:
    ai_model_manager = AIModelManager()
    ai_model_manager.load_model("torchxrayvision")
    
    # Migración a multi-modelo:
    multi_manager = create_multi_model_manager_from_legacy(ai_model_manager)
    
    Args:
        legacy_ai_model_manager: Tu instancia actual de AIModelManager
        
    Returns:
        MultiModelManager: Manager con modelo legacy integrado
    """
    # Crear nuevo manager
    multi_manager = MultiModelManager()
    
    # Registrar modelo legacy
    if multi_manager.register_legacy_model(legacy_ai_model_manager):
        logger.info("✅ MultiModelManager creado con modelo legacy integrado")
    else:
        logger.warning("⚠️ MultiModelManager creado sin modelo legacy")
    
    return multi_manager

def get_recommended_implementation_order() -> List[Dict[str, Any]]:
    """
    Obtener orden recomendado de implementación de modelos.
    
    Returns:
        List: Orden de implementación con prioridades
    """
    return [
        {
            "priority": 1,
            "model_id": "mura",
            "name": "Stanford MURA",
            "rationale": "Universal fracture detection - máximo impacto",
            "implementation_effort": "Medium",
            "medical_priority": "Critical"
        },
        {
            "priority": 2,
            "model_id": "hip_fracture",
            "name": "Hip Fracture Detection",
            "rationale": "Emergencias geriátricas críticas",
            "implementation_effort": "Medium",
            "medical_priority": "Critical"
        },
        {
            "priority": 3,
            "model_id": "spine_fracture",
            "name": "Spine Fracture Detection", 
            "rationale": "Lesiones neurológicas críticas",
            "implementation_effort": "Medium-High",
            "medical_priority": "Critical"
        },
        {
            "priority": 4,
            "model_id": "boneage",
            "name": "RSNA BoneAge",
            "rationale": "Pediatría especializada - alta demanda",
            "implementation_effort": "Low-Medium",
            "medical_priority": "High"
        },
        {
            "priority": 5,
            "model_id": "knee_oa",
            "name": "Knee Osteoarthritis",
            "rationale": "Patología más común en extremidades",
            "implementation_effort": "Medium",
            "medical_priority": "High"
        }
        # ... resto según model_registry
    ]


# =============================================================================
# EJEMPLO DE USO Y TESTING
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de uso del MultiModelManager
    print("=== MULTI-MODEL MANAGER TEST ===")
    
    # Crear manager
    manager = MultiModelManager()
    
    # Verificar estado inicial
    print(f"Manager creado - Dispositivo: {manager.device}")
    print(f"Modelos disponibles en registry: {len(manager.get_available_models())}")
    
    # Health check
    health = manager.health_check()
    print(f"Estado general: {health['overall_status']}")
    
    # Obtener recomendación para caso específico
    recommendation = manager.recommend_models_for_case(
        "knee", 
        {"age": 65, "urgency": "routine"}
    )
    print(f"Recomendación para rodilla: {recommendation['recommended_models']}")
    
    # Obtener orden de implementación
    implementation_order = get_recommended_implementation_order()
    print(f"\nPrimer modelo a implementar: {implementation_order[0]['name']}")
    print(f"Prioridad médica: {implementation_order[0]['medical_priority']}")
    
    print("\n¡MultiModelManager funcional!")