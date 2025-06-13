#!/usr/bin/env python3
"""
Router Inteligente para Modelos de Radiología IA
================================================

Sistema de enrutamiento avanzado que coordina múltiples modelos de IA médica
para obtener la máxima precisión en análisis radiológicos.

Características:
- Detección automática de tipo de imagen
- Selección inteligente de modelos
- Ensemble optimizado por especialización
- Escalabilidad para nuevos modelos
- Trazabilidad médica completa
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import cv2
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ============================================================================
# SISTEMA DE TIPOS Y ENUMS
# ============================================================================

class ImageType(Enum):
    """Tipos de imagen médica detectados automáticamente."""
    CHEST_XRAY = "chest_xray"
    EXTREMITY = "extremity" 
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE = "spine"
    UNKNOWN = "unknown"

class StudyType(Enum):
    """Tipos de estudio radiológico."""
    PA_CHEST = "pa_chest"
    LATERAL_CHEST = "lateral_chest"
    BONE_FRACTURE = "bone_fracture"
    TRAUMA = "trauma"
    ROUTINE = "routine"
    EMERGENCY = "emergency"

class ModelCapability(Enum):
    """Capacidades especializadas de cada modelo."""
    CHEST_PATHOLOGY = "chest_pathology"
    FRACTURE_DETECTION = "fracture_detection"
    PNEUMONIA_SPECIALIST = "pneumonia_specialist"
    UNIVERSAL_MEDICAL = "universal_medical"

class ConfidenceLevel(Enum):
    """Niveles de confianza en predicciones."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

# ============================================================================
# DATACLASSES PARA STRUCTURED DATA
# ============================================================================

@dataclass
class ImageAnalysis:
    """Análisis automático de características de imagen."""
    image_type: ImageType
    study_type: StudyType
    quality_score: float
    contrast_level: float
    noise_level: float
    anatomical_region: str
    has_trauma_indicators: bool
    estimated_age_group: str
    technical_quality: str
    
class ModelSpec:
    """Especificación de modelo registrado en el router."""
    def __init__(self, name: str, capabilities: List[ModelCapability], 
                 specialization_weight: float, device: str = "auto"):
        self.name = name
        self.capabilities = capabilities
        self.specialization_weight = specialization_weight
        self.device = device
        self.is_loaded = False
        self.load_time = None
        self.error_count = 0
        self.success_count = 0
        self.avg_inference_time = 0.0

@dataclass
class PredictionResult:
    """Resultado de predicción de un modelo individual."""
    model_name: str
    predictions: Dict[str, float]
    confidence: float
    inference_time: float
    model_specific_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleResult:
    """Resultado final del ensemble de modelos."""
    final_predictions: Dict[str, float]
    individual_results: List[PredictionResult]
    consensus_analysis: Dict[str, Any]
    medical_recommendation: Dict[str, Any]
    confidence_score: float
    models_used: List[str]
    processing_time: float
    image_analysis: ImageAnalysis

# ============================================================================
# BASE INTERFACE PARA MODELOS
# ============================================================================

class MedicalModelInterface(ABC):
    """Interface base que deben implementar todos los modelos médicos."""
    
    @abstractmethod
    def load_model(self) -> bool:
        """Carga el modelo y retorna True si exitoso."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Realiza predicción sobre la imagen."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[ModelCapability]:
        """Retorna las capacidades del modelo."""
        pass
    
    @property
    @abstractmethod
    def specialization_weight(self) -> float:
        """Peso de especialización (0.0 a 1.0)."""
        pass

# ============================================================================
# ADAPTADORES PARA MODELOS EXISTENTES
# ============================================================================

class ToraxModelAdapter(MedicalModelInterface):
    """Adaptador para ToraxModel (TorchXRayVision)."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self._model = None
        self._manager = None
    
    def load_model(self) -> bool:
        try:
            from ..models.torax_model import AIModelManager
            self._manager = AIModelManager(device=self.device)
            return self._manager.load_model()
        except Exception as e:
            logger.error(f"Error cargando ToraxModel: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self._manager:
            raise RuntimeError("Modelo no cargado")
        return {"predictions": self._manager.predict(image)}
    
    def get_model_info(self) -> Dict[str, Any]:
        return self._manager.get_model_info() if self._manager else {}
    
    @property
    def capabilities(self) -> List[ModelCapability]:
        return [ModelCapability.CHEST_PATHOLOGY]
    
    @property
    def specialization_weight(self) -> float:
        return 0.9  # Altamente especializado en tórax

class FracturasModelAdapter(MedicalModelInterface):
    """Adaptador para FracturasModel (MIMIC)."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self._manager = None
    
    def load_model(self) -> bool:
        try:
            from ..models.fracturas_generales_model import FracturasManager
            self._manager = FracturasManager(device=self.device)
            return self._manager.load_model()
        except Exception as e:
            logger.error(f"Error cargando FracturasModel: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self._manager:
            raise RuntimeError("Modelo no cargado")
        return self._manager.predict(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        return self._manager.get_model_info() if self._manager else {}
    
    @property
    def capabilities(self) -> List[ModelCapability]:
        return [ModelCapability.FRACTURE_DETECTION]
    
    @property
    def specialization_weight(self) -> float:
        return 0.95  # Muy especializado en fracturas

class CheXNetModelAdapter(MedicalModelInterface):
    """Adaptador para CheXNetModel (Stanford)."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self._manager = None
    
    def load_model(self) -> bool:
        try:
            from ..models.chexnet_model import CheXNetManager
            self._manager = CheXNetManager(device=self.device)
            return self._manager.load_model()
        except Exception as e:
            logger.error(f"Error cargando CheXNetModel: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self._manager:
            raise RuntimeError("Modelo no cargado")
        return self._manager.predict(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        return self._manager.get_model_info() if self._manager else {}
    
    @property
    def capabilities(self) -> List[ModelCapability]:
        return [ModelCapability.PNEUMONIA_SPECIALIST, ModelCapability.CHEST_PATHOLOGY]
    
    @property
    def specialization_weight(self) -> float:
        return 0.85  # Especializado en neumonía

class RadImageNetModelAdapter(MedicalModelInterface):
    """Adaptador para RadImageNetModel (Universal)."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self._manager = None
    
    def load_model(self) -> bool:
        try:
            from ..models.radimagenet_model import RadImageNetManager
            self._manager = RadImageNetManager(device=self.device)
            return self._manager.load_model()
        except Exception as e:
            logger.error(f"Error cargando RadImageNetModel: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self._manager:
            raise RuntimeError("Modelo no cargado")
        return {"predictions": self._manager.predict(image)}
    
    def get_model_info(self) -> Dict[str, Any]:
        return self._manager.get_model_info() if self._manager else {}
    
    @property
    def capabilities(self) -> List[ModelCapability]:
        return [ModelCapability.UNIVERSAL_MEDICAL]
    
    @property
    def specialization_weight(self) -> float:
        return 0.6  # Modelo universal, menos especializado

# ============================================================================
# ANALIZADOR DE IMÁGENES MÉDICAS
# ============================================================================

class MedicalImageAnalyzer:
    """Analizador inteligente de características de imagen médica."""
    
    @staticmethod
    def analyze_image(image: np.ndarray) -> ImageAnalysis:
        """
        Analiza automáticamente las características de una imagen médica.
        
        Args:
            image: Imagen médica como array numpy
            
        Returns:
            ImageAnalysis: Análisis completo de la imagen
        """
        try:
            # Análisis de calidad básica
            quality_score = MedicalImageAnalyzer._assess_image_quality(image)
            contrast_level = MedicalImageAnalyzer._calculate_contrast(image)
            noise_level = MedicalImageAnalyzer._estimate_noise(image)
            
            # Detección de tipo de imagen
            image_type = MedicalImageAnalyzer._detect_image_type(image)
            study_type = MedicalImageAnalyzer._classify_study_type(image, image_type)
            
            # Análisis anatómico
            anatomical_region = MedicalImageAnalyzer._identify_anatomical_region(image, image_type)
            
            # Detección de indicadores
            has_trauma = MedicalImageAnalyzer._detect_trauma_indicators(image)
            
            # Estimaciones adicionales
            age_group = MedicalImageAnalyzer._estimate_age_group(image)
            tech_quality = MedicalImageAnalyzer._assess_technical_quality(quality_score, contrast_level)
            
            return ImageAnalysis(
                image_type=image_type,
                study_type=study_type,
                quality_score=quality_score,
                contrast_level=contrast_level,
                noise_level=noise_level,
                anatomical_region=anatomical_region,
                has_trauma_indicators=has_trauma,
                estimated_age_group=age_group,
                technical_quality=tech_quality
            )
            
        except Exception as e:
            logger.error(f"Error en análisis de imagen: {e}")
            # Retornar análisis por defecto
            return ImageAnalysis(
                image_type=ImageType.UNKNOWN,
                study_type=StudyType.ROUTINE,
                quality_score=0.5,
                contrast_level=0.5,
                noise_level=0.5,
                anatomical_region="unknown",
                has_trauma_indicators=False,
                estimated_age_group="adult",
                technical_quality="moderate"
            )
    
    @staticmethod
    def _assess_image_quality(image: np.ndarray) -> float:
        """Evalúa la calidad general de la imagen."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Usar varianza de Laplaciano para detectar blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalizar a escala 0-1
        quality = min(1.0, max(0.0, laplacian_var / 1000.0))
        return quality
    
    @staticmethod
    def _calculate_contrast(image: np.ndarray) -> float:
        """Calcula el nivel de contraste."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # RMS contrast
        contrast = gray.std() / 255.0
        return min(1.0, contrast * 2.0)  # Normalizar
    
    @staticmethod
    def _estimate_noise(image: np.ndarray) -> float:
        """Estima el nivel de ruido."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Usar filtro de mediana para estimar ruido
        median_filtered = cv2.medianBlur(gray, 5)
        noise = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
        
        return min(1.0, noise / 50.0)  # Normalizar
    
    @staticmethod
    def _detect_image_type(image: np.ndarray) -> ImageType:
        """Detecta el tipo de imagen radiológica."""
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Heurísticas básicas para clasificación
        if 0.8 <= aspect_ratio <= 1.2:  # Aproximadamente cuadrada
            if height > 400:  # Imagen grande, probablemente tórax
                return ImageType.CHEST_XRAY
            else:
                return ImageType.EXTREMITY
        elif aspect_ratio > 1.2:  # Más ancha que alta
            return ImageType.CHEST_XRAY
        else:  # Más alta que ancha
            return ImageType.EXTREMITY
        
        return ImageType.UNKNOWN
    
    @staticmethod
    def _classify_study_type(image: np.ndarray, image_type: ImageType) -> StudyType:
        """Clasifica el tipo de estudio."""
        if image_type == ImageType.CHEST_XRAY:
            # Análisis de orientación para PA vs Lateral
            height, width = image.shape[:2]
            if width > height * 1.1:
                return StudyType.PA_CHEST
            else:
                return StudyType.LATERAL_CHEST
        elif image_type == ImageType.EXTREMITY:
            return StudyType.BONE_FRACTURE
        
        return StudyType.ROUTINE
    
    @staticmethod
    def _identify_anatomical_region(image: np.ndarray, image_type: ImageType) -> str:
        """Identifica la región anatómica."""
        region_map = {
            ImageType.CHEST_XRAY: "thorax",
            ImageType.EXTREMITY: "extremity",
            ImageType.ABDOMEN: "abdomen",
            ImageType.PELVIS: "pelvis",
            ImageType.SPINE: "spine"
        }
        
        return region_map.get(image_type, "unknown")
    
    @staticmethod
    def _detect_trauma_indicators(image: np.ndarray) -> bool:
        """Detecta indicadores visuales de trauma."""
        # Implementación básica - puede expandirse con ML
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detectar contornos irregulares que pueden indicar fracturas
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Heurística simple: muchos contornos pequeños pueden indicar trauma
        small_contours = [c for c in contours if cv2.contourArea(c) < 100]
        
        return len(small_contours) > 20
    
    @staticmethod
    def _estimate_age_group(image: np.ndarray) -> str:
        """Estima grupo etario basado en características óseas."""
        # Implementación simple - puede mejorarse con ML
        return "adult"  # Por defecto
    
    @staticmethod
    def _assess_technical_quality(quality_score: float, contrast_level: float) -> str:
        """Evalúa la calidad técnica general."""
        overall_score = (quality_score + contrast_level) / 2
        
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "moderate"
        elif overall_score >= 0.2:
            return "poor"
        else:
            return "very_poor"

# ============================================================================
# ROUTER INTELIGENTE PRINCIPAL
# ============================================================================

class IntelligentMedicalRouter:
    """
    Router inteligente para coordinación de múltiples modelos de IA médica.
    
    Características:
    - Registro dinámico de modelos
    - Selección automática basada en análisis de imagen
    - Ensemble inteligente por especialización
    - Trazabilidad médica completa
    - Escalabilidad para nuevos modelos
    """
    
    def __init__(self, device: str = "auto"):
        """
        Inicializa el router inteligente.
        
        Args:
            device: Dispositivo para los modelos ('auto', 'cpu', 'cuda')
        """
        self.device = device
        self.registered_models: Dict[str, MedicalModelInterface] = {}
        self.model_specs: Dict[str, ModelSpec] = {}
        self.image_analyzer = MedicalImageAnalyzer()
        self.is_initialized = False
        
        logger.info(f"IntelligentMedicalRouter inicializado - Dispositivo: {device}")
    
    def register_model(self, name: str, model: MedicalModelInterface) -> bool:
        """
        Registra un nuevo modelo en el router.
        
        Args:
            name: Nombre único del modelo
            model: Instancia del modelo que implementa MedicalModelInterface
            
        Returns:
            bool: True si el registro fue exitoso
        """
        try:
            # Crear especificación del modelo
            spec = ModelSpec(
                name=name,
                capabilities=model.capabilities,
                specialization_weight=model.specialization_weight,
                device=self.device
            )
            
            # Registrar modelo y especificación
            self.registered_models[name] = model
            self.model_specs[name] = spec
            
            logger.info(f"✅ Modelo '{name}' registrado - Capacidades: {[c.value for c in model.capabilities]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registrando modelo '{name}': {e}")
            return False
    
    def initialize_models(self) -> Dict[str, bool]:
        """
        Inicializa todos los modelos registrados.
        
        Returns:
            Dict[str, bool]: Estado de carga de cada modelo
        """
        load_results = {}
        start_time = time.time()
        
        logger.info(f"🔧 Inicializando {len(self.registered_models)} modelos...")
        
        for name, model in self.registered_models.items():
            try:
                logger.info(f"📦 Cargando modelo '{name}'...")
                model_start = time.time()
                
                success = model.load_model()
                load_time = time.time() - model_start
                
                # Actualizar especificaciones
                spec = self.model_specs[name]
                spec.is_loaded = success
                spec.load_time = load_time
                
                load_results[name] = success
                
                if success:
                    logger.info(f"✅ Modelo '{name}' cargado en {load_time:.2f}s")
                else:
                    logger.error(f"❌ Error cargando modelo '{name}'")
                    
            except Exception as e:
                logger.error(f"❌ Excepción cargando modelo '{name}': {e}")
                load_results[name] = False
        
        # Verificar si hay al menos un modelo cargado
        loaded_count = sum(load_results.values())
        total_time = time.time() - start_time
        
        if loaded_count > 0:
            self.is_initialized = True
            logger.info(f"✅ Router inicializado: {loaded_count}/{len(self.registered_models)} modelos en {total_time:.2f}s")
        else:
            logger.error("❌ No se pudo cargar ningún modelo")
        
        return load_results
    
    def analyze_and_route(self, image: np.ndarray, 
                         force_models: Optional[List[str]] = None) -> EnsembleResult:
        """
        Analiza la imagen y ejecuta el ensemble de modelos apropiados.
        
        Args:
            image: Imagen médica como array numpy
            force_models: Lista de modelos específicos a usar (opcional)
            
        Returns:
            EnsembleResult: Resultado completo del análisis ensemble
        """
        if not self.is_initialized:
            raise RuntimeError("Router no inicializado. Ejecutar initialize_models() primero.")
        
        start_time = time.time()
        
        # PASO 1: Analizar imagen automáticamente
        logger.info("🔍 Analizando características de imagen...")
        image_analysis = self.image_analyzer.analyze_image(image)
        
        logger.info(f"📊 Imagen detectada: {image_analysis.image_type.value} - "
                   f"Estudio: {image_analysis.study_type.value} - "
                   f"Calidad: {image_analysis.technical_quality}")
        
        # PASO 2: Seleccionar modelos apropiados
        if force_models:
            selected_models = force_models
            logger.info(f"🎯 Usando modelos forzados: {selected_models}")
        else:
            selected_models = self._select_optimal_models(image_analysis)
            logger.info(f"🎯 Modelos seleccionados automáticamente: {selected_models}")
        
        # PASO 3: Ejecutar predicciones en paralelo conceptual
        individual_results = []
        for model_name in selected_models:
            if model_name in self.registered_models and self.model_specs[model_name].is_loaded:
                try:
                    result = self._run_single_model(model_name, image)
                    individual_results.append(result)
                    
                    # Actualizar estadísticas del modelo
                    spec = self.model_specs[model_name]
                    spec.success_count += 1
                    spec.avg_inference_time = (spec.avg_inference_time + result.inference_time) / 2
                    
                except Exception as e:
                    logger.error(f"❌ Error en modelo '{model_name}': {e}")
                    self.model_specs[model_name].error_count += 1
        
        # PASO 4: Crear ensemble inteligente
        ensemble_result = self._create_ensemble(individual_results, image_analysis)
        ensemble_result.processing_time = time.time() - start_time
        
        logger.info(f"✅ Análisis ensemble completado en {ensemble_result.processing_time:.3f}s - "
                   f"Modelos: {len(individual_results)}, Confianza: {ensemble_result.confidence_score:.3f}")
        
        return ensemble_result
    
    def _select_optimal_models(self, image_analysis: ImageAnalysis) -> List[str]:
        """
        Selecciona los modelos óptimos basado en el análisis de imagen.
        
        Args:
            image_analysis: Análisis de la imagen
            
        Returns:
            List[str]: Nombres de modelos seleccionados
        """
        selected = []
        
        # Selección basada en tipo de imagen y estudio
        if image_analysis.image_type == ImageType.CHEST_XRAY:
            # Para tórax: ToraxModel + CheXNet siempre
            if "torax_model" in self.registered_models:
                selected.append("torax_model")
            if "chexnet_model" in self.registered_models:
                selected.append("chexnet_model")
        
        # Para cualquier imagen con sospecha de fractura
        if (image_analysis.has_trauma_indicators or 
            image_analysis.study_type == StudyType.BONE_FRACTURE or
            image_analysis.image_type == ImageType.EXTREMITY):
            if "fracturas_model" in self.registered_models:
                selected.append("fracturas_model")
        
        # RadImageNet como modelo universal siempre
        if "radimagenet_model" in self.registered_models:
            selected.append("radimagenet_model")
        
        # Si no hay selección específica, usar todos los disponibles
        if not selected:
            selected = [name for name, spec in self.model_specs.items() if spec.is_loaded]
        
        # Limitar a máximo 4 modelos para performance
        return selected[:4]
    
    def _run_single_model(self, model_name: str, image: np.ndarray) -> PredictionResult:
        """
        Ejecuta predicción en un modelo individual.
        
        Args:
            model_name: Nombre del modelo
            image: Imagen a analizar
            
        Returns:
            PredictionResult: Resultado de la predicción
        """
        start_time = time.time()
        
        model = self.registered_models[model_name]
        result = model.predict(image)
        
        inference_time = time.time() - start_time
        
        # Extraer predicciones según formato del modelo
        if "predictions" in result:
            predictions = result["predictions"]
            model_specific = {k: v for k, v in result.items() if k != "predictions"}
        else:
            predictions = result
            model_specific = {}
        
        # Calcular confianza del modelo
        confidence = self._calculate_model_confidence(predictions, model_name)
        
        return PredictionResult(
            model_name=model_name,
            predictions=predictions,
            confidence=confidence,
            inference_time=inference_time,
            model_specific_analysis=model_specific,
            metadata={"model_spec": self.model_specs[model_name]}
        )
    
    def _calculate_model_confidence(self, predictions: Dict[str, float], model_name: str) -> float:
        """
        Calcula la confianza del modelo basada en sus predicciones.
        
        Args:
            predictions: Predicciones del modelo
            model_name: Nombre del modelo
            
        Returns:
            float: Score de confianza (0.0 a 1.0)
        """
        if not predictions:
            return 0.0
        
        # Obtener peso de especialización del modelo
        spec_weight = self.model_specs[model_name].specialization_weight
        
        # Calcular confianza basada en distribución de probabilidades
        probs = list(predictions.values())
        max_prob = max(probs)
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
        normalized_entropy = entropy / np.log(len(probs)) if len(probs) > 1 else 0
        
        # Confianza combinada
        confidence = (max_prob * 0.4 + (1 - normalized_entropy) * 0.3 + spec_weight * 0.3)
        
        return min(1.0, max(0.0, confidence))
    
    def _create_ensemble(self, individual_results: List[PredictionResult], 
                        image_analysis: ImageAnalysis) -> EnsembleResult:
        """
        Crea el resultado ensemble combinando predicciones individuales.
        
        Args:
            individual_results: Resultados de modelos individuales
            image_analysis: Análisis de la imagen
            
        Returns:
            EnsembleResult: Resultado final del ensemble
        """
        if not individual_results:
            raise ValueError("No hay resultados individuales para crear ensemble")
        
        # Recopilar todas las patologías
        all_pathologies = set()
        for result in individual_results:
            all_pathologies.update(result.predictions.keys())
        
        # Crear predicciones ensemble ponderadas
        final_predictions = {}
        for pathology in all_pathologies:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for result in individual_results:
                if pathology in result.predictions:
                    # Peso basado en confianza del modelo y especialización
                    model_weight = result.confidence * self.model_specs[result.model_name].specialization_weight
                    weighted_sum += result.predictions[pathology] * model_weight
                    total_weight += model_weight
            
            if total_weight > 0:
                final_predictions[pathology] = weighted_sum / total_weight
            else:
                final_predictions[pathology] = 0.0
        
        # Análisis de consenso entre modelos
        consensus_analysis = self._analyze_consensus(individual_results, all_pathologies)
        
        # Generar recomendación médica
        medical_recommendation = self._generate_medical_recommendation(
            final_predictions, consensus_analysis, image_analysis
        )
        
        # Calcular confianza general del ensemble
        confidence_score = self._calculate_ensemble_confidence(individual_results, consensus_analysis)
        
        return EnsembleResult(
            final_predictions=final_predictions,
            individual_results=individual_results,
            consensus_analysis=consensus_analysis,
            medical_recommendation=medical_recommendation,
            confidence_score=confidence_score,
            models_used=[r.model_name for r in individual_results],
            processing_time=0.0,  # Se actualiza después
            image_analysis=image_analysis
        )
    
    def _analyze_consensus(self, individual_results: List[PredictionResult], 
                          all_pathologies: set) -> Dict[str, Any]:
        """
        Analiza el nivel de consenso entre modelos.
        
        Args:
            individual_results: Resultados individuales
            all_pathologies: Todas las patologías detectadas
            
        Returns:
            Dict[str, Any]: Análisis de consenso
        """
        consensus_analysis = {
            "high_agreement": [],      # Patologías con alto acuerdo
            "moderate_agreement": [],  # Patologías con acuerdo moderado
            "low_agreement": [],       # Patologías con bajo acuerdo
            "conflicting": [],         # Patologías conflictivas
            "agreement_scores": {},    # Scores de acuerdo por patología
            "model_correlations": {}   # Correlaciones entre modelos
        }
        
        for pathology in all_pathologies:
            predictions = []
            for result in individual_results:
                if pathology in result.predictions:
                    predictions.append(result.predictions[pathology])
            
            if len(predictions) >= 2:
                # Calcular varianza como medida de acuerdo
                variance = np.var(predictions)
                mean_pred = np.mean(predictions)
                
                # Score de acuerdo (menor varianza = mayor acuerdo)
                agreement_score = 1.0 / (1.0 + variance * 10)
                consensus_analysis["agreement_scores"][pathology] = agreement_score
                
                # Clasificar nivel de acuerdo
                if agreement_score >= 0.8:
                    consensus_analysis["high_agreement"].append(pathology)
                elif agreement_score >= 0.6:
                    consensus_analysis["moderate_agreement"].append(pathology)
                elif agreement_score >= 0.4:
                    consensus_analysis["low_agreement"].append(pathology)
                else:
                    consensus_analysis["conflicting"].append(pathology)
        
        return consensus_analysis
    
    def _generate_medical_recommendation(self, final_predictions: Dict[str, float],
                                       consensus_analysis: Dict[str, Any],
                                       image_analysis: ImageAnalysis) -> Dict[str, Any]:
        """
        Genera recomendaciones médicas basadas en el análisis ensemble.
        
        Args:
            final_predictions: Predicciones finales del ensemble
            consensus_analysis: Análisis de consenso
            image_analysis: Análisis de imagen
            
        Returns:
            Dict[str, Any]: Recomendaciones médicas estructuradas
        """
        # Identificar hallazgos significativos
        significant_findings = []
        critical_findings = []
        
        for pathology, probability in final_predictions.items():
            if probability >= 0.7:
                critical_findings.append({
                    "pathology": pathology,
                    "probability": probability,
                    "urgency": "high"
                })
            elif probability >= 0.4:
                significant_findings.append({
                    "pathology": pathology,
                    "probability": probability,
                    "urgency": "moderate"
                })
        
        # Determinar nivel de urgencia general
        if critical_findings:
            urgency_level = "urgent"
            recommendation_text = "Evaluación médica urgente recomendada"
        elif significant_findings:
            urgency_level = "priority"
            recommendation_text = "Evaluación médica prioritaria recomendada"
        else:
            urgency_level = "routine"
            recommendation_text = "Seguimiento rutinario apropiado"
        
        # Recomendaciones específicas por consenso
        consensus_recommendations = []
        if consensus_analysis["high_agreement"]:
            consensus_recommendations.append(
                f"Alto consenso en: {', '.join(consensus_analysis['high_agreement'])}"
            )
        
        if consensus_analysis["conflicting"]:
            consensus_recommendations.append(
                f"Hallazgos conflictivos requieren evaluación adicional: {', '.join(consensus_analysis['conflicting'])}"
            )
        
        # Recomendaciones por calidad de imagen
        image_recommendations = []
        if image_analysis.technical_quality in ["poor", "very_poor"]:
            image_recommendations.append("Considerar repetir estudio por calidad técnica subóptima")
        
        if image_analysis.noise_level > 0.7:
            image_recommendations.append("Alto nivel de ruido puede afectar interpretación")
        
        return {
            "urgency_level": urgency_level,
            "primary_recommendation": recommendation_text,
            "significant_findings": significant_findings,
            "critical_findings": critical_findings,
            "consensus_recommendations": consensus_recommendations,
            "image_quality_recommendations": image_recommendations,
            "followup_needed": len(critical_findings) > 0 or len(significant_findings) > 2,
            "specialist_referral": len(critical_findings) > 0
        }
    
    def _calculate_ensemble_confidence(self, individual_results: List[PredictionResult],
                                     consensus_analysis: Dict[str, Any]) -> float:
        """
        Calcula la confianza general del ensemble.
        
        Args:
            individual_results: Resultados individuales
            consensus_analysis: Análisis de consenso
            
        Returns:
            float: Score de confianza del ensemble
        """
        if not individual_results:
            return 0.0
        
        # Confianza promedio de modelos individuales
        avg_model_confidence = np.mean([r.confidence for r in individual_results])
        
        # Factor de consenso (más modelos de acuerdo = mayor confianza)
        total_agreements = len(consensus_analysis.get("high_agreement", []))
        total_conflicts = len(consensus_analysis.get("conflicting", []))
        
        if total_agreements + total_conflicts > 0:
            consensus_factor = total_agreements / (total_agreements + total_conflicts)
        else:
            consensus_factor = 0.5
        
        # Factor de cantidad de modelos
        model_count_factor = min(1.0, len(individual_results) / 3.0)  # Óptimo con 3+ modelos
        
        # Confianza final ponderada
        ensemble_confidence = (
            avg_model_confidence * 0.5 + 
            consensus_factor * 0.3 + 
            model_count_factor * 0.2
        )
        
        return min(1.0, max(0.0, ensemble_confidence))
    
    def get_router_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del router y sus modelos.
        
        Returns:
            Dict[str, Any]: Estado completo del router
        """
        model_status = {}
        for name, spec in self.model_specs.items():
            model_status[name] = {
                "loaded": spec.is_loaded,
                "load_time": spec.load_time,
                "success_count": spec.success_count,
                "error_count": spec.error_count,
                "avg_inference_time": spec.avg_inference_time,
                "capabilities": [c.value for c in spec.capabilities],
                "specialization_weight": spec.specialization_weight
            }
        
        loaded_models = [name for name, spec in self.model_specs.items() if spec.is_loaded]
        
        return {
            "router_initialized": self.is_initialized,
            "total_models": len(self.registered_models),
            "loaded_models": len(loaded_models),
            "loaded_model_names": loaded_models,
            "device": self.device,
            "model_status": model_status,
            "capabilities_coverage": self._get_capabilities_coverage()
        }
    
    def _get_capabilities_coverage(self) -> Dict[str, List[str]]:
        """
        Analiza la cobertura de capacidades por los modelos cargados.
        
        Returns:
            Dict[str, List[str]]: Capacidades cubiertas y modelos que las proveen
        """
        coverage = {}
        for capability in ModelCapability:
            models_with_capability = []
            for name, spec in self.model_specs.items():
                if spec.is_loaded and capability in spec.capabilities:
                    models_with_capability.append(name)
            
            coverage[capability.value] = models_with_capability
        
        return coverage
    
    def predict_single_model(self, model_name: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta predicción en un modelo específico (útil para testing).
        
        Args:
            model_name: Nombre del modelo
            image: Imagen a analizar
            
        Returns:
            Dict[str, Any]: Resultado del modelo específico
        """
        if model_name not in self.registered_models:
            raise ValueError(f"Modelo '{model_name}' no registrado")
        
        if not self.model_specs[model_name].is_loaded:
            raise RuntimeError(f"Modelo '{model_name}' no cargado")
        
        result = self._run_single_model(model_name, image)
        
        return {
            "model_name": result.model_name,
            "predictions": result.predictions,
            "confidence": result.confidence,
            "inference_time": result.inference_time,
            "model_specific_analysis": result.model_specific_analysis
        }

# ============================================================================
# FACTORY PARA CREACIÓN AUTOMÁTICA DEL ROUTER
# ============================================================================

class MedicalRouterFactory:
    """
    Factory para crear y configurar el router inteligente automáticamente.
    """
    
    @staticmethod
    def create_default_router(device: str = "auto") -> IntelligentMedicalRouter:
        """
        Crea un router con todos los modelos disponibles registrados.
        
        Args:
            device: Dispositivo para los modelos
            
        Returns:
            IntelligentMedicalRouter: Router configurado
        """
        router = IntelligentMedicalRouter(device=device)
        
        # Registrar todos los modelos disponibles
        models_to_register = [
            ("torax_model", ToraxModelAdapter),
            ("fracturas_model", FracturasModelAdapter),
            ("chexnet_model", CheXNetModelAdapter),
            ("radimagenet_model", RadImageNetModelAdapter)
        ]
        
        registered_count = 0
        for name, adapter_class in models_to_register:
            try:
                adapter = adapter_class(device=device)
                if router.register_model(name, adapter):
                    registered_count += 1
                    logger.info(f"✅ {name} registrado exitosamente")
                else:
                    logger.warning(f"⚠️ Falló registro de {name}")
            except Exception as e:
                logger.error(f"❌ Error creando adaptador {name}: {e}")
        
        if registered_count == 0:
            logger.error("❌ No se pudo registrar ningún modelo")
            return None
        
        logger.info(f"🎯 Router creado con {registered_count}/4 modelos registrados")
        return router
    
    @staticmethod
    def create_custom_router(model_configs: List[Dict[str, Any]], 
                           device: str = "auto") -> IntelligentMedicalRouter:
        """
        Crea un router con configuración personalizada.
        
        Args:
            model_configs: Lista de configuraciones de modelo
            device: Dispositivo para los modelos
            
        Returns:
            IntelligentMedicalRouter: Router configurado
        """
        router = IntelligentMedicalRouter(device=device)
        
        for config in model_configs:
            name = config["name"]
            adapter_class = config["adapter_class"]
            adapter_args = config.get("args", {})
            
            try:
                adapter = adapter_class(device=device, **adapter_args)
                router.register_model(name, adapter)
            except Exception as e:
                logger.error(f"❌ Error registrando modelo personalizado {name}: {e}")
        
        return router

# ============================================================================
# MANAGER PRINCIPAL PARA INTEGRACIÓN
# ============================================================================

class AdvancedMedicalAIManager:
    """
    Manager principal que integra el router inteligente con la API.
    
    Esta clase reemplaza el AIModelManager simple y proporciona
    capacidades avanzadas de ensemble y routing automático.
    """
    
    def __init__(self, model_path: str = "./models/", device: str = "auto"):
        """
        Inicializa el manager avanzado.
        
        Args:
            model_path: Directorio de modelos (para compatibilidad)
            device: Dispositivo para los modelos
        """
        self.model_path = Path(model_path)
        self.device = device
        self.router = None
        self.is_initialized = False
        
        logger.info(f"AdvancedMedicalAIManager inicializado - Dispositivo: {device}")
    
    def load_model(self, model_name: str = "torchxrayvision") -> bool:
        """
        Carga el router inteligente con todos los modelos.
        
        Args:
            model_name: Nombre del modelo (para compatibilidad con API existente)
            
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            logger.info("🚀 Inicializando sistema de IA médica avanzado...")
            
            # Crear router con todos los modelos
            self.router = MedicalRouterFactory.create_default_router(device=self.device)
            
            if self.router is None:
                logger.error("❌ No se pudo crear el router")
                return False
            
            # Inicializar todos los modelos
            load_results = self.router.initialize_models()
            
            # Verificar que al menos un modelo se cargó
            loaded_count = sum(load_results.values())
            
            if loaded_count > 0:
                self.is_initialized = True
                logger.info(f"✅ Sistema IA avanzado inicializado con {loaded_count} modelos")
                return True
            else:
                logger.error("❌ No se pudo cargar ningún modelo")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema IA: {e}")
            return False
    
    def predict(self, image: np.ndarray, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Realiza predicción usando el sistema avanzado.
        
        Args:
            image: Imagen médica como array numpy
            use_ensemble: Si usar ensemble (True) o modelo simple (False)
            
        Returns:
            Dict[str, Any]: Resultado de la predicción
        """
        if not self.is_initialized:
            raise RuntimeError("Sistema no inicializado. Ejecutar load_model() primero.")
        
        if use_ensemble:
            # Usar ensemble inteligente
            ensemble_result = self.router.analyze_and_route(image)
            
            return {
                "predictions": ensemble_result.final_predictions,
                "analysis_type": "intelligent_ensemble",
                "models_used": ensemble_result.models_used,
                "confidence": ensemble_result.confidence_score,
                "processing_time": ensemble_result.processing_time,
                "medical_recommendation": ensemble_result.medical_recommendation,
                "consensus_analysis": ensemble_result.consensus_analysis,
                "image_analysis": {
                    "type": ensemble_result.image_analysis.image_type.value,
                    "study_type": ensemble_result.image_analysis.study_type.value,
                    "quality": ensemble_result.image_analysis.technical_quality,
                    "trauma_indicators": ensemble_result.image_analysis.has_trauma_indicators
                },
                "individual_results": [
                    {
                        "model": r.model_name,
                        "confidence": r.confidence,
                        "inference_time": r.inference_time
                    }
                    for r in ensemble_result.individual_results
                ]
            }
        else:
            # Usar solo modelo principal (compatibilidad)
            try:
                result = self.router.predict_single_model("torax_model", image)
                return {
                    "predictions": result["predictions"],
                    "analysis_type": "single_model",
                    "model_used": result["model_name"],
                    "confidence": result["confidence"],
                    "inference_time": result["inference_time"]
                }
            except Exception as e:
                logger.error(f"Error en predicción single model: {e}")
                # Fallback a ensemble si falla single model
                return self.predict(image, use_ensemble=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del sistema IA avanzado.
        
        Returns:
            Dict[str, Any]: Información completa del sistema
        """
        if not self.is_initialized:
            return {
                "status": "No inicializado",
                "error": "Sistema IA no inicializado"
            }
        
        router_status = self.router.get_router_status()
        
        return {
            "status": "Sistema IA Avanzado Operacional",
            "system_type": "IntelligentMedicalRouter",
            "total_models": router_status["total_models"],
            "loaded_models": router_status["loaded_models"],
            "loaded_model_names": router_status["loaded_model_names"],
            "device": router_status["device"],
            "capabilities": {
                "intelligent_routing": True,
                "ensemble_analysis": True,
                "automatic_model_selection": True,
                "medical_recommendations": True,
                "consensus_analysis": True,
                "image_quality_assessment": True,
                "multi_model_validation": True
            },
            "model_details": router_status["model_status"],
            "capabilities_coverage": router_status["capabilities_coverage"],
            "pathologies_supported": "Variable según modelos activos",
            "advanced_features": [
                "Análisis automático de imagen",
                "Selección inteligente de modelos",
                "Ensemble ponderado por especialización",
                "Detección de consenso entre modelos",
                "Recomendaciones médicas automáticas",
                "Evaluación de calidad técnica",
                "Escalabilidad para nuevos modelos"
            ]
        }
    
    def get_available_models(self) -> List[str]:
        """
        Obtiene lista de modelos disponibles.
        
        Returns:
            List[str]: Nombres de modelos disponibles
        """
        if not self.is_initialized:
            return []
        
        status = self.router.get_router_status()
        return status["loaded_model_names"]
    
    def predict_with_specific_models(self, image: np.ndarray, 
                                   model_names: List[str]) -> Dict[str, Any]:
        """
        Realiza predicción con modelos específicos.
        
        Args:
            image: Imagen médica
            model_names: Lista de modelos a usar
            
        Returns:
            Dict[str, Any]: Resultado de predicción
        """
        if not self.is_initialized:
            raise RuntimeError("Sistema no inicializado")
        
        ensemble_result = self.router.analyze_and_route(image, force_models=model_names)
        
        return {
            "predictions": ensemble_result.final_predictions,
            "analysis_type": "custom_ensemble",
            "models_used": ensemble_result.models_used,
            "confidence": ensemble_result.confidence_score,
            "medical_recommendation": ensemble_result.medical_recommendation
        }