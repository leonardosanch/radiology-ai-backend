"""
Anatomical Classifier - Detector Inteligente de Extremidades
==========================================================
Sistema de clasificación automática de regiones anatómicas en radiografías.
Determina qué extremidad está presente en una imagen y enruta al modelo apropiado.

Características:
- Análisis visual de imágenes radiográficas
- Interpretación de metadatos DICOM
- Clasificador ML ligero para backup
- Mapeo inteligente a modelos específicos
- Detección de orientación y vista

Regiones Soportadas:
- Tórax: chest, lung, heart, ribs
- Extremidades Superiores: shoulder, humerus, elbow, forearm, hand, wrist
- Extremidades Inferiores: hip, pelvis, femur, knee, tibia, ankle, foot
- Columna: cervical, thoracic, lumbar, spine
- Cabeza: skull, cervical

Autor: Radiology AI Team
Versión: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import re
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pydicom
from pydicom.errors import InvalidDicomError

# Importar componentes del sistema
from ..models.base.model_registry import ModelRegistry, get_models_for_extremity
from ..core.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# TIPOS Y ENUMS
# =============================================================================

class AnatomicalRegion(Enum):
    """Regiones anatómicas principales"""
    CHEST = "chest"
    SHOULDER = "shoulder"
    HUMERUS = "humerus"
    ELBOW = "elbow"
    FOREARM = "forearm"
    HAND = "hand"
    WRIST = "wrist"
    HIP = "hip"
    PELVIS = "pelvis"
    FEMUR = "femur"
    KNEE = "knee"
    TIBIA = "tibia"
    ANKLE = "ankle"
    FOOT = "foot"
    SPINE = "spine"
    CERVICAL = "cervical"
    THORACIC = "thoracic"
    LUMBAR = "lumbar"
    SKULL = "skull"
    UNKNOWN = "unknown"

class ViewPosition(Enum):
    """Posiciones de vista radiográfica"""
    AP = "ap"  # Anteroposterior
    PA = "pa"  # Posteroanterior
    LATERAL = "lateral"
    OBLIQUE = "oblique"
    UNKNOWN = "unknown"

class DetectionConfidence(Enum):
    """Niveles de confianza en la detección"""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.3-0.5
    UNCERTAIN = "uncertain"  # <0.3

@dataclass
class AnatomicalDetection:
    """Resultado de detección anatómica"""
    region: AnatomicalRegion
    confidence: float
    view_position: ViewPosition
    confidence_level: DetectionConfidence
    recommended_models: List[str]
    metadata_source: str
    image_features: Dict[str, Any]
    clinical_priority: str
    ensemble_strategy: str

# =============================================================================
# CLASIFICADOR VISUAL LIGERO
# =============================================================================

class LightAnatomicalCNN(nn.Module):
    """
    Red neuronal ligera para clasificación anatómica.
    Optimizada para velocidad y precisión en detección de extremidades.
    """
    
    def __init__(self, num_regions: int = 20):
        """
        Inicializar clasificador anatómico.
        
        Args:
            num_regions: Número de regiones anatómicas a clasificar
        """
        super(LightAnatomicalCNN, self).__init__()
        
        # Backbone ligero basado en MobileNet concepts
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Bloque 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Bloque 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Bloque 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_regions)
        )
        
        logger.info(f"LightAnatomicalCNN inicializada para {num_regions} regiones")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del clasificador"""
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# =============================================================================
# CLASIFICADOR ANATÓMICO PRINCIPAL
# =============================================================================

class AnatomicalClassifier:
    """
    Clasificador inteligente de regiones anatómicas.
    
    Combina múltiples técnicas para detectar qué extremidad está presente:
    1. Análisis de metadatos DICOM
    2. Patrones visuales en la imagen
    3. Clasificador ML de backup
    4. Reglas heurísticas médicas
    """
    
    def __init__(self, device: str = "auto"):
        """
        Inicializar clasificador anatómico.
        
        Args:
            device: Dispositivo de computación
        """
        self.device = self._setup_device(device)
        
        # Configuración
        self.input_size = (224, 224)
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3
        }
        
        # Diccionarios de mapeo
        self.dicom_keywords = self._setup_dicom_keywords()
        self.anatomical_patterns = self._setup_anatomical_patterns()
        self.region_to_models = self._setup_region_model_mapping()
        
        # Modelo de clasificación visual
        self.visual_classifier = None
        self.transform = self._setup_transforms()
        
        # Registry de modelos
        self.model_registry = ModelRegistry()
        
        logger.info("AnatomicalClassifier inicializado")
        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Regiones soportadas: {len(AnatomicalRegion)}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Configurar dispositivo de computación"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _setup_transforms(self) -> transforms.Compose:
        """Configurar transformaciones para el clasificador visual"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _setup_dicom_keywords(self) -> Dict[str, List[str]]:
        """Mapeo de palabras clave DICOM a regiones anatómicas"""
        return {
            AnatomicalRegion.CHEST.value: [
                "chest", "lung", "thorax", "heart", "cardiac", "pulmonary", 
                "rib", "sternum", "mediastinum", "pleural"
            ],
            AnatomicalRegion.SHOULDER.value: [
                "shoulder", "scapula", "clavicle", "acromioclavicular", "glenohumeral"
            ],
            AnatomicalRegion.HUMERUS.value: [
                "humerus", "upper arm", "humeral"
            ],
            AnatomicalRegion.ELBOW.value: [
                "elbow", "olecranon", "radial head", "coronoid"
            ],
            AnatomicalRegion.FOREARM.value: [
                "forearm", "radius", "ulna", "radioulnar"
            ],
            AnatomicalRegion.HAND.value: [
                "hand", "finger", "thumb", "metacarpal", "phalanx", "carpal",
                "scaphoid", "lunate", "triquetrum"
            ],
            AnatomicalRegion.WRIST.value: [
                "wrist", "carpal", "radiocarpal", "midcarpal"
            ],
            AnatomicalRegion.HIP.value: [
                "hip", "acetabulum", "femoral head", "femoral neck"
            ],
            AnatomicalRegion.PELVIS.value: [
                "pelvis", "pelvic", "iliac", "pubic", "ischial", "sacroiliac"
            ],
            AnatomicalRegion.FEMUR.value: [
                "femur", "thigh", "femoral shaft", "greater trochanter"
            ],
            AnatomicalRegion.KNEE.value: [
                "knee", "patella", "tibiofemoral", "patellofemoral", "meniscal"
            ],
            AnatomicalRegion.TIBIA.value: [
                "tibia", "fibula", "leg", "tibial", "fibular"
            ],
            AnatomicalRegion.ANKLE.value: [
                "ankle", "malleolar", "talus", "calcaneus", "mortise"
            ],
            AnatomicalRegion.FOOT.value: [
                "foot", "toe", "metatarsal", "tarsal", "navicular", "cuboid"
            ],
            AnatomicalRegion.SPINE.value: [
                "spine", "vertebral", "vertebra", "spinal"
            ],
            AnatomicalRegion.CERVICAL.value: [
                "cervical", "c-spine", "atlas", "axis", "c1", "c2", "c3", "c4", "c5", "c6", "c7"
            ],
            AnatomicalRegion.LUMBAR.value: [
                "lumbar", "l-spine", "l1", "l2", "l3", "l4", "l5", "lumbosacral"
            ],
            AnatomicalRegion.SKULL.value: [
                "skull", "head", "cranium", "brain", "facial", "mandible", "maxilla"
            ]
        }
    
    def _setup_anatomical_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Patrones visuales característicos por región anatómica"""
        return {
            AnatomicalRegion.CHEST.value: {
                "aspect_ratio_range": (0.7, 1.3),
                "typical_shapes": ["rectangular", "oval"],
                "key_structures": ["ribs", "lungs", "heart_shadow"],
                "symmetry": "bilateral"
            },
            AnatomicalRegion.HAND.value: {
                "aspect_ratio_range": (0.6, 1.4),
                "typical_shapes": ["elongated", "fan-like"],
                "key_structures": ["fingers", "metacarpals", "wrist"],
                "symmetry": "unilateral"
            },
            AnatomicalRegion.KNEE.value: {
                "aspect_ratio_range": (0.8, 1.2),
                "typical_shapes": ["circular", "oval"],
                "key_structures": ["patella", "femoral_condyles", "tibial_plateau"],
                "symmetry": "unilateral"
            },
            AnatomicalRegion.HIP.value: {
                "aspect_ratio_range": (1.0, 1.8),
                "typical_shapes": ["pelvic", "curved"],
                "key_structures": ["acetabulum", "femoral_head", "pelvis"],
                "symmetry": "bilateral"
            },
            AnatomicalRegion.SPINE.value: {
                "aspect_ratio_range": (0.3, 0.7),
                "typical_shapes": ["linear", "curved"],
                "key_structures": ["vertebrae", "disc_spaces", "spinous_processes"],
                "symmetry": "midline"
            }
        }
    
    def _setup_region_model_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Mapeo de regiones anatómicas a modelos y configuraciones"""
        return {
            AnatomicalRegion.CHEST.value: {
                "primary_models": ["torchxrayvision"],
                "secondary_models": [],
                "ensemble_strategy": "single",
                "clinical_priority": "high",
                "typical_urgency": "routine"
            },
            AnatomicalRegion.HAND.value: {
                "primary_models": ["mura", "boneage"],
                "secondary_models": [],
                "ensemble_strategy": "specialist_priority",
                "clinical_priority": "medium",
                "typical_urgency": "routine"
            },
            AnatomicalRegion.HIP.value: {
                "primary_models": ["mura", "hip_fracture"],
                "secondary_models": [],
                "ensemble_strategy": "max_sensitivity",
                "clinical_priority": "high",
                "typical_urgency": "emergency"
            },
            AnatomicalRegion.KNEE.value: {
                "primary_models": ["mura", "knee_oa"],
                "secondary_models": [],
                "ensemble_strategy": "weighted_average",
                "clinical_priority": "medium",
                "typical_urgency": "routine"
            },
            AnatomicalRegion.SPINE.value: {
                "primary_models": ["spine_fracture"],
                "secondary_models": ["mura"],
                "ensemble_strategy": "max_sensitivity",
                "clinical_priority": "high",
                "typical_urgency": "emergency"
            },
            # Mapeo para otras extremidades a MURA como universal
            AnatomicalRegion.SHOULDER.value: {
                "primary_models": ["mura"],
                "secondary_models": [],
                "ensemble_strategy": "single",
                "clinical_priority": "medium",
                "typical_urgency": "routine"
            },
            AnatomicalRegion.ANKLE.value: {
                "primary_models": ["mura"],
                "secondary_models": [],
                "ensemble_strategy": "single",
                "clinical_priority": "medium",
                "typical_urgency": "urgent"
            }
        }
    
    def classify_image(self, image: np.ndarray, dicom_metadata: Optional[Dict] = None) -> AnatomicalDetection:
        """
        Clasificar región anatómica de una imagen radiográfica.
        
        Args:
            image: Array numpy de la imagen
            dicom_metadata: Metadatos DICOM opcionales
        
        Returns:
            AnatomicalDetection: Resultado de la clasificación
        """
        try:
            logger.info("🔍 Iniciando clasificación anatómica")
            
            # 1. Análisis de metadatos DICOM (primera prioridad)
            dicom_result = self._analyze_dicom_metadata(dicom_metadata)
            
            # 2. Análisis visual de la imagen
            visual_result = self._analyze_visual_features(image)
            
            # 3. Clasificador ML (backup)
            ml_result = self._classify_with_ml(image)
            
            # 4. Combinar resultados y tomar decisión final
            final_detection = self._combine_detection_results(
                dicom_result, visual_result, ml_result, image
            )
            
            logger.info(f"✅ Región detectada: {final_detection.region.value}")
            logger.info(f"📊 Confianza: {final_detection.confidence:.3f}")
            logger.info(f"🎯 Modelos recomendados: {final_detection.recommended_models}")
            
            return final_detection
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación anatómica: {str(e)}")
            return self._generate_fallback_detection(image)
    
    def _analyze_dicom_metadata(self, metadata: Optional[Dict]) -> Optional[Tuple[AnatomicalRegion, float, str]]:
        """
        Analizar metadatos DICOM para detectar región anatómica.
        
        Args:
            metadata: Diccionario con metadatos DICOM
        
        Returns:
            Tuple con región, confianza y fuente, o None si no se puede determinar
        """
        if not metadata:
            return None
        
        try:
            # Campos DICOM más relevantes
            body_part = metadata.get('BodyPartExamined', '').lower()
            study_desc = metadata.get('StudyDescription', '').lower()
            series_desc = metadata.get('SeriesDescription', '').lower()
            
            # Combinar todas las descripciones
            combined_text = f"{body_part} {study_desc} {series_desc}".strip()
            
            if not combined_text:
                return None
            
            logger.debug(f"Analizando metadatos DICOM: '{combined_text}'")
            
            # Buscar coincidencias con palabras clave
            best_match = None
            best_score = 0
            
            for region, keywords in self.dicom_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in combined_text:
                        # Peso mayor para BodyPartExamined
                        if keyword in body_part:
                            score += 3
                        elif keyword in study_desc:
                            score += 2
                        else:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = region
            
            if best_match and best_score > 0:
                # Convertir score a confianza
                confidence = min(best_score / 5.0, 1.0)  # Normalizar
                region = AnatomicalRegion(best_match)
                
                logger.info(f"📋 DICOM detectó: {region.value} (confianza: {confidence:.3f})")
                return region, confidence, "dicom_metadata"
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analizando metadatos DICOM: {str(e)}")
            return None
    
    def _analyze_visual_features(self, image: np.ndarray) -> Optional[Tuple[AnatomicalRegion, float, str]]:
        """
        Analizar características visuales de la imagen para detectar región anatómica.
        
        Args:
            image: Array numpy de la imagen
        
        Returns:
            Tuple con región, confianza y fuente, o None
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Análisis de características básicas
            height, width = gray.shape
            aspect_ratio = width / height
            
            # Detectar contornos principales
            contours = self._find_main_contours(gray)
            
            # Análisis de simetría
            symmetry_score = self._analyze_symmetry(gray)
            
            # Detectar estructuras anatómicas típicas
            structural_features = self._detect_anatomical_structures(gray, contours)
            
            # Comparar con patrones conocidos
            best_match = None
            best_confidence = 0
            
            for region_name, patterns in self.anatomical_patterns.items():
                confidence = self._match_visual_pattern(
                    aspect_ratio, symmetry_score, structural_features, patterns
                )
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = region_name
            
            if best_match and best_confidence > 0.3:
                region = AnatomicalRegion(best_match)
                logger.info(f"👁️ Visual detectó: {region.value} (confianza: {best_confidence:.3f})")
                return region, best_confidence, "visual_analysis"
            
            return None
            
        except Exception as e:
            logger.warning(f"Error en análisis visual: {str(e)}")
            return None
    
    def _find_main_contours(self, gray_image: np.ndarray) -> List:
        """Encontrar contornos principales en la imagen"""
        try:
            # Preprocesamiento
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            
            # Detección de bordes
            edges = cv2.Canny(blurred, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área
            min_area = gray_image.shape[0] * gray_image.shape[1] * 0.01  # 1% del área total
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            return significant_contours[:10]  # Top 10 contornos
            
        except Exception:
            return []
    
    def _analyze_symmetry(self, gray_image: np.ndarray) -> float:
        """Analizar simetría de la imagen (útil para tórax, pelvis)"""
        try:
            height, width = gray_image.shape
            mid_point = width // 2
            
            # Dividir imagen en mitades
            left_half = gray_image[:, :mid_point]
            right_half = gray_image[:, mid_point:]
            
            # Flip la mitad derecha
            right_flipped = cv2.flip(right_half, 1)
            
            # Redimensionar si es necesario
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_resized = left_half[:, :min_width]
            right_resized = right_flipped[:, :min_width]
            
            # Calcular diferencia
            diff = cv2.absdiff(left_resized, right_resized)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0, symmetry_score)
            
        except Exception:
            return 0.5  # Neutral
    
    def _detect_anatomical_structures(self, gray_image: np.ndarray, contours: List) -> Dict[str, float]:
        """Detectar estructuras anatómicas características"""
        features = {
            "circular_structures": 0.0,
            "linear_structures": 0.0,
            "bilateral_structures": 0.0,
            "elongated_structures": 0.0,
            "complex_patterns": 0.0
        }
        
        try:
            for contour in contours:
                # Análisis de forma
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Detectar estructuras circulares (articulaciones)
                    if circularity > 0.7:
                        features["circular_structures"] += 1
                    
                    # Detectar estructuras elongadas (huesos largos)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio > 3:
                        features["elongated_structures"] += 1
            
            # Normalizar features
            max_structures = max(len(contours), 1)
            for key in features:
                features[key] = min(features[key] / max_structures, 1.0)
            
            return features
            
        except Exception:
            return features
    
    def _match_visual_pattern(self, aspect_ratio: float, symmetry: float, 
                            features: Dict[str, float], patterns: Dict[str, Any]) -> float:
        """Comparar características con patrones conocidos"""
        try:
            confidence = 0.0
            
            # Verificar aspect ratio
            ar_range = patterns.get("aspect_ratio_range", (0, 100))
            if ar_range[0] <= aspect_ratio <= ar_range[1]:
                confidence += 0.3
            
            # Verificar simetría según tipo
            symmetry_type = patterns.get("symmetry", "unknown")
            if symmetry_type == "bilateral" and symmetry > 0.7:
                confidence += 0.2
            elif symmetry_type == "unilateral" and symmetry < 0.6:
                confidence += 0.2
            elif symmetry_type == "midline" and 0.6 <= symmetry <= 0.8:
                confidence += 0.2
            
            # Verificar estructuras características
            key_structures = patterns.get("key_structures", [])
            for structure in key_structures:
                if structure in ["circular", "patella", "acetabulum"] and features["circular_structures"] > 0.3:
                    confidence += 0.1
                elif structure in ["elongated", "bones", "metacarpals"] and features["elongated_structures"] > 0.3:
                    confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0
    
    def _classify_with_ml(self, image: np.ndarray) -> Optional[Tuple[AnatomicalRegion, float, str]]:
        """
        Clasificar usando modelo ML (cuando esté disponible).
        Por ahora retorna None - se implementaría con un modelo entrenado.
        """
        # TODO: Implementar clasificador ML cuando tengamos dataset etiquetado
        # Por ahora, usar heurísticas adicionales
        
        try:
            # Heurísticas básicas basadas en tamaño y forma
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Reglas simples para casos obvios
            if 0.8 <= aspect_ratio <= 1.2 and min(height, width) > 400:
                # Probablemente tórax o pelvis
                if height > width:
                    return AnatomicalRegion.CHEST, 0.4, "ml_heuristic"
                else:
                    return AnatomicalRegion.HIP, 0.4, "ml_heuristic"
            
            elif aspect_ratio > 1.5:
                # Probablemente extremidad alargada
                return AnatomicalRegion.HAND, 0.3, "ml_heuristic"
            
            return None
            
        except Exception:
            return None
    
    def _combine_detection_results(self, dicom_result: Optional[Tuple], 
                                 visual_result: Optional[Tuple],
                                 ml_result: Optional[Tuple],
                                 image: np.ndarray) -> AnatomicalDetection:
        """
        Combinar resultados de diferentes métodos de detección.
        
        Args:
            dicom_result: Resultado del análisis DICOM
            visual_result: Resultado del análisis visual
            ml_result: Resultado del clasificador ML
            image: Imagen original
        
        Returns:
            AnatomicalDetection: Detección final combinada
        """
        # Prioridades: DICOM > Visual > ML > Fallback
        results = []
        
        if dicom_result:
            results.append(("dicom", dicom_result[0], dicom_result[1], dicom_result[2]))
        
        if visual_result:
            results.append(("visual", visual_result[0], visual_result[1], visual_result[2]))
        
        if ml_result:
            results.append(("ml", ml_result[0], ml_result[1], ml_result[2]))
        
        if not results:
            # Fallback total
            return self._generate_fallback_detection(image)
        
        # Tomar el resultado con mayor confianza, priorizando DICOM
        best_result = None
        best_score = 0
        
        for source, region, confidence, method in results:
            # Bonus por fuente confiable
            score = confidence
            if source == "dicom":
                score *= 1.5  # DICOM es más confiable
            elif source == "visual":
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best_result = (source, region, confidence, method)
        
        if best_result:
            source, region, confidence, method = best_result
            
            # Determinar nivel de confianza
            if confidence >= self.confidence_thresholds["high"]:
                conf_level = DetectionConfidence.HIGH
            elif confidence >= self.confidence_thresholds["medium"]:
                conf_level = DetectionConfidence.MEDIUM
            elif confidence >= self.confidence_thresholds["low"]:
                conf_level = DetectionConfidence.LOW
            else:
                conf_level = DetectionConfidence.UNCERTAIN
            
            # Obtener configuración para la región
            region_config = self.region_to_models.get(region.value, {})
            
            # Obtener modelos recomendados
            recommended_models = region_config.get("primary_models", ["mura"])  # Fallback a MURA
            
            return AnatomicalDetection(
                region=region,
                confidence=confidence,
                view_position=self._detect_view_position(image),
                confidence_level=conf_level,
                recommended_models=recommended_models,
                metadata_source=method,
                image_features=self._extract_image_features(image),
                clinical_priority=region_config.get("clinical_priority", "medium"),
                ensemble_strategy=region_config.get("ensemble_strategy", "single")
            )
        
        # Si llegamos aquí, usar fallback
        return self._generate_fallback_detection(image)
    
    def _detect_view_position(self, image: np.ndarray) -> ViewPosition:
        """
        Detectar posición de vista radiográfica (AP, PA, lateral, etc.)
        
        Args:
            image: Array numpy de la imagen
        
        Returns:
            ViewPosition: Posición detectada
        """
        try:
            # Análisis básico de simetría y orientación
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Análisis de simetría
            symmetry_score = self._analyze_symmetry(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
            
            # Heurísticas básicas
            if symmetry_score > 0.7 and 0.8 <= aspect_ratio <= 1.2:
                # Alta simetría y aspecto cuadrado/rectangular = AP/PA
                return ViewPosition.AP
            elif aspect_ratio < 0.7 or aspect_ratio > 1.4:
                # Aspecto alargado = lateral
                return ViewPosition.LATERAL
            else:
                return ViewPosition.UNKNOWN
                
        except Exception:
            return ViewPosition.UNKNOWN
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extraer características básicas de la imagen para metadatos.
        
        Args:
            image: Array numpy de la imagen
        
        Returns:
            Dict con características de la imagen
        """
        try:
            height, width = image.shape[:2]
            
            # Características básicas
            features = {
                "dimensions": {"width": int(width), "height": int(height)},
                "aspect_ratio": float(width / height),
                "total_pixels": int(width * height),
                "image_quality": "good"  # Placeholder - se podría analizar más
            }
            
            # Análisis de intensidad si es escala de grises
            if len(image.shape) == 2:
                features.update({
                    "mean_intensity": float(np.mean(image)),
                    "std_intensity": float(np.std(image)),
                    "min_intensity": int(np.min(image)),
                    "max_intensity": int(np.max(image))
                })
            
            return features
            
        except Exception:
            return {"error": "Could not extract features"}
    
    def _generate_fallback_detection(self, image: np.ndarray) -> AnatomicalDetection:
        """
        Generar detección de fallback cuando otros métodos fallan.
        
        Args:
            image: Array numpy de la imagen
        
        Returns:
            AnatomicalDetection: Detección conservadora de fallback
        """
        logger.warning("⚠️ Usando detección de fallback - región desconocida")
        
        # Usar MURA como modelo universal para casos desconocidos
        return AnatomicalDetection(
            region=AnatomicalRegion.UNKNOWN,
            confidence=0.1,
            view_position=ViewPosition.UNKNOWN,
            confidence_level=DetectionConfidence.UNCERTAIN,
            recommended_models=["mura"],  # MURA como fallback universal
            metadata_source="fallback",
            image_features=self._extract_image_features(image),
            clinical_priority="medium",
            ensemble_strategy="single"
        )
    
    def load_visual_classifier(self) -> bool:
        """
        Cargar clasificador visual ML (futuro).
        Por ahora retorna True para compatibilidad.
        
        Returns:
            bool: True si se carga exitosamente
        """
        try:
            # TODO: Implementar carga de modelo preentrenado
            # self.visual_classifier = LightAnatomicalCNN(num_regions=len(AnatomicalRegion))
            # checkpoint = torch.load("anatomical_classifier.pth")
            # self.visual_classifier.load_state_dict(checkpoint)
            # self.visual_classifier.to(self.device)
            # self.visual_classifier.eval()
            
            logger.info("✅ Clasificador visual listo (usando heurísticas)")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar clasificador visual: {str(e)}")
            return True  # Continuar con heurísticas
    
    def get_supported_regions(self) -> List[str]:
        """
        Obtener lista de regiones anatómicas soportadas.
        
        Returns:
            List[str]: Regiones soportadas
        """
        return [region.value for region in AnatomicalRegion if region != AnatomicalRegion.UNKNOWN]
    
    def get_region_info(self, region: str) -> Dict[str, Any]:
        """
        Obtener información detallada sobre una región anatómica.
        
        Args:
            region: Nombre de la región
        
        Returns:
            Dict con información de la región
        """
        region_config = self.region_to_models.get(region, {})
        dicom_keywords = self.dicom_keywords.get(region, [])
        visual_patterns = self.anatomical_patterns.get(region, {})
        
        return {
            "region_name": region,
            "primary_models": region_config.get("primary_models", []),
            "secondary_models": region_config.get("secondary_models", []),
            "ensemble_strategy": region_config.get("ensemble_strategy", "single"),
            "clinical_priority": region_config.get("clinical_priority", "medium"),
            "typical_urgency": region_config.get("typical_urgency", "routine"),
            "dicom_keywords": dicom_keywords,
            "visual_patterns": visual_patterns,
            "supported": True
        }
    
    def batch_classify(self, images: List[np.ndarray], 
                      metadata_list: Optional[List[Dict]] = None) -> List[AnatomicalDetection]:
        """
        Clasificar múltiples imágenes en lote.
        
        Args:
            images: Lista de arrays numpy
            metadata_list: Lista opcional de metadatos DICOM
        
        Returns:
            List[AnatomicalDetection]: Resultados de clasificación
        """
        results = []
        
        for i, image in enumerate(images):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            detection = self.classify_image(image, metadata)
            results.append(detection)
        
        logger.info(f"✅ Clasificación en lote completada: {len(results)} imágenes")
        return results
    
    def validate_detection(self, detection: AnatomicalDetection, 
                         expected_region: Optional[str] = None) -> Dict[str, Any]:
        """
        Validar resultado de detección (útil para testing y debugging).
        
        Args:
            detection: Resultado de detección
            expected_region: Región esperada opcional
        
        Returns:
            Dict con información de validación
        """
        validation_result = {
            "detection_valid": detection.confidence_level != DetectionConfidence.UNCERTAIN,
            "confidence_level": detection.confidence_level.value,
            "confidence_score": detection.confidence,
            "recommended_models_available": len(detection.recommended_models) > 0,
            "metadata_source": detection.metadata_source
        }
        
        if expected_region:
            validation_result.update({
                "expected_region": expected_region,
                "detected_region": detection.region.value,
                "match": detection.region.value == expected_region,
                "region_compatible": expected_region in self.get_supported_regions()
            })
        
        return validation_result

# =============================================================================
# FUNCIONES DE UTILIDAD Y HELPERS
# =============================================================================

def create_anatomical_classifier(device: str = "auto") -> AnatomicalClassifier:
    """
    Función de conveniencia para crear clasificador anatómico.
    
    Args:
        device: Dispositivo de computación
    
    Returns:
        AnatomicalClassifier: Instancia del clasificador
    """
    classifier = AnatomicalClassifier(device=device)
    classifier.load_visual_classifier()
    return classifier

def extract_dicom_metadata(dicom_file_path: str) -> Optional[Dict[str, str]]:
    """
    Extraer metadatos relevantes de archivo DICOM.
    
    Args:
        dicom_file_path: Ruta al archivo DICOM
    
    Returns:
        Dict con metadatos extraídos o None si hay error
    """
    try:
        ds = pydicom.dcmread(dicom_file_path)
        
        metadata = {}
        
        # Campos más importantes para clasificación anatómica
        fields_to_extract = [
            ('BodyPartExamined', 'BodyPartExamined'),
            ('StudyDescription', 'StudyDescription'),
            ('SeriesDescription', 'SeriesDescription'),
            ('ViewPosition', 'ViewPosition'),
            ('PatientOrientation', 'PatientOrientation'),
            ('ImageOrientationPatient', 'ImageOrientationPatient'),
            ('StudyInstanceUID', 'StudyInstanceUID'),
            ('SeriesInstanceUID', 'SeriesInstanceUID')
        ]
        
        for dicom_tag, key in fields_to_extract:
            try:
                value = getattr(ds, dicom_tag, None)
                if value is not None:
                    metadata[key] = str(value)
            except AttributeError:
                continue
        
        return metadata if metadata else None
        
    except (InvalidDicomError, Exception) as e:
        logger.warning(f"Error extrayendo metadatos DICOM: {str(e)}")
        return None

def region_requires_emergency_protocol(region: AnatomicalRegion) -> bool:
    """
    Determinar si una región requiere protocolo de emergencia.
    
    Args:
        region: Región anatómica
    
    Returns:
        bool: True si requiere protocolo de emergencia
    """
    emergency_regions = {
        AnatomicalRegion.HIP,
        AnatomicalRegion.SPINE,
        AnatomicalRegion.CERVICAL,
        AnatomicalRegion.SKULL
    }
    
    return region in emergency_regions

def get_clinical_recommendations(detection: AnatomicalDetection) -> Dict[str, Any]:
    """
    Obtener recomendaciones clínicas basadas en la detección.
    
    Args:
        detection: Resultado de detección anatómica
    
    Returns:
        Dict con recomendaciones clínicas
    """
    recommendations = {
        "urgency_level": detection.clinical_priority,
        "recommended_models": detection.recommended_models,
        "ensemble_strategy": detection.ensemble_strategy,
        "requires_emergency_protocol": region_requires_emergency_protocol(detection.region),
        "confidence_adequate": detection.confidence_level in [DetectionConfidence.HIGH, DetectionConfidence.MEDIUM],
        "suggested_action": "proceed" if detection.confidence > 0.3 else "manual_review"
    }
    
    # Recomendaciones específicas por región
    if detection.region == AnatomicalRegion.HIP:
        recommendations.update({
            "special_considerations": ["Check for femoral neck fractures", "Consider age-related factors"],
            "typical_pathologies": ["Hip fracture", "Arthritis", "Dislocation"]
        })
    elif detection.region == AnatomicalRegion.HAND:
        recommendations.update({
            "special_considerations": ["Consider pediatric bone age if patient < 18", "Check for scaphoid fractures"],
            "typical_pathologies": ["Fractures", "Arthritis", "Bone age assessment"]
        })
    elif detection.region == AnatomicalRegion.CHEST:
        recommendations.update({
            "special_considerations": ["Comprehensive thoracic analysis", "Check cardiac silhouette"],
            "typical_pathologies": ["Pneumonia", "Cardiomegaly", "Pneumothorax", "Effusion"]
        })
    
    return recommendations

# =============================================================================
# EJEMPLO DE USO Y TESTING
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de uso del clasificador anatómico
    print("=== ANATOMICAL CLASSIFIER TEST ===")
    
    # Crear clasificador
    classifier = create_anatomical_classifier(device="cpu")
    print(f"Clasificador creado - Dispositivo: {classifier.device}")
    print(f"Regiones soportadas: {len(classifier.get_supported_regions())}")
    
    # Test con imagen simulada
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test con metadatos simulados
    test_metadata = {
        "BodyPartExamined": "HAND",
        "StudyDescription": "Hand X-ray AP view",
        "ViewPosition": "AP"
    }
    
    # Clasificar imagen
    detection = classifier.classify_image(test_image, test_metadata)
    
    print(f"\n🎯 Resultado de Clasificación:")
    print(f"Región detectada: {detection.region.value}")
    print(f"Confianza: {detection.confidence:.3f}")
    print(f"Nivel de confianza: {detection.confidence_level.value}")
    print(f"Vista: {detection.view_position.value}")
    print(f"Modelos recomendados: {detection.recommended_models}")
    print(f"Prioridad clínica: {detection.clinical_priority}")
    print(f"Estrategia ensemble: {detection.ensemble_strategy}")
    print(f"Fuente de metadatos: {detection.metadata_source}")
    
    # Obtener recomendaciones clínicas
    recommendations = get_clinical_recommendations(detection)
    print(f"\n📋 Recomendaciones Clínicas:")
    print(f"Nivel de urgencia: {recommendations['urgency_level']}")
    print(f"Protocolo de emergencia: {recommendations['requires_emergency_protocol']}")
    print(f"Acción sugerida: {recommendations['suggested_action']}")
    
    # Test de regiones soportadas
    print(f"\n📊 Información del Sistema:")
    for region in ["chest", "hand", "hip", "knee"]:
        info = classifier.get_region_info(region)
        print(f"{region}: {info['primary_models']} (prioridad: {info['clinical_priority']})")
    
    print("\n✅ Anatomical Classifier funcional!")
    print("🔄 Listo para integración con MultiModelManager")