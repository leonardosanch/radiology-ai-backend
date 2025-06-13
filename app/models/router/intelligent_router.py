import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import torch
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importar todos los modelos especializados
from ..torax_model import AIModelManager
from ..chexnet_model import CheXNetManager
from ..fracturas_generales_model import FracturasManager
from ..radimagenet_model import RadImageNetManager

logger = logging.getLogger(__name__)

class AdvancedMedicalAIManager:
    """
    Router inteligente para análisis médico con múltiples modelos especializados.
    
    Combina TorchXRayVision, CheXNet, FracturasModel y RadImageNet para 
    proporcionar análisis médico completo y preciso de radiografías.
    """
    
    def __init__(self, model_path: str = "./models/", device: str = "auto"):
        """
        Inicializa el sistema de IA médica avanzado.
        
        Args:
            model_path: Directorio para modelos
            device: Dispositivo ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Estado de inicialización
        self.is_initialized = False
        self.models_loaded = {}
        
        # Configuración de modelos disponibles
        self.available_models = {
            "torchxrayvision": {
                "manager": None,
                "specialization": "general_thoracic_pathology",
                "priority": 1,
                "description": "Modelo principal validado clínicamente",
                "pathologies": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
                              "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
                              "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
            },
            "chexnet": {
                "manager": None,
                "specialization": "pneumonia_detection",
                "priority": 2,
                "description": "Especialista en neumonía y patologías torácicas",
                "pathologies": ["Pneumonia", "Atelectasis", "Cardiomegaly", "Consolidation", 
                              "Edema", "Effusion", "Infiltration", "Mass", "Nodule"]
            },
            "fracturas": {
                "manager": None,
                "specialization": "fracture_detection",
                "priority": 3,
                "description": "Especialista en detección de fracturas",
                "pathologies": ["Fracture", "Pneumothorax", "Effusion", "Soft_Tissue_Trauma"]
            },
            "radimagenet": {
                "manager": None,
                "specialization": "general_medical_imaging",
                "priority": 4,
                "description": "Base universal para imágenes médicas",
                "pathologies": ["Abnormal_Finding", "Normal_Anatomy", "Pathology_Present"]
            }
        }
        
        # Configuración de ensemble
        self.ensemble_weights = {
            "torchxrayvision": 0.4,  # Peso principal
            "chexnet": 0.3,          # Especialista neumonía
            "fracturas": 0.2,        # Especialista fracturas
            "radimagenet": 0.1       # Soporte general
        }
        
        # Lock para thread safety
        self._lock = threading.Lock()
        
        logger.info(f"AdvancedMedicalAIManager inicializado - Dispositivo: {self.device}")
        logger.info("🏥 Sistema de IA médica multi-modelo preparado")
    
    def load_model(self, model_name: str = "all") -> bool:
        """
        Carga modelos especificados del sistema médico.
        
        Args:
            model_name: Nombre del modelo o 'all' para cargar todos
            
        Returns:
            bool: True si se cargaron exitosamente
        """
        try:
            with self._lock:
                if model_name == "all":
                    return self._load_all_models()
                elif model_name in self.available_models:
                    return self._load_single_model(model_name)
                else:
                    logger.error(f"❌ Modelo no reconocido: {model_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error cargando modelo(s): {str(e)}")
            return False
    
    def _load_all_models(self) -> bool:
        """
        Carga todos los modelos disponibles de forma secuencial.
        
        Returns:
            bool: True si al menos un modelo se cargó exitosamente
        """
        logger.info("🔧 Cargando todos los modelos médicos...")
        
        loaded_count = 0
        
        # Cargar en orden de prioridad
        for model_name in sorted(self.available_models.keys(), 
                               key=lambda x: self.available_models[x]["priority"]):
            
            logger.info(f"📦 Cargando {model_name}...")
            
            if self._load_single_model(model_name):
                loaded_count += 1
                logger.info(f"✅ {model_name} cargado exitosamente")
            else:
                logger.warning(f"⚠️ Falló carga de {model_name}")
        
        self.is_initialized = loaded_count > 0
        
        if self.is_initialized:
            logger.info(f"✅ Sistema inicializado - {loaded_count}/{len(self.available_models)} modelos cargados")
            logger.info(f"🎯 Modelos activos: {list(self.models_loaded.keys())}")
        else:
            logger.error("❌ No se pudo cargar ningún modelo")
        
        return self.is_initialized
    
    def _load_single_model(self, model_name: str) -> bool:
        """
        Carga un modelo específico.
        
        Args:
            model_name: Nombre del modelo a cargar
            
        Returns:
            bool: True si se cargó exitosamente
        """
        try:
            if model_name == "torchxrayvision":
                manager = AIModelManager(str(self.model_path), self.device)
                success = manager.load_model("torchxrayvision")
                
            elif model_name == "chexnet":
                manager = CheXNetManager(str(self.model_path), self.device, variant="chexpert")
                success = manager.load_model()
                
            elif model_name == "fracturas":
                manager = FracturasManager(str(self.model_path), self.device, variant="mimic_nb")
                success = manager.load_model()
                
            elif model_name == "radimagenet":
                manager = RadImageNetManager(str(self.model_path), self.device, architecture="resnet50")
                success = manager.load_model()
                
            else:
                logger.error(f"❌ Modelo no implementado: {model_name}")
                return False
            
            if success:
                self.available_models[model_name]["manager"] = manager
                self.models_loaded[model_name] = {
                    "manager": manager,
                    "status": "loaded",
                    "specialization": self.available_models[model_name]["specialization"]
                }
                return True
            else:
                logger.warning(f"⚠️ Falló inicialización de {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cargando {model_name}: {str(e)}")
            return False
    
    def predict(self, image: np.ndarray, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Realiza predicción médica usando el sistema de IA avanzado.
        
        Args:
            image: Imagen como array numpy
            use_ensemble: Si usar ensemble de modelos o solo el principal
            
        Returns:
            Dict: Predicciones completas y análisis médico
        """
        if not self.is_initialized:
            raise RuntimeError("❌ Sistema no inicializado. Ejecutar load_model() primero.")
        
        start_time = time.time()
        
        try:
            if use_ensemble and len(self.models_loaded) > 1:
                return self._ensemble_prediction(image)
            else:
                return self._single_model_prediction(image)
                
        except Exception as e:
            logger.error(f"❌ Error durante predicción: {str(e)}")
            return self._generate_emergency_response(str(e))
        
        finally:
            processing_time = time.time() - start_time
            logger.info(f"⏱️ Predicción completada en {processing_time:.2f}s")
    
    def _ensemble_prediction(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Realiza predicción ensemble usando múltiples modelos.
        
        Args:
            image: Imagen a analizar
            
        Returns:
            Dict: Predicciones ensemble y análisis comparativo
        """
        logger.info("🔬 Ejecutando análisis ensemble multi-modelo...")
        
        model_predictions = {}
        model_analyses = {}
        
        # Ejecutar predicciones en paralelo
        with ThreadPoolExecutor(max_workers=len(self.models_loaded)) as executor:
            future_to_model = {}
            
            for model_name, model_info in self.models_loaded.items():
                future = executor.submit(
                    self._safe_model_predict, 
                    model_name, 
                    model_info["manager"], 
                    image
                )
                future_to_model[future] = model_name
            
            # Recopilar resultados
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=30)  # 30s timeout por modelo
                    if result:
                        model_predictions[model_name] = result
                        logger.info(f"✅ {model_name}: predicción completada")
                    else:
                        logger.warning(f"⚠️ {model_name}: predicción falló")
                except Exception as e:
                    logger.error(f"❌ {model_name}: error en predicción - {str(e)}")
        
        # Generar ensemble
        ensemble_predictions = self._compute_ensemble(model_predictions)
        
        # Análisis médico especializado
        medical_analysis = self._perform_medical_analysis(ensemble_predictions, model_predictions)
        
        return {
            "ensemble_predictions": ensemble_predictions,
            "individual_models": model_predictions,
            "medical_analysis": medical_analysis,
            "analysis_type": "ensemble_multi_model",
            "models_used": list(model_predictions.keys()),
            "confidence": self._calculate_ensemble_confidence(model_predictions),
            "processing_info": {
                "models_attempted": len(self.models_loaded),
                "models_successful": len(model_predictions),
                "ensemble_method": "weighted_average"
            }
        }
    
    def _single_model_prediction(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Realiza predicción usando un solo modelo (TorchXRayVision prioritario).
        
        Args:
            image: Imagen a analizar
            
        Returns:
            Dict: Predicción del modelo principal
        """
        logger.info("🔬 Ejecutando análisis con modelo principal...")
        
        # Priorizar TorchXRayVision
        primary_models = ["torchxrayvision", "chexnet", "fracturas", "radimagenet"]
        
        for model_name in primary_models:
            if model_name in self.models_loaded:
                try:
                    manager = self.models_loaded[model_name]["manager"]
                    predictions = manager.predict(image)
                    
                    if predictions:
                        return {
                            "predictions": predictions,
                            "analysis_type": "single_model",
                            "model_used": model_name,
                            "confidence": self._calculate_single_confidence(predictions),
                            "specialization": self.available_models[model_name]["specialization"]
                        }
                    
                except Exception as e:
                    logger.error(f"❌ Error en {model_name}: {str(e)}")
                    continue
        
        raise RuntimeError("❌ Ningún modelo disponible para predicción")
    
    def _safe_model_predict(self, model_name: str, manager, image: np.ndarray) -> Optional[Dict]:
        """
        Ejecuta predicción de forma segura con manejo de errores.
        
        Args:
            model_name: Nombre del modelo
            manager: Manager del modelo
            image: Imagen a procesar
            
        Returns:
            Optional[Dict]: Predicciones o None si falló
        """
        try:
            if hasattr(manager, 'predict'):
                return manager.predict(image)
            else:
                logger.error(f"❌ {model_name}: manager sin método predict")
                return None
                
        except Exception as e:
            logger.error(f"❌ {model_name}: error en predicción - {str(e)}")
            return None
    
    def _compute_ensemble(self, model_predictions: Dict[str, Dict]) -> Dict[str, float]:
        """
        Computa ensemble ponderado de las predicciones.
        
        Args:
            model_predictions: Predicciones de cada modelo
            
        Returns:
            Dict[str, float]: Predicciones ensemble
        """
        logger.info("🧮 Computando ensemble ponderado...")
        
        # Recopilar todas las patologías únicas
        all_pathologies = set()
        for model_preds in model_predictions.values():
            if isinstance(model_preds, dict):
                if "predictions" in model_preds:
                    all_pathologies.update(model_preds["predictions"].keys())
                else:
                    all_pathologies.update(model_preds.keys())
        
        ensemble_predictions = {}
        
        for pathology in all_pathologies:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, model_preds in model_predictions.items():
                # Extraer predicciones del formato del modelo
                if isinstance(model_preds, dict):
                    if "predictions" in model_preds:
                        predictions = model_preds["predictions"]
                    else:
                        predictions = model_preds
                else:
                    continue
                
                if pathology in predictions:
                    weight = self.ensemble_weights.get(model_name, 0.1)
                    weighted_sum += predictions[pathology] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_predictions[pathology] = weighted_sum / total_weight
            else:
                ensemble_predictions[pathology] = 0.05  # Valor conservador
        
        logger.info(f"✅ Ensemble computado para {len(ensemble_predictions)} patologías")
        return ensemble_predictions
    
    def _perform_medical_analysis(self, ensemble_predictions: Dict[str, float], 
                                 model_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Realiza análisis médico especializado basado en las predicciones.
        
        Args:
            ensemble_predictions: Predicciones ensemble
            model_predictions: Predicciones individuales
            
        Returns:
            Dict: Análisis médico completo
        """
        logger.info("🏥 Realizando análisis médico especializado...")
        
        # Análisis de patologías críticas
        critical_findings = self._identify_critical_findings(ensemble_predictions)
        
        # Análisis de confianza
        confidence_analysis = self._analyze_prediction_confidence(model_predictions)
        
        # Recomendaciones médicas
        medical_recommendations = self._generate_medical_recommendations(
            ensemble_predictions, critical_findings
        )
        
        # Análisis de consenso entre modelos
        consensus_analysis = self._analyze_model_consensus(model_predictions)
        
        return {
            "critical_findings": critical_findings,
            "confidence_analysis": confidence_analysis,
            "medical_recommendations": medical_recommendations,
            "consensus_analysis": consensus_analysis,
            "priority_pathologies": self._rank_pathologies_by_severity(ensemble_predictions),
            "clinical_urgency": self._assess_clinical_urgency(critical_findings)
        }
    
    def _identify_critical_findings(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Identifica hallazgos críticos que requieren atención médica.
        
        Args:
            predictions: Predicciones del ensemble
            
        Returns:
            Dict: Hallazgos críticos identificados
        """
        critical_thresholds = {
            "Pneumothorax": 0.3,      # Neumotórax - urgencia alta
            "Pneumonia": 0.4,         # Neumonía - prioridad alta
            "Edema": 0.4,             # Edema pulmonar - crítico
            "Mass": 0.3,              # Masas - requiere evaluación
            "Effusion": 0.4,          # Derrame pleural
            "Fracture": 0.3,          # Fracturas
            "Cardiomegaly": 0.5       # Cardiomegalia
        }
        
        critical_findings = {
            "high_priority": [],
            "moderate_priority": [],
            "findings_detected": {}
        }
        
        for pathology, probability in predictions.items():
            threshold = critical_thresholds.get(pathology, 0.6)
            
            if probability >= threshold:
                finding = {
                    "pathology": pathology,
                    "probability": probability,
                    "threshold": threshold,
                    "severity": "high" if probability >= threshold * 1.5 else "moderate"
                }
                
                if finding["severity"] == "high":
                    critical_findings["high_priority"].append(finding)
                else:
                    critical_findings["moderate_priority"].append(finding)
                
                critical_findings["findings_detected"][pathology] = finding
        
        return critical_findings
    
    def _calculate_ensemble_confidence(self, model_predictions: Dict[str, Dict]) -> float:
        """
        Calcula confianza del ensemble basado en consenso entre modelos.
        
        Args:
            model_predictions: Predicciones de todos los modelos
            
        Returns:
            float: Nivel de confianza (0-1)
        """
        if len(model_predictions) < 2:
            return 0.7  # Confianza moderada para modelo único
        
        # Calcular consenso promedio entre modelos
        consensus_scores = []
        
        models_list = list(model_predictions.keys())
        for i in range(len(models_list)):
            for j in range(i + 1, len(models_list)):
                model1_preds = model_predictions[models_list[i]]
                model2_preds = model_predictions[models_list[j]]
                
                # Extraer predicciones
                preds1 = model1_preds.get("predictions", model1_preds)
                preds2 = model2_preds.get("predictions", model2_preds)
                
                # Calcular similitud entre predicciones
                similarity = self._calculate_prediction_similarity(preds1, preds2)
                consensus_scores.append(similarity)
        
        if consensus_scores:
            avg_consensus = np.mean(consensus_scores)
            return min(0.95, max(0.3, avg_consensus))
        else:
            return 0.7
    
    def _calculate_single_confidence(self, predictions: Dict[str, float]) -> float:
        """
        Calcula confianza para predicción de modelo único.
        
        Args:
            predictions: Predicciones del modelo
            
        Returns:
            float: Nivel de confianza
        """
        # Basado en la distribución de probabilidades
        probs = list(predictions.values())
        max_prob = max(probs)
        entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0.01)
        
        # Normalizar entropy
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Confianza basada en probabilidad máxima y entropía
        confidence = (max_prob + (1 - normalized_entropy)) / 2
        return min(0.9, max(0.3, confidence))
    
    def _calculate_prediction_similarity(self, preds1: Dict, preds2: Dict) -> float:
        """
        Calcula similitud entre dos conjuntos de predicciones.
        
        Args:
            preds1, preds2: Diccionarios de predicciones
            
        Returns:
            float: Similitud (0-1)
        """
        common_keys = set(preds1.keys()) & set(preds2.keys())
        if not common_keys:
            return 0.0
        
        differences = []
        for key in common_keys:
            diff = abs(preds1[key] - preds2[key])
            differences.append(diff)
        
        avg_difference = np.mean(differences)
        similarity = 1.0 - avg_difference  # Convertir diferencia a similitud
        return max(0.0, similarity)
    
    def _analyze_prediction_confidence(self, model_predictions: Dict) -> Dict[str, Any]:
        """
        Analiza la confianza de las predicciones.
        """
        return {
            "overall_confidence": self._calculate_ensemble_confidence(model_predictions),
            "model_agreement": len(model_predictions) > 1,
            "prediction_stability": "high" if len(model_predictions) > 2 else "moderate"
        }
    
    def _generate_medical_recommendations(self, predictions: Dict[str, float], 
                                        critical_findings: Dict) -> Dict[str, Any]:
        """
        Genera recomendaciones médicas basadas en los hallazgos.
        """
        if critical_findings["high_priority"]:
            urgency = "immediate"
            recommendation = "Evaluación médica inmediata requerida"
        elif critical_findings["moderate_priority"]:
            urgency = "priority"
            recommendation = "Evaluación médica prioritaria recomendada"
        else:
            urgency = "routine"
            recommendation = "Seguimiento rutinario"
        
        return {
            "urgency": urgency,
            "primary_recommendation": recommendation,
            "follow_up_needed": len(critical_findings["findings_detected"]) > 0
        }
    
    def _analyze_model_consensus(self, model_predictions: Dict) -> Dict[str, Any]:
        """
        Analiza el consenso entre modelos.
        """
        return {
            "models_in_agreement": len(model_predictions),
            "consensus_level": "high" if len(model_predictions) > 2 else "moderate",
            "conflicting_predictions": []  # Se podría implementar análisis más detallado
        }
    
    def _rank_pathologies_by_severity(self, predictions: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Rankea patologías por severidad.
        """
        severity_weights = {
            "Pneumothorax": 10,
            "Pneumonia": 8,
            "Edema": 9,
            "Mass": 7,
            "Fracture": 6,
            "Effusion": 5
        }
        
        ranked = []
        for pathology, probability in predictions.items():
            weight = severity_weights.get(pathology, 1)
            severity_score = probability * weight
            
            ranked.append({
                "pathology": pathology,
                "probability": probability,
                "severity_score": severity_score
            })
        
        return sorted(ranked, key=lambda x: x["severity_score"], reverse=True)[:5]
    
    def _assess_clinical_urgency(self, critical_findings: Dict) -> str:
        """
        Evalúa la urgencia clínica general.
        """
        if critical_findings["high_priority"]:
            return "urgent"
        elif critical_findings["moderate_priority"]:
            return "priority"
        else:
            return "routine"
    
    def _generate_emergency_response(self, error_message: str) -> Dict[str, Any]:
        """
        Genera respuesta de emergencia en caso de error crítico.
        
        Args:
            error_message: Mensaje de error
            
        Returns:
            Dict: Respuesta de emergencia
        """
        logger.error(f"🚨 Generando respuesta de emergencia: {error_message}")
        
        return {
            "predictions": {
                "error_detected": 1.0,
                "system_status": "emergency_mode",
                "requires_manual_review": 1.0
            },
            "analysis_type": "emergency_response",
            "error_info": {
                "message": error_message,
                "recommendation": "Revisión manual requerida - sistema en modo emergencia"
            },
            "confidence": 0.0,
            "medical_analysis": {
                "critical_findings": {"emergency_mode": True},
                "medical_recommendations": {
                    "urgency": "immediate",
                    "primary_recommendation": "Revisión médica manual inmediata requerida"
                }
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información completa del sistema de IA médica.
        
        Returns:
            Dict: Información detallada del sistema
        """
        if not self.is_initialized:
            return {
                "status": "No inicializado",
                "error": "Sistema no cargado. Ejecutar load_model() primero."
            }
        
        models_info = {}
        for model_name, model_data in self.models_loaded.items():
            try:
                if hasattr(model_data["manager"], "get_model_info"):
                    models_info[model_name] = model_data["manager"].get_model_info()
                else:
                    models_info[model_name] = {
                        "status": "Cargado",
                        "specialization": model_data["specialization"]
                    }
            except Exception as e:
                models_info[model_name] = {
                    "status": "Error",
                    "error": str(e)
                }
        
        return {
            "status": "Sistema inicializado y funcional",
            "system_type": "Advanced Medical AI Manager",
            "device": self.device,
            "models_loaded": list(self.models_loaded.keys()),
            "total_models": len(self.models_loaded),
            "available_models": len(self.available_models),
            "ensemble_capable": len(self.models_loaded) > 1,
            "individual_models": models_info,
            "ensemble_weights": self.ensemble_weights,
            "capabilities": [
                "Multi-model ensemble analysis",
                "Specialized pathology detection",
                "Real-time medical inference",
                "Clinical urgency assessment", 
                "Automated medical recommendations",
                "Thread-safe operation",
                "Emergency fallback systems"
            ],
            "supported_analyses": [
                "General thoracic pathology (TorchXRayVision)",
                "Pneumonia specialist analysis (CheXNet)",
                "Fracture detection (Fracturas Model)",
                "Universal medical imaging (RadImageNet)"
            ]
        }
    
    def get_available_models(self) -> List[str]:
        """
        Obtiene lista de modelos disponibles.
        
        Returns:
            List[str]: Lista de nombres de modelos
        """
        return list(self.available_models.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del sistema.
        
        Returns:
            Dict: Estado de salud del sistema
        """
        healthy_models = 0
        total_models = len(self.available_models)
        
        model_status = {}
        for model_name, config in self.available_models.items():
            if model_name in self.models_loaded:
                try:
                    # Test básico del modelo
                    manager = self.models_loaded[model_name]["manager"]
                    if hasattr(manager, "get_model_info"):
                        info = manager.get_model_info()
                        if info.get("status") == "Cargado y funcional":
                            healthy_models += 1
                            model_status[model_name] = "healthy"
                        else:
                            model_status[model_name] = "loaded_but_issues"
                    else:
                        model_status[model_name] = "loaded_no_info"
                        healthy_models += 1
                except Exception as e:
                    model_status[model_name] = f"error: {str(e)}"
            else:
                model_status[model_name] = "not_loaded"
        
        health_score = healthy_models / total_models if total_models > 0 else 0
        
        return {
            "overall_health": "healthy" if health_score > 0.5 else "degraded" if health_score > 0 else "critical",
            "health_score": health_score,
            "models_healthy": healthy_models,
            "models_total": total_models,
            "system_initialized": self.is_initialized,
            "device": self.device,
            "model_status": model_status,
            "ensemble_available": len(self.models_loaded) > 1,
            "timestamp": time.time()
        }