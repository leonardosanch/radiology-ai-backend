from typing import Dict, Any, List
import logging
from datetime import datetime
import uuid

# Configurar logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generador de reportes médicos para análisis radiológico usando información dinámica del modelo.
    
    Esta clase genera reportes médicos completos basados en predicciones de IA
    y utiliza únicamente la información real proporcionada por TorchXRayVision.
    """
    
    def __init__(self):
        """
        Inicializa el generador de reportes médicos.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("ReportGenerator inicializado - Usando información dinámica del modelo")
    
    def generate_full_report(self, predictions: Dict[str, float], image_info: Dict, model_info: Dict = None) -> Dict[str, Any]:
        """
        Genera un reporte médico completo usando información dinámica del modelo.
        
        Args:
            predictions: Predicciones del modelo {patología: confianza}
            image_info: Información de la imagen procesada
            model_info: Información del modelo de IA utilizado (datos reales de TorchXRayVision)
        
        Returns:
            Dict: Reporte médico completo estructurado
        """
        try:
            self.logger.info("Generando reporte médico completo con información dinámica")
            
            # Obtener información del modelo (usar datos reales o valores por defecto)
            model_name = "Unknown AI Model"
            model_architecture = "Unknown Architecture"
            pathologies_evaluated = []
            validation_status = "Unknown"
            
            if model_info:
                model_name = model_info.get("model_type", "Unknown AI Model")
                model_architecture = model_info.get("model_architecture", "Unknown Architecture")
                pathologies_evaluated = model_info.get("pathologies_supported", [])
                validation_status = model_info.get("validation_status", "Unknown")
            
            # Generar timestamp único
            timestamp = datetime.now().isoformat()
            report_id = str(uuid.uuid4())
            
            # Clasificar hallazgos por nivel de confianza
            findings_by_confidence = self._classify_findings_by_confidence(predictions)
            
            # Generar interpretación médica
            medical_interpretation = self._generate_medical_interpretation(
                findings_by_confidence, model_info
            )
            
            # Estructurar reporte completo
            medical_report = {
                # Información del estudio
                "study_info": {
                    "report_id": report_id,
                    "timestamp": timestamp,
                    "study_type": "Chest X-Ray Analysis",
                    "modality": "Digital Radiography",
                    "view": "Chest PA/AP (estimated)"
                },
                
                # Información técnica del análisis (usando datos reales)
                "analysis_details": {
                    "ai_model_used": model_name,
                    "model_architecture": model_architecture,
                    "pathologies_evaluated": len(pathologies_evaluated),
                    "supported_pathologies": pathologies_evaluated,
                    "validation_status": validation_status,
                    "image_quality": image_info.get("estimated_quality", "unknown"),
                    "processing_notes": self._get_processing_notes(image_info)
                },
                
                # Hallazgos principales
                "primary_findings": {
                    "high_confidence": findings_by_confidence["high"],
                    "moderate_confidence": findings_by_confidence["moderate"],
                    "low_confidence": findings_by_confidence["low"],
                    "total_findings": len([f for f in findings_by_confidence.values() for f in f])
                },
                
                # Interpretación médica
                "medical_interpretation": medical_interpretation,
                
                # Análisis detallado por patología
                "detailed_analysis": self._generate_detailed_pathology_analysis(predictions, model_info),
                
                # Recomendaciones clínicas
                "clinical_recommendations": self._generate_clinical_recommendations(
                    findings_by_confidence, model_info
                ),
                
                # Limitaciones y consideraciones
                "limitations_and_notes": {
                    "ai_limitations": [
                        "Los resultados de IA requieren validación por radiólogo certificado",
                        "La interpretación debe considerar el contexto clínico del paciente",
                        "La calidad de la imagen puede afectar la precisión del análisis"
                    ],
                    "model_specific_notes": self._get_model_specific_notes(model_info),
                    "quality_indicators": {
                        "image_quality": image_info.get("estimated_quality", "unknown"),
                        "processing_quality": image_info.get("processing_quality", "unknown"),
                        "confidence_calibration": model_info.get("confidence_calibration", "unknown") if model_info else "unknown"
                    }
                },
                
                # Métricas de confianza
                "confidence_metrics": {
                    "overall_confidence": self._calculate_overall_confidence(predictions),
                    "highest_confidence_finding": self._get_highest_confidence_finding(predictions),
                    "confidence_distribution": self._analyze_confidence_distribution(predictions)
                }
            }
            
            self.logger.info(f"Reporte médico generado exitosamente - ID: {report_id}")
            return medical_report
            
        except Exception as e:
            self.logger.error(f"Error generando reporte médico: {str(e)}")
            raise
    
    def _classify_findings_by_confidence(self, predictions: Dict[str, float]) -> Dict[str, List[Dict]]:
        """
        Clasifica los hallazgos por nivel de confianza usando umbrales dinámicos.
        
        Args:
            predictions: Predicciones del modelo
            
        Returns:
            Dict: Hallazgos clasificados por confianza
        """
        # Umbrales de confianza (estos podrían venir de settings)
        HIGH_THRESHOLD = 0.7
        MODERATE_THRESHOLD = 0.3
        
        findings = {
            "high": [],
            "moderate": [],
            "low": []
        }
        
        for pathology, confidence in predictions.items():
            finding = {
                "pathology": pathology,
                "confidence": round(confidence, 3),
                "confidence_percentage": f"{round(confidence * 100, 1)}%",
                "clinical_significance": self._get_clinical_significance(pathology, confidence)
            }
            
            if confidence >= HIGH_THRESHOLD:
                findings["high"].append(finding)
            elif confidence >= MODERATE_THRESHOLD:
                findings["moderate"].append(finding)
            else:
                findings["low"].append(finding)
        
        # Ordenar por confianza descendente
        for category in findings.values():
            category.sort(key=lambda x: x["confidence"], reverse=True)
        
        return findings
    
    def _generate_medical_interpretation(self, findings_by_confidence: Dict, model_info: Dict = None) -> Dict[str, Any]:
        """
        Genera interpretación médica basada en los hallazgos y la información del modelo.
        
        Args:
            findings_by_confidence: Hallazgos clasificados por confianza
            model_info: Información del modelo utilizado
            
        Returns:
            Dict: Interpretación médica estructurada
        """
        high_findings = findings_by_confidence["high"]
        moderate_findings = findings_by_confidence["moderate"]
        
        # Determinar impresión general
        if high_findings:
            impression = "Hallazgos significativos detectados que requieren atención médica"
            urgency = "Alta prioridad"
        elif moderate_findings:
            impression = "Hallazgos moderados detectados que requieren evaluación médica"
            urgency = "Prioridad moderada"
        else:
            impression = "No se detectaron hallazgos significativos en el análisis de IA"
            urgency = "Prioridad rutinaria"
        
        # Generar resumen de hallazgos principales
        main_findings_summary = []
        for finding in high_findings[:3]:  # Top 3 hallazgos de alta confianza
            main_findings_summary.append(
                f"{finding['pathology']}: {finding['confidence_percentage']} de confianza"
            )
        
        # Obtener nombre del modelo para la interpretación
        model_name = "AI Model"
        if model_info:
            model_name = model_info.get("model_type", "AI Model")
        
        return {
            "overall_impression": impression,
            "clinical_urgency": urgency,
            "main_findings_summary": main_findings_summary,
            "analysis_method": f"Análisis automatizado con {model_name}",
            "recommendation_summary": self._get_recommendation_summary(high_findings, moderate_findings),
            "follow_up_required": len(high_findings) > 0 or len(moderate_findings) > 2
        }
    
    def _generate_detailed_pathology_analysis(self, predictions: Dict[str, float], model_info: Dict = None) -> List[Dict]:
        """
        Genera análisis detallado por patología usando información real del modelo.
        
        Args:
            predictions: Predicciones del modelo
            model_info: Información del modelo
            
        Returns:
            List: Análisis detallado por patología
        """
        detailed_analysis = []
        
        # Obtener patologías soportadas del modelo si está disponible
        supported_pathologies = []
        if model_info:
            supported_pathologies = model_info.get("pathologies_supported", [])
        
        for pathology, confidence in predictions.items():
            # Verificar si la patología está en la lista de soportadas
            is_supported = pathology in supported_pathologies if supported_pathologies else True
            
            analysis = {
                "pathology_name": pathology,
                "confidence_score": round(confidence, 3),
                "confidence_level": self._get_confidence_level_text(confidence),
                "clinical_description": self._get_pathology_description(pathology),
                "typical_presentation": self._get_typical_presentation(pathology),
                "recommended_action": self._get_recommended_action(pathology, confidence),
                "model_support_status": "Directly supported" if is_supported else "Inferred"
            }
            detailed_analysis.append(analysis)
        
        # Ordenar por confianza descendente
        detailed_analysis.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        return detailed_analysis
    
    def _generate_clinical_recommendations(self, findings_by_confidence: Dict, model_info: Dict = None) -> Dict[str, Any]:
        """
        Genera recomendaciones clínicas basadas en hallazgos y capacidades del modelo.
        
        Args:
            findings_by_confidence: Hallazgos clasificados
            model_info: Información del modelo
            
        Returns:
            Dict: Recomendaciones clínicas estructuradas
        """
        high_findings = findings_by_confidence["high"]
        moderate_findings = findings_by_confidence["moderate"]
        
        # Recomendaciones inmediatas
        immediate_actions = []
        if high_findings:
            immediate_actions.append("Revisión por radiólogo certificado requerida")
            immediate_actions.append("Correlación con historia clínica y examen físico")
            
            # Recomendaciones específicas para hallazgos de alta confianza
            for finding in high_findings:
                pathology = finding["pathology"]
                if pathology.lower() in ["pneumothorax", "pneumonia"]:
                    immediate_actions.append(f"Evaluación urgente para {pathology}")
        
        # Recomendaciones de seguimiento
        follow_up_actions = []
        if moderate_findings:
            follow_up_actions.append("Seguimiento clínico recomendado")
            follow_up_actions.append("Considerar estudios complementarios si clínicamente indicado")
        
        # Obtener información del modelo para contexto
        model_context = "Análisis realizado con modelo de IA"
        if model_info:
            model_name = model_info.get("model_type", "Unknown")
            validation_status = model_info.get("validation_status", "Unknown")
            model_context = f"Análisis realizado con {model_name} ({validation_status})"
        
        return {
            "immediate_actions": immediate_actions,
            "follow_up_actions": follow_up_actions,
            "general_recommendations": [
                "Los resultados de IA deben ser interpretados por profesional médico calificado",
                "Considerar el contexto clínico del paciente en la interpretación",
                "Validar hallazgos significativos con métodos diagnósticos adicionales si es necesario"
            ],
            "model_context": model_context,
            "quality_assurance": "Reporte generado automáticamente - Requiere validación médica"
        }
    
    def _get_model_specific_notes(self, model_info: Dict = None) -> List[str]:
        """
        Obtiene notas específicas del modelo basadas en la información real.
        
        Args:
            model_info: Información del modelo
            
        Returns:
            List: Notas específicas del modelo
        """
        notes = []
        
        if model_info:
            # Agregar notas basadas en la información real del modelo
            model_type = model_info.get("model_type", "")
            architecture = model_info.get("model_architecture", "")
            validation_status = model_info.get("validation_status", "")
            capabilities = model_info.get("capabilities", [])
            
            notes.append(f"Modelo utilizado: {model_type}")
            
            if architecture:
                notes.append(f"Arquitectura: {architecture}")
            
            if validation_status:
                notes.append(f"Estado de validación: {validation_status}")
            
            # Agregar información sobre capacidades si está disponible
            if capabilities:
                notes.append("Capacidades del modelo:")
                for capability in capabilities[:3]:  # Limitar a 3 principales
                    notes.append(f"  - {capability}")
            
            # Información sobre patologías soportadas
            num_pathologies = model_info.get("num_pathologies", 0)
            if num_pathologies > 0:
                notes.append(f"Evalúa {num_pathologies} patologías diferentes")
        else:
            notes.append("Información del modelo no disponible")
        
        return notes
    
    def _calculate_overall_confidence(self, predictions: Dict[str, float]) -> float:
        """
        Calcula la confianza general del análisis.
        
        Args:
            predictions: Predicciones del modelo
            
        Returns:
            float: Confianza general
        """
        if not predictions:
            return 0.0
        
        # Calcular promedio ponderado dando más peso a las confianzas altas
        total_weighted = sum(conf * conf for conf in predictions.values())
        total_weights = sum(conf for conf in predictions.values())
        
        if total_weights == 0:
            return 0.0
        
        return round(total_weighted / total_weights, 3)
    
    def _get_highest_confidence_finding(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Obtiene el hallazgo con mayor confianza.
        
        Args:
            predictions: Predicciones del modelo
            
        Returns:
            Dict: Hallazgo con mayor confianza
        """
        if not predictions:
            return {"pathology": "None", "confidence": 0.0}
        
        max_pathology = max(predictions.items(), key=lambda x: x[1])
        
        return {
            "pathology": max_pathology[0],
            "confidence": round(max_pathology[1], 3),
            "confidence_percentage": f"{round(max_pathology[1] * 100, 1)}%"
        }
    
    def _analyze_confidence_distribution(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Analiza la distribución de confianzas.
        
        Args:
            predictions: Predicciones del modelo
            
        Returns:
            Dict: Análisis de distribución de confianzas
        """
        if not predictions:
            return {"high": 0, "moderate": 0, "low": 0}
        
        confidences = list(predictions.values())
        
        high_count = sum(1 for c in confidences if c >= 0.7)
        moderate_count = sum(1 for c in confidences if 0.3 <= c < 0.7)
        low_count = sum(1 for c in confidences if c < 0.3)
        
        return {
            "high_confidence_findings": high_count,
            "moderate_confidence_findings": moderate_count,
            "low_confidence_findings": low_count,
            "average_confidence": round(sum(confidences) / len(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "min_confidence": round(min(confidences), 3)
        }
    
    def _get_clinical_significance(self, pathology: str, confidence: float) -> str:
        """
        Determina la significancia clínica de un hallazgo.
        
        Args:
            pathology: Nombre de la patología
            confidence: Nivel de confianza
            
        Returns:
            str: Significancia clínica
        """
        if confidence >= 0.7:
            if pathology.lower() in ["pneumothorax", "pneumonia", "mass"]:
                return "Altamente significativo - Requiere atención inmediata"
            else:
                return "Significativo - Requiere evaluación médica"
        elif confidence >= 0.3:
            return "Moderadamente significativo - Seguimiento recomendado"
        else:
            return "Baja significancia - Monitoreo rutinario"
    
    def _get_confidence_level_text(self, confidence: float) -> str:
        """
        Convierte nivel de confianza numérico a texto descriptivo.
        
        Args:
            confidence: Nivel de confianza numérico
            
        Returns:
            str: Descripción textual del nivel de confianza
        """
        if confidence >= 0.7:
            return "Alta confianza"
        elif confidence >= 0.3:
            return "Confianza moderada"
        else:
            return "Baja confianza"
    
    def _get_pathology_description(self, pathology: str) -> str:
        """
        Obtiene descripción clínica de una patología.
        
        Args:
            pathology: Nombre de la patología
            
        Returns:
            str: Descripción clínica
        """
        descriptions = {
            "Atelectasis": "Colapso parcial o completo del pulmón o lóbulo pulmonar",
            "Cardiomegaly": "Agrandamiento del corazón visible en radiografía",
            "Effusion": "Acumulación de líquido en el espacio pleural",
            "Infiltration": "Presencia de material inflamatorio en el tejido pulmonar",
            "Mass": "Lesión sólida localizada en el pulmón",
            "Nodule": "Pequeña lesión redondeada en el pulmón",
            "Pneumonia": "Infección e inflamación del tejido pulmonar",
            "Pneumothorax": "Presencia de aire en el espacio pleural",
            "Consolidation": "Llenado de los alvéolos con material inflamatorio",
            "Edema": "Acumulación de líquido en los pulmones",
            "Emphysema": "Destrucción de las paredes alveolares",
            "Fibrosis": "Cicatrización y engrosamiento del tejido pulmonar",
            "Pleural_Thickening": "Engrosamiento de la pleura",
            "Hernia": "Protrusión de órganos abdominales hacia el tórax"
        }
        
        return descriptions.get(pathology, f"Hallazgo radiológico: {pathology}")
    
    def _get_typical_presentation(self, pathology: str) -> str:
        """
        Obtiene la presentación típica de una patología en radiografía.
        
        Args:
            pathology: Nombre de la patología
            
        Returns:
            str: Presentación radiológica típica
        """
        presentations = {
            "Atelectasis": "Opacidad lineal o triangular con pérdida de volumen",
            "Cardiomegaly": "Silueta cardíaca > 50% del diámetro torácico en PA",
            "Effusion": "Opacidad homogénea en base pulmonar con menisco",
            "Infiltration": "Opacidades heterogéneas difusas o localizadas",
            "Mass": "Lesión redondeada > 3cm con bordes definidos",
            "Nodule": "Lesión redondeada ≤ 3cm bien circunscrita",
            "Pneumonia": "Consolidación lobar o bronconeumonía",
            "Pneumothorax": "Hiperlucencia periférica sin trama vascular",
            "Consolidation": "Opacidad homogénea con broncograma aéreo",
            "Edema": "Opacidades difusas bilaterales simétricas",
            "Emphysema": "Hiperlucencia difusa con trama vascular disminuida",
            "Fibrosis": "Opacidades reticulares o reticulonodulares",
            "Pleural_Thickening": "Engrosamiento pleural localizado o difuso",
            "Hernia": "Estructura abdominal visible en hemitórax"
        }
        
        return presentations.get(pathology, f"Alteración radiológica compatible con {pathology}")
    
    def _get_recommended_action(self, pathology: str, confidence: float) -> str:
        """
        Determina la acción recomendada basada en patología y confianza.
        
        Args:
            pathology: Nombre de la patología
            confidence: Nivel de confianza
            
        Returns:
            str: Acción recomendada
        """
        if confidence >= 0.7:
            urgent_pathologies = ["pneumothorax", "pneumonia", "mass"]
            if pathology.lower() in urgent_pathologies:
                return "Evaluación médica urgente requerida"
            else:
                return "Correlación clínica y evaluación por especialista"
        elif confidence >= 0.3:
            return "Seguimiento clínico y consideración de estudios adicionales"
        else:
            return "Monitoreo rutinario, repetir estudio si indicado clínicamente"
    
    def _get_recommendation_summary(self, high_findings: List, moderate_findings: List) -> str:
        """
        Genera resumen de recomendaciones basado en hallazgos.
        
        Args:
            high_findings: Hallazgos de alta confianza
            moderate_findings: Hallazgos de confianza moderada
            
        Returns:
            str: Resumen de recomendaciones
        """
        if high_findings:
            return "Se requiere evaluación médica prioritaria para validar hallazgos significativos"
        elif moderate_findings:
            return "Se recomienda seguimiento clínico y correlación con síntomas"
        else:
            return "Continuar con cuidado médico rutinario según indicación clínica"
    
    def _get_processing_notes(self, image_info: Dict) -> List[str]:
        """
        Genera notas sobre el procesamiento de la imagen.
        
        Args:
            image_info: Información de la imagen procesada
            
        Returns:
            List: Notas de procesamiento
        """
        notes = []
        
        quality = image_info.get("estimated_quality", "unknown")
        notes.append(f"Calidad de imagen estimada: {quality}")
        
        if "shape" in image_info:
            shape = image_info["shape"]
            notes.append(f"Resolución procesada: {shape[0]}x{shape[1]}")
        
        if "processing_quality" in image_info:
            notes.append(f"Calidad de procesamiento: {image_info['processing_quality']}")
        
        return notes