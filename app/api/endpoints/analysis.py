from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
import traceback
import time
import asyncio
from pathlib import Path
import uuid

# Importar componentes de la aplicación
from ...utils.validators import validate_upload_file, get_file_info
from ...services.image_processor import ImageProcessor
from ...services.report_generator import ReportGenerator
from ...models.torax_model import AIModelManager
from ...core.config import settings

# Configurar logging para este módulo
logger = logging.getLogger(__name__)

# Crear router para endpoints de análisis radiológico
router = APIRouter(prefix="/analysis", tags=["Análisis Radiológico"])

# Instancia global del gestor de modelos de IA
model_manager: Optional[AIModelManager] = None

def get_model_manager() -> AIModelManager:
    """
    Dependency injection para obtener el gestor de modelos de IA.
    Inicializa el modelo la primera vez que se solicita.
    
    Returns:
        AIModelManager: Instancia del gestor de modelos
        
    Raises:
        HTTPException: Si no se puede inicializar el modelo
    """
    global model_manager
    
    if model_manager is None:
        try:
            logger.info("Inicializando gestor de modelos de IA...")
            
            # Crear instancia del gestor con configuración
            model_manager = AIModelManager(
                model_path=settings.model_path,
                device=settings.device
            )
            
            # Cargar el modelo (TorchXRayVision)
            success = model_manager.load_model(settings.model_name)
            
            if not success:
                logger.error("No se pudo cargar el modelo de IA")
                raise HTTPException(
                    status_code=503,
                    detail="Servicio de IA no disponible. El modelo no se pudo cargar."
                )
            
            logger.info("Gestor de modelos inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error crítico inicializando modelo: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Error inicializando servicio de IA: {str(e)}"
            )
    
    return model_manager

async def cleanup_temp_files(file_paths: list):
    """
    Tarea en background para limpiar archivos temporales.
    
    Args:
        file_paths: Lista de rutas de archivos a eliminar
    """
    try:
        await asyncio.sleep(300)  # Esperar 5 minutos antes de limpiar
        
        for file_path in file_paths:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.debug(f"Archivo temporal eliminado: {file_path}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal {file_path}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error en limpieza de archivos temporales: {str(e)}")

@router.post("/upload", 
             summary="Analizar Radiografía de Tórax",
             description="Sube una radiografía de tórax para análisis automático con IA")
async def analyze_radiography(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Archivo de radiografía (JPG, PNG, DICOM)"),
    model_mgr: AIModelManager = Depends(get_model_manager)
) -> JSONResponse:
    """
    Endpoint principal para análisis automático de radiografías de tórax.
    
    Este endpoint utiliza TorchXRayVision para analizar radiografías
    y generar reportes médicos detallados con las patologías detectadas.
    
    Args:
        background_tasks: Tareas en background de FastAPI
        file: Archivo de imagen radiográfica subido
        model_mgr: Gestor del modelo de IA (inyectado automáticamente)
    
    Returns:
        JSONResponse: Análisis completo con diagnóstico médico
        
    Raises:
        HTTPException: Si hay errores en validación, procesamiento o análisis
    """
    analysis_id = str(uuid.uuid4())  # ID único para rastrear análisis
    start_time = time.time()
    
    logger.info(f"[{analysis_id}] Iniciando análisis de radiografía: {file.filename}")
    
    try:
        # =============================================================================
        # FASE 1: VALIDACIÓN DEL ARCHIVO
        # =============================================================================
        logger.info(f"[{analysis_id}] Fase 1: Validando archivo...")
        
        # Obtener información del archivo para logging
        file_info = get_file_info(file)
        logger.info(f"[{analysis_id}] Archivo: {file_info}")
        
        # Validar archivo subido (formato, tamaño, tipo, calidad médica)
        validate_upload_file(file)
        
        validation_time = time.time() - start_time
        logger.info(f"[{analysis_id}] Validación completada en {validation_time:.2f}s")
        
        # =============================================================================
        # FASE 2: PROCESAMIENTO DE IMAGEN
        # =============================================================================
        logger.info(f"[{analysis_id}] Fase 2: Procesando imagen médica...")
        
        # Inicializar procesador de imágenes médicas
        processor = ImageProcessor()
        
        # Cargar imagen desde upload con soporte para DICOM y formatos estándar
        image_array = processor.load_image_from_upload(file)
        logger.info(f"[{analysis_id}] Imagen cargada - Shape: {image_array.shape}, Dtype: {image_array.dtype}")
        
        # Obtener información detallada de la imagen para el reporte
        image_info = processor.get_image_info(image_array)
        logger.info(f"[{analysis_id}] Calidad de imagen: {image_info.get('estimated_quality', 'unknown')}")
        
        # Mejorar contraste específicamente para radiografías médicas
        enhanced_image = processor.enhance_medical_contrast(image_array)
        logger.debug(f"[{analysis_id}] Contraste médico mejorado")
        
        # Preprocesar imagen para el modelo de IA (normalización, redimensionado, denoising)
        processed_image = processor.preprocess_for_model(enhanced_image)
        logger.info(f"[{analysis_id}] Imagen preprocesada para modelo de IA")
        
        processing_time = time.time() - validation_time - start_time
        logger.info(f"[{analysis_id}] Procesamiento completado en {processing_time:.2f}s")
        
        # =============================================================================
        # FASE 3: ANÁLISIS CON INTELIGENCIA ARTIFICIAL
        # =============================================================================
        logger.info(f"[{analysis_id}] Fase 3: Ejecutando análisis de IA...")
        
        # Obtener información del modelo ANTES del análisis
        model_info = model_mgr.get_model_info()
        
        # Realizar predicción con TorchXRayVision
        ai_start_time = time.time()
        predictions = model_mgr.predict(processed_image)
        ai_time = time.time() - ai_start_time
        
        logger.info(f"[{analysis_id}] Análisis de IA completado en {ai_time:.2f}s")
        logger.info(f"[{analysis_id}] Predicciones generadas para {len(predictions)} patologías")
        
        # Log de hallazgos significativos
        significant_findings = [
            pathology for pathology, confidence in predictions.items() 
            if confidence > settings.confidence_threshold_moderate
        ]
        if significant_findings:
            logger.info(f"[{analysis_id}] Hallazgos significativos: {', '.join(significant_findings)}")
        else:
            logger.info(f"[{analysis_id}] No se detectaron hallazgos significativos")
        
        # =============================================================================
        # FASE 4: GENERACIÓN DE REPORTE MÉDICO
        # =============================================================================
        logger.info(f"[{analysis_id}] Fase 4: Generando reporte médico...")
        
        # Generar reporte médico profesional completo CON model_info
        report_generator = ReportGenerator()
        medical_report = report_generator.generate_full_report(predictions, image_info, model_info)
        
        logger.info(f"[{analysis_id}] Reporte médico generado")
        
        # =============================================================================
        # FASE 5: PREPARAR RESPUESTA COMPLETA
        # =============================================================================
        total_time = time.time() - start_time
        
        # Estructurar respuesta completa usando información dinámica del modelo
        response_data = {
            # Información del análisis
            "analysis_id": analysis_id,
            "status": "success",
            "message": "Análisis radiológico completado exitosamente",
            
            # Información del archivo procesado
            "file_info": {
                "original_filename": file.filename,
                "file_size_mb": file_info.get("size_mb", 0),
                "file_type": file_info.get("extension", "unknown"),
                "content_type": file.content_type
            },
            
            # Reporte médico completo
            "medical_analysis": medical_report,
            
            # Información técnica del modelo (dinámica)
            "model_information": {
                "ai_model": model_info.get("model_type", "Unknown"),
                "model_architecture": model_info.get("model_architecture", "Unknown"),
                "device_used": model_info.get("device", "Unknown"),
                "pathologies_evaluated": model_info.get("num_pathologies", len(predictions)),
                "analysis_confidence": "Real AI Analysis" if model_info.get("status") == "Cargado y funcional" else "Fallback Mode",
                "validation_status": model_info.get("validation_status", "Unknown")
            },
            
            # Métricas de rendimiento
            "performance_metrics": {
                "total_processing_time_seconds": round(total_time, 2),
                "validation_time_seconds": round(validation_time, 2),
                "image_processing_time_seconds": round(processing_time, 2),
                "ai_inference_time_seconds": round(ai_time, 2),
                "report_generation_time_seconds": round(total_time - ai_time - processing_time - validation_time, 2)
            },
            
            # Metadatos adicionales
            "metadata": {
                "analysis_timestamp": medical_report["study_info"]["timestamp"],
                "system_version": "Radiology AI Backend v1.0",
                "api_version": "v1",
                "processing_quality": image_info.get("estimated_quality", "unknown")
            }
        }
        
        # Programar limpieza de archivos temporales
        temp_files = []  # Agregar rutas de archivos temporales si se crean
        if temp_files:
            background_tasks.add_task(cleanup_temp_files, temp_files)
        
        logger.info(f"[{analysis_id}] Análisis completo exitoso en {total_time:.2f}s")
        
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException as e:
        # Re-lanzar HTTPExceptions con el ID de análisis
        logger.error(f"[{analysis_id}] Error HTTP: {e.detail}")
        raise HTTPException(
            status_code=e.status_code,
            detail=f"[{analysis_id}] {e.detail}"
        )
        
    except Exception as e:
        # Manejar errores inesperados
        error_trace = traceback.format_exc()
        logger.error(f"[{analysis_id}] Error crítico durante análisis:")
        logger.error(error_trace)
        
        raise HTTPException(
            status_code=500,
            detail=f"[{analysis_id}] Error interno durante el análisis: {str(e)}"
        )

@router.get("/health",
           summary="Estado del Servicio",
           description="Verifica el estado de salud del servicio de análisis radiológico")
async def health_check(
    model_mgr: AIModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Endpoint de verificación de salud del servicio de análisis radiológico.
    
    Verifica que todos los componentes estén funcionando correctamente:
    - Modelo de IA cargado y operativo
    - Procesamiento de imágenes funcional
    - Generación de reportes activa
    
    Args:
        model_mgr: Gestor del modelo de IA
        
    Returns:
        Dict: Estado detallado del servicio
    """
    try:
        start_time = time.time()
        
        # Verificar estado del modelo de IA
        model_info = model_mgr.get_model_info()
        
        # Verificar configuración del sistema
        system_info = settings.get_system_info()
        
        # Verificar directorios necesarios
        directories_status = {
            "models": Path(settings.model_path).exists(),
            "uploads": Path(settings.upload_dir).exists(),
            "temp": Path(settings.temp_dir).exists(),
            "logs": Path(settings.logs_dir).exists()
        }
        
        # Calcular tiempo de respuesta
        response_time = time.time() - start_time
        
        # Determinar estado general
        is_healthy = (
            model_info.get("status") == "Cargado y funcional" and
            all(directories_status.values())
        )
        
        health_status = {
            "service_status": "healthy" if is_healthy else "degraded",
            "timestamp": time.time(),
            "response_time_ms": round(response_time * 1000, 2),
            
            # Estado del servicio
            "service_info": {
                "name": "Radiology AI Analysis Service",
                "version": "1.0.0",
                "mode": "development" if settings.debug else "production",
                "uptime_check": "operational"
            },
            
            # Estado del modelo de IA (información dinámica)
            "ai_model_status": {
                "status": model_info.get("status", "unknown"),
                "model_type": model_info.get("model_type", "unknown"),
                "model_architecture": model_info.get("model_architecture", "unknown"),
                "device": model_info.get("device", "unknown"),
                "pathologies_supported": model_info.get("num_pathologies", 0),
                "capabilities": model_info.get("capabilities", []),
                "validation_status": model_info.get("validation_status", "unknown")
            },
            
            # Configuración del sistema
            "system_configuration": {
                "max_file_size_mb": settings.max_file_size / (1024 * 1024),
                "allowed_extensions": settings.allowed_extensions,
                "confidence_thresholds": {
                    "low": settings.confidence_threshold_low,
                    "moderate": settings.confidence_threshold_moderate,
                    "high": settings.confidence_threshold_high
                },
                "processing_timeout_seconds": settings.model_inference_timeout
            },
            
            # Estado de directorios
            "directories_status": directories_status,
            
            # Información del sistema
            "system_info": system_info,
            
            # Límites operacionales
            "operational_limits": {
                "max_concurrent_requests": settings.max_concurrent_requests,
                "request_timeout_seconds": settings.request_timeout,
                "inference_timeout_seconds": settings.model_inference_timeout
            }
        }
        
        logger.info(f"Health check completado - Estado: {health_status['service_status']}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Error verificando estado del servicio: {str(e)}"
        )

@router.get("/model/info",
           summary="Información del Modelo de IA",
           description="Obtiene información detallada del modelo de IA cargado")
async def get_model_information(
    model_mgr: AIModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Obtiene información detallada del modelo de IA radiológica.
    
    Incluye especificaciones técnicas, capacidades diagnósticas y
    configuración del modelo actual.
    
    Args:
        model_mgr: Gestor del modelo de IA
        
    Returns:
        Dict: Información completa del modelo
    """
    try:
        logger.info("Consultando información del modelo de IA")
        
        # Obtener información base del modelo
        model_info = model_mgr.get_model_info()
        
        # Obtener configuración específica del modelo
        model_config = settings.get_model_config()
        
        # Estructurar información detallada usando datos reales
        detailed_info = {
            # Información principal del modelo (dinámica)
            "model_details": {
                "name": model_config.get("model_name", settings.model_name),
                "type": model_info.get("model_type", "Unknown"),
                "architecture": model_info.get("model_architecture", "Unknown"),
                "status": model_info.get("status", "Unknown"),
                "device": model_info.get("device", "Unknown"),
                "model_weights": model_info.get("model_weights", "Unknown")
            },
            
            # Capacidades diagnósticas (usando datos reales)
            "diagnostic_capabilities": {
                "pathologies_detected": model_info.get("pathologies_supported", []),
                "total_pathologies": model_info.get("num_pathologies", 0),
                "mapped_pathologies": model_info.get("mapped_pathologies", 0),
                "direct_mappings": model_info.get("direct_mappings", []),
                "specialization": "Chest X-ray Analysis",
                "medical_focus": "Thoracic Radiology"
            },
            
            # Especificaciones técnicas (usando datos reales)
            "technical_specifications": {
                "input_format": "Medical Images (DICOM, JPG, PNG)",
                "input_resolution": model_info.get("input_resolution", "Variable"),
                "output_format": "Multi-label Classification Probabilities",
                "preprocessing": model_info.get("preprocessing", "Standard pipeline"),
                "confidence_calibration": model_info.get("confidence_calibration", "Standard")
            },
            
            # Configuración de confianza
            "confidence_configuration": model_config.get("confidence_thresholds", {}),
            
            # Información de entrenamiento (usando datos reales)
            "training_information": {
                "training_data": model_info.get("training_data", "Unknown"),
                "validation_status": model_info.get("validation_status", "Unknown"),
                "regulatory_note": "Research and development use"
            },
            
            # Capacidades del sistema (usando datos reales)
            "system_capabilities": model_info.get("capabilities", []),
            
            # Limitaciones conocidas
            "limitations": [
                "Optimized specifically for chest X-rays",
                "Performance may vary with image quality",
                "Requires medical professional validation",
                "Not a replacement for radiologist interpretation",
                "May have reduced performance on rare pathologies"
            ],
            
            # Recomendaciones de uso
            "usage_recommendations": [
                "Use high-quality chest X-ray images",
                "Ensure proper patient positioning",
                "Review all findings with qualified radiologist",
                "Consider clinical context in interpretation",
                "Validate critical findings independently"
            ]
        }
        
        logger.info("Información del modelo consultada exitosamente")
        
        return detailed_info
        
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo información del modelo: {str(e)}"
        )

@router.post("/demo",
            summary="Análisis de Demostración",
            description="Genera un análisis de demostración con datos simulados para testing")
async def demo_analysis(
    model_mgr: AIModelManager = Depends(get_model_manager)
) -> JSONResponse:
    """
    Endpoint de demostración que genera un análisis simulado usando
    la información real del modelo cargado.
    
    Returns:
        JSONResponse: Análisis de demostración completo
    """
    try:
        logger.info("Generando análisis de demostración")
        
        # Obtener información real del modelo para el demo
        model_info = model_mgr.get_model_info()
        pathologies = model_info.get("pathologies_supported", [])
        
        # Generar predicciones de demostración realistas usando las patologías reales
        demo_predictions = {}
        import random
        random.seed(42)  # Para resultados consistentes
        
        for pathology in pathologies:
            # Generar probabilidades realistas
            demo_predictions[pathology] = round(random.uniform(0.02, 0.45), 3)
        
        # Hacer que una patología tenga mayor probabilidad para demo
        if "Pneumonia" in demo_predictions:
            demo_predictions["Pneumonia"] = 0.42
        
        # Generar información de imagen simulada
        demo_image_info = {
            "shape": (512, 512, 3),
            "dtype": "uint8",
            "estimated_quality": "good",
            "contrast_ratio": 0.75,
            "processing_quality": "optimized"
        }
        
        # Generar reporte médico con datos de demo
        report_generator = ReportGenerator()
        demo_report = report_generator.generate_full_report(
            demo_predictions, 
            demo_image_info,
            model_info  # Pasar model_info al reporte
        )
        
        # Respuesta de demostración usando información real del modelo
        demo_response = {
            "analysis_id": "demo-" + str(uuid.uuid4())[:8],
            "status": "demo",
            "message": "Análisis de demostración generado exitosamente",
            
            "file_info": {
                "original_filename": "demo_chest_xray.jpg",
                "file_size_mb": 2.5,
                "file_type": ".jpg",
                "content_type": "image/jpeg"
            },
            
            "medical_analysis": demo_report,
            
            # Información del modelo usando datos reales
            "model_information": {
                "ai_model": f"{model_info.get('model_type', 'Unknown')} (Demo Mode)",
                "model_architecture": model_info.get('model_architecture', 'Unknown'),
                "device_used": model_info.get('device', 'Unknown'),
                "pathologies_evaluated": model_info.get('num_pathologies', len(demo_predictions)),
                "analysis_confidence": "Demonstration Data"
            },
            
            "performance_metrics": {
                "total_processing_time_seconds": 1.23,
                "validation_time_seconds": 0.15,
                "image_processing_time_seconds": 0.45,
                "ai_inference_time_seconds": 0.38,
                "report_generation_time_seconds": 0.25
            },
            
            "metadata": {
                "analysis_timestamp": demo_report["study_info"]["timestamp"],
                "system_version": "Radiology AI Backend v1.0 (Demo)",
                "api_version": "v1",
                "demo_note": "Este es un análisis de demostración con datos simulados"
            }
        }
        
        logger.info("Análisis de demostración generado exitosamente")
        
        return JSONResponse(content=demo_response, status_code=200)
        
    except Exception as e:
        logger.error(f"Error generando demostración: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generando análisis de demostración: {str(e)}"
        )

@router.get("/statistics",
           summary="Estadísticas del Servicio",
           description="Obtiene estadísticas de uso y rendimiento del servicio")
async def get_service_statistics() -> Dict[str, Any]:
    """
    Obtiene estadísticas de uso y rendimiento del servicio.
    
    Returns:
        Dict: Estadísticas del servicio
    """
    try:
        # En una implementación real, estas estadísticas vendrían de una base de datos
        # Por ahora, retornamos estadísticas simuladas
        
        statistics = {
            "service_statistics": {
                "total_analyses_performed": 0,  # Se incrementaría con cada análisis
                "successful_analyses": 0,
                "failed_analyses": 0,
                "average_processing_time_seconds": 2.5,
                "uptime_hours": 0
            },
            
            "pathology_detection_stats": {
                "most_detected_pathology": "Infiltration",
                "least_detected_pathology": "Hernia",
                "average_confidence_score": 0.23
            },
            
            "system_performance": {
                "average_response_time_ms": 150,
                "peak_concurrent_requests": 1,
                "memory_usage_mb": 2048,
                "cpu_usage_percent": 15
            },
            
            "model_performance": {
                "model_accuracy": "Research Mode",
                "false_positive_rate": "Under Evaluation",
                "false_negative_rate": "Under Evaluation",
                "overall_performance": "Development Phase"
            }
        }
        
        logger.info("Estadísticas del servicio consultadas")
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas del servicio: {str(e)}"
        )