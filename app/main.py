#!/usr/bin/env python3
"""
Main.py - Sistema de Radiología IA con Arquitectura Desacoplada
==============================================================

Sistema backend avanzado que soporta múltiples modelos de IA médica
con router inteligente y arquitectura escalable.

Características:
- Router inteligente para múltiples modelos
- Arquitectura completamente desacoplada
- Escalabilidad automática para nuevos modelos
- Ensemble optimizado por especialización
- API RESTful para integración con Liferay
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uvicorn
from pathlib import Path

# Importar componentes de la aplicación
from .core.config import settings
from .core.cors import setup_cors
from .api.endpoints.analysis import router as analysis_router

# Importar sistema de IA avanzado
from .models.router.intelligent_router import AdvancedMedicalAIManager

# Configurar logging global de la aplicación
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            settings.logs_dir + "/" + settings.log_file,
            mode="a",
            encoding="utf-8"
        )
    ]
)

# Logger específico para este módulo
logger = logging.getLogger(__name__)

# ============================================================================
# GESTOR GLOBAL DEL SISTEMA IA
# ============================================================================

# Instancia global del sistema IA avanzado
ai_system: Optional[AdvancedMedicalAIManager] = None

def get_ai_system() -> AdvancedMedicalAIManager:
    """
    Obtiene la instancia global del sistema IA.
    
    Returns:
        AdvancedMedicalAIManager: Sistema IA inicializado
        
    Raises:
        RuntimeError: Si el sistema no está inicializado
    """
    global ai_system
    if ai_system is None or not ai_system.is_initialized:
        raise RuntimeError("Sistema IA no inicializado")
    return ai_system

def initialize_ai_system() -> bool:
    """
    Inicializa el sistema IA avanzado.
    
    Returns:
        bool: True si la inicialización fue exitosa
    """
    global ai_system
    
    try:
        logger.info("🚀 Inicializando Sistema IA Médica Avanzado...")
        
        # Crear manager avanzado
        ai_system = AdvancedMedicalAIManager(
            model_path=settings.model_path,
            device=settings.device
        )
        
        # Cargar todos los modelos disponibles
        success = ai_system.load_model("intelligent_router")
        
        if success:
            # Obtener información del sistema inicializado
            system_info = ai_system.get_model_info()
            logger.info(f"✅ Sistema IA inicializado:")
            logger.info(f"   • Tipo: {system_info.get('system_type', 'Unknown')}")
            logger.info(f"   • Modelos cargados: {system_info.get('loaded_models', 0)}")
            logger.info(f"   • Modelos activos: {', '.join(system_info.get('loaded_model_names', []))}")
            logger.info(f"   • Dispositivo: {system_info.get('device', 'Unknown')}")
            
            # Mostrar capacidades avanzadas
            capabilities = system_info.get('capabilities', {})
            logger.info("🎯 Capacidades avanzadas:")
            for capability, enabled in capabilities.items():
                status = "✅" if enabled else "❌"
                logger.info(f"   • {capability}: {status}")
                
            logger.info("🏥 Sistema listo para análisis médico avanzado")
            return True
        else:
            logger.error("❌ Falló la inicialización del sistema IA")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error crítico inicializando sistema IA: {str(e)}")
        logger.error(traceback.format_exc())
        ai_system = None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación FastAPI con sistema IA avanzado.
    
    Args:
        app: Instancia de la aplicación FastAPI
    """
    # =========================================================================
    # STARTUP - Inicialización de la aplicación
    # =========================================================================
    startup_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO RADIOLOGY AI BACKEND API - SISTEMA AVANZADO")
    logger.info("=" * 80)
    
    try:
        # Mostrar información de configuración
        logger.info(f"📋 Configuración del sistema:")
        logger.info(f"   • Modo: {'🔧 Desarrollo' if settings.debug else '🏭 Producción'}")
        logger.info(f"   • Host: {settings.host}:{settings.port}")
        logger.info(f"   • Dispositivo IA: {settings.device}")
        logger.info(f"   • Arquitectura: Desacoplada + Router Inteligente")
        logger.info(f"   • Tamaño máximo archivo: {settings.max_file_size / (1024*1024):.1f}MB")
        
        # Verificar directorios críticos
        logger.info(f"📁 Verificando directorios:")
        for directory_name, directory_path in [
            ("Modelos", settings.model_path),
            ("Uploads", settings.upload_dir),
            ("Logs", settings.logs_dir),
            ("Cache", settings.cache_dir)
        ]:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"   • {directory_name}: {directory_path} ✅")
        
        # Mostrar información del sistema
        system_info = settings.get_system_info()
        logger.info(f"💻 Información del sistema:")
        logger.info(f"   • Python: {system_info['python_version']}")
        logger.info(f"   • PyTorch: {system_info['torch_version']}")
        logger.info(f"   • CUDA disponible: {'✅' if system_info['cuda_available'] else '❌'}")
        logger.info(f"   • Plataforma: {system_info['platform']}")
        
        # Mostrar configuración de CORS para Liferay
        logger.info(f"🌐 Configuración CORS para Liferay:")
        for origin in settings.cors_origins[:3]:  # Mostrar solo los primeros 3
            logger.info(f"   • {origin}")
        if len(settings.cors_origins) > 3:
            logger.info(f"   • ... y {len(settings.cors_origins) - 3} más")
        
        # Inicializar sistema IA avanzado
        logger.info("🧠 Inicializando sistema de IA médica...")
        
        ai_init_success = initialize_ai_system()
        
        if not ai_init_success:
            logger.error("❌ FALLO CRÍTICO: No se pudo inicializar el sistema IA")
            logger.error("💡 La API funcionará pero sin capacidades de análisis")
        
        startup_duration = time.time() - startup_time
        logger.info(f"✅ Inicialización completada en {startup_duration:.2f} segundos")
        
        if ai_init_success:
            logger.info("🏥 API BACKEND CON IA AVANZADA LISTA PARA LIFERAY")
        else:
            logger.warning("⚠️ API BACKEND EN MODO LIMITADO (SIN IA)")
            
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Error crítico durante inicialización: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Punto de yield - la aplicación está corriendo
    yield
    
    # =========================================================================
    # SHUTDOWN - Cierre limpio de la aplicación
    # =========================================================================
    logger.info("=" * 80)
    logger.info("🛑 CERRANDO RADIOLOGY AI BACKEND API - SISTEMA AVANZADO")
    logger.info("=" * 80)
    
    try:
        # Limpiar recursos del sistema IA
        global ai_system
        if ai_system:
            logger.info("🧹 Limpiando recursos del sistema IA...")
            # Aquí se pueden agregar operaciones de limpieza específicas
            ai_system = None
        
        logger.info("✅ Cierre limpio completado")
        
    except Exception as e:
        logger.error(f"❌ Error durante cierre: {str(e)}")
    
    logger.info("👋 RADIOLOGY AI BACKEND API FINALIZADO")
    logger.info("=" * 80)

# ============================================================================
# CREAR APLICACIÓN FASTAPI PARA BACKEND API AVANZADO
# ============================================================================

app = FastAPI(
    title="Radiology AI Backend API - Sistema Avanzado",
    description="""
    **API Backend Avanzado para Análisis Radiológico con Inteligencia Artificial**
    
    Sistema de nueva generación con router inteligente y ensemble de múltiples modelos
    de IA médica para análisis radiológico de máxima precisión.
    
    ## 🧠 Sistema IA Avanzado
    
    - **Router Inteligente**: Selección automática de modelos según tipo de imagen
    - **Ensemble Multi-Modelo**: Combinación inteligente de 4 modelos especializados
    - **Análisis de Consenso**: Validación cruzada entre modelos
    - **Recomendaciones Médicas**: Generación automática de recomendaciones clínicas
    
    ## 🏥 Modelos Médicos Integrados
    
    1. **ToraxModel** (TorchXRayVision) - Patologías torácicas generales
    2. **FracturasModel** (MIMIC-MIT) - Detección especializada de fracturas
    3. **CheXNetModel** (Stanford) - Especialista en neumonía y tórax
    4. **RadImageNetModel** (Universal) - Análisis médico multi-modalidad
    
    ## 📊 Endpoints Principales
    
    - `POST /api/v1/analysis/upload` - Análisis inteligente con ensemble
    - `POST /api/v1/analysis/upload?use_ensemble=false` - Análisis modelo único
    - `GET /api/v1/analysis/health` - Estado del sistema IA avanzado
    - `GET /api/v1/analysis/model/info` - Información completa del sistema
    - `GET /api/v1/analysis/models/available` - Modelos disponibles
    - `POST /api/v1/analysis/demo` - Análisis de demostración avanzado
    
    ## 🎯 Patologías Detectadas
    
    **Sistema combinado detecta 20+ patologías:**
    - Tórax: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, 
             Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, 
             Fibrosis, Pleural Thickening, Hernia
    - Fracturas: Simple, Complex, Displaced, Hairline, Compression, etc.
    - Universal: Abnormal findings, Inflammation, Degeneration, etc.
    
    ## 🔍 Capacidades Avanzadas
    
    - **Análisis automático de calidad de imagen**
    - **Detección de tipo de estudio radiológico**
    - **Selección inteligente de modelos especializados**
    - **Ensemble ponderado por confianza y especialización**
    - **Análisis de consenso entre modelos**
    - **Recomendaciones médicas automáticas**
    - **Evaluación de urgencia clínica**
    - **Trazabilidad completa de decisiones**
    
    ## 📁 Formatos Soportados
    
    DICOM (.dcm), JPEG (.jpg), PNG (.png), TIFF (.tiff), BMP (.bmp)
    
    ## ⚡ Performance
    
    - **Análisis ensemble**: ~2-4 segundos
    - **Análisis modelo único**: ~0.5-1 segundo
    - **Procesamiento paralelo** conceptual de modelos
    - **Cache inteligente** para optimización
    
    ⚠️ **Importante**: Herramienta de apoyo diagnóstico avanzado. 
    Requiere validación médica profesional.
    """,
    version="2.0.0",
    
    # Configuración de documentación
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    
    # Gestión del ciclo de vida
    lifespan=lifespan,
    
    # Metadatos para integración
    contact={
        "name": "Radiology AI Advanced Team",
        "email": "advanced-backend@radiologyai.com"
    },
    license_info={
        "name": "Medical AI License",
        "url": "https://radiologyai.com/license"
    },
    tags_metadata=[
        {
            "name": "analysis",
            "description": "Análisis radiológico avanzado con IA",
        },
        {
            "name": "health",
            "description": "Monitoreo del sistema IA",
        },
        {
            "name": "models",
            "description": "Información de modelos IA",
        },
    ]
)

# ============================================================================
# CONFIGURAR MIDDLEWARE PARA INTEGRACION CON LIFERAY
# ============================================================================

# Configurar CORS específicamente para Liferay
setup_cors(app)

# Middleware de compresión para optimizar transferencia de datos
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware de seguridad para producción
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*.yourdomain.com"]
    )

# ============================================================================
# MIDDLEWARE PERSONALIZADO PARA API AVANZADA
# ============================================================================

@app.middleware("http")
async def advanced_api_logging_middleware(request: Request, call_next):
    """
    Middleware para logging avanzado de API requests.
    
    Args:
        request: Request HTTP entrante
        call_next: Siguiente función en la cadena
        
    Returns:
        Response con headers de API avanzada
    """
    start_time = time.time()
    
    # Generar ID único para el request
    request_id = f"ai_{int(start_time * 1000000) % 1000000}"
    
    # Log del request para API
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(
        f"[{request_id}] API {request.method} {request.url.path} from {client_ip}"
    )
    
    try:
        # Procesar request
        response = await call_next(request)
        
        # Calcular tiempo de procesamiento
        process_time = time.time() - start_time
        
        # Headers específicos para API backend avanzada
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        response.headers["X-Request-ID"] = request_id
        response.headers["X-API-Version"] = "v2.0"
        response.headers["X-Backend-Service"] = "radiology-ai-advanced"
        response.headers["X-AI-System"] = "intelligent-router"
        
        # Agregar información del sistema IA si está disponible
        try:
            ai_sys = get_ai_system()
            available_models = ai_sys.get_available_models()
            response.headers["X-AI-Models-Active"] = str(len(available_models))
            response.headers["X-AI-Capabilities"] = "ensemble,routing,consensus"
        except:
            response.headers["X-AI-Status"] = "unavailable"
        
        # Log de respuesta
        logger.info(
            f"[{request_id}] {response.status_code} in {process_time:.3f}s"
        )
        
        # Log especial para análisis completados
        if request.url.path.endswith("/upload") and response.status_code == 200:
            logger.info(f"[{request_id}] 🧠 Análisis IA avanzado para Liferay completado")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] API Error after {process_time:.3f}s: {str(e)}")
        raise

@app.middleware("http")
async def ai_system_middleware(request: Request, call_next):
    """
    Middleware específico para el sistema IA avanzado.
    
    Args:
        request: Request HTTP
        call_next: Siguiente función
        
    Returns:
        Response con información del sistema IA
    """
    response = await call_next(request)
    
    # Headers específicos para sistema IA
    response.headers["X-AI-Architecture"] = "decoupled-intelligent-router"
    response.headers["X-AI-Ensemble"] = "available"
    
    # Headers de cache específicos para análisis IA
    if request.url.path.startswith("/api/v1/analysis"):
        # No cache para análisis médicos (siempre fresh)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif request.url.path.endswith("/model/info"):
        # Cache corto para info de modelos
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutos
    
    return response

# ============================================================================
# INCLUIR ROUTERS DE API
# ============================================================================

# Router principal de análisis radiológico
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])

# ============================================================================
# ENDPOINTS BÁSICOS DE API AVANZADA
# ============================================================================

@app.get("/",
         summary="API Root - Sistema Avanzado",
         description="Información básica de la API backend avanzada",
         tags=["root"])
async def api_root():
    """
    Endpoint raíz de la API con información del sistema avanzado.
    
    Returns:
        Dict: Información de la API avanzada
    """
    # Obtener información del sistema IA si está disponible
    ai_info = {}
    try:
        ai_sys = get_ai_system()
        system_info = ai_sys.get_model_info()
        ai_info = {
            "ai_system_status": "operational",
            "ai_system_type": system_info.get("system_type", "Unknown"),
            "loaded_models": system_info.get("loaded_models", 0),
            "active_models": system_info.get("loaded_model_names", []),
            "ai_capabilities": list(system_info.get("capabilities", {}).keys())
        }
    except:
        ai_info = {
            "ai_system_status": "unavailable",
            "ai_system_type": "none",
            "loaded_models": 0,
            "active_models": [],
            "ai_capabilities": []
        }
    
    return {
        "service": "Radiology AI Backend API - Sistema Avanzado",
        "version": "2.0.0",
        "status": "operational",
        "mode": "development" if settings.debug else "production",
        "architecture": "decoupled_intelligent_router",
        "api_base": "/api/v1",
        "endpoints": {
            "analysis_ensemble": "/api/v1/analysis/upload",
            "analysis_single": "/api/v1/analysis/upload?use_ensemble=false",
            "health": "/api/v1/analysis/health",
            "demo": "/api/v1/analysis/demo",
            "model_info": "/api/v1/analysis/model/info",
            "available_models": "/api/v1/analysis/models/available"
        },
        "capabilities": {
            "intelligent_routing": True,
            "ensemble_analysis": True,
            "consensus_validation": True,
            "medical_recommendations": True,
            "automatic_model_selection": True,
            "image_quality_assessment": True,
            "pathologies_detected": "20+",
            "formats_supported": settings.allowed_extensions,
            "max_file_size_mb": settings.max_file_size / (1024 * 1024),
            "real_time_analysis": True,
            "liferay_integration": True
        },
        "ai_system": ai_info,
        "cors_configured": True,
        "timestamp": time.time()
    }

@app.get("/health",
         summary="Quick Health Check - Sistema Avanzado",
         description="Verificación rápida de estado para monitoreo",
         tags=["health"])
async def quick_health():
    """
    Health check rápido para monitoreo automático del sistema avanzado.
    
    Returns:
        Dict: Estado básico del sistema
    """
    # Verificar estado del sistema IA
    ai_status = "unknown"
    ai_models_count = 0
    
    try:
        ai_sys = get_ai_system()
        ai_status = "operational"
        ai_models_count = len(ai_sys.get_available_models())
    except:
        ai_status = "unavailable"
    
    health_status = "healthy" if ai_status == "operational" else "degraded"
    
    return {
        "status": health_status,
        "service": "radiology-ai-backend-advanced",
        "version": "2.0.0",
        "timestamp": time.time(),
        "uptime": True,
        "api_operational": True,
        "ai_system": {
            "status": ai_status,
            "models_loaded": ai_models_count,
            "capabilities": ["ensemble", "routing", "consensus"] if ai_status == "operational" else []
        }
    }

@app.get("/ping",
         summary="Ping Test - Sistema Avanzado",
         description="Test de conectividad simple",
         tags=["connectivity"])
async def ping():
    """
    Endpoint simple para test de conectividad desde Liferay.
    
    Returns:
        Dict: Respuesta de ping con info del sistema
    """
    return {
        "ping": "pong",
        "timestamp": time.time(),
        "service": "radiology-ai-backend-advanced",
        "architecture": "intelligent_router",
        "system_ready": True
    }

@app.get("/system/status",
         summary="Estado Completo del Sistema",
         description="Estado detallado del sistema IA y todos sus componentes",
         tags=["system"])
async def system_status():
    """
    Endpoint para obtener estado completo del sistema IA avanzado.
    
    Returns:
        Dict: Estado detallado del sistema
    """
    try:
        ai_sys = get_ai_system()
        system_info = ai_sys.get_model_info()
        
        return {
            "system_status": "operational",
            "ai_system": system_info,
            "performance": {
                "models_loaded": system_info.get("loaded_models", 0),
                "models_available": system_info.get("total_models", 0),
                "active_models": system_info.get("loaded_model_names", [])
            },
            "capabilities_status": {
                capability: "available" for capability in system_info.get("capabilities", {})
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {e}")
        return {
            "system_status": "error",
            "error": str(e),
            "ai_system": {"status": "unavailable"},
            "timestamp": time.time()
        }

# ============================================================================
# MANEJADORES DE ERRORES PARA API AVANZADA
# ============================================================================

@app.exception_handler(404)
async def advanced_api_not_found_handler(request: Request, exc):
    """
    Manejador de 404 optimizado para API avanzada.
    
    Args:
        request: Request que causó el error
        exc: Excepción 404
        
    Returns:
        JSONResponse: Error estructurado para API
    """
    logger.warning(f"API 404: {request.url.path}")
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "endpoint_not_found",
            "message": f"API endpoint '{request.url.path}' not found",
            "available_endpoints": [
                "/api/v1/analysis/upload",
                "/api/v1/analysis/health",
                "/api/v1/analysis/demo", 
                "/api/v1/analysis/model/info",
                "/api/v1/analysis/models/available",
                "/health",
                "/ping",
                "/system/status"
            ],
            "api_version": "v2.0",
            "system_type": "advanced_ai",
            "timestamp": time.time()
        }
    )

@app.exception_handler(500)
async def advanced_api_internal_error_handler(request: Request, exc):
    """
    Manejador de errores internos para API avanzada.
    
    Args:
        request: Request que causó el error
        exc: Excepción interna
        
    Returns:
        JSONResponse: Error estructurado
    """
    error_id = f"ai_err_{int(time.time() * 1000) % 1000000}"
    
    logger.error(f"API Error [{error_id}] en {request.url.path}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Verificar si es error del sistema IA
    ai_error = False
    try:
        get_ai_system()
    except:
        ai_error = True
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "error_id": error_id,
            "message": "Internal API error occurred",
            "ai_system_error": ai_error,
            "system_type": "advanced_ai",
            "timestamp": time.time(),
            "request_path": str(request.url.path)
        }
    )

@app.exception_handler(HTTPException)
async def advanced_api_http_exception_handler(request: Request, exc: HTTPException):
    """
    Manejador de HTTPExceptions para API avanzada.
    
    Args:
        request: Request que causó la excepción
        exc: HTTPException
        
    Returns:
        JSONResponse: Error HTTP estructurado
    """
    logger.warning(f"API HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"http_{exc.status_code}",
            "detail": exc.detail,
            "api_version": "v2.0",
            "system_type": "advanced_ai",
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# ============================================================================
# ENDPOINTS ESPECÍFICOS PARA SISTEMA IA AVANZADO
# ============================================================================

@app.get("/api/v1/ai/models/status",
         summary="Estado de Modelos IA",
         description="Estado detallado de todos los modelos IA",
         tags=["models"])
async def ai_models_status():
    """
    Obtiene estado detallado de todos los modelos IA.
    
    Returns:
        Dict: Estado de cada modelo
    """
    try:
        ai_sys = get_ai_system()
        system_info = ai_sys.get_model_info()
        
        return {
            "models_status": system_info.get("model_details", {}),
            "capabilities_coverage": system_info.get("capabilities_coverage", {}),
            "total_models": system_info.get("total_models", 0),
            "loaded_models": system_info.get("loaded_models", 0),
            "system_ready": True,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "error": "ai_system_unavailable",
            "message": str(e),
            "system_ready": False,
            "timestamp": time.time()
        }

@app.get("/api/v1/ai/capabilities",
         summary="Capacidades del Sistema IA",
         description="Lista completa de capacidades del sistema IA",
         tags=["capabilities"])
async def ai_capabilities():
    """
    Obtiene lista completa de capacidades del sistema IA.
    
    Returns:
        Dict: Capacidades disponibles
    """
    try:
        ai_sys = get_ai_system()
        system_info = ai_sys.get_model_info()
        
        return {
            "capabilities": system_info.get("capabilities", {}),
            "advanced_features": system_info.get("advanced_features", []),
            "pathologies_supported": system_info.get("pathologies_supported", "Variable"),
            "system_type": system_info.get("system_type", "Unknown"),
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "error": "ai_system_unavailable",
            "message": str(e),
            "capabilities": {},
            "timestamp": time.time()
        }

# ============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR LA APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Iniciando API Backend Avanzado para Liferay...")
    
    uvicorn_config = {
        "app": "app.main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": settings.debug,
        "log_level": settings.log_level.lower(),
        "access_log": True
    }
    
    if not settings.debug:
        uvicorn_config.update({
            "workers": settings.workers,
            "reload": False
        })
    
    # Información de endpoints
    logger.info(f"🌐 API disponible en: http://{settings.host}:{settings.port}")
    logger.info(f"🧠 Endpoint ensemble: http://{settings.host}:{settings.port}/api/v1/analysis/upload")
    logger.info(f"🏥 Endpoint modelo único: http://{settings.host}:{settings.port}/api/v1/analysis/upload?use_ensemble=false")
    logger.info(f"📊 Health check avanzado: http://{settings.host}:{settings.port}/api/v1/analysis/health")
    logger.info(f"🔍 Estado del sistema: http://{settings.host}:{settings.port}/system/status")
    logger.info(f"📚 Documentación: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(**uvicorn_config)