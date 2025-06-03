from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn

# Importar componentes de la aplicación
from .core.config import settings
from .core.cors import setup_cors
from .api.endpoints.analysis import router as analysis_router

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación FastAPI.
    
    Maneja la inicialización y cierre limpio de recursos:
    - Carga inicial del modelo de IA
    - Verificación de componentes del sistema
    - Limpieza de recursos al cerrar
    
    Args:
        app: Instancia de la aplicación FastAPI
    """
    # =========================================================================
    # STARTUP - Inicialización de la aplicación
    # =========================================================================
    startup_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO RADIOLOGY AI BACKEND API")
    logger.info("=" * 80)
    
    try:
        # Mostrar información de configuración
        logger.info(f"📋 Configuración del sistema:")
        logger.info(f"   • Modo: {'🔧 Desarrollo' if settings.debug else '🏭 Producción'}")
        logger.info(f"   • Host: {settings.host}:{settings.port}")
        logger.info(f"   • Dispositivo IA: {settings.device}")
        logger.info(f"   • Modelo: {settings.model_name}")
        logger.info(f"   • Tamaño máximo archivo: {settings.max_file_size / (1024*1024):.1f}MB")
        
        # Verificar directorios críticos
        logger.info(f"📁 Verificando directorios:")
        logger.info(f"   • Modelos: {settings.model_path}")
        logger.info(f"   • Uploads: {settings.upload_dir}")
        logger.info(f"   • Logs: {settings.logs_dir}")
        logger.info(f"   • Cache: {settings.cache_dir}")
        
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
        
        # Pre-inicializar componentes críticos
        logger.info("🔧 Inicializando componentes del sistema...")
        
        # Verificar que el modelo se puede cargar
        try:
            from .models.ai_model import AIModelManager
            test_manager = AIModelManager(model_path=settings.model_path, device="cpu")
            logger.info("✅ Gestor de modelos IA inicializado correctamente")
        except Exception as e:
            logger.warning(f"⚠️ Advertencia al verificar modelo IA: {str(e)}")
        
        startup_duration = time.time() - startup_time
        logger.info(f"✅ Inicialización completada en {startup_duration:.2f} segundos")
        logger.info("🏥 API BACKEND LISTO PARA LIFERAY")
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
    logger.info("🛑 CERRANDO RADIOLOGY AI BACKEND API")
    logger.info("=" * 80)
    
    try:
        # Limpiar recursos si es necesario
        logger.info("🧹 Limpiando recursos del sistema...")
        
        # Cerrar conexiones, limpiar archivos temporales, etc.
        
        logger.info("✅ Cierre limpio completado")
        
    except Exception as e:
        logger.error(f"❌ Error durante cierre: {str(e)}")
    
    logger.info("👋 RADIOLOGY AI BACKEND API FINALIZADO")
    logger.info("=" * 80)

# ============================================================================
# CREAR APLICACIÓN FASTAPI PARA BACKEND API
# ============================================================================

app = FastAPI(
    title="Radiology AI Backend API",
    description="""
    **API Backend para Análisis Radiológico con Inteligencia Artificial**
    
    Sistema especializado para análisis automático de radiografías de tórax 
    utilizando Google CXR Foundation. Diseñado para integración con Liferay.
    
    ## Endpoints Principales
    
    - `POST /api/v1/analysis/upload` - Analizar radiografía
    - `GET /api/v1/analysis/health` - Estado del sistema
    - `GET /api/v1/analysis/model/info` - Información del modelo IA
    - `POST /api/v1/analysis/demo` - Análisis de demostración
    
    ## Patologías Detectadas (14 total)
    
    Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, 
    Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, 
    Fibrosis, Pleural Thickening, Hernia
    
    ## Formatos Soportados
    
    DICOM (.dcm), JPEG (.jpg), PNG (.png), TIFF (.tiff), BMP (.bmp)
    
    ⚠️ **Importante**: Herramienta de apoyo diagnóstico. Requiere validación médica profesional.
    """,
    version="1.0.0",
    
    # Configuración de documentación
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    
    # Gestión del ciclo de vida
    lifespan=lifespan,
    
    # Metadatos para integración
    contact={
        "name": "Radiology AI Backend Team",
        "email": "backend@radiologyai.com"
    },
    
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
# MIDDLEWARE PERSONALIZADO PARA API
# ============================================================================

@app.middleware("http")
async def api_logging_middleware(request: Request, call_next):
    """
    Middleware para logging de API requests optimizado para backend.
    
    Args:
        request: Request HTTP entrante
        call_next: Siguiente función en la cadena
        
    Returns:
        Response con headers de API
    """
    start_time = time.time()
    
    # Generar ID único para el request
    request_id = f"api_{int(start_time * 1000000) % 1000000}"
    
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
        
        # Headers específicos para API backend
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        response.headers["X-Request-ID"] = request_id
        response.headers["X-API-Version"] = "v1"
        response.headers["X-Backend-Service"] = "radiology-ai"
        
        # Log de respuesta
        logger.info(
            f"[{request_id}] {response.status_code} in {process_time:.3f}s"
        )
        
        # Log especial para análisis completados
        if request.url.path.endswith("/upload") and response.status_code == 200:
            logger.info(f"[{request_id}] 🏥 Análisis radiológico para Liferay completado")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] API Error after {process_time:.3f}s: {str(e)}")
        raise

@app.middleware("http")
async def liferay_integration_middleware(request: Request, call_next):
    """
    Middleware específico para optimizar integración con Liferay.
    
    Args:
        request: Request HTTP
        call_next: Siguiente función
        
    Returns:
        Response optimizada para Liferay
    """
    response = await call_next(request)
    
    # Headers específicos para Liferay
    response.headers["X-Liferay-Compatible"] = "true"
    response.headers["X-Content-Source"] = "radiology-ai-backend"
    
    # Headers de cache para optimizar requests desde Liferay
    if request.url.path.startswith("/api/v1/analysis"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response

# ============================================================================
# INCLUIR ROUTERS DE API
# ============================================================================

# Router principal de análisis radiológico
app.include_router(analysis_router, prefix="/api/v1")

# ============================================================================
# ENDPOINTS BÁSICOS DE API
# ============================================================================

@app.get("/",
         summary="API Root",
         description="Información básica de la API backend")
async def api_root():
    """
    Endpoint raíz de la API con información básica para Liferay.
    
    Returns:
        Dict: Información de la API
    """
    return {
        "service": "Radiology AI Backend API",
        "version": "1.0.0",
        "status": "operational",
        "mode": "development" if settings.debug else "production",
        "api_base": "/api/v1",
        "endpoints": {
            "analysis": "/api/v1/analysis/upload",
            "health": "/api/v1/analysis/health",
            "demo": "/api/v1/analysis/demo",
            "model_info": "/api/v1/analysis/model/info"
        },
        "capabilities": {
            "pathologies_detected": 14,
            "formats_supported": settings.allowed_extensions,
            "max_file_size_mb": settings.max_file_size / (1024 * 1024),
            "real_time_analysis": True,
            "liferay_integration": True
        },
        "cors_configured": True,
        "timestamp": time.time()
    }

@app.get("/health",
         summary="Quick Health Check",
         description="Verificación rápida de estado para monitoreo")
async def quick_health():
    """
    Health check rápido para monitoreo automático.
    
    Returns:
        Dict: Estado básico del sistema
    """
    return {
        "status": "healthy",
        "service": "radiology-ai-backend",
        "version": "1.0.0",
        "timestamp": time.time(),
        "uptime": True,
        "api_operational": True
    }

@app.get("/ping",
         summary="Ping Test",
         description="Test de conectividad simple")
async def ping():
    """
    Endpoint simple para test de conectividad desde Liferay.
    
    Returns:
        Dict: Respuesta de ping
    """
    return {
        "ping": "pong",
        "timestamp": time.time(),
        "service": "radiology-ai-backend"
    }

# ============================================================================
# MANEJADORES DE ERRORES PARA API
# ============================================================================

@app.exception_handler(404)
async def api_not_found_handler(request: Request, exc):
    """
    Manejador de 404 optimizado para API.
    
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
                "/health",
                "/ping"
            ],
            "timestamp": time.time()
        }
    )

@app.exception_handler(500)
async def api_internal_error_handler(request: Request, exc):
    """
    Manejador de errores internos para API.
    
    Args:
        request: Request que causó el error
        exc: Excepción interna
        
    Returns:
        JSONResponse: Error estructurado
    """
    error_id = f"err_{int(time.time() * 1000) % 1000000}"
    
    logger.error(f"API Error [{error_id}] en {request.url.path}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "error_id": error_id,
            "message": "Internal API error occurred",
            "timestamp": time.time(),
            "request_path": str(request.url.path)
        }
    )

@app.exception_handler(HTTPException)
async def api_http_exception_handler(request: Request, exc: HTTPException):
    """
    Manejador de HTTPExceptions para API.
    
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
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Iniciando API Backend para Liferay...")
    
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
    

    # POR ESTAS (agregar línea de documentación):
    logger.info(f"🌐 API disponible en: http://{settings.host}:{settings.port}")
    logger.info(f"🏥 Endpoint principal: http://{settings.host}:{settings.port}/api/v1/analysis/upload")
    logger.info(f"📊 Health check: http://{settings.host}:{settings.port}/api/v1/analysis/health")
    logger.info(f"📚 Documentación: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(**uvicorn_config)