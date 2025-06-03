from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.
    Maneja todas las variables de entorno y configuraciones del sistema.
    """
    
    # =============================================================================
    # CONFIGURACIÓN DEL SERVIDOR
    # =============================================================================
    host: str = "0.0.0.0"                    # Host del servidor (0.0.0.0 para acceso externo)
    port: int = 8002                         # Puerto del servidor web
    debug: bool = True                       # Modo debug (mostrar errores detallados)
    workers: int = 1                         # Número de workers para producción
    
    # =============================================================================
    # CONFIGURACIÓN DE CORS (Cross-Origin Resource Sharing)
    # =============================================================================
    cors_origins: str = "http://localhost:3000,http://localhost:8080,http://localhost:8002,https://localhost:3000,http://127.0.0.1:3000,http://127.0.0.1:8080"  # Orígenes CORS
    cors_credentials: bool = True            # Permitir cookies y credenciales
    cors_methods: str = "*"                  # Métodos HTTP permitidos (string)
    cors_headers: str = "*"                  # Headers permitidos (string)
    
    # =============================================================================
    # CONFIGURACIÓN DE ARCHIVOS Y UPLOADS
    # =============================================================================
    max_file_size: int = 52428800           # Tamaño máximo de archivo (50MB en bytes)
    allowed_extensions: str = "jpg,jpeg,png,dcm,dicom,tiff,tif,bmp"  # Extensiones permitidas
    
    # =============================================================================
    # CONFIGURACIÓN DEL MODELO DE IA - ACTUALIZADA PARA TORCHXRAYVISION
    # =============================================================================
    model_path: str = "./models/"           # Directorio para almacenar modelos descargados
    model_name: str = "torchxrayvision"     # CAMBIADO: Modelo principal TorchXRayVision
    device: str = "auto"                    # Dispositivo de computación ('auto', 'cpu', 'cuda')
    
    # Configuración específica de TorchXRayVision
    torchxrayvision_weights: str = "densenet121-res224-all"  # NUEVO: Pesos específicos del modelo
    enable_model_validation: bool = True    # NUEVO: Validar que el modelo carga correctamente
    model_warmup: bool = True               # NUEVO: Pre-calentar modelo al iniciar
    
    # DEPRECATED: Configuración del modelo Google CXR Foundation (mantenido para compatibilidad)
    # Estas configuraciones ya no se usan pero se mantienen para evitar errores
    huggingface_model_name: str = "google/cxr-foundation"  # No usado con TorchXRayVision
    model_cache_dir: Optional[str] = None   # No usado con TorchXRayVision
    trust_remote_code: bool = True          # No usado con TorchXRayVision
    torch_dtype: str = "float32"            # Tipo de datos para PyTorch (aún relevante)
    
    # =============================================================================
    # CONFIGURACIÓN DE DIRECTORIOS
    # =============================================================================
    upload_dir: str = "./uploads/"          # Directorio temporal para archivos subidos
    temp_dir: str = "./temp/"               # Directorio para archivos temporales
    logs_dir: str = "./logs/"               # Directorio para archivos de log
    cache_dir: str = "./cache/"             # Directorio para cache de la aplicación
    
    # =============================================================================
    # CONFIGURACIÓN DE LOGGING
    # =============================================================================
    log_level: str = "INFO"                 # Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    log_file: str = "radiology_ai.log"      # Nombre del archivo de log
    log_rotation: str = "1 week"            # Rotación de logs
    log_retention: str = "1 month"          # Retención de logs
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Formato de log
    
    # =============================================================================
    # CONFIGURACIÓN DE RENDIMIENTO
    # =============================================================================
    max_concurrent_requests: int = 10       # Máximo de requests concurrentes
    request_timeout: int = 300              # Timeout de request en segundos (5 minutos)
    model_inference_timeout: int = 120      # Timeout para inferencia del modelo (2 minutos)
    cache_ttl: int = 3600                   # TTL del cache en segundos (1 hora)
    
    # =============================================================================
    # CONFIGURACIÓN DE SEGURIDAD
    # =============================================================================
    api_key_required: bool = False          # Requerir API key para acceso
    api_key: Optional[str] = None           # API key para autenticación
    rate_limit_requests: int = 100          # Límite de requests por minuto
    rate_limit_window: int = 60             # Ventana de tiempo para rate limiting
    
    # =============================================================================
    # CONFIGURACIÓN MÉDICA ESPECÍFICA
    # =============================================================================
    # Umbrales de confianza para diagnósticos
    confidence_threshold_low: float = 0.3   # Umbral mínimo para reportar hallazgo
    confidence_threshold_moderate: float = 0.6  # Umbral para confianza moderada
    confidence_threshold_high: float = 0.8  # Umbral para alta confianza
    
    # Configuración de calidad de imagen
    min_image_resolution: int = 224         # Resolución mínima requerida
    max_image_resolution: int = 2048        # Resolución máxima procesable
    image_quality_threshold: float = 0.5    # Umbral mínimo de calidad de imagen
    
    # =============================================================================
    # CONFIGURACIÓN DE DESARROLLO Y DEBUG
    # =============================================================================
    enable_model_fallback: bool = False     # CAMBIADO: No hay fallbacks con TorchXRayVision único
    generate_demo_data: bool = False        # Generar datos de demostración
    save_processed_images: bool = False     # Guardar imágenes procesadas para debug
    detailed_error_messages: bool = True    # Mostrar mensajes de error detallados
    
    # =============================================================================
    # CONFIGURACIÓN DE LA BASE DE DATOS (para futuras versiones)
    # =============================================================================
    database_url: Optional[str] = None      # URL de conexión a base de datos
    redis_url: Optional[str] = None         # URL de Redis para cache
    
    # =============================================================================
    # CONFIGURACIÓN DE MONITOREO
    # =============================================================================
    enable_metrics: bool = True             # Habilitar métricas de rendimiento
    metrics_port: int = 9090                # Puerto para métricas Prometheus
    health_check_interval: int = 30         # Intervalo de health check en segundos
    
    class Config:
        """Configuración de Pydantic para carga de variables de entorno."""
        env_file = ".env"                   # Archivo de variables de entorno
        env_file_encoding = "utf-8"         # Codificación del archivo .env
        case_sensitive = False              # No distinguir mayúsculas/minúsculas
        extra = "ignore"                    # Ignorar variables extra en .env
        
    def __init__(self, **kwargs):
        """
        Inicializa la configuración y crea directorios necesarios.
        
        Args:
            **kwargs: Argumentos adicionales para configuración
        """
        super().__init__(**kwargs)
        self._create_directories()
        self._validate_configuration()
        self._setup_logging_config()
    
    def _create_directories(self) -> None:
        """
        Crea todos los directorios necesarios para la aplicación.
        """
        directories = [
            self.model_path,
            self.upload_dir,
            self.temp_dir,
            self.logs_dir,
            self.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_configuration(self) -> None:
        """
        Valida que la configuración sea coherente y segura.
        
        Raises:
            ValueError: Si hay configuraciones inválidas
        """
        # Validar tamaños de archivo
        if self.max_file_size < 1024 * 1024:  # Mínimo 1MB
            raise ValueError("max_file_size debe ser al menos 1MB")
        
        if self.max_file_size > 100 * 1024 * 1024:  # Máximo 100MB
            raise ValueError("max_file_size no debe exceder 100MB")
        
        # Validar umbrales de confianza
        if not (0 <= self.confidence_threshold_low <= 1):
            raise ValueError("confidence_threshold_low debe estar entre 0 y 1")
        
        if not (self.confidence_threshold_low <= self.confidence_threshold_moderate <= self.confidence_threshold_high <= 1):
            raise ValueError("Los umbrales de confianza deben estar en orden ascendente")
        
        # Validar configuración de dispositivo
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError("device debe ser 'auto', 'cpu' o 'cuda'")
        
        # Validar extensiones de archivo
        valid_extensions = ["jpg", "jpeg", "png", "dcm", "dicom", "tiff", "tif", "bmp"]
        for ext in self.get_allowed_extensions_list():
            if ext.lower() not in valid_extensions:
                raise ValueError(f"Extensión no soportada: {ext}")
    
    def _setup_logging_config(self) -> None:
        """
        Configura el sistema de logging de la aplicación.
        """
        import logging
        import logging.handlers
        
        # Crear directorio de logs si no existe
        log_path = Path(self.logs_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar nivel de logging
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format=self.log_format,
            handlers=[
                # Handler para consola
                logging.StreamHandler(),
                # Handler para archivo con rotación
                logging.handlers.RotatingFileHandler(
                    log_path / self.log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
            ]
        )
    
    # =============================================================================
    # MÉTODOS HELPER PARA CONVERTIR STRINGS A LISTAS
    # =============================================================================
    
    def get_cors_origins_list(self) -> List[str]:
        """Convierte la string de CORS origins a lista."""
        if self.cors_origins:
            return [origin.strip() for origin in self.cors_origins.split(',')]
        return ["http://localhost:3000", "http://localhost:8080"]

    def get_cors_methods_list(self) -> List[str]:
        """Convierte la string de CORS methods a lista."""
        if self.cors_methods == "*":
            return ["*"]
        return [method.strip() for method in self.cors_methods.split(',')]

    def get_cors_headers_list(self) -> List[str]:
        """Convierte la string de CORS headers a lista."""
        if self.cors_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_headers.split(',')]

    def get_allowed_extensions_list(self) -> List[str]:
        """Convierte la string de extensiones a lista."""
        if self.allowed_extensions:
            return [ext.strip() for ext in self.allowed_extensions.split(',')]
        return ["jpg", "jpeg", "png", "dcm", "dicom"]
    
    def get_model_config(self) -> dict:
        """
        Obtiene la configuración específica del modelo de IA.
        ACTUALIZADO para TorchXRayVision.
        
        Returns:
            dict: Configuración del modelo
        """
        return {
            # Configuración principal para TorchXRayVision
            "model_type": "torchxrayvision",
            "model_weights": self.torchxrayvision_weights,
            "cache_dir": self.model_cache_dir or self.model_path,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "enable_validation": self.enable_model_validation,
            "warmup_enabled": self.model_warmup,
            
            # Umbrales de confianza
            "confidence_thresholds": {
                "low": self.confidence_threshold_low,
                "moderate": self.confidence_threshold_moderate,
                "high": self.confidence_threshold_high
            },
            
            # Configuración legacy (mantenida para compatibilidad)
            "legacy_model_name": self.huggingface_model_name,
            "trust_remote_code": self.trust_remote_code
        }
    
    def get_torchxrayvision_config(self) -> dict:
        """
        NUEVO: Obtiene configuración específica para TorchXRayVision.
        
        Returns:
            dict: Configuración específica de TorchXRayVision
        """
        return {
            "weights": self.torchxrayvision_weights,
            "device": self.device,
            "validation_enabled": self.enable_model_validation,
            "warmup_enabled": self.model_warmup,
            "model_path": self.model_path,
            "input_resolution": 224,  # Resolución estándar de TorchXRayVision
            "pathologies_count": 14   # Número de patologías que detectamos
        }
    
    def get_cors_config(self) -> dict:
        """
        Obtiene la configuración de CORS para FastAPI.
        
        Returns:
            dict: Configuración de CORS
        """
        return {
            "allow_origins": self.get_cors_origins_list(),
            "allow_credentials": self.cors_credentials,
            "allow_methods": self.get_cors_methods_list(),
            "allow_headers": self.get_cors_headers_list()
        }
    
    def get_upload_config(self) -> dict:
        """
        Obtiene la configuración para manejo de archivos.
        
        Returns:
            dict: Configuración de uploads
        """
        return {
            "max_file_size": self.max_file_size,
            "allowed_extensions": self.get_allowed_extensions_list(),
            "upload_dir": self.upload_dir,
            "temp_dir": self.temp_dir
        }
    
    def is_development(self) -> bool:
        """
        Determina si la aplicación está en modo desarrollo.
        
        Returns:
            bool: True si está en desarrollo
        """
        return self.debug
    
    def is_production(self) -> bool:
        """
        Determina si la aplicación está en modo producción.
        
        Returns:
            bool: True si está en producción
        """
        return not self.debug
    
    def get_system_info(self) -> dict:
        """
        Obtiene información del sistema para diagnóstico.
        ACTUALIZADO para incluir info de TorchXRayVision.
        
        Returns:
            dict: Información del sistema
        """
        import platform
        import torch
        
        # Verificar si TorchXRayVision está disponible
        try:
            import torchxrayvision as xrv
            torchxrayvision_available = True
            torchxrayvision_version = getattr(xrv, '__version__', 'unknown')
        except ImportError:
            torchxrayvision_available = False
            torchxrayvision_version = None
        
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_configured": self.device,
            "model_name": self.model_name,
            "model_type": "TorchXRayVision",
            "torchxrayvision_available": torchxrayvision_available,
            "torchxrayvision_version": torchxrayvision_version,
            "model_weights": self.torchxrayvision_weights,
            "debug_mode": self.debug
        }

# Instancia global de configuración
# Esta instancia se importa en otros módulos para acceder a la configuración
settings = Settings()

# Configurar logging al importar este módulo
import logging
logger = logging.getLogger(__name__)
logger.info("Configuración cargada exitosamente")
logger.info(f"Modo: {'Desarrollo' if settings.is_development() else 'Producción'}")
logger.info(f"Dispositivo configurado: {settings.device}")
logger.info(f"Modelo: {settings.model_name} ({settings.torchxrayvision_weights})")