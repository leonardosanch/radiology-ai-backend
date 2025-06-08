from typing import Optional, List, Tuple
from fastapi import HTTPException, UploadFile
from PIL import Image
import io
import magic
import pydicom
from pathlib import Path
import logging

# Importar configuración de la aplicación
from ..core.config import settings

# Configurar logging para este módulo
logger = logging.getLogger(__name__)

class ImageValidator:
    """
    Validador especializado para imágenes médicas radiológicas.
    Valida formato, tamaño, calidad y características específicas de radiografías.
    """
    
    # Tipos MIME permitidos para imágenes médicas
    ALLOWED_MIME_TYPES = {
        'image/jpeg': ['.jpg', '.jpeg'],
        'image/png': ['.png'],
        'image/tiff': ['.tiff', '.tif'],
        'image/bmp': ['.bmp'],
        'application/dicom': ['.dcm', '.dicom'],
        'application/octet-stream': ['.dcm', '.dicom']  # DICOM a veces se detecta así
    }
    
    # Resoluciones mínimas y máximas para radiografías válidas
    MIN_RESOLUTION = (224, 224)      # Resolución mínima para análisis de IA
    MAX_RESOLUTION = (4096, 4096)    # Resolución máxima procesable
    RECOMMENDED_RESOLUTION = (512, 512)  # Resolución recomendada para mejor análisis
    
    @staticmethod
    def validate_image_file(file: UploadFile) -> bool:
        """
        Validación completa del archivo de imagen subido.
        
        Args:
            file: Archivo subido por FastAPI
            
        Returns:
            bool: True si el archivo es válido
            
        Raises:
            HTTPException: Si el archivo no es válido con detalles específicos
        """
        logger.info(f"Iniciando validación de archivo: {file.filename}")
        
        # Validación 1: Verificar que existe nombre de archivo
        if not file.filename:
            logger.error("Archivo sin nombre")
            raise HTTPException(
                status_code=400, 
                detail="El archivo debe tener un nombre válido"
            )
        
        # Validación 2: Verificar extensión del archivo
        ImageValidator._validate_file_extension(file.filename)
        
        # Validación 3: Verificar tamaño del archivo
        ImageValidator._validate_file_size(file)
        
        # Validación 4: Verificar tipo MIME del contenido
        ImageValidator._validate_mime_type(file)
        
        # Validación 5: Verificar que el contenido es realmente una imagen válida
        ImageValidator._validate_image_content(file)
        
        # Validación 6: Verificar calidad y características de imagen médica
        ImageValidator._validate_medical_image_quality(file)
        
        logger.info(f"Archivo {file.filename} validado exitosamente")
        return True
    
    @staticmethod
    def _validate_file_extension(filename: str) -> None:
        """
        Valida que la extensión del archivo esté permitida.
        
        Args:
            filename: Nombre del archivo
            
        Raises:
            HTTPException: Si la extensión no está permitida
        """
        file_extension = Path(filename).suffix.lower()
        
        if not file_extension:
            raise HTTPException(
                status_code=400,
                detail="El archivo debe tener una extensión válida"
            )
        
        # Verificar extensión contra lista permitida (sin el punto)
        extension_without_dot = file_extension[1:]  # Remover el punto inicial
        
        if extension_without_dot not in settings.allowed_extensions:
            allowed_exts = ', '.join(settings.allowed_extensions)
            logger.warning(f"Extensión no permitida: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"Extensión '{file_extension}' no permitida. "
                      f"Extensiones válidas: {allowed_exts}"
            )
        
        logger.debug(f"Extensión válida: {file_extension}")
    
    @staticmethod
    def _validate_file_size(file: UploadFile) -> None:
        """
        Valida el tamaño del archivo subido.
        
        Args:
            file: Archivo a validar
            
        Raises:
            HTTPException: Si el archivo es demasiado grande o pequeño
        """
        # Leer contenido para verificar tamaño real
        file.file.seek(0, 2)  # Ir al final del archivo
        file_size = file.file.tell()  # Obtener posición = tamaño
        file.file.seek(0)  # Volver al inicio
        
        logger.debug(f"Tamaño de archivo: {file_size} bytes")
        
        # Verificar tamaño mínimo (1KB)
        min_size = 1024
        if file_size < min_size:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo es demasiado pequeño ({file_size} bytes). "
                      f"Tamaño mínimo: {min_size} bytes"
            )
        
        # Verificar tamaño máximo
        if file_size > settings.max_file_size:
            max_size_mb = settings.max_file_size / (1024 * 1024)
            current_size_mb = file_size / (1024 * 1024)
            logger.warning(f"Archivo demasiado grande: {current_size_mb:.2f}MB")
            raise HTTPException(
                status_code=400,
                detail=f"Archivo demasiado grande ({current_size_mb:.2f}MB). "
                      f"Tamaño máximo permitido: {max_size_mb:.1f}MB"
            )
    
    @staticmethod
    def _validate_mime_type(file: UploadFile) -> None:
        """
        Valida el tipo MIME del archivo usando detección mágica.
        
        Args:
            file: Archivo a validar
            
        Raises:
            HTTPException: Si el tipo MIME no está permitido
        """
        try:
            # Leer una muestra del archivo para detección MIME
            file.file.seek(0)
            file_sample = file.file.read(8192)  # Leer primeros 8KB
            file.file.seek(0)  # Volver al inicio
            
            # Detectar tipo MIME usando python-magic
            detected_mime = magic.from_buffer(file_sample, mime=True)
            logger.debug(f"Tipo MIME detectado: {detected_mime}")
            
            # Verificar si el tipo MIME está en la lista permitida
            if detected_mime not in ImageValidator.ALLOWED_MIME_TYPES:
                # Verificar tipos MIME alternativos para DICOM
                if detected_mime in ['application/octet-stream'] and file.filename.lower().endswith(('.dcm', '.dicom')):
                    logger.info("Archivo DICOM detectado como octet-stream (normal)")
                    return
                
                allowed_types = list(ImageValidator.ALLOWED_MIME_TYPES.keys())
                logger.warning(f"Tipo MIME no permitido: {detected_mime}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Tipo de archivo no soportado: {detected_mime}. "
                          f"Tipos permitidos: {', '.join(allowed_types)}"
                )
            
        except Exception as e:
            logger.error(f"Error validando tipo MIME: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="No se pudo determinar el tipo de archivo. "
                      "Verifique que sea una imagen válida."
            )
    
    @staticmethod
    def _validate_image_content(file: UploadFile) -> None:
        """
        Valida que el contenido del archivo sea realmente una imagen válida.
        
        Args:
            file: Archivo a validar
            
        Raises:
            HTTPException: Si el contenido no es una imagen válida
        """
        try:
            file.file.seek(0)
            file_contents = file.file.read()
            file.file.seek(0)  # Resetear posición
            
            # Intentar detectar si es DICOM primero
            if ImageValidator._is_dicom_file(file_contents):
                ImageValidator._validate_dicom_content(file_contents)
                return
            
            # Validar como imagen estándar
            ImageValidator._validate_standard_image_content(file_contents)
            
        except HTTPException:
            # Re-lanzar excepciones HTTP sin modificar
            raise
        except Exception as e:
            logger.error(f"Error validando contenido de imagen: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"El archivo no contiene una imagen válida: {str(e)}"
            )
    
    @staticmethod
    def _is_dicom_file(file_contents: bytes) -> bool:
        """
        Determina si el archivo es un DICOM válido.
        
        Args:
            file_contents: Contenido del archivo en bytes
            
        Returns:
            bool: True si es un archivo DICOM
        """
        try:
            # Los archivos DICOM tienen un preámbulo específico
            if len(file_contents) < 132:
                return False
            
            # Verificar el identificador DICM en la posición 128
            dicom_identifier = file_contents[128:132]
            return dicom_identifier == b'DICM'
            
        except Exception:
            return False
    
    @staticmethod
    def _validate_dicom_content(file_contents: bytes) -> None:
        """
        Valida específicamente el contenido de un archivo DICOM.
        
        Args:
            file_contents: Contenido del archivo DICOM
            
        Raises:
            HTTPException: Si el DICOM no es válido
        """
        try:
            # Crear un objeto BytesIO para pydicom
            dicom_file = io.BytesIO(file_contents)
            
            # Intentar leer el archivo DICOM
            dicom_data = pydicom.dcmread(dicom_file)
            
            # Verificar que tiene datos de imagen
            if not hasattr(dicom_data, 'pixel_array'):
                raise HTTPException(
                    status_code=400,
                    detail="El archivo DICOM no contiene datos de imagen"
                )
            
            # Verificar dimensiones de la imagen
            pixel_array = dicom_data.pixel_array
            if len(pixel_array.shape) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Los datos de imagen DICOM no tienen dimensiones válidas"
                )
            
            # Verificar modalidad si está disponible
            if hasattr(dicom_data, 'Modality'):
                modality = dicom_data.Modality
                # Modalidades típicas para radiografías de tórax
                chest_modalities = ['CR', 'DX', 'RF', 'XA']
                if modality not in chest_modalities:
                    logger.warning(f"Modalidad DICOM no típica para tórax: {modality}")
            
            logger.info(f"DICOM válido - Dimensiones: {pixel_array.shape}")
            
        except Exception as e:
            logger.error(f"Error validando DICOM: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Archivo DICOM inválido o corrupto: {str(e)}"
            )
    
    @staticmethod
    def _validate_standard_image_content(file_contents: bytes) -> None:
        """
        Valida el contenido de una imagen estándar (JPG, PNG, etc.).
        
        Args:
            file_contents: Contenido de la imagen
            
        Raises:
            HTTPException: Si la imagen no es válida
        """
        try:
            # Crear imagen PIL desde bytes
            image = Image.open(io.BytesIO(file_contents))
            
            # Verificar que la imagen se puede cargar completamente
            image.verify()
            
            # Reabrir la imagen para obtener información (verify() la cierra)
            image = Image.open(io.BytesIO(file_contents))
            
            # Verificar dimensiones mínimas
            width, height = image.size
            min_width, min_height = ImageValidator.MIN_RESOLUTION
            
            if width < min_width or height < min_height:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resolución demasiado baja ({width}x{height}). "
                          f"Mínimo requerido: {min_width}x{min_height}"
                )
            
            # Verificar dimensiones máximas
            max_width, max_height = ImageValidator.MAX_RESOLUTION
            if width > max_width or height > max_height:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resolución demasiado alta ({width}x{height}). "
                          f"Máximo permitido: {max_width}x{max_height}"
                )
            
            # Verificar modo de color soportado
            if image.mode not in ['L', 'RGB', 'RGBA', 'P']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modo de color no soportado: {image.mode}. "
                          f"Modos válidos: L (escala de grises), RGB, RGBA"
                )
            
            logger.info(f"Imagen estándar válida - {width}x{height}, modo: {image.mode}")
            
        except HTTPException:
            # Re-lanzar excepciones HTTP
            raise
        except Exception as e:
            logger.error(f"Error validando imagen estándar: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"La imagen está corrupta o no es válida: {str(e)}"
            )
    
    @staticmethod
    def _validate_medical_image_quality(file: UploadFile) -> None:
        """
        Valida características específicas de calidad para imágenes médicas.
        
        Args:
            file: Archivo de imagen médica
            
        Raises:
            HTTPException: Si la calidad no es adecuada para análisis médico
        """
        try:
            file.file.seek(0)
            file_contents = file.file.read()
            file.file.seek(0)
            
            # Para DICOM, validaciones específicas
            if ImageValidator._is_dicom_file(file_contents):
                ImageValidator._validate_dicom_medical_quality(file_contents)
            else:
                ImageValidator._validate_standard_medical_quality(file_contents)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"No se pudo evaluar calidad médica: {str(e)}")
            # No es crítico, solo registrar warning
    
    @staticmethod
    def _validate_dicom_medical_quality(file_contents: bytes) -> None:
        """
        Valida calidad específica de archivos DICOM médicos.
        
        Args:
            file_contents: Contenido del archivo DICOM
        """
        try:
            dicom_data = pydicom.dcmread(io.BytesIO(file_contents))
            
            # Verificar que es una imagen de tórax si la información está disponible
            if hasattr(dicom_data, 'BodyPartExamined'):
                body_part = dicom_data.BodyPartExamined.upper()
                chest_keywords = ['CHEST', 'THORAX', 'LUNG', 'PECHO', 'TORAX']
                
                if not any(keyword in body_part for keyword in chest_keywords):
                    logger.warning(f"Parte del cuerpo no es tórax: {body_part}")
                    # No lanzar error, solo warning
            
            # Verificar orientación si está disponible
            if hasattr(dicom_data, 'ViewPosition'):
                view_position = dicom_data.ViewPosition
                valid_views = ['PA', 'AP', 'LAT', 'LATERAL']
                if view_position not in valid_views:
                    logger.warning(f"Vista no estándar para tórax: {view_position}")
            
            logger.debug("Calidad DICOM médica validada")
            
        except Exception as e:
            logger.warning(f"Error validando calidad DICOM médica: {str(e)}")
    
    @staticmethod
    def _validate_standard_medical_quality(file_contents: bytes) -> None:
        """
        Valida calidad de imagen estándar para uso médico.
        
        Args:
            file_contents: Contenido de la imagen
        """
        try:
            import numpy as np
            
            image = Image.open(io.BytesIO(file_contents))
            
            # Convertir a array numpy para análisis
            img_array = np.array(image)
            
            # Verificar que no es una imagen completamente negra o blanca
            if np.all(img_array == 0):
                raise HTTPException(
                    status_code=400,
                    detail="La imagen está completamente negra. "
                          "Verifique que sea una radiografía válida."
                )
            
            if len(img_array.shape) >= 2:
                max_possible = 255 if img_array.dtype == np.uint8 else np.max(img_array)
                if np.all(img_array == max_possible):
                    raise HTTPException(
                        status_code=400,
                        detail="La imagen está completamente blanca. "
                              "Verifique que sea una radiografía válida."
                    )
            
            # Verificar contraste mínimo
            if len(img_array.shape) >= 2:
                std_dev = np.std(img_array)
                if std_dev < 10:  # Muy poco contraste
                    logger.warning("Imagen con contraste muy bajo detectada")
            
            logger.debug("Calidad médica estándar validada")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Error validando calidad médica estándar: {str(e)}")

def validate_upload_file(file: UploadFile) -> bool:
    """
    Función principal de validación para archivos subidos.
    Punto de entrada único para todas las validaciones.
    
    Args:
        file: Archivo subido por FastAPI
        
    Returns:
        bool: True si el archivo es válido
        
    Raises:
        HTTPException: Si el archivo no pasa alguna validación
    """
    try:
        logger.info(f"Iniciando validación completa de archivo: {file.filename}")
        
        # Realizar todas las validaciones usando el validador especializado
        result = ImageValidator.validate_image_file(file)
        
        logger.info(f"Archivo {file.filename} validado exitosamente")
        return result
        
    except HTTPException as e:
        logger.error(f"Validación falló para {file.filename}: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado en validación: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno durante la validación del archivo"
        )

def get_file_info(file: UploadFile) -> dict:
    """
    Obtiene información detallada de un archivo para logging y diagnóstico.
    
    Args:
        file: Archivo a analizar
        
    Returns:
        dict: Información del archivo
    """
    try:
        file.file.seek(0, 2)  # Ir al final
        file_size = file.file.tell()
        file.file.seek(0)  # Volver al inicio
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "extension": Path(file.filename).suffix.lower() if file.filename else None
        }
    except Exception as e:
        logger.error(f"Error obteniendo información de archivo: {str(e)}")
        return {"error": str(e)}