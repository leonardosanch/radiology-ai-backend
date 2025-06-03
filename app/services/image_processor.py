import numpy as np
import cv2
from PIL import Image
import io
from typing import Tuple, Optional, Dict, Any
from fastapi import UploadFile, HTTPException
import logging
import pydicom
from pathlib import Path

# Configurar logging para este módulo
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Procesador especializado de imágenes radiológicas para el sistema de IA.
    Maneja la carga, preprocesamiento y optimización de radiografías de tórax.
    """
    
    # Configuraciones estándar para procesamiento de radiografías
    STANDARD_SIZE = (512, 512)  # Tamaño estándar para procesamiento inicial
    MODEL_INPUT_SIZE = (224, 224)  # Tamaño requerido por el modelo de IA
    
    @staticmethod
    def load_image_from_upload(file: UploadFile) -> np.ndarray:
        """
        Carga una imagen desde un archivo subido por el usuario.
        Soporta múltiples formatos: JPG, PNG, DICOM.
        
        Args:
            file: Archivo subido por FastAPI
            
        Returns:
            np.ndarray: Imagen como array numpy en formato RGB
            
        Raises:
            HTTPException: Si hay error al procesar la imagen
        """
        try:
            # Leer el contenido del archivo
            contents = file.file.read()
            file.file.seek(0)  # Resetear puntero para futuros usos
            
            logger.info(f"Procesando archivo: {file.filename}, tamaño: {len(contents)} bytes")
            
            # Detectar tipo de archivo por extensión
            file_extension = Path(file.filename).suffix.lower() if file.filename else ""
            
            if file_extension in ['.dcm', '.dicom']:
                # Procesar archivo DICOM (formato médico estándar)
                image_array = ImageProcessor._load_dicom_image(contents)
            else:
                # Procesar imagen estándar (JPG, PNG, etc.)
                image_array = ImageProcessor._load_standard_image(contents)
            
            logger.info(f"Imagen cargada exitosamente - Shape: {image_array.shape}, Dtype: {image_array.dtype}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error al cargar imagen desde upload: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error al procesar la imagen: {str(e)}"
            )
    
    @staticmethod
    def _load_dicom_image(contents: bytes) -> np.ndarray:
        """
        Carga una imagen DICOM desde bytes.
        DICOM es el formato estándar para imágenes médicas.
        
        Args:
            contents: Contenido del archivo DICOM en bytes
            
        Returns:
            np.ndarray: Imagen extraída del DICOM
        """
        try:
            # Crear archivo temporal en memoria para pydicom
            dicom_file = io.BytesIO(contents)
            
            # Leer el archivo DICOM
            dicom_data = pydicom.dcmread(dicom_file)
            
            # Extraer la imagen pixel array
            pixel_array = dicom_data.pixel_array
            
            # Aplicar transformaciones DICOM si existen
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                # Aplicar rescaling DICOM: pixel_value = slope * raw_value + intercept
                slope = float(dicom_data.RescaleSlope)
                intercept = float(dicom_data.RescaleIntercept)
                pixel_array = slope * pixel_array + intercept
            
            # Normalizar valores si están fuera del rango estándar
            if pixel_array.max() > 255:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Convertir a RGB si es escala de grises
            if len(pixel_array.shape) == 2:
                # Imagen en escala de grises, convertir a RGB
                rgb_image = np.stack([pixel_array] * 3, axis=-1)
            else:
                rgb_image = pixel_array
            
            logger.info("Imagen DICOM procesada exitosamente")
            return rgb_image.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error procesando DICOM: {str(e)}")
            raise ValueError(f"Archivo DICOM inválido o corrupto: {str(e)}")
    
    @staticmethod
    def _load_standard_image(contents: bytes) -> np.ndarray:
        """
        Carga una imagen estándar (JPG, PNG, etc.) desde bytes.
        
        Args:
            contents: Contenido de la imagen en bytes
            
        Returns:
            np.ndarray: Imagen como array numpy RGB
        """
        try:
            # Convertir bytes a imagen PIL
            pil_image = Image.open(io.BytesIO(contents))
            
            # Convertir a RGB si es necesario (eliminar canal alpha, convertir escala de grises)
            if pil_image.mode == 'RGBA':
                # Convertir RGBA a RGB con fondo blanco
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[-1])  # Usar canal alpha como máscara
                pil_image = rgb_image
            elif pil_image.mode == 'L':
                # Convertir escala de grises a RGB
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode != 'RGB':
                # Convertir cualquier otro modo a RGB
                pil_image = pil_image.convert('RGB')
            
            # Convertir PIL a numpy array
            image_array = np.array(pil_image)
            
            logger.info("Imagen estándar procesada exitosamente")
            return image_array
            
        except Exception as e:
            logger.error(f"Error procesando imagen estándar: {str(e)}")
            raise ValueError(f"Formato de imagen no soportado o archivo corrupto: {str(e)}")
    
    @staticmethod
    def preprocess_for_model(image: np.ndarray, target_size: Tuple[int, int] = MODEL_INPUT_SIZE) -> np.ndarray:
        """
        Preprocesa la imagen específicamente para el modelo de IA.
        Optimiza contraste, tamaño y formato para máxima precisión diagnóstica.
        
        Args:
            image: Imagen original como array numpy
            target_size: Tamaño objetivo (ancho, alto) para el modelo
            
        Returns:
            np.ndarray: Imagen preprocesada lista para el modelo
        """
        try:
            logger.info(f"Preprocesando imagen para modelo - Tamaño objetivo: {target_size}")
            
            # Paso 1: Mejorar el contraste específicamente para radiografías
            enhanced_image = ImageProcessor.enhance_medical_contrast(image)
            
            # Paso 2: Redimensionar manteniendo proporción
            resized_image = ImageProcessor._smart_resize(enhanced_image, target_size)
            
            # Paso 3: Normalización específica para imágenes médicas
            normalized_image = ImageProcessor._medical_normalize(resized_image)
            
            # Paso 4: Aplicar filtros de reducción de ruido
            denoised_image = ImageProcessor._denoise_medical_image(normalized_image)
            
            logger.info("Preprocesamiento completado exitosamente")
            return denoised_image
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento para modelo: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail="Error al preparar imagen para análisis"
            )
    
    @staticmethod
    def enhance_medical_contrast(image: np.ndarray) -> np.ndarray:
        """
        Mejora el contraste específicamente optimizado para radiografías médicas.
        Utiliza técnicas especializadas para resaltar estructuras anatómicas.
        
        Args:
            image: Imagen original
            
        Returns:
            np.ndarray: Imagen con contraste mejorado
        """
        try:
            # Convertir a escala de grises para procesamiento
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Técnica 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Especializado para imágenes médicas - evita sobre-amplificación
            clahe = cv2.createCLAHE(
                clipLimit=2.5,      # Límite para evitar ruido
                tileGridSize=(8,8)  # Tamaño de ventana adaptativa
            )
            clahe_enhanced = clahe.apply(gray)
            
            # Técnica 2: Mejora de bordes para estructuras anatómicas
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # Kernel de sharpening
            sharpened = cv2.filter2D(clahe_enhanced, -1, kernel)
            
            # Técnica 3: Combinación ponderada de ambas técnicas
            alpha = 0.7  # Peso para CLAHE
            beta = 0.3   # Peso para sharpening
            combined = cv2.addWeighted(clahe_enhanced, alpha, sharpened, beta, 0)
            
            # Asegurar rango válido [0, 255]
            enhanced = np.clip(combined, 0, 255).astype(np.uint8)
            
            # Convertir de vuelta a RGB si la imagen original era RGB
            if len(image.shape) == 3:
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                return enhanced_rgb
            
            logger.info("Contraste médico mejorado exitosamente")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error mejorando contraste médico: {str(e)}")
            return image  # Retornar imagen original si falla
    
    @staticmethod
    def _smart_resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Redimensiona la imagen manteniendo proporción y centrando el contenido.
        Utiliza interpolación optimizada para imágenes médicas.
        
        Args:
            image: Imagen a redimensionar
            target_size: Tamaño objetivo (ancho, alto)
            
        Returns:
            np.ndarray: Imagen redimensionada
        """
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calcular escala manteniendo proporción
            scale = min(target_w / w, target_h / h)
            
            # Calcular nuevas dimensiones
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Redimensionar con interpolación de alta calidad
            resized = cv2.resize(
                image, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LANCZOS4  # Mejor para imágenes médicas
            )
            
            # Crear imagen de salida con fondo negro (típico en radiografías)
            if len(image.shape) == 3:
                output = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            else:
                output = np.zeros((target_h, target_w), dtype=image.dtype)
            
            # Centrar la imagen redimensionada
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            if len(image.shape) == 3:
                output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            else:
                output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return output
            
        except Exception as e:
            logger.error(f"Error en redimensionado inteligente: {str(e)}")
            # Fallback: redimensionado simple
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def _medical_normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalización específica para imágenes médicas.
        Optimiza los valores de píxel para análisis por IA.
        
        Args:
            image: Imagen a normalizar
            
        Returns:
            np.ndarray: Imagen normalizada
        """
        try:
            # Convertir a float32 para cálculos precisos
            normalized = image.astype(np.float32)
            
            # Normalización percentil (más robusta que min-max para imágenes médicas)
            p1, p99 = np.percentile(normalized, (1, 99))
            
            # Aplicar normalización percentil
            normalized = (normalized - p1) / (p99 - p1)
            
            # Clampear valores al rango [0, 1]
            normalized = np.clip(normalized, 0, 1)
            
            # Convertir de vuelta a uint8 si es necesario
            if image.dtype == np.uint8:
                normalized = (normalized * 255).astype(np.uint8)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error en normalización médica: {str(e)}")
            return image
    
    @staticmethod
    def _denoise_medical_image(image: np.ndarray) -> np.ndarray:
        """
        Reduce ruido específicamente en imágenes médicas.
        Preserva bordes importantes mientras elimina ruido de fondo.
        
        Args:
            image: Imagen con ruido
            
        Returns:
            np.ndarray: Imagen con ruido reducido
        """
        try:
            if len(image.shape) == 3:
                # Para imágenes RGB, aplicar denoising por canal
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, 
                    None, 
                    10,  # Fuerza del filtro para componente luminancia
                    10,  # Fuerza del filtro para componentes de color
                    7,   # Tamaño de ventana de template
                    21   # Tamaño de ventana de búsqueda
                )
            else:
                # Para imágenes en escala de grises
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    10,  # Fuerza del filtro
                    7,   # Tamaño de ventana de template
                    21   # Tamaño de ventana de búsqueda
                )
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error en reducción de ruido: {str(e)}")
            return image  # Retornar imagen original si falla
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict[str, Any]:
        """
        Extrae información detallada de la imagen para análisis y logging.
        
        Args:
            image: Imagen a analizar
            
        Returns:
            Dict: Información completa de la imagen
        """
        try:
            info = {
                # Información básica de dimensiones
                "shape": image.shape,
                "dtype": str(image.dtype),
                "size_bytes": image.nbytes,
                
                # Estadísticas de píxeles
                "min_value": float(np.min(image)),
                "max_value": float(np.max(image)),
                "mean_value": float(np.mean(image)),
                "std_value": float(np.std(image)),
                
                # Información de calidad
                "dynamic_range": float(np.max(image) - np.min(image)),
                "contrast_ratio": float(np.std(image) / np.mean(image)) if np.mean(image) > 0 else 0,
                
                # Información médica específica
                "is_grayscale": len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1),
                "aspect_ratio": image.shape[1] / image.shape[0] if len(image.shape) >= 2 else 1,
                
                # Calidad estimada
                "estimated_quality": ImageProcessor._estimate_image_quality(image)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo información de imagen: {str(e)}")
            return {"error": str(e), "shape": getattr(image, 'shape', 'unknown')}
    
    @staticmethod
    def _estimate_image_quality(image: np.ndarray) -> str:
        """
        Estima la calidad de la imagen radiológica.
        
        Args:
            image: Imagen a evaluar
            
        Returns:
            str: Calidad estimada ('excellent', 'good', 'fair', 'poor')
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calcular métricas de calidad
            
            # 1. Varianza de Laplaciano (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Contraste (desviación estándar)
            contrast = np.std(gray)
            
            # 3. Rango dinámico
            dynamic_range = np.max(gray) - np.min(gray)
            
            # Clasificar calidad basada en métricas
            if laplacian_var > 500 and contrast > 50 and dynamic_range > 150:
                return "excellent"
            elif laplacian_var > 200 and contrast > 30 and dynamic_range > 100:
                return "good"
            elif laplacian_var > 50 and contrast > 15 and dynamic_range > 50:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error estimando calidad de imagen: {str(e)}")
            return "unknown"