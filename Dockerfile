# ============================================================================
# DOCKERFILE OPTIMIZADO PARA RADIOLOGY AI BACKEND
# Puerto 8002 para evitar conflictos
# ============================================================================

# Usar imagen base Python 3.11 slim (más ligera)
FROM python:3.11-slim

# Información del mantenedor
LABEL maintainer="Radiology AI Team"
LABEL description="Backend de análisis radiológico con IA - Google CXR Foundation"
LABEL version="1.0.0"

# ============================================================================
# VARIABLES DE ENTORNO
# ============================================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8002
ENV HOST=0.0.0.0

# Variables específicas para el modelo de IA
ENV MODEL_CACHE_DIR=/app/models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface

# ============================================================================
# INSTALAR DEPENDENCIAS DEL SISTEMA
# ============================================================================
RUN apt-get update && apt-get install -y \
    # Dependencias para OpenCV y procesamiento de imágenes
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    # Dependencias para DICOM y imágenes médicas
    libfontconfig1 \
    libfreetype6 \
    libxft2 \
    # Herramientas del sistema
    curl \
    wget \
    git \
    # Dependencias para python-magic
    libmagic1 \
    # Herramientas de compilación (necesarias para algunos paquetes Python)
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# CONFIGURAR DIRECTORIO DE TRABAJO
# ============================================================================
WORKDIR /app

# ============================================================================
# INSTALAR DEPENDENCIAS DE PYTHON
# ============================================================================
# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Actualizar pip y instalar dependencias
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# CREAR ESTRUCTURA DE DIRECTORIOS
# ============================================================================
RUN mkdir -p /app/uploads \
    /app/models \
    /app/models/transformers \
    /app/models/huggingface \
    /app/temp \
    /app/logs \
    /app/cache

# ============================================================================
# COPIAR CÓDIGO DE LA APLICACIÓN
# ============================================================================
# Copiar código fuente
COPY app/ ./app/

# Copiar archivo de configuración
COPY .env .env

# Copiar archivos de configuración adicionales si existen
COPY README.md ./
COPY requirements.txt ./

# ============================================================================
# CONFIGURACIÓN DE SEGURIDAD
# ============================================================================
# Crear usuario no privilegiado para seguridad
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# ============================================================================
# CONFIGURACIÓN DE PUERTOS
# ============================================================================
# Exponer puerto 8002 (cambiado desde 8000)
EXPOSE 8002

# ============================================================================
# CONFIGURACIÓN DE SALUD
# ============================================================================
# Health check mejorado específico para el puerto 8002
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# ============================================================================
# SCRIPT DE INICIO
# ============================================================================
# Crear script de inicio para mejor control
RUN echo '#!/bin/bash\n\
    set -e\n\
    echo "🏥 Iniciando Radiology AI Backend en puerto 8002..."\n\
    echo "📋 Verificando directorios..."\n\
    ls -la /app/\n\
    echo "🔧 Verificando configuración..."\n\
    python -c "from app.core.config import settings; print(f\"Puerto configurado: {settings.port}\")" || echo "⚠️ Error en configuración"\n\
    echo "🚀 Iniciando servidor..."\n\
    exec uvicorn app.main:app --host $HOST --port $PORT --workers 1\n\
    ' > /app/start.sh && \
    chmod +x /app/start.sh

# ============================================================================
# CAMBIAR A USUARIO NO PRIVILEGIADO
# ============================================================================
USER appuser

# ============================================================================
# COMANDO POR DEFECTO
# ============================================================================
# Usar script de inicio personalizado
CMD ["/app/start.sh"]

# ============================================================================
# CONFIGURACIÓN ALTERNATIVA PARA DEVELOPMENT
# ============================================================================
# Para desarrollo con hot reload, usar:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]