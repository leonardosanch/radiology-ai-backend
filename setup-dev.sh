#!/bin/bash

# ============================================================================
# SCRIPT DE SETUP PARA DESARROLLADORES
# Radiology AI Backend - Configuración automática
# ============================================================================

set -e

echo "🏥 RADIOLOGY AI BACKEND - SETUP AUTOMÁTICO"
echo "=========================================="

# Verificar que Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Por favor instala Docker Compose primero."
    exit 1
fi

echo "✅ Docker y Docker Compose detectados"

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p uploads models logs temp cache
mkdir -p models/torchxrayvision

# Configurar permisos para el sistema anfitrión
echo "🔧 Configurando permisos..."
chmod 777 uploads models logs temp cache 2>/dev/null || echo "⚠️ No se pudieron configurar algunos permisos (requiere sudo)"

# Verificar archivo .env
if [ ! -f .env ]; then
    echo "⚠️ Archivo .env no encontrado. Copiando desde .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "❌ No se encontró .env.example. Crea el archivo .env manualmente."
        exit 1
    fi
fi

echo "✅ Configuración completada"
echo ""
echo "🚀 Para iniciar el sistema:"
echo "   docker-compose up --build -d"
echo ""
echo "📋 Para ver logs:"
echo "   docker-compose logs -f radiology-ai-backend"
echo ""
echo "🔍 Para verificar salud:"
echo "   curl http://localhost:8002/health"
echo ""
echo "📖 Documentación API:"
echo "   http://localhost:8002/docs"