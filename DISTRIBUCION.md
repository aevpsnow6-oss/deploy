# 📦 Guía de Distribución - ILO Document Evaluator

Esta guía explica cómo distribuir la aplicación a usuarios finales.

---

## 🎯 Métodos de Distribución Recomendados

### 1. **Distribución Simple (Recomendado para la mayoría)**

#### Para usuarios técnicos (conocen Python):

**Pasos:**
1. Comprimir la carpeta completa en `ilo-evaluator.zip`
2. Excluir del ZIP:
   - Carpeta `venv/` si existe
   - Carpeta `data/`
   - Archivo `.env` (por seguridad)
   - Carpeta `__pycache__/`

**Comando para crear el ZIP:**
```bash
# Mac/Linux
zip -r ilo-evaluator.zip . -x "venv/*" "data/*" ".env" "__pycache__/*" ".git/*"

# Windows PowerShell
Compress-Archive -Path . -DestinationPath ilo-evaluator.zip -Exclude venv,data,.env,__pycache__
```

**Instrucciones para el usuario:**
```
INSTALACIÓN RÁPIDA:

1. Extraer ilo-evaluator.zip
2. Copiar .env.example a .env
3. Editar .env y agregar tu OPENAI_API_KEY
4. Windows: Doble clic en start.bat
   Mac/Linux: Ejecutar ./start.sh en terminal
5. Abrir navegador en http://localhost:8501
```

---

### 2. **Distribución con Docker (Recomendado para no técnicos)**

Docker elimina todos los problemas de dependencias y configuración de Python.

#### Opción A: Distribuir imagen Docker pre-construida

**Paso 1: Construir la imagen**
```bash
docker build -t ilo-evaluator:1.0.0 .
```

**Paso 2: Guardar imagen a archivo**
```bash
docker save ilo-evaluator:1.0.0 -o ilo-evaluator-v1.0.0.tar
```

**Paso 3: Comprimir (opcional, reduce tamaño ~50%)**
```bash
gzip ilo-evaluator-v1.0.0.tar
# Resulta en: ilo-evaluator-v1.0.0.tar.gz
```

**Paso 4: Distribuir archivo .tar.gz + instrucciones**

Crear archivo `INSTRUCCIONES-DOCKER.txt`:
```
INSTALACIÓN CON DOCKER:

Requisitos:
- Docker Desktop instalado (descargar de docker.com)

Pasos:

1. Cargar la imagen Docker:
   docker load -i ilo-evaluator-v1.0.0.tar.gz

2. Crear archivo .env con tu clave de API:
   OPENAI_API_KEY=tu-clave-aqui

3. Ejecutar:
   docker run -d -p 8501:8501 --env-file .env --name ilo-app ilo-evaluator:1.0.0

4. Abrir navegador en: http://localhost:8501

Para detener:
   docker stop ilo-app

Para reiniciar:
   docker start ilo-app

Para ver logs:
   docker logs -f ilo-app
```

#### Opción B: Distribuir Dockerfile + scripts

**Ventaja:** Archivo más pequeño, pero requiere que el usuario construya la imagen.

**Pasos:**
1. Incluir todos los archivos fuente + Dockerfile + docker-compose.yml
2. Usuario ejecuta: `docker-compose up -d`

---

### 3. **Distribución como Aplicación de Escritorio**

Usar PyInstaller para crear ejecutable standalone.

#### Paso 1: Crear archivo launcher.py

```python
# launcher.py
import streamlit.web.cli as stcli
import sys
import os

if __name__ == '__main__':
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        app_dir = sys._MEIPASS
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to main script
    script_path = os.path.join(app_dir, "oli_v6_deploy.py")

    sys.argv = [
        "streamlit",
        "run",
        script_path,
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.port=8501"
    ]
    sys.exit(stcli.main())
```

#### Paso 2: Crear ejecutable

```bash
# Instalar PyInstaller
pip install pyinstaller

# Crear ejecutable
pyinstaller --onefile \
    --add-data "oli_v6_deploy.py:." \
    --add-data "Rubricas_6ago2025.xlsx:." \
    --add-data ".env.example:." \
    --hidden-import=streamlit \
    --hidden-import=pandas \
    --hidden-import=openai \
    --name "ILO-Evaluator" \
    launcher.py
```

#### Paso 3: Distribuir

Distribuir carpeta `dist/` que contiene:
- `ILO-Evaluator.exe` (Windows) o `ILO-Evaluator` (Mac/Linux)
- `.env.example` → Usuario debe crear `.env`

**Nota:** Este método produce ejecutables grandes (200-500MB) porque incluye Python completo.

---

## 📋 Checklist de Distribución

Antes de distribuir, verificar:

- [ ] Eliminar archivo `.env` con claves reales
- [ ] Incluir `.env.example` como plantilla
- [ ] Incluir `README.md` con instrucciones claras
- [ ] Incluir archivo `Rubricas_6ago2025.xlsx`
- [ ] Probar en máquina limpia (sin Python/Docker instalado)
- [ ] Documentar requisitos del sistema
- [ ] Incluir información de contacto/soporte
- [ ] Especificar versión en el nombre del archivo

---

## 📁 Estructura de Distribución Recomendada

```
ilo-evaluator-v1.0.0/
│
├── README.md                  # Guía completa de uso
├── CHANGELOG.md              # Lista de cambios por versión
├── LICENSE                   # Licencia del software
│
├── oli_v6_deploy.py          # Aplicación principal
├── Rubricas_6ago2025.xlsx    # Datos de rúbricas
├── requirements.txt          # Dependencias Python
│
├── .env.example              # Plantilla configuración
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
│
├── start.bat                 # Launcher Windows
├── start.sh                  # Launcher Mac/Linux
├── start-docker.bat          # Docker launcher Windows
├── start-docker.sh           # Docker launcher Mac/Linux
│
└── docs/                     # Documentación adicional
    ├── INSTALACION-WINDOWS.md
    ├── INSTALACION-MAC.md
    ├── INSTALACION-DOCKER.md
    └── TROUBLESHOOTING.md
```

---

## 🚀 Estrategias de Distribución por Audiencia

### Para desarrolladores / usuarios técnicos:
✅ **Método:** GitHub Repository + Releases
- Subir código a GitHub
- Crear releases con tags (v1.0.0, v1.1.0, etc.)
- Incluir instrucciones en README.md

### Para usuarios de negocio / no técnicos:
✅ **Método:** Docker Image + Script de inicio
- Distribuir imagen Docker pre-construida
- Incluir script simple de inicio
- Video tutorial de instalación

### Para organizaciones empresariales:
✅ **Método:** Docker + Docker Compose
- Desplegar en servidor interno
- Múltiples usuarios acceden vía URL
- Administración centralizada

---

## 💡 Consejos para Distribución Efectiva

1. **Versionado Semántico:**
   - v1.0.0 = Primera versión estable
   - v1.1.0 = Nuevas funcionalidades
   - v1.0.1 = Corrección de bugs

2. **Changelog:**
   - Documentar todos los cambios
   - Facilita que usuarios sepan qué actualizar

3. **Testing:**
   - Probar en máquinas limpias (Windows, Mac, Linux)
   - Verificar que scripts de inicio funcionen
   - Validar que Docker image sea funcional

4. **Documentación:**
   - README claro con capturas de pantalla
   - Videos tutoriales de instalación
   - FAQ de problemas comunes

5. **Soporte:**
   - Email/formulario de contacto
   - GitHub Issues para reportar bugs
   - Canal de Slack/Discord para comunidad

---

## 🔐 Seguridad en Distribución

**NUNCA incluir:**
- ❌ Archivo `.env` con claves reales
- ❌ Credenciales de API
- ❌ Datos sensibles de clientes
- ❌ Archivos de log con información personal

**SIEMPRE incluir:**
- ✅ `.env.example` como plantilla
- ✅ Instrucciones de cómo obtener API keys
- ✅ Advertencias sobre seguridad de claves
- ✅ Recomendaciones de no compartir `.env`

---

## 📊 Métricas de Éxito

Después de distribuir, monitorear:
- Número de instalaciones exitosas
- Problemas comunes reportados
- Tiempo promedio de instalación
- Satisfacción de usuarios
- Solicitudes de nuevas funcionalidades

---

## 🔄 Plan de Actualizaciones

### Actualizaciones Menores (bug fixes):
- Notificar a usuarios vía email
- Instrucciones: descargar y reemplazar archivos

### Actualizaciones Mayores (nuevas funcionalidades):
- Release notes detallados
- Video demostrando nuevas funcionalidades
- Período de testing beta con usuarios voluntarios

---

## 📞 Plantilla de Email de Distribución

```
Asunto: Nueva Herramienta Disponible - ILO Document Evaluator v1.0

Estimado/a [Nombre],

Me complace informarte que la nueva versión de la herramienta
ILO Document Evaluator está disponible para descarga.

🎯 Características principales:
- Evaluación automatizada de documentos con criterios de la OIT
- Análisis usando GPT-5 con razonamiento avanzado
- Exportación de resultados a Excel
- Interfaz intuitiva basada en web

📦 Instalación:
Opción 1 (Simple): Descargar ZIP y ejecutar start.bat/start.sh
Opción 2 (Docker): Usar Docker Desktop para instalación sin dependencias

📖 Documentación completa incluida en README.md

⚠️ Requisitos:
- Clave API de OpenAI (instrucciones incluidas)
- Python 3.10+ o Docker Desktop

🔗 Descarga: [ENLACE]

Para soporte, contactar: [EMAIL/SLACK]

Saludos,
[Tu nombre]
```

---

## ✅ Checklist Final Pre-Distribución

```
[ ] Código funciona en Windows
[ ] Código funciona en Mac
[ ] Código funciona en Linux
[ ] Docker image construye correctamente
[ ] Scripts de inicio funcionan
[ ] README está actualizado
[ ] .env.example incluido
[ ] Dependencias en requirements.txt están actualizadas
[ ] Versión documentada en archivos
[ ] Changelog creado
[ ] Tests ejecutados exitosamente
[ ] Documentación revisada
[ ] Plan de soporte definido
[ ] Canales de comunicación establecidos
```

---

¡Listo para distribuir! 🚀
