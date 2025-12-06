from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
import os
import psutil

from models.loader import get_model_loader
from models.inference import BoneAnalyzer
from grafana_push import GrafanaCloudPusher

app = FastAPI(title="Bone Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  MÉTRICAS PROMETHEUS

predicciones_total = Counter('bone_predicciones_total', 'Total de predicciones')
predicciones_exitosas = Counter('bone_predicciones_exitosas', 'Predicciones exitosas')
predicciones_fallidas = Counter('bone_predicciones_fallidas', 'Predicciones fallidas', ['tipo_error'])
fracturas_detectadas = Counter('bone_fracturas_total', 'Total fracturas detectadas')
fracturas_criticas = Counter('bone_fracturas_criticas', 'Fracturas críticas')
region_upper = Counter('bone_region_upper', 'Análisis región upper')
region_lower = Counter('bone_region_lower', 'Análisis región lower')
region_rechazada = Counter('bone_region_rechazada', 'Imágenes rechazadas')
tiempo_inferencia = Histogram('bone_tiempo_inferencia_segundos', 'Tiempo de inferencia')
tiempo_request = Histogram('bone_request_segundos', 'Duración total request')
confianza_actual = Gauge('bone_confianza_ultima', 'Confianza última predicción')
memoria_mb = Gauge('bone_memoria_mb', 'Memoria RAM usada')


GRAFANA_ENABLED = os.getenv('GRAFANA_ENABLED', 'false').lower() == 'true'
GRAFANA_REMOTE_WRITE_URL = os.getenv('GRAFANA_REMOTE_WRITE_URL', '')
GRAFANA_USERNAME = os.getenv('GRAFANA_USERNAME', '')
GRAFANA_API_KEY = os.getenv('GRAFANA_API_KEY', '')

bone_analyzer = None
startup_time = time.time()
grafana_pusher = None

@app.on_event("startup")
async def startup():
    global bone_analyzer, grafana_pusher
    
    print(" Iniciando Bone Analysis API...")
    print(" Cargando modelos desde HuggingFace...")
    
    try:
        loader = get_model_loader()
        models = loader.get_models()
        bone_analyzer = BoneAnalyzer(*models)
        print(" Modelos cargados exitosamente")
    except Exception as e:
        print(f" Error cargando modelos: {e}")
        raise

    # Iniciar envío a Grafana Cloud si está habilitado
    if GRAFANA_ENABLED and GRAFANA_REMOTE_WRITE_URL:
        try:
            grafana_pusher = GrafanaCloudPusher(
                GRAFANA_REMOTE_WRITE_URL,
                GRAFANA_USERNAME,
                GRAFANA_API_KEY
            )
            grafana_pusher.start()
            print(" Grafana Cloud Pusher activado")
        except Exception as e:
            print(f"  Grafana pusher no disponible: {e}")
    
    print(" API lista en http://localhost:8000")
    print(" Documentación en http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown():
    if grafana_pusher:
        grafana_pusher.stop()
    print(" API detenida")

# ENDPOINTS

@app.get("/")
async def root():
    uptime = time.time() - startup_time
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    
    return {
        "message": "Bone Analysis API",
        "status": "online",
        "version": "1.0.0",
        "uptime": f"{hours}h {minutes}m",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "features": {
            "regions": ["upper", "lower"],
            "gpu_enabled": True,
            "grafana_push": GRAFANA_ENABLED
        }
    }

@app.get("/health")
async def health():
    models_loaded = bone_analyzer is not None
    
    # Obtener uso de memoria
    try:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
    except:
        mem_mb = 0
    
    return {
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "memory_mb": round(mem_mb, 2),
        "uptime_seconds": round(time.time() - startup_time, 2)
    }

@app.get("/metrics")
async def metrics():
    """Endpoint para Prometheus"""
    try:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        memoria_mb.set(mem_mb)
    except:
        pass

    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    """Endpoint principal de análisis de radiografías"""
    request_start = time.time()

    try:
        predicciones_total.inc()

        # Validar tipo de archivo
        if not image.content_type.startswith('image/'):
            predicciones_fallidas.labels(tipo_error='tipo_invalido').inc()
            raise HTTPException(400, "El archivo debe ser una imagen")

        # Leer imagen
        image_bytes = await image.read()

        # Validar tamaño (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            predicciones_fallidas.labels(tipo_error='muy_grande').inc()
            raise HTTPException(400, "Imagen muy grande (máximo 10MB)")

        # Verificar que los modelos estén cargados
        if bone_analyzer is None:
            predicciones_fallidas.labels(tipo_error='modelos_no_cargados').inc()
            raise HTTPException(503, "Modelos aún no cargados. Espera unos momentos e intenta de nuevo.")

        # Análisis con métricas
        inference_start = time.time()
        result = bone_analyzer.analyze_image(image_bytes)
        inference_time = time.time() - inference_start
        tiempo_inferencia.observe(inference_time)

        # Actualizar métricas según resultado
        if result.get('status') == 'success':
            predicciones_exitosas.inc()

            # Métricas de región
            region = result['region_analysis']['region']
            if region == 'upper':
                region_upper.inc()
            elif region == 'lower':
                region_lower.inc()

            # Confianza
            confidence = result['region_analysis']['confidence']
            confianza_actual.set(confidence)

            # Fracturas
            fracture_data = result.get('fracture_analysis', {})
            total_fracturas = fracture_data.get('total_fractures', 0)

            if total_fracturas > 0:
                fracturas_detectadas.inc(total_fracturas)

                if fracture_data.get('requires_immediate_attention'):
                    fracturas_criticas.inc()
        else:
            region_rechazada.inc()
            predicciones_fallidas.labels(tipo_error='rechazada').inc()

        # Tiempo total de request
        request_time = time.time() - request_start
        tiempo_request.observe(request_time)

        # Agregar metadata de tiempo
        result['processing_time'] = {
            'inference_seconds': round(inference_time, 3),
            'total_seconds': round(request_time, 3)
        }

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        predicciones_fallidas.labels(tipo_error='error_interno').inc()
        raise HTTPException(500, f"Error interno: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)