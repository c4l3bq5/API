import sys
import requests
import base64
import json
from pathlib import Path


def test_health(base_url):
    """Test de health check"""
    print(" Testing /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        data = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Health: {data.get('status')}")
        print(f"   Models: {'' if data.get('models_loaded') else ''}")
        print(f"   Memory: {data.get('memory_mb', 0):.2f} MB")
        print(f"   Uptime: {data.get('uptime_seconds', 0):.1f}s")
        
        return data.get('models_loaded', False)
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_root(base_url):
    """Test de endpoint raíz"""
    print("\n Testing /...")
    try:
        response = requests.get(base_url, timeout=5)
        data = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Message: {data.get('message')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Uptime: {data.get('uptime')}")
        
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_analyze(base_url, image_path):
    """Test de análisis de imagen"""
    print(f"\n Testing /analyze con {image_path}...")
    
    if not Path(image_path).exists():
        print(f"    Archivo no encontrado: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{base_url}/analyze",
                files=files,
                timeout=60  # 60s para inferencia
            )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"    Error: {response.text}")
            return False
        
        result = response.json()
        
        # Mostrar resultados
        print(f"\n RESULTADOS:")
        print(f"   Status: {result.get('status')}")
        
        if result.get('status') == 'success':
            # Región
            region_info = result.get('region_analysis', {})
            print(f"\n    REGIÓN:")
            print(f"      Tipo: {region_info.get('region', 'N/A').upper()}")
            print(f"      Confianza: {region_info.get('confidence', 0):.2%}")
            
            # Huesos detectados
            bones = result.get('detected_bones', [])
            print(f"\n    HUESOS DETECTADOS: {len(bones)}")
            for bone in bones[:5]:  # Top 5
                print(f"      - {bone['name']}: {bone['coverage']:.1f}%")
            if len(bones) > 5:
                print(f"      ... y {len(bones) - 5} más")
            
            # Fracturas
            fracture_info = result.get('fracture_analysis', {})
            total_fracturas = fracture_info.get('total_fractures', 0)
            
            print(f"\n    FRACTURAS: {total_fracturas}")
            
            if total_fracturas > 0:
                alert_count = fracture_info.get('alert_count', 0)
                sick_count = fracture_info.get('sickness_count', 0)
                major_count = fracture_info.get('major_fractures', 0)
                micro_count = fracture_info.get('microfractures', 0)
                lux_count = fracture_info.get('luxation_count', 0)
                
                if alert_count > 0:
                    print(f"       ALERTAS CRÍTICAS: {alert_count}")
                if sick_count > 0:
                    print(f"       ENFERMEDADES: {sick_count}")
                if major_count > 0:
                    print(f"       FRACTURAS MAYORES: {major_count}")
                if lux_count > 0:
                    print(f"       LUXACIONES: {lux_count}")
                if micro_count > 0:
                    print(f"       MICROFRACTURAS SOSPECHADAS: {micro_count}")
                
                # Detalles de fracturas
                print(f"\n    DETALLES:")
                for idx, frac in enumerate(fracture_info.get('fractures', [])[:3], 1):
                    frac_type = frac.get('fracture_type', {})
                    print(f"\n      [{idx}] {frac.get('description', 'N/A')}")
                    print(f"          Tipo: {frac.get('type', 'N/A')}")
                    print(f"          Hueso: {frac.get('affected_bone', 'N/A')}")
                    print(f"          Clasificación: {frac_type.get('classification', 'N/A')}")
                    
                    if frac_type.get('severity'):
                        print(f"          Severidad: {frac_type['severity']}")
                    if frac_type.get('management'):
                        print(f"          Manejo: {frac_type['management']}")
                
                if len(fracture_info.get('fractures', [])) > 3:
                    print(f"\n      ... y {len(fracture_info['fractures']) - 3} hallazgos más")
                
                # Atención inmediata
                if fracture_info.get('requires_immediate_attention'):
                    print(f"\n        REQUIERE ATENCIÓN INMEDIATA")
            
            # Tiempo de procesamiento
            proc_time = result.get('processing_time', {})
            if proc_time:
                print(f"\n     TIEMPOS:")
                print(f"      Inferencia: {proc_time.get('inference_seconds', 0):.3f}s")
                print(f"      Total: {proc_time.get('total_seconds', 0):.3f}s")
            
            # Guardar imagen anotada
            if result.get('annotated_image'):
                output_path = Path(image_path).stem + "_anotada.jpg"
                img_data = base64.b64decode(result['annotated_image'])
                with open(output_path, 'wb') as f:
                    f.write(img_data)
                print(f"\n    Imagen anotada guardada: {output_path}")
            
            # Guardar JSON completo
            json_path = Path(image_path).stem + "_resultado.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    Resultado completo: {json_path}")
            
            return True
            
        elif result.get('status') == 'rejected':
            print(f"\n     IMAGEN RECHAZADA")
            print(f"      Razón: {result.get('message')}")
            return False
            
        elif result.get('status') == 'low_confidence':
            print(f"\n     CONFIANZA BAJA")
            print(f"      Razón: {result.get('message')}")
            return False
            
        else:
            print(f"\n    Estado desconocido: {result.get('status')}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"    Timeout - La inferencia tomó más de 60s")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    base_url = "http://localhost:8000"
    
    print("="*70)
    print(" BONE ANALYSIS API - TEST SUITE")
    print("="*70)
    
    # Test 1: Root endpoint
    if not test_root(base_url):
        print("\n API no está respondiendo. ¿Está el servidor corriendo?")
        sys.exit(1)
    
    # Test 2: Health check
    if not test_health(base_url):
        print("\n Modelos no están cargados. Espera unos minutos y vuelve a intentar.")
        sys.exit(1)
    
    # Test 3: Análisis (si se proporciona imagen)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        success = test_analyze(base_url, image_path)
        
        if success:
            print("\n" + "="*70)
            print(" TODAS LAS PRUEBAS PASARON")
            print("="*70)
        else:
            print("\n" + "="*70)
            print(" ALGUNAS PRUEBAS FALLARON")
            print("="*70)
            sys.exit(1)
    else:
        print("\n" + "="*70)
        print(" TESTS BÁSICOS PASARON")
        print("="*70)
        print("\nPara probar análisis de imagen:")
        print(f"   python {sys.argv[0]} <ruta_imagen.jpg>")


if __name__ == "__main__":
    main()