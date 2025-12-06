import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image as PILImage, ImageDraw, ImageFont
from config import Config


def analyze_fractures_advanced(pred_mask, detected_bones, region):
    """
    Análisis avanzado con soporte para:
    - ALERTAS (Epifisiolisis)
    - ENFERMEDADES (Osgood-Schlatter)
    - FRACTURAS
    - LUXACIONES
    """
    # Convertir mask a numpy si viene como lista
    if isinstance(pred_mask, list):
        pred_mask = np.array(pred_mask)

    bone_names = Config.BONE_NAMES_SPANISH_UPPER if region == 'upper' else Config.BONE_NAMES_SPANISH_LOWER

    # Keywords por categoría
    fracture_keywords = ['fractura', 'linea', 'colles', 'smith',
                         'galeazzi', 'barton', 'tillaux']
    luxation_keywords = ['luxacion', 'luxación', 'dislocation']
    sickness_keywords = ['osgood']
    alert_keywords = ['epifisiolisis', 'epífisis']

    fractured_analysis = []

    # Separar por categorías (PRIORIDAD: alerta > enfermedad > luxación > fractura)
    fracture_detections = []
    luxation_detections = []
    sickness_detections = []
    alert_detections = []
    bone_detections = []

    for bone_info in detected_bones:
        class_id = bone_info['id']
        bone_name = bone_info['name']
        percentage = bone_info['coverage']
        bone_lower = bone_name.lower()

        # Clasificar por prioridad
        if any(kw in bone_lower for kw in alert_keywords):
            alert_detections.append((class_id, bone_name, percentage))
        elif any(kw in bone_lower for kw in sickness_keywords):
            sickness_detections.append((class_id, bone_name, percentage))
        elif any(kw in bone_lower for kw in luxation_keywords):
            luxation_detections.append((class_id, bone_name, percentage))
        elif any(kw in bone_lower for kw in fracture_keywords):
            fracture_detections.append((class_id, bone_name, percentage))
        else:
            bone_detections.append((class_id, bone_name, percentage))

    # ============ CASO 1: ALERTAS CRÍTICAS ============
    for alert_id, alert_name, alert_pct in alert_detections:
        alert_mask = (pred_mask == alert_id).astype(np.uint8)
        contours, _ = cv2.findContours(alert_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)

            fractured_analysis.append({
                'type': 'ALERT',
                'source': 'SEGNET_DIRECT',
                'alert_class_id': int(alert_id),
                'alert_name': alert_name,
                'affected_bone': 'Epífisis',
                'description': 'ALERTA CRÍTICA: Epifisiolisis detectada',
                'fracture_bbox': [int(x), int(y), int(w), int(h)],
                'fracture_type': {
                    'classification': 'Epifisiolisis',
                    'mechanism': 'Deslizamiento de la placa de crecimiento',
                    'deformity': 'Desplazamiento epifisario',
                    'management': ' URGENCIA ORTOPÉDICA - Riesgo de cierre prematuro de fisis',
                    'stability': 'muy_inestable',
                    'is_major': True,
                    'age_group': 'Adolescente (10-16 años)',
                    'severity': 'CRÍTICO'
                },
                'measurements': {
                    'coverage_percentage': float(alert_pct)
                }
            })

    # ============ CASO 2: ENFERMEDADES ============
    for sick_id, sick_name, sick_pct in sickness_detections:
        sick_mask = (pred_mask == sick_id).astype(np.uint8)
        contours, _ = cv2.findContours(sick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)

            fractured_analysis.append({
                'type': 'SICKNESS',
                'source': 'SEGNET_DIRECT',
                'sickness_class_id': int(sick_id),
                'sickness_name': sick_name,
                'affected_bone': 'Tuberosidad tibial anterior',
                'description': 'Enfermedad de Osgood-Schlatter detectada',
                'fracture_bbox': [int(x), int(y), int(w), int(h)],
                'fracture_type': {
                    'classification': 'Enfermedad de Osgood-Schlatter',
                    'mechanism': 'Tracción repetitiva del tendón rotuliano',
                    'deformity': 'Inflamación de la tuberosidad tibial',
                    'management': 'Reposo deportivo + Fisioterapia + AINEs. Autolimitada.',
                    'stability': 'no_aplica',
                    'is_major': False,
                    'age_group': 'Adolescente activo (10-15 años)',
                    'severity': 'LEVE-MODERADO'
                },
                'measurements': {
                    'coverage_percentage': float(sick_pct)
                }
            })

    # ============ CASO 3: FRACTURAS ============
    for frac_id, frac_name, frac_pct in fracture_detections:
        fracture_mask = (pred_mask == frac_id).astype(np.uint8)
        contours, _ = cv2.findContours(fracture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            arc_length = cv2.arcLength(main_contour, False)
            area = cv2.contourArea(main_contour)
            x, y, w, h = cv2.boundingRect(main_contour)

            fracture_info = classify_fracture_type(frac_name, arc_length, area)

            fractured_analysis.append({
                'type': 'FRACTURE',
                'source': 'SEGNET_DIRECT',
                'fracture_class_id': int(frac_id),
                'fracture_name': frac_name,
                'affected_bone': fracture_info['affected_bone'],
                'description': fracture_info['classification'],
                'fracture_type': fracture_info,
                'fracture_bbox': [int(x), int(y), int(w), int(h)],
                'measurements': {
                    'length_mm': float(arc_length * 0.2),
                    'area_mm2': float(area * 0.04),
                    'coverage_percentage': float(frac_pct)
                }
            })

    # ============ CASO 4: LUXACIONES ============
    for lux_id, lux_name, lux_pct in luxation_detections:
        luxation_mask = (pred_mask == lux_id).astype(np.uint8)
        contours, _ = cv2.findContours(luxation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)

            fractured_analysis.append({
                'type': 'LUXATION',
                'source': 'SEGNET_DIRECT',
                'luxation_class_id': int(lux_id),
                'luxation_name': lux_name,
                'affected_bone': lux_name.split('-')[0].strip() if '-' in lux_name else lux_name,
                'description': f'Luxación: {lux_name}',
                'fracture_bbox': [int(x), int(y), int(w), int(h)],
                'fracture_type': {
                    'classification': 'Luxación',
                    'management': 'Reducción urgente + inmovilización',
                    'stability': 'inestable',
                    'is_major': True
                }
            })

    # ============ CASO 5: MICROFRACTURAS (HEURÍSTICO) ============
    for bone_id, bone_name, bone_pct in bone_detections:
        bone_mask = (pred_mask == bone_id).astype(np.uint8)

        if np.sum(bone_mask) < 100:
            continue

        try:
            # Detectar discontinuidades con operaciones morfológicas
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(bone_mask, kernel, iterations=2)

            # Contar regiones conectadas
            num_labels, labels = cv2.connectedComponents(eroded)

            if num_labels > 2:  # > 2 porque background es 0
                component_centroids = []
                for i in range(1, num_labels):
                    component_mask = (labels == i).astype(np.uint8)
                    moments = cv2.moments(component_mask)
                    if moments['m00'] > 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        component_centroids.append((cx, cy))

                if len(component_centroids) >= 2:
                    min_dist = float('inf')
                    gap_x, gap_y = 0, 0

                    for i in range(len(component_centroids)):
                        for j in range(i + 1, len(component_centroids)):
                            dist = np.sqrt(
                                (component_centroids[i][0] - component_centroids[j][0]) ** 2 +
                                (component_centroids[i][1] - component_centroids[j][1]) ** 2
                            )
                            if dist < min_dist:
                                min_dist = dist
                                gap_x = (component_centroids[i][0] + component_centroids[j][0]) // 2
                                gap_y = (component_centroids[i][1] + component_centroids[j][1]) // 2

                    # Microfractura
                    if 5 < min_dist < 30:
                        padding = 20

                        fractured_analysis.append({
                            'type': 'FRACTURE',
                            'source': 'HEURISTIC_MICROFRACTURE',
                            'affected_bone': bone_name,
                            'affected_bone_id': int(bone_id),
                            'description': f'Microfractura sospechada en {bone_name}',
                            'fracture_bbox': [
                                max(0, int(gap_x - padding)),
                                max(0, int(gap_y - padding)),
                                padding * 2,
                                padding * 2
                            ],
                            'fracture_type': {
                                'classification': 'Microfractura (sospecha)',
                                'management': 'TC de alta resolución recomendada',
                                'stability': 'probablemente_estable',
                                'is_major': False
                            },
                            'measurements': {
                                'gap_distance_mm': float(min_dist * 0.2),
                                'num_fragments': int(num_labels - 1)
                            },
                            'confidence': 'media'
                        })
        except Exception:
            pass

    # Calcular resumen por tipo
    alert_count = len([f for f in fractured_analysis if f.get('type') == 'ALERT'])
    sickness_count = len([f for f in fractured_analysis if f.get('type') == 'SICKNESS'])
    major_count = len([f for f in fractured_analysis if f.get('type') == 'FRACTURE' and f.get('fracture_type', {}).get('is_major', True)])
    micro_count = len([f for f in fractured_analysis if f.get('type') == 'FRACTURE' and not f.get('fracture_type', {}).get('is_major', True)])
    luxation_count = len([f for f in fractured_analysis if f.get('type') == 'LUXATION'])

    return {
        'total_fractures': len(fractured_analysis),
        'alert_count': alert_count,
        'sickness_count': sickness_count,
        'major_fractures': major_count,
        'microfractures': micro_count,
        'luxation_count': luxation_count,
        'fractures': fractured_analysis,
        'requires_immediate_attention': alert_count > 0 or major_count > 0
    }


def classify_fracture_type(fracture_name, arc_length, area):
    """Clasificación detallada del tipo de fractura"""
    frac_lower = fracture_name.lower()

    if 'colles' in frac_lower:
        return {
            'classification': 'Fractura de Colles',
            'affected_bone': 'Radio',
            'mechanism': 'Caída con mano extendida',
            'deformity': 'Dorso de tenedor',
            'management': 'Reducción cerrada + yeso 4-6 semanas',
            'stability': 'estable' if arc_length < 40 else 'inestable',
            'is_major': True
        }
    elif 'smith' in frac_lower:
        return {
            'classification': 'Fractura de Smith',
            'affected_bone': 'Radio',
            'mechanism': 'Caída sobre dorso de mano',
            'deformity': 'Pala de jardín',
            'management': 'Frecuentemente requiere fijación quirúrgica',
            'stability': 'inestable',
            'is_major': True
        }
    elif 'galeazzi' in frac_lower:
        return {
            'classification': 'Fractura de Galeazzi',
            'affected_bone': 'Radio + Cúbito',
            'mechanism': 'Fractura radio + luxación cubital',
            'deformity': 'Desplazamiento del cúbito distal',
            'management': ' QUIRÚRGICO - RAFI obligatoria',
            'stability': 'inestable',
            'is_major': True
        }
    elif 'barton' in frac_lower:
        return {
            'classification': 'Fractura de Barton',
            'affected_bone': 'Radio',
            'mechanism': 'Fractura-luxación intrarticular',
            'deformity': 'Subluxación carpiana',
            'management': ' QUIRÚRGICO - Placa volar',
            'stability': 'inestable',
            'is_major': True
        }
    elif 'tillaux' in frac_lower:
        return {
            'classification': 'Fractura de Tillaux',
            'affected_bone': 'Tibia',
            'mechanism': 'Avulsión epífisis tibial (adolescente)',
            'deformity': 'Fractura vertical epífisis',
            'management': 'Si >2mm desplazamiento → cirugía',
            'stability': 'variable',
            'is_major': True
        }
    else:
        return {
            'classification': 'Línea de fractura',
            'affected_bone': fracture_name,
            'mechanism': 'Indeterminado',
            'management': 'Evaluación clínica completa',
            'stability': 'indeterminado',
            'is_major': True
        }


def generate_annotated_image(original_image_bytes, pred_mask, fractured_analysis, detected_bones, region):
    """
    Genera imagen con anotaciones de colores por tipo
    """
    try:
        img = PILImage.open(BytesIO(original_image_bytes)).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Colores por tipo
        ALERT_COLOR = (255, 0, 255)       # Magenta para alertas
        SICKNESS_COLOR = (255, 128, 0)    # Naranja para enfermedades
        FRACTURE_COLOR = (255, 0, 0)      # Rojo para fracturas
        LUXATION_COLOR = (255, 255, 0)    # Amarillo para luxaciones
        MICRO_COLOR = (255, 165, 0)       # Naranja claro para microfracturas

        line_width = max(3, img.width // 300)

        for item in fractured_analysis:
            if 'fracture_bbox' not in item:
                continue

            x, y, w, h = item['fracture_bbox']

            # Determinar color y etiqueta según tipo
            item_type = item.get('type', 'FRACTURE')

            if item_type == 'ALERT':
                color = ALERT_COLOR
                label = " ALERTA"
            elif item_type == 'SICKNESS':
                color = SICKNESS_COLOR
                label = " ENFERMEDAD"
            elif item_type == 'LUXATION':
                color = LUXATION_COLOR
                label = " LUXACIÓN"
            elif item.get('source') == 'HEURISTIC_MICROFRACTURE':
                color = MICRO_COLOR
                label = " SOSPECHA"
            else:
                color = FRACTURE_COLOR
                label = " FRACTURA"

            # Dibujar rectángulo
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=color,
                width=line_width
            )

            # Agregar etiqueta
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                         max(16, img.width // 50))
            except:
                font = ImageFont.load_default()

            # Fondo para texto
            text_bbox = draw.textbbox((x, y - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x + 5, y - 22), label, fill=(255, 255, 255), font=font)

        # Convertir a base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return img_base64

    except Exception as e:
        print(f"Error generando imagen anotada: {e}")
        return None