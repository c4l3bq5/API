import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
from config import Config
from utils.fracture_analysis import analyze_fractures_advanced, generate_annotated_image


class BoneAnalyzer:
    def __init__(self, model_ternary, model_upper, model_lower):
        self.model_ternary = model_ternary
        self.model_upper = model_upper
        self.model_lower = model_lower

    def preprocess_image(self, image_bytes, target_size):
        """Preprocesa imagen para los modelos"""
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def classify_region(self, image_bytes):
        """Clasifica si es upper/lower/neither"""
        img_tensor = self.preprocess_image(image_bytes, Config.IMG_SIZE_CLASSIFIER)
        
        predictions = self.model_ternary.predict(img_tensor, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        region_map = {0: 'upper', 1: 'lower', 2: 'neither'}
        region = region_map[class_idx]
        
        return {
            'region': region,
            'confidence': confidence,
            'probabilities': {
                'upper': float(predictions[0][0]),
                'lower': float(predictions[0][1]),
                'neither': float(predictions[0][2])
            }
        }

    def segment_bones(self, image_bytes, region):
        """Segmenta huesos según región"""
        model = self.model_upper if region == 'upper' else self.model_lower
        num_classes = Config.NUM_CLASSES_UPPER if region == 'upper' else Config.NUM_CLASSES_LOWER
        
        img_tensor = self.preprocess_image(image_bytes, Config.IMG_SIZE_SEGNET)
        
        predictions = model.predict(img_tensor, verbose=0)
        pred_mask = np.argmax(predictions[0], axis=-1)
        
        return pred_mask, predictions[0]

    def extract_detected_bones(self, pred_mask, region):
        """Extrae información de huesos detectados"""
        bone_names = Config.BONE_NAMES_SPANISH_UPPER if region == 'upper' else Config.BONE_NAMES_SPANISH_LOWER
        
        unique_classes = np.unique(pred_mask)
        detected_bones = []
        
        total_pixels = pred_mask.size
        
        for class_id in unique_classes:
            if class_id == 0 or class_id == 2:  # Skip background y fondo
                continue
                
            class_mask = (pred_mask == class_id)
            class_pixels = np.sum(class_mask)
            coverage = (class_pixels / total_pixels) * 100
            
            if coverage > Config.FRACTURE_THRESHOLD:
                detected_bones.append({
                    'id': int(class_id),
                    'name': bone_names.get(int(class_id), f'Clase {class_id}'),
                    'coverage': float(coverage)
                })
        
        # Ordenar por cobertura descendente
        detected_bones.sort(key=lambda x: x['coverage'], reverse=True)
        
        return detected_bones

    def analyze_image(self, image_bytes):
        """Pipeline completo de análisis"""
        try:
            # Paso 1: Clasificar región
            region_result = self.classify_region(image_bytes)
            
            if region_result['region'] == 'neither':
                return {
                    'status': 'rejected',
                    'message': 'La imagen no corresponde a una radiografía de extremidades válida',
                    'region_analysis': region_result
                }
            
            if region_result['confidence'] < Config.CONFIDENCE_THRESHOLD:
                return {
                    'status': 'low_confidence',
                    'message': f'Confianza baja en clasificación ({region_result["confidence"]:.2%})',
                    'region_analysis': region_result
                }
            
            # Paso 2: Segmentar huesos
            pred_mask, predictions = self.segment_bones(image_bytes, region_result['region'])
            
            # Paso 3: Extraer huesos detectados
            detected_bones = self.extract_detected_bones(pred_mask, region_result['region'])
            
            # Paso 4: Análisis de fracturas
            fracture_analysis = analyze_fractures_advanced(
                pred_mask, 
                detected_bones, 
                region_result['region']
            )
            
            # Paso 5: Generar imagen anotada
            annotated_image = generate_annotated_image(
                image_bytes,
                pred_mask,
                fracture_analysis['fractures'],
                detected_bones,
                region_result['region']
            )
            
            return {
                'status': 'success',
                'region_analysis': region_result,
                'detected_bones': detected_bones,
                'fracture_analysis': fracture_analysis,
                'annotated_image': annotated_image,
                'metadata': {
                    'total_bones_detected': len(detected_bones),
                    'segmentation_shape': pred_mask.shape
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error durante el análisis: {str(e)}'
            }