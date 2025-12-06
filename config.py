class Config:
    HF_MODEL_TERNARY = "c4l3bq5/bone-ternary-classifier"
    HF_MODEL_UPPER = "c4l3bq5/bone-segnet-upper"
    HF_MODEL_LOWER = "c4l3bq5/bone-segnet-lower"

    HF_FILE_TERNARY = "bone-ternary-classifier-v3.0.keras"
    HF_FILE_UPPER = "bone-segnet-upper-v3.0.keras"
    HF_FILE_LOWER = "bone-segnet-lower-v3.0.keras"

    IMG_SIZE_CLASSIFIER = (224, 224)
    IMG_SIZE_SEGNET = (128, 128)

    CONFIDENCE_THRESHOLD = 0.70
    FRACTURE_THRESHOLD = 0.05

    NUM_CLASSES_UPPER = 43
    NUM_CLASSES_LOWER = 38
    NUM_CLASSES_TERNARY = 3

    BONE_NAMES_SPANISH_UPPER = {
    0: 'Región genérica',
    1: 'Brazo completo',
    2: 'Fondo',
    3: 'Cúbito',
    4: 'Articulación del codo',
    5: 'Epífisis',
    6: 'Escafoides',
    7: 'Falange distal del anular',
    8: 'Falange distal del índice',
    9: 'Falange distal del medio',
    10: 'Falange distal del meñique',
    11: 'Falange distal del pulgar',
    12: 'Falange media del anular',
    13: 'Falange media del índice',
    14: 'Falange media del medio',
    15: 'Falange media del meñique',
    16: 'Falange proximal del anular',
    17: 'Falange proximal del índice',
    18: 'Falange proximal del medio',
    19: 'Falange proximal del meñique',
    20: 'Falange proximal del pulgar',
    21: 'Ganchoso',
    22: 'Hueso grande',
    23: 'Mano completa',
    24: 'Húmero',
    25: 'Línea de fractura',
    26: 'Luxación cúbito - Barton',
    27: 'Luxación cúbito - Galeazzi',
    28: 'Metacarpiano del anular (4to)',
    29: 'Metacarpiano del índice (2do)',
    30: 'Metacarpiano del medio (3ro)',
    31: 'Metacarpiano del meñique (5to)',
    32: 'Metacarpiano del pulgar (1ro)',
    33: 'Piramidal',
    34: 'Pisiforme',
    35: 'Radio',
    36: 'Fractura Barton del radio',
    37: 'Fractura Colles del radio',
    38: 'Fractura Galeazzi del radio',
    39: 'Fractura Smith del radio',
    40: 'Semilunar',
    41: 'Trapecio',
    42: 'Trapezoide'
}

    BONE_NAMES_SPANISH_LOWER = {
    0: 'Región genérica',
    1: 'Astrágalo',
    2: 'Fondo',
    3: 'Calcáneo',
    4: 'Cúbito',
    5: 'Cuboides',
    6: 'Cuneiforme A',
    7: 'Cuneiforme B',
    8: 'Cuneiforme C',
    9: 'Epífisis',
    10: 'Escafoides del pie',
    11: 'Falange distal del 4to dedo',
    12: 'Falange distal del Hallux (1er dedo)',
    13: 'Falange distal del meñique del pie (5to dedo)',
    14: 'Falange distal del 2do dedo',
    15: 'Falange distal del 3er dedo',
    16: 'Falange media del 4to dedo',
    17: 'Falange media del meñique del pie (5to dedo)',
    18: 'Falange media del 2do dedo',
    19: 'Falange media del 3er dedo',
    20: 'Falange proximal del 4to dedo',
    21: 'Falange proximal del Hallux (1er dedo)',
    22: 'Falange proximal del meñique del pie (5to dedo)',
    23: 'Falange proximal del 2do dedo',
    24: 'Falange proximal del 3er dedo',
    25: 'Fémur',
    26: 'Línea de fractura',
    27: 'Metatarso del 4to dedo',
    28: 'Metatarso del Hallux (1er dedo)',
    29: 'Metatarso del meñique del pie (5to dedo)',
    30: 'Metatarso del 2do dedo',
    31: 'Metatarso del 3er dedo',
    32: 'Osgood-Schlatter (tibia)',
    33: 'Peroné',
    34: 'Rótula',
    35: 'Tibia',
    36: 'Tibia con fractura',
    37: 'Tibia Tillaux (fractura)'
}

# Fracturas específicas a detectar
FRACTURE_TYPES = {
    'tillaux': {'region': 'lower', 'bone': 'tibia', 'age_group': 'adolescent'},
    'galeazzi': {'region': 'upper', 'bone': 'radio', 'severity': 'high'},
    'colles': {'region': 'upper', 'bone': 'radio', 'severity': 'moderate'},
    'smith': {'region': 'upper', 'bone': 'radio', 'severity': 'moderate'},
    'barton': {'region': 'upper', 'bone': 'radio', 'severity': 'high'},
    'osgood_schlatter': {'region': 'lower', 'bone': 'tibia', 'age_group': 'adolescent'},
    'epifisiolisis': {'region': 'both', 'bone': 'epifisis', 'age_group': 'adolescent'}
}