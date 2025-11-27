#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES PAR TRAITEMENT D'IMAGE - V2
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

Version améliorée avec détection robuste des couleurs basée sur
l'analyse HSV et le filtrage morphologique.

Techniques de traitement de signal utilisées:
- Filtrage bilatéral (préservation des bords)
- Morphologie mathématique (ouverture, fermeture)
- Segmentation couleur en espace HSV
- Analyse de profil vertical

Temps de traitement: < 1 seconde
=====================================================================
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import sys
import glob


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class DetectedBand:
    """Représente une bande de couleur détectée"""
    color: str
    value: int
    y_position: int
    area: int
    confidence: float


@dataclass
class ResistanceResult:
    """Résultat de l'analyse"""
    value_ohms: float
    formatted: str
    tolerance: str
    bands: List[str]
    processing_time: float


# =============================================================================
# CONFIGURATION DES COULEURS HSV
# =============================================================================

# Configuration des couleurs avec plages HSV optimisées
# Format: 'nom': {'lower': [H,S,V], 'upper': [H,S,V], 'value': int}
# H: 0-180, S: 0-255, V: 0-255

COLOR_CONFIG = {
    'black': {
        'lower': [0, 0, 0],
        'upper': [180, 255, 50],
        'value': 0,
        'priority': 1
    },
    'brown': {
        'lower': [0, 80, 30],
        'upper': [15, 200, 100],
        'value': 1,
        'priority': 2
    },
    'red_low': {
        'lower': [0, 120, 70],
        'upper': [8, 255, 255],
        'value': 2,
        'priority': 10  # Haute priorité pour le rouge
    },
    'red_high': {
        'lower': [172, 120, 70],
        'upper': [180, 255, 255],
        'value': 2,
        'priority': 10
    },
    'orange': {
        'lower': [8, 150, 170],
        'upper': [20, 255, 255],
        'value': 3,
        'priority': 5
    },
    'yellow': {
        'lower': [22, 120, 170],
        'upper': [35, 255, 255],
        'value': 4,
        'priority': 5
    },
    'green': {
        'lower': [40, 60, 60],
        'upper': [80, 255, 255],
        'value': 5,
        'priority': 5
    },
    'blue': {
        'lower': [90, 60, 60],
        'upper': [125, 255, 255],
        'value': 6,
        'priority': 5
    },
    'violet': {
        'lower': [125, 50, 50],
        'upper': [160, 255, 255],
        'value': 7,
        'priority': 5
    },
    'gray': {
        'lower': [0, 0, 80],
        'upper': [180, 40, 180],
        'value': 8,
        'priority': 1
    },
    'white': {
        'lower': [0, 0, 220],
        'upper': [180, 30, 255],
        'value': 9,
        'priority': 1
    },
    'gold': {
        'lower': [18, 100, 100],  # Or: saturation moyenne-haute, différent du beige
        'upper': [30, 220, 200],
        'value': -1,
        'priority': 3
    },
    'silver': {
        'lower': [0, 0, 150],
        'upper': [180, 35, 210],
        'value': -2,
        'priority': 1
    }
}

TOLERANCE_MAP = {
    'brown': '±1%',
    'red': '±2%',
    'green': '±0.5%',
    'blue': '±0.25%',
    'violet': '±0.1%',
    'gray': '±0.05%',
    'gold': '±5%',
    'silver': '±10%',
    'none': '±20%'
}


# =============================================================================
# FONCTIONS DE PRÉTRAITEMENT
# =============================================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Applique le prétraitement pour améliorer la détection des couleurs.

    Pipeline:
    1. Amélioration du contraste
    2. Filtre bilatéral (lisse le bruit, préserve les bords)
    """
    # Amélioration légère du contraste
    enhanced = cv2.convertScaleAbs(image, alpha=1.15, beta=10)

    # Filtre bilatéral
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered


def find_resistor_body(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Détecte automatiquement le corps de la résistance dans l'image.

    Returns:
        Tuple (x, y, w, h) ou None si non trouvé
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Le corps de la résistance a généralement:
    # - Une couleur beige/crème (H: 15-25, S: 30-100, V: 130-220)
    # - Ou des bandes colorées (S > 80)

    # Masque pour le beige + couleurs saturées
    beige_mask = cv2.inRange(hsv, np.array([10, 25, 110]), np.array([30, 130, 235]))
    colored_mask = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([180, 255, 255]))

    combined = cv2.bitwise_or(beige_mask, colored_mask)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

    # Trouver les contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Prendre le plus grand contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Vérifier que c'est assez grand
    img_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(largest) < img_area * 0.005:
        return None

    return (x, y, w, h)


# =============================================================================
# DÉTECTION DES BANDES
# =============================================================================

def create_resistor_mask(roi_hsv: np.ndarray) -> np.ndarray:
    """
    Crée un masque pour isoler les bandes de la résistance
    en excluant le corps beige et le fond blanc.
    """
    h, s, v = cv2.split(roi_hsv)

    # Exclure le fond blanc (S < 30 et V > 200)
    white_mask = cv2.inRange(roi_hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))

    # Exclure le corps beige (S: 30-90, V: 150-230, H: 15-25)
    beige_mask = cv2.inRange(roi_hsv, np.array([12, 30, 150]), np.array([28, 100, 235]))

    # Zone utile = tout sauf blanc et beige
    excluded = cv2.bitwise_or(white_mask, beige_mask)
    useful = cv2.bitwise_not(excluded)

    # Nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    useful = cv2.morphologyEx(useful, cv2.MORPH_OPEN, kernel)

    return useful


def detect_color_bands(roi: np.ndarray, roi_hsv: np.ndarray) -> List[DetectedBand]:
    """
    Détecte les bandes de couleur dans la ROI.

    Utilise une approche par couleur: on cherche chaque couleur séparément
    puis on fusionne les résultats.
    """
    roi_h, roi_w = roi.shape[:2]
    detected = []

    # Créer le masque pour exclure le corps et le fond
    useful_mask = create_resistor_mask(roi_hsv)

    for color_name, config in COLOR_CONFIG.items():
        lower = np.array(config['lower'])
        upper = np.array(config['upper'])
        value = config['value']
        priority = config['priority']

        # Créer le masque pour cette couleur
        color_mask = cv2.inRange(roi_hsv, lower, upper)

        # Appliquer le masque des zones utiles (exclure beige/blanc)
        # Seulement pour les couleurs qui peuvent être confondues
        if color_name in ['gold', 'brown', 'orange']:
            color_mask = cv2.bitwise_and(color_mask, useful_mask)

        # Morphologie
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Trouver les contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filtres de taille
            if area < 60:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Une bande doit être assez haute et pas trop large
            if h < roi_h * 0.05 or w > roi_w * 0.5:
                continue

            # Calculer le centre Y
            M = cv2.moments(cnt)
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else y + h // 2

            # Calculer la confiance basée sur la taille et la forme
            shape_score = min(1.0, h / (w + 1))  # Préférer les formes verticales
            size_score = min(1.0, area / 500)
            confidence = (shape_score + size_score) / 2 * (priority / 10)

            # Nettoyer le nom de couleur (enlever _low, _high)
            clean_name = color_name.replace('_low', '').replace('_high', '')

            detected.append(DetectedBand(
                color=clean_name,
                value=value,
                y_position=cy,
                area=area,
                confidence=confidence
            ))

    return detected


def merge_duplicate_bands(bands: List[DetectedBand], min_distance: int = 15) -> List[DetectedBand]:
    """
    Fusionne les bandes détectées qui sont au même endroit.
    Garde celle avec la meilleure confiance ou la plus grande aire.
    """
    if not bands:
        return []

    # Trier par position Y
    bands = sorted(bands, key=lambda b: b.y_position)

    merged = []
    for band in bands:
        # Chercher si une bande similaire existe déjà
        found = False
        for i, existing in enumerate(merged):
            if abs(band.y_position - existing.y_position) < min_distance:
                # Même position: garder la meilleure
                if band.confidence > existing.confidence:
                    merged[i] = band
                elif band.confidence == existing.confidence and band.area > existing.area:
                    merged[i] = band
                found = True
                break

        if not found:
            merged.append(band)

    return sorted(merged, key=lambda b: b.y_position)


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def correct_orientation(bands: List[DetectedBand]) -> List[DetectedBand]:
    """
    Corrige l'orientation: or/argent doit être à la fin (tolérance).
    """
    if len(bands) < 3:
        return bands

    first_val = bands[0].value
    last_val = bands[-1].value

    # Si or/argent au début mais pas à la fin -> inverser
    if first_val < 0 and last_val >= 0:
        return list(reversed(bands))

    return bands


def calculate_resistance(bands: List[DetectedBand]) -> Optional[ResistanceResult]:
    """
    Calcule la valeur de résistance à partir des bandes.
    """
    if len(bands) < 3:
        return None

    # Corriger l'orientation
    bands = correct_orientation(bands)

    # Déterminer la tolérance
    tolerance = 'none'
    working_bands = bands

    if len(bands) >= 4 and bands[-1].value < 0:
        tolerance = bands[-1].color
        working_bands = bands[:-1]

    if len(working_bands) < 3:
        return None

    # Extraire les valeurs
    d1 = working_bands[0].value
    d2 = working_bands[1].value
    mult_val = working_bands[2].value

    # Vérifier la validité
    if d1 < 0 or d2 < 0:
        return None

    # Calculer le multiplicateur
    if mult_val >= 0:
        multiplier = 10 ** mult_val
    elif mult_val == -1:  # Gold
        multiplier = 0.1
    elif mult_val == -2:  # Silver
        multiplier = 0.01
    else:
        return None

    # Calculer la résistance
    base_value = d1 * 10 + d2
    resistance = base_value * multiplier

    # Formater
    if resistance >= 1_000_000:
        formatted = f"{resistance / 1_000_000:.2f} MΩ"
    elif resistance >= 1_000:
        formatted = f"{resistance / 1_000:.2f} kΩ"
    else:
        formatted = f"{resistance:.1f} Ω"

    tolerance_str = TOLERANCE_MAP.get(tolerance, '±20%')

    band_names = [b.color for b in bands]

    return ResistanceResult(
        value_ohms=resistance,
        formatted=formatted,
        tolerance=tolerance_str,
        bands=band_names,
        processing_time=0
    )


# =============================================================================
# INTERFACE UTILISATEUR
# =============================================================================

def interactive_roi_selection(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Permet à l'utilisateur de sélectionner manuellement la ROI.
    """
    roi_selected = False
    drawing = False
    start_point = (0, 0)
    end_point = (0, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, roi_selected

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            roi_selected = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            roi_selected = True

    window_name = 'Selectionnez la resistance - ENTREE=valider, R=reset, ESC=quitter'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 60)
    print("  MODE INTERACTIF")
    print("=" * 60)
    print("  1. Tracez un rectangle serré autour de la résistance")
    print("  2. Appuyez sur ENTRÉE pour valider")
    print("  3. Appuyez sur R pour recommencer")
    print("  4. Appuyez sur ESC pour quitter")
    print("=" * 60)

    while True:
        display = image.copy()

        if drawing or roi_selected:
            cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and roi_selected:  # ENTRÉE
            break
        elif key == ord('r'):
            roi_selected = False
            drawing = False
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyWindow(window_name)

    x1 = min(start_point[0], end_point[0])
    y1 = min(start_point[1], end_point[1])
    x2 = max(start_point[0], end_point[0])
    y2 = max(start_point[1], end_point[1])

    if x2 - x1 < 20 or y2 - y1 < 10:
        print("❌ Zone trop petite")
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def display_result(image: np.ndarray, roi_coords: Tuple[int, int, int, int],
                   result: ResistanceResult, bands: List[DetectedBand]):
    """
    Affiche le résultat de l'analyse.
    """
    output = image.copy()
    x, y, w, h = roi_coords

    # Dessiner le rectangle
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher la valeur
    cv2.rectangle(output, (x, y - 40), (x + w + 100, y), (0, 0, 0), -1)
    cv2.putText(output, f"{result.formatted} {result.tolerance}",
                (x + 5, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Afficher les bandes détectées
    bands_text = " > ".join(result.bands)
    cv2.putText(output, f"Bandes: {bands_text}",
                (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print("\n" + "=" * 60)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 60)
    print(f"\n  📊 Bandes: {bands_text}")
    print(f"  ⏱️  Temps: {result.processing_time * 1000:.1f} ms")
    print("=" * 60)

    cv2.imshow('RESULTAT', output)
    print("\n💡 Appuyez sur une touche pour fermer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# FONCTION PRINCIPALE D'ANALYSE
# =============================================================================

def analyze_resistance_image(image_path: str, interactive: bool = True) -> Optional[ResistanceResult]:
    """
    Analyse une image de résistance et retourne la valeur.
    """
    start_time = time.time()

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur: Impossible de charger '{image_path}'")
        return None

    print(f"📁 Image chargée: {image.shape}")

    # Redimensionner
    height, width = image.shape[:2]
    target_width = 800
    scale = target_width / width
    image = cv2.resize(image, (target_width, int(height * scale)))

    # Prétraitement
    preprocessed = preprocess_image(image)

    # Obtenir la ROI
    if interactive:
        roi_coords = interactive_roi_selection(preprocessed)
    else:
        # Détection automatique
        roi_coords = find_resistor_body(preprocessed)

    if roi_coords is None:
        print("❌ Impossible de localiser la résistance")
        return None

    x, y, w, h = roi_coords
    roi = preprocessed[y:y + h, x:x + w]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    print(f"📐 ROI: {roi.shape}")

    # Détecter les bandes
    bands = detect_color_bands(roi, roi_hsv)
    print(f"\n📊 Bandes brutes détectées: {len(bands)}")

    # Fusionner les doublons
    bands = merge_duplicate_bands(bands)
    print(f"📊 Bandes après fusion: {len(bands)}")

    for i, b in enumerate(bands):
        print(f"   {i + 1}. {b.color.upper():8s} (val={b.value:2d}, y={b.y_position:3d})")

    # Calculer la résistance
    result = calculate_resistance(bands)

    if result:
        result.processing_time = time.time() - start_time

        # Afficher
        display_result(image, roi_coords, result, bands)

        return result
    else:
        print(f"\n❌ Impossible de calculer la valeur ({len(bands)} bandes)")
        return None


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def find_image_file() -> Optional[str]:
    """Recherche automatiquement une image de résistance."""
    patterns = [
        "resistance*.jpg", "resistance*.png",
        "résistance*.jpg", "résistance*.png",
        "Resistance*.jpg", "Resistance*.png",
    ]

    for pattern in patterns:
        try:
            files = glob.glob(pattern)
            if files:
                return files[0]
        except:
            pass

    return None


def main():
    print("\n" + "=" * 60)
    print("  ANALYSEUR DE RÉSISTANCES - V2")
    print("  Projet de Traitement de Signal - EPHEC 2025")
    print("=" * 60)

    # Obtenir le chemin de l'image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = find_image_file()
        if image_path is None:
            print("\n❌ Aucune image trouvée.")
            print("   Usage: python resistance_analyzer_v2.py [image.jpg]")
            return None

    print(f"\n📁 Image: {image_path}")

    result = analyze_resistance_image(image_path, interactive=True)

    if result:
        print(f"\n✅ Analyse terminée: {result.formatted}")
    else:
        print("\n❌ Échec de l'analyse")

    return result


if __name__ == "__main__":
    main()