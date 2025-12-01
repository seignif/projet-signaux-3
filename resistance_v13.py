#!/usr/bin/env python3
"""
=====================================================================
    ResistorReader V5 - Détection des BANDES uniquement
    Méthode: Masques HSV + Contours + Filtrage du corps
    Projet de Traitement de Signal - EPHEC 2025

    Auteur: Noah R.
=====================================================================

AMÉLIORATIONS V5:
- Détecte UNIQUEMENT les bandes de couleur (pas le corps beige)
- Utilise les contours pour trouver les vraies bandes
- Filtre le fond (corps de la résistance)
- Validation par forme (bandes = rectangles verticaux)

=====================================================================
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

MULTIPLIERS = {
    0: 1, 1: 10, 2: 100, 3: 1000, 4: 10000,
    5: 100000, 6: 1000000, 7: 10000000, 8: 100000000, 9: 1000000000,
    -1: 0.1, -2: 0.01
}

TOLERANCES = {
    1: '±1%', 2: '±2%', 5: '±0.5%', 6: '±0.25%',
    7: '±0.1%', 8: '±0.05%', -1: '±5%', -2: '±10%'
}

COLOR_VALUES = {
    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,
    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,
    'GOLD': -1, 'SILVER': -2
}

COLOR_DISPLAY = {
    'BLACK': (0, 0, 0),
    'BROWN': (0, 51, 102),
    'RED': (0, 0, 255),
    'ORANGE': (0, 128, 255),
    'YELLOW': (0, 255, 255),
    'GREEN': (0, 255, 0),
    'BLUE': (255, 0, 0),
    'VIOLET': (255, 0, 127),
    'GRAY': (128, 128, 128),
    'WHITE': (255, 255, 255),
    'GOLD': (0, 215, 255),
    'SILVER': (192, 192, 192),
}

# =============================================================================
# PLAGES HSV POUR CHAQUE COULEUR (BANDES UNIQUEMENT)
# =============================================================================

# Format: [(H_min, S_min, V_min), (H_max, S_max, V_max)]
# Ces plages sont calibrées pour détecter les BANDES, pas le corps

COLOUR_RANGES = {
    # NOIR: très sombre
    'BLACK': [(0, 0, 0), (180, 255, 65)],

    # MARRON: H rouge-orange, S élevé, V moyen
    'BROWN': [(5, 100, 50), (20, 255, 165)],

    # ROUGE: H proche de 0 ou 180, S très élevé
    'RED': [(0, 130, 80), (8, 255, 255)],
    'RED2': [(172, 130, 80), (180, 255, 255)],  # Rouge haut

    # ORANGE: H orange, S et V élevés
    'ORANGE': [(8, 150, 150), (22, 255, 255)],

    # JAUNE
    'YELLOW': [(22, 100, 150), (38, 255, 255)],

    # VERT
    'GREEN': [(38, 50, 50), (85, 255, 255)],

    # BLEU
    'BLUE': [(85, 50, 50), (130, 255, 255)],

    # VIOLET
    'VIOLET': [(130, 30, 50), (165, 255, 255)],

    # GRIS
    'GRAY': [(0, 0, 60), (180, 50, 180)],

    # BLANC
    'WHITE': [(0, 0, 200), (180, 40, 255)],

    # GOLD: S moyenne-basse, V élevé (différent du corps!)
    'GOLD': [(18, 30, 130), (32, 150, 255)],

    # ARGENT
    'SILVER': [(0, 0, 160), (180, 40, 230)],
}


# =============================================================================
# DÉTECTION DES BANDES PAR CONTOURS
# =============================================================================

def find_color_contours(hsv: np.ndarray, thresh: np.ndarray,
                        color_name: str, ranges: list,
                        min_area: int = 50) -> List[Tuple]:
    """
    Trouve les contours pour une couleur donnée.
    """
    low, high = ranges
    mask = cv2.inRange(hsv, np.array(low), np.array(high))

    # Combiner avec le seuillage pour isoler la résistance
    mask = cv2.bitwise_and(mask, thresh)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Vérifier la forme (une bande est plus haute que large)
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        ratio = w / h
        # Une bande a un ratio < 1 (plus haute que large)
        if ratio > 1.2:
            continue

        # Calculer le centre
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        valid_contours.append({
            'color': color_name,
            'contour': contour,
            'x': cx,
            'y': cy,
            'area': area,
            'bbox': (x, y, w, h)
        })

    return valid_contours


def find_bands(image: np.ndarray, min_area: int = 80) -> List[Dict]:
    """
    Trouve les bandes de couleur sur la résistance.
    """
    # Prétraitement
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Seuillage adaptatif pour isoler la résistance du fond blanc
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    thresh = cv2.bitwise_not(thresh)

    # Dilater pour connecter les zones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    all_bands = []

    # Détecter chaque couleur
    for color_name, ranges in COLOUR_RANGES.items():
        # Cas spécial pour RED (deux plages)
        if color_name == 'RED2':
            continue

        contours = find_color_contours(hsv, thresh, color_name, ranges, min_area)

        # Ajouter RED2 pour le rouge
        if color_name == 'RED':
            contours2 = find_color_contours(hsv, thresh, 'RED', COLOUR_RANGES['RED2'], min_area)
            contours.extend(contours2)

        all_bands.extend(contours)

    # Trier par position X
    all_bands.sort(key=lambda b: b['x'])

    # Éliminer les doublons (bandes trop proches)
    filtered_bands = []
    for band in all_bands:
        # Vérifier si une bande similaire existe déjà à proximité
        is_duplicate = False
        idx_to_remove = -1

        for idx, existing in enumerate(filtered_bands):
            if abs(band['x'] - existing['x']) < 25:
                # Même position, garder celle avec la plus grande aire
                if band['area'] > existing['area']:
                    idx_to_remove = idx
                else:
                    is_duplicate = True
                break

        if idx_to_remove >= 0:
            filtered_bands.pop(idx_to_remove)
            filtered_bands.append(band)
        elif not is_duplicate:
            filtered_bands.append(band)

    # Re-trier après modifications
    filtered_bands.sort(key=lambda b: b['x'])

    return filtered_bands


def identify_band_color(hsv: np.ndarray, bgr: np.ndarray,
                        x: int, y: int, w: int, h: int) -> str:
    """
    Identifie la couleur d'une bande en analysant ses pixels.
    """
    # Extraire la zone de la bande
    zone_hsv = hsv[y:y + h, x:x + w]
    zone_bgr = bgr[y:y + h, x:x + w]

    H = np.mean(zone_hsv[:, :, 0])
    S = np.mean(zone_hsv[:, :, 1])
    V = np.mean(zone_hsv[:, :, 2])
    R = np.mean(zone_bgr[:, :, 2])
    G = np.mean(zone_bgr[:, :, 1])
    B = np.mean(zone_bgr[:, :, 0])

    # Classification basée sur HSV et RGB
    ratio_rg = R / G if G > 0 else 2

    # NOIR: très sombre
    if V < 70:
        return 'BLACK'

    # BLANC: très lumineux, peu saturé
    if S < 40 and V > 200:
        return 'WHITE'

    # GRIS
    if S < 50 and 70 <= V <= 200:
        return 'GRAY'

    # ROUGE: H proche de 0 ou 180, S élevé
    if (H <= 10 or H >= 170) and S > 130 and V > 80:
        return 'RED'

    # ORANGE
    if 10 <= H <= 22 and S > 150 and V > 150:
        return 'ORANGE'

    # MARRON: H rouge-orange, S élevé, V moyen, ratio R/G élevé
    if 5 <= H <= 22 and S > 100 and 60 <= V <= 170 and ratio_rg > 1.3:
        return 'BROWN'

    # GOLD: H jaune-orange, S moyenne, V élevé, ratio R/G équilibré
    if 15 <= H <= 32 and 30 <= S <= 150 and V > 120 and ratio_rg <= 1.35:
        return 'GOLD'

    # JAUNE
    if 22 <= H <= 38 and S > 100 and V > 150:
        return 'YELLOW'

    # VERT
    if 38 <= H <= 85 and S > 50:
        return 'GREEN'

    # BLEU
    if 85 <= H <= 130 and S > 50:
        return 'BLUE'

    # VIOLET
    if 130 <= H <= 165:
        return 'VIOLET'

    # ARGENT
    if S < 50 and 150 <= V <= 230:
        return 'SILVER'

    return 'UNKNOWN'


def analyze_bands(image: np.ndarray, bands: List[Dict]) -> List[Dict]:
    """
    Analyse et confirme la couleur de chaque bande détectée.
    """
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    for band in bands:
        x, y, w, h = band['bbox']
        confirmed_color = identify_band_color(hsv, filtered, x, y, w, h)
        band['confirmed_color'] = confirmed_color

    return bands


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def calculate_resistance(bands: List[Dict]) -> Optional[Dict]:
    """
    Calcule la valeur de la résistance.
    """
    if len(bands) < 3:
        return None

    # Utiliser les couleurs confirmées
    colors = [b.get('confirmed_color', b['color']) for b in bands]

    # Filtrer les UNKNOWN
    colors = [c for c in colors if c != 'UNKNOWN']

    if len(colors) < 3:
        return None

    # Vérifier la tolérance à la fin
    tolerance = None
    tolerance_name = None
    if colors[-1] in ['GOLD', 'SILVER']:
        tolerance = COLOR_VALUES[colors[-1]]
        tolerance_name = colors[-1]
        colors = colors[:-1]

    if len(colors) < 3:
        return None

    # Obtenir les valeurs
    values = []
    for c in colors:
        if c in COLOR_VALUES:
            values.append(COLOR_VALUES[c])
        else:
            return None

    # Vérifier que les valeurs sont valides
    if values[0] < 0 or values[0] > 9:
        return None
    if values[1] < 0 or values[1] > 9:
        return None
    if values[2] not in MULTIPLIERS:
        return None

    d1 = values[0]
    d2 = values[1]
    mult = MULTIPLIERS[values[2]]

    resistance = (d1 * 10 + d2) * mult
    tol_str = TOLERANCES.get(tolerance, '±20%') if tolerance else '±20%'

    # Formater
    if resistance >= 1e9:
        formatted = f"{resistance / 1e9:.2f} GΩ"
    elif resistance >= 1e6:
        formatted = f"{resistance / 1e6:.2f} MΩ"
    elif resistance >= 1e3:
        formatted = f"{resistance / 1e3:.2f} kΩ"
    else:
        formatted = f"{resistance:.2f} Ω"

    return {
        'value': resistance,
        'formatted': formatted,
        'tolerance': tol_str,
        'bands': colors + ([tolerance_name] if tolerance_name else []),
        'd1': d1,
        'd2': d2,
        'multiplier': values[2],
        'mult_name': colors[2],
        'tolerance_band': tolerance_name
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def draw_bands(image: np.ndarray, bands: List[Dict]) -> np.ndarray:
    """Dessine les bandes détectées."""
    display = image.copy()

    for band in bands:
        color_name = band.get('confirmed_color', band['color'])
        color_bgr = COLOR_DISPLAY.get(color_name, (255, 255, 255))
        x, y, w, h = band['bbox']

        # Rectangle autour de la bande
        cv2.rectangle(display, (x, y), (x + w, y + h), color_bgr, 2)

        # Label
        label = color_name[:3]
        cv2.putText(display, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

    return display


# =============================================================================
# INTERFACE
# =============================================================================

def select_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """Sélection de la région d'intérêt."""
    print("\n  📐 Sélectionnez la zone de la résistance")
    print("     Dessinez un rectangle → ENTRÉE/ESPACE pour valider")

    h, w = image.shape[:2]
    max_dim = 900
    scale = 1.0
    display = image.copy()

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(image, (int(w * scale), int(h * scale)))

    roi = cv2.selectROI("ResistorReader V5 - Selection", display, fromCenter=False)
    cv2.destroyWindow("ResistorReader V5 - Selection")

    x, y, rw, rh = roi
    return int(x / scale), int(y / scale), int(rw / scale), int(rh / scale)


def process_image(image_path: str) -> None:
    """Traite une image de résistance."""
    print("\n" + "=" * 65)
    print("  ResistorReader V5 - Détection des BANDES uniquement")
    print("  Méthode: Contours + Filtrage du corps")
    print("=" * 65)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur: Impossible de charger '{image_path}'")
        return

    print(f"\n📷 Image: {image_path}")
    print(f"   Dimensions: {image.shape[1]}x{image.shape[0]}")

    # Sélection
    x, y, w, h = select_roi(image)
    if w == 0 or h == 0:
        print("❌ Aucune région sélectionnée")
        return

    print(f"   Région: ({x}, {y}) - {w}x{h}")

    # Extraire la ROI
    roi = image[y:y + h, x:x + w]

    # Aire minimale adaptative
    min_area = max(50, (w * h) // 500)
    print(f"   Aire minimale: {min_area} px²")

    # Détecter les bandes
    print("\n📊 Détection des bandes de couleur...")
    bands = find_bands(roi, min_area=min_area)

    # Confirmer les couleurs
    bands = analyze_bands(roi, bands)

    print(f"\n   Bandes détectées: {len(bands)}")
    for i, band in enumerate(bands):
        color = band.get('confirmed_color', band['color'])
        print(f"     {i + 1}. {color:8s} (x={band['x']:3d}, aire={band['area']:.0f}px²)")

    # Calculer
    result = calculate_resistance(bands)

    # Afficher les résultats
    print("\n" + "-" * 65)
    print("  RÉSULTATS")
    print("-" * 65)

    if result:
        print(f"\n  Bandes: {' - '.join(result['bands'])}")
        print(f"\n" + "=" * 65)
        print(f"  🎯 Résistance = {result['formatted']} {result['tolerance']}")
        print("=" * 65)

        print(f"\n  📝 Calcul:")
        print(f"     Bande 1: {result['bands'][0]} = {result['d1']}")
        print(f"     Bande 2: {result['bands'][1]} = {result['d2']}")
        print(f"     Bande 3: {result['mult_name']} = ×{MULTIPLIERS[result['multiplier']]}")
        if result['tolerance_band']:
            print(f"     Bande 4: {result['tolerance_band']} = {result['tolerance']}")
        print(
            f"\n     → ({result['d1']}×10 + {result['d2']}) × {MULTIPLIERS[result['multiplier']]} = {result['value']:.0f} Ω")
    else:
        print("\n  ❌ Impossible de calculer la résistance")
        if bands:
            colors = [b.get('confirmed_color', b['color']) for b in bands]
            print(f"     Bandes détectées: {colors}")

    # Visualisation
    display_roi = draw_bands(roi, bands)
    cv2.imshow("ResistorReader V5 - Bandes", display_roi)

    # Image complète
    display = image.copy()
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if result:
        text = f"{result['formatted']} {result['tolerance']}"
        cv2.putText(display, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    h_d, w_d = display.shape[:2]
    if max(h_d, w_d) > 800:
        scale = 800 / max(h_d, w_d)
        display = cv2.resize(display, (int(w_d * scale), int(h_d * scale)))

    cv2.imshow("ResistorReader V5 - Resultat", display)
    print("\n  Appuyez sur une touche pour fermer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  ____           _     _             ____                _           ")
    print(" |  _ \\ ___  ___(_)___| |_ ___  _ __|  _ \\ ___  __ _  __| | ___ _ __ ")
    print(" | |_) / _ \\/ __| / __| __/ _ \\| '__| |_) / _ \\/ _` |/ _` |/ _ \\ '__|")
    print(" |  _ <  __/\\__ \\ \\__ \\ || (_) | |  |  _ <  __/ (_| | (_| |  __/ |   ")
    print(" |_| \\_\\___||___/_|___/\\__\\___/|_|  |_| \\_\\___|\\__,_|\\__,_|\\___|_|   ")
    print()
    print("  Traitement de Signal III - EPHEC 2025")
    print("  Version 5 - Détection des BANDES uniquement")
    print("=" * 65)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        for pattern in ['*.jpg', '*.png', '*.jpeg']:
            files = glob.glob(pattern)
            if files:
                image_path = files[0]
                break
        else:
            print("\n❌ Aucune image trouvée!")
            print("   Usage: python resistor_reader_v5.py <image.jpg>")
            return

    print(f"\n📷 Image: {image_path}")
    process_image(image_path)


if __name__ == "__main__":
    main()