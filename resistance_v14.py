#!/usr/bin/env python3
"""
=====================================================================
    ResistorReader V7 - Détection de couleur ULTRA-OPTIMISÉE
    Projet de Traitement de Signal - EPHEC 2025

    Auteur: Noah R.
=====================================================================

CORRECTIONS V7:
- ROUGE vs VIOLET: Rouge rosé (H > 160) maintenant correctement détecté
- JAUNE vs OR: Jaune (S > 150, V > 180) vs Or (S < 150)
- MARRON vs OR: Marron (V < 130, S > 100) vs Or (V > 120)
- NOIR vs VERT: Noir (V < 70) détecté en priorité

ORDRE DE PRIORITÉ:
1. NOIR (V < 70) - en premier!
2. BLANC (S < 35, V > 200)
3. ROUGE (H < 12 ou H > 160, S > 100, V > 100)
4. JAUNE (H 20-50, S > 150, V > 180) - avant OR!
5. ORANGE (H 8-25, S > 150, V > 160)
6. MARRON (H 8-28, S > 80, V 60-150) - avant OR!
7. OR (H 12-38, S 40-150, V > 120)
8. VERT, BLEU, VIOLET, etc.

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
    'noir': 0, 'marron': 1, 'rouge': 2, 'orange': 3, 'jaune': 4,
    'vert': 5, 'bleu': 6, 'violet': 7, 'gris': 8, 'blanc': 9,
    'or': -1, 'argent': -2
}

COLOR_DISPLAY = {
    'noir': (0, 0, 0),
    'marron': (0, 51, 102),
    'rouge': (0, 0, 255),
    'orange': (0, 128, 255),
    'jaune': (0, 255, 255),
    'vert': (0, 255, 0),
    'bleu': (255, 0, 0),
    'violet': (255, 0, 127),
    'gris': (128, 128, 128),
    'blanc': (255, 255, 255),
    'or': (0, 215, 255),
    'argent': (192, 192, 192),
}

# Plages pour la DÉTECTION des contours (larges)
DETECTION_RANGES = {
    'noir': [(0, 0, 0), (180, 255, 80)],
    'marron': [(5, 60, 40), (28, 255, 180)],
    'rouge': [(0, 80, 70), (12, 255, 255)],
    'rouge2': [(160, 80, 70), (180, 255, 255)],
    'orange': [(8, 120, 140), (25, 255, 255)],
    'jaune': [(20, 100, 150), (50, 255, 255)],
    'vert': [(35, 40, 40), (95, 255, 255)],
    'bleu': [(85, 40, 40), (135, 255, 255)],
    'violet': [(125, 25, 40), (165, 255, 255)],
    'gris': [(0, 0, 50), (180, 60, 200)],
    'blanc': [(0, 0, 180), (180, 50, 255)],
    'or': [(12, 30, 90), (40, 200, 250)],
    'argent': [(0, 0, 140), (180, 50, 250)],
}


# =============================================================================
# IDENTIFICATION DES COULEURS - V7 ULTRA-OPTIMISÉE
# =============================================================================

def identify_color_v7(h: float, s: float, v: float) -> Tuple[str, float]:
    """
    Identification des couleurs ULTRA-OPTIMISÉE.

    ORDRE DE PRIORITÉ (crucial pour éviter les confusions):
    1. NOIR     - V très bas
    2. BLANC    - S très bas, V très haut
    3. ROUGE    - H proche de 0 OU > 160 (rouge rosé!)
    4. JAUNE    - S très élevé, V très élevé (avant OR!)
    5. ORANGE   - S élevé, V élevé
    6. MARRON   - S moyen-élevé, V moyen (avant OR!)
    7. OR       - S moyen, V moyen-élevé
    8. VERT     - H 35-95
    9. BLEU     - H 85-135
    10. VIOLET  - H 125-160 (après ROUGE!)
    11. GRIS    - S bas, V moyen
    12. ARGENT  - S bas, V élevé
    """

    # =========================================
    # ÉTAPE 1: NOIR (priorité maximale)
    # =========================================
    # Le noir a une luminosité très basse
    if v < 70:
        # Vérifier que ce n'est pas un vert très foncé
        if not (35 <= h <= 95 and s > 80):
            return ('noir', 100 - v)

    # =========================================
    # ÉTAPE 2: BLANC
    # =========================================
    if s < 35 and v > 200:
        return ('blanc', 80 + (v - 200) / 2)

    # =========================================
    # ÉTAPE 3: ROUGE (AVANT VIOLET!)
    # =========================================
    # Rouge bas (H proche de 0)
    if h <= 12 and s > 80 and v > 80:
        score = 95 - h * 2
        return ('rouge', score)

    # Rouge haut / rosé (H proche de 180) - CRUCIAL!
    if h >= 160 and s > 60 and v > 80:
        score = 95 - (180 - h) * 2
        return ('rouge', score)

    # =========================================
    # ÉTAPE 4: JAUNE (AVANT OR!)
    # =========================================
    # Le jaune est TRÈS saturé et TRÈS lumineux
    if 20 <= h <= 50 and s > 140 and v > 170:
        score = 90 - abs(h - 35) * 1.5
        return ('jaune', score)

    # =========================================
    # ÉTAPE 5: ORANGE
    # =========================================
    # Orange: H 8-25, S très élevé, V élevé
    if 8 <= h <= 25 and s > 140 and v > 150:
        score = 88 - abs(h - 16) * 2
        return ('orange', score)

    # =========================================
    # ÉTAPE 6: MARRON (AVANT OR!)
    # =========================================
    # Marron: H rouge-orange, S moyen-élevé, V MOYEN (plus sombre que or)
    if 6 <= h <= 30 and s > 70 and 50 <= v <= 160:
        score = 85 - abs(h - 15) * 1.5
        # Bonus si saturation élevée
        if s > 100:
            score += 5
        return ('marron', score)

    # =========================================
    # ÉTAPE 7: OR
    # =========================================
    # Or: H jaune-orange, S MOYENNE (pas élevée!), V moyen-élevé
    if 12 <= h <= 40 and 30 <= s <= 160 and v > 110:
        score = 80 - abs(h - 25) * 1.5
        # Pénalité si trop saturé (c'est probablement jaune ou orange)
        if s > 140:
            score -= 20
        return ('or', score)

    # =========================================
    # ÉTAPE 8: VERT
    # =========================================
    if 35 <= h <= 95 and s > 35 and v > 35:
        score = 90 - abs(h - 65)
        return ('vert', score)

    # =========================================
    # ÉTAPE 9: BLEU
    # =========================================
    if 85 <= h <= 135 and s > 40 and v > 40:
        score = 90 - abs(h - 110)
        return ('bleu', score)

    # =========================================
    # ÉTAPE 10: VIOLET (APRÈS ROUGE!)
    # =========================================
    # Violet: H 125-160, mais PAS le rouge rosé!
    if 125 <= h <= 160 and s > 25 and v > 40:
        score = 85 - abs(h - 145) * 1.2
        return ('violet', score)

    # =========================================
    # ÉTAPE 11: GRIS
    # =========================================
    if s < 60 and 50 <= v <= 200:
        score = 70 - s
        return ('gris', score)

    # =========================================
    # ÉTAPE 12: ARGENT
    # =========================================
    if s < 55 and 130 <= v <= 240:
        score = 65 - s
        return ('argent', score)

    # Inconnu
    return ('inconnu', 0)


# =============================================================================
# DÉTECTION DES BANDES PAR CONTOURS
# =============================================================================

def find_contours_for_color(hsv: np.ndarray, thresh: np.ndarray,
                            color_name: str, ranges: list,
                            min_area: int = 50) -> List[Dict]:
    """Trouve les contours pour une couleur donnée."""
    low, high = ranges
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    mask = cv2.bitwise_and(mask, thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        ratio = w / h
        if ratio > 1.2:  # Bande = plus haute que large
            continue

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        valid.append({
            'contour': contour,
            'x': cx, 'y': cy,
            'area': area,
            'bbox': (x, y, w, h)
        })

    return valid


def find_bands(image: np.ndarray, min_area: int = 80) -> List[Dict]:
    """Trouve les bandes de couleur sur la résistance."""
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    all_bands = []

    for color_name, ranges in DETECTION_RANGES.items():
        if color_name == 'rouge2':
            continue

        contours = find_contours_for_color(hsv, thresh, color_name, ranges, min_area)

        if color_name == 'rouge':
            contours2 = find_contours_for_color(hsv, thresh, 'rouge', DETECTION_RANGES['rouge2'], min_area)
            contours.extend(contours2)

        all_bands.extend(contours)

    all_bands.sort(key=lambda b: b['x'])

    # Éliminer les doublons
    filtered_bands = []
    for band in all_bands:
        is_duplicate = False
        idx_to_remove = -1

        for idx, existing in enumerate(filtered_bands):
            if abs(band['x'] - existing['x']) < 20:
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

    filtered_bands.sort(key=lambda b: b['x'])
    return filtered_bands


def identify_bands(image: np.ndarray, bands: List[Dict]) -> List[Dict]:
    """Identifie la couleur de chaque bande avec la fonction V7."""
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    for band in bands:
        x, y, w, h = band['bbox']
        zone_hsv = hsv[y:y + h, x:x + w]

        H = np.mean(zone_hsv[:, :, 0])
        S = np.mean(zone_hsv[:, :, 1])
        V = np.mean(zone_hsv[:, :, 2])

        color, confidence = identify_color_v7(H, S, V)

        band['color'] = color
        band['confidence'] = confidence
        band['h'] = H
        band['s'] = S
        band['v'] = V

    return bands


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def calculate_resistance(bands: List[Dict]) -> Optional[Dict]:
    """Calcule la valeur de la résistance."""
    if len(bands) < 3:
        return None

    colors = [b['color'] for b in bands if b['color'] != 'inconnu']

    if len(colors) < 3:
        return None

    # Tolérance à la fin
    tolerance = None
    tolerance_name = None
    if colors[-1] in ['or', 'argent']:
        tolerance = COLOR_VALUES[colors[-1]]
        tolerance_name = colors[-1]
        colors = colors[:-1]

    if len(colors) < 3:
        return None

    values = [COLOR_VALUES.get(c, -99) for c in colors]

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
        'bands': [b['color'] for b in bands],
        'd1': d1, 'd2': d2,
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
        color_name = band.get('color', 'inconnu')
        color_bgr = COLOR_DISPLAY.get(color_name, (255, 255, 255))
        x, y, w, h = band['bbox']

        cv2.rectangle(display, (x, y), (x + w, y + h), color_bgr, 2)

        label = color_name[:3].upper()
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
    print("     Dessinez un rectangle → ENTRÉE/ESPACE")

    h, w = image.shape[:2]
    max_dim = 900
    scale = 1.0
    display = image.copy()

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(image, (int(w * scale), int(h * scale)))

    roi = cv2.selectROI("ResistorReader V7 - Selection", display, fromCenter=False)
    cv2.destroyWindow("ResistorReader V7 - Selection")

    x, y, rw, rh = roi
    return int(x / scale), int(y / scale), int(rw / scale), int(rh / scale)


def process_image(image_path: str) -> None:
    """Traite une image de résistance."""
    print("\n" + "=" * 65)
    print("  ResistorReader V7 - Détection ULTRA-OPTIMISÉE")
    print("=" * 65)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur: Impossible de charger '{image_path}'")
        return

    print(f"\n📷 Image: {image_path}")
    print(f"   Dimensions: {image.shape[1]}x{image.shape[0]}")

    x, y, w, h = select_roi(image)
    if w == 0 or h == 0:
        print("❌ Aucune région sélectionnée")
        return

    print(f"   Région: ({x}, {y}) - {w}x{h}")

    roi = image[y:y + h, x:x + w]
    min_area = max(50, (w * h) // 500)

    print("\n📊 Détection des bandes...")
    bands = find_bands(roi, min_area=min_area)
    bands = identify_bands(roi, bands)

    print(f"\n   Bandes détectées: {len(bands)}")
    for i, band in enumerate(bands):
        color = band.get('color', '?')
        h_val = band.get('h', 0)
        s_val = band.get('s', 0)
        v_val = band.get('v', 0)
        print(f"     {i + 1}. {color.upper():8s} HSV=({h_val:.0f}, {s_val:.0f}, {v_val:.0f})")

    result = calculate_resistance(bands)

    print("\n" + "-" * 65)
    print("  RÉSULTATS")
    print("-" * 65)

    if result:
        band_names = [b.upper() for b in result['bands']]
        print(f"\n  Bandes: {' - '.join(band_names)}")
        print(f"\n" + "=" * 65)
        print(f"  🎯 Résistance = {result['formatted']} {result['tolerance']}")
        print("=" * 65)

        print(f"\n  📝 Calcul:")
        print(f"     {result['bands'][0].upper()} = {result['d1']}")
        print(f"     {result['bands'][1].upper()} = {result['d2']}")
        print(f"     {result['mult_name'].upper()} = ×{MULTIPLIERS[result['multiplier']]}")
        if result['tolerance_band']:
            print(f"     {result['tolerance_band'].upper()} = {result['tolerance']}")
        print(
            f"\n     → ({result['d1']}×10 + {result['d2']}) × {MULTIPLIERS[result['multiplier']]} = {result['value']:.0f} Ω")
    else:
        print("\n  ❌ Impossible de calculer")
        if bands:
            print(f"     Bandes: {[b.get('color', '?') for b in bands]}")

    display_roi = draw_bands(roi, bands)
    cv2.imshow("ResistorReader V7 - Bandes", display_roi)

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

    cv2.imshow("ResistorReader V7 - Resultat", display)
    print("\n  Appuyez sur une touche pour fermer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n" + "=" * 65)
    print("  ResistorReader V7 - Détection ULTRA-OPTIMISÉE")
    print("  Traitement de Signal III - EPHEC 2025")
    print("=" * 65)
    print("\n  CORRECTIONS V7:")
    print("  ✓ Rouge rosé détecté (avant: confondu avec violet)")
    print("  ✓ Jaune vif détecté (avant: confondu avec or)")
    print("  ✓ Marron foncé détecté (avant: confondu avec or)")
    print("  ✓ Noir détecté (avant: parfois confondu avec vert)")

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
            return

    print(f"\n📷 Image: {image_path}")
    process_image(image_path)


if __name__ == "__main__":
    main()