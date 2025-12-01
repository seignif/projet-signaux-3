#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION FINALE (V5)
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

Cette version utilise:
1. Détection par masques de couleur HSV
2. Filtrage des faux positifs
3. Détermination automatique du sens de lecture
4. Fond blanc paramétré par défaut

CODE COULEUR:
┌─────────┬────────┬───────────────┬───────────┐
│ Couleur │ Valeur │ Multiplicateur│ Tolérance │
├─────────┼────────┼───────────────┼───────────┤
│ Noir    │   0    │ ×1            │     -     │
│ Marron  │   1    │ ×10           │   ±1%     │
│ Rouge   │   2    │ ×100          │   ±2%     │
│ Orange  │   3    │ ×1kΩ          │     -     │
│ Jaune   │   4    │ ×10kΩ         │     -     │
│ Vert    │   5    │ ×100kΩ        │  ±0.5%    │
│ Bleu    │   6    │ ×1MΩ          │ ±0.25%    │
│ Violet  │   7    │ ×10MΩ         │  ±0.1%    │
│ Gris    │   8    │ ×100MΩ        │ ±0.05%    │
│ Blanc   │   9    │ ×1GΩ          │     -     │
│ Or      │   -    │ ×0.1          │   ±5%     │
│ Argent  │   -    │ ×0.01         │  ±10%     │
└─────────┴────────┴───────────────┴───────────┘

SENS DE LECTURE:
→ La bande de tolérance (or/argent) est TOUJOURS à DROITE
→ Elle est souvent plus espacée des autres bandes
→ On lit de GAUCHE vers DROITE: Chiffre1-Chiffre2-Multiplicateur-Tolérance

=====================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import sys
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Le fond est considéré comme BLANC par défaut
# (saturation faible + luminosité haute)
FOND_BLANC = True
FOND_S_MAX = 50  # Saturation max pour être considéré comme fond blanc
FOND_V_MIN = 190  # Luminosité min pour être considéré comme fond blanc

# =============================================================================
# DÉFINITION DES COULEURS (HSV)
# =============================================================================

# Format: 'couleur': {'hsv': [(H_min, H_max, S_min, S_max, V_min, V_max), ...], 'value': int, 'tol': str}
COLORS = {
    'noir': {'hsv': [(0, 180, 0, 255, 0, 50)], 'value': 0, 'mult': 1, 'tol': None},
    'marron': {'hsv': [(0, 18, 60, 200, 30, 120)], 'value': 1, 'mult': 10, 'tol': '±1%'},
    'rouge': {'hsv': [(0, 10, 100, 255, 60, 255), (170, 180, 100, 255, 60, 255)], 'value': 2, 'mult': 100,
              'tol': '±2%'},
    'orange': {'hsv': [(10, 22, 130, 255, 160, 255)], 'value': 3, 'mult': 1000, 'tol': None},
    'jaune': {'hsv': [(22, 38, 100, 255, 170, 255)], 'value': 4, 'mult': 10000, 'tol': None},
    'vert': {'hsv': [(38, 85, 50, 255, 50, 255)], 'value': 5, 'mult': 100000, 'tol': '±0.5%'},
    'bleu': {'hsv': [(85, 128, 50, 255, 50, 255)], 'value': 6, 'mult': 1000000, 'tol': '±0.25%'},
    'violet': {'hsv': [(128, 165, 40, 255, 40, 255)], 'value': 7, 'mult': 10000000, 'tol': '±0.1%'},
    'gris': {'hsv': [(0, 180, 0, 50, 70, 190)], 'value': 8, 'mult': 100000000, 'tol': '±0.05%'},
    'blanc': {'hsv': [(0, 180, 0, 30, 220, 255)], 'value': 9, 'mult': 1000000000, 'tol': None},
    'or': {'hsv': [(15, 28, 100, 200, 80, 170)], 'value': -1, 'mult': 0.1, 'tol': '±5%'},
    'argent': {'hsv': [(0, 180, 0, 40, 140, 210)], 'value': -2, 'mult': 0.01, 'tol': '±10%'},
}


# =============================================================================
# STRUCTURES
# =============================================================================

@dataclass
class Band:
    color: str
    value: int
    position: int
    area: int


@dataclass
class Result:
    ohms: float
    formatted: str
    tolerance: str
    bands: List[str]
    time_ms: float


# =============================================================================
# FONCTIONS
# =============================================================================

def preprocess(image: np.ndarray) -> np.ndarray:
    """Prétraitement: amélioration contraste + filtre bilatéral."""
    enhanced = cv2.convertScaleAbs(image, alpha=1.4, beta=15)
    return cv2.bilateralFilter(enhanced, 9, 75, 75)


def create_mask(hsv: np.ndarray, color_info: dict) -> np.ndarray:
    """Crée un masque pour une couleur donnée."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_min, h_max, s_min, s_max, v_min, v_max in color_info['hsv']:
        m = cv2.inRange(hsv, np.array([h_min, s_min, v_min]),
                        np.array([h_max, s_max, v_max]))
        mask = cv2.bitwise_or(mask, m)
    return mask


def find_bands_for_color(hsv: np.ndarray, color_name: str, color_info: dict) -> List[Band]:
    """Trouve les bandes d'une couleur spécifique."""
    mask = create_mask(hsv, color_info)

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 80:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                # Position selon l'axe principal (Y pour vertical, X pour horizontal)
                h, w = hsv.shape[:2]
                if h > w:  # Vertical
                    pos = int(M['m01'] / M['m00'])
                else:  # Horizontal
                    pos = int(M['m10'] / M['m00'])

                bands.append(Band(color=color_name, value=color_info['value'],
                                  position=pos, area=area))

    return bands


def detect_all_bands(hsv: np.ndarray) -> List[Band]:
    """Détecte toutes les bandes de couleur."""
    all_bands = []

    for color_name, color_info in COLORS.items():
        bands = find_bands_for_color(hsv, color_name, color_info)
        all_bands.extend(bands)

    # Trier par position
    all_bands.sort(key=lambda b: b.position)

    # Fusionner les bandes proches (même position)
    merged = []
    for band in all_bands:
        if not merged or abs(band.position - merged[-1].position) > 15:
            merged.append(band)
        elif band.area > merged[-1].area:
            merged[-1] = band

    return merged


def filter_tolerance_bands(bands: List[Band]) -> List[Band]:
    """
    Filtre les faux positifs pour les bandes de tolérance (or/argent).

    Règle: Il y a au maximum UNE bande de tolérance par résistance,
    située à une extrémité. Si on en détecte plusieurs, on garde
    celle avec la plus grande aire.
    """
    tolerance_bands = [b for b in bands if b.value < 0]
    other_bands = [b for b in bands if b.value >= 0]

    if len(tolerance_bands) > 1:
        # Garder seulement la plus grande
        best = max(tolerance_bands, key=lambda b: b.area)
        tolerance_bands = [best]

    result = other_bands + tolerance_bands
    result.sort(key=lambda b: b.position)
    return result


def determine_orientation(bands: List[Band]) -> List[Band]:
    """
    Détermine le sens de lecture correct.

    La tolérance (or/argent) est toujours à la FIN (côté droit).
    Si elle est au début, on inverse l'ordre.
    """
    if len(bands) < 2:
        return bands

    # Si or/argent au début -> inverser
    if bands[0].value < 0 and bands[-1].value >= 0:
        return list(reversed(bands))

    return bands


def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    """Calcule la valeur de la résistance."""
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    # Retirer la tolérance si présente à la fin
    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLORS[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    # Vérifier que les 3 premières bandes ont des valeurs valides
    if any(b.value < 0 for b in working[:3]):
        return None

    d1 = working[0].value
    d2 = working[1].value
    mult = working[2].value

    base = d1 * 10 + d2
    multiplier = 10 ** mult if mult >= 0 else (0.1 if mult == -1 else 0.01)
    ohms = base * multiplier

    # Formater
    if ohms >= 1e9:
        fmt = f"{ohms / 1e9:.2f} GΩ"
    elif ohms >= 1e6:
        fmt = f"{ohms / 1e6:.2f} MΩ"
    elif ohms >= 1e3:
        fmt = f"{ohms / 1e3:.2f} kΩ"
    else:
        fmt = f"{ohms:.2f} Ω"

    return Result(
        ohms=ohms,
        formatted=fmt,
        tolerance=tolerance,
        bands=[b.color.upper() for b in bands],
        time_ms=0
    )


# =============================================================================
# INTERFACE
# =============================================================================

def select_roi(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Sélection interactive de la ROI."""
    drawing = False
    p1, p2 = (0, 0), (0, 0)
    done = False

    def mouse(event, x, y, flags, param):
        nonlocal drawing, p1, p2, done
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, done = True, False
            p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            p2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing, done = False, True
            p2 = (x, y)

    win = 'Selectionnez la resistance [ENTREE=OK | R=Reset | ESC=Quitter]'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)

    print("\n" + "=" * 55)
    print("  SÉLECTION DE LA ZONE")
    print("=" * 55)
    print("  → Dessinez un rectangle SERRÉ autour du corps")
    print("  → ENTRÉE pour valider")
    print("  → R pour recommencer")
    print("  → ESC pour quitter")
    print("=" * 55)

    while True:
        disp = image.copy()
        if drawing or done:
            cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and done:
            break
        elif key == ord('r'):
            done = False
        elif key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyWindow(win)

    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

    return (x1, y1, x2 - x1, y2 - y1) if x2 - x1 > 20 and y2 - y1 > 10 else None


def show_result(image: np.ndarray, roi: Tuple[int, int, int, int], result: Result):
    """Affiche le résultat."""
    out = image.copy()
    rx, ry, rw, rh = roi

    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.rectangle(out, (rx, ry - 50), (rx + rw + 120, ry), (0, 0, 0), -1)
    cv2.putText(out, f"{result.formatted} {result.tolerance}",
                (rx + 5, ry - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("\n" + "=" * 55)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 55)
    print(f"  Bandes: {' > '.join(result.bands)}")
    print(f"  Temps:  {result.time_ms:.1f} ms")
    print("=" * 55)

    cv2.imshow('RESULTAT - Appuyez sur une touche', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# ANALYSE PRINCIPALE
# =============================================================================

def analyze(image_path: str, interactive: bool = True) -> Optional[Result]:
    """Analyse une image de résistance."""
    start = time.time()

    # Charger
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Impossible de charger: {image_path}")
        return None

    # Redimensionner
    h, w = img.shape[:2]
    if w > 1000:
        scale = 1000 / w
        img = cv2.resize(img, (1000, int(h * scale)))

    print(f"📷 Image: {image_path}")
    print(f"   Dimensions: {img.shape[1]}x{img.shape[0]}")

    # Prétraitement
    processed = preprocess(img)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

    # Sélection ROI
    if interactive:
        roi = select_roi(processed)
        if roi is None:
            return None
    else:
        # Auto-détection (simplifiée)
        roi = (0, 0, img.shape[1], img.shape[0])

    rx, ry, rw, rh = roi
    roi_hsv = hsv[ry:ry + rh, rx:rx + rw]

    # Détection
    bands = detect_all_bands(roi_hsv)
    print(f"\n📊 Bandes brutes: {len(bands)}")

    # Filtrer les faux positifs de tolérance
    bands = filter_tolerance_bands(bands)
    print(f"📊 Après filtrage: {len(bands)}")

    # Orientation
    bands = determine_orientation(bands)

    for i, b in enumerate(bands):
        print(f"   {i + 1}. {b.color.upper()}")

    # Calcul
    result = calculate_resistance(bands)

    if result:
        result.time_ms = (time.time() - start) * 1000
        show_result(img, roi, result)
        return result

    print(f"\n❌ Impossible de calculer (bandes insuffisantes)")
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 55)
    print("  ANALYSEUR DE RÉSISTANCES - V5 (FINALE)")
    print("  Projet Traitement de Signal - EPHEC 2025")
    print("=" * 55)

    print("\n📖 SENS DE LECTURE:")
    print("   → Tolérance (or/argent) à DROITE")
    print("   → Lecture: Chiffre1-Chiffre2-Multiplicateur-Tolérance")

    # Image
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        for pattern in ['resistance*.jpg', 'resistance*.png', '*.jpg']:
            files = glob.glob(pattern)
            if files:
                path = files[0]
                break
        else:
            print("\n❌ Aucune image trouvée")
            print("   Usage: python resistance_v5.py [image.jpg]")
            return

    result = analyze(path)
    print(f"\n{'✅' if result else '❌'} Analyse {'réussie' if result else 'échouée'}")


if __name__ == "__main__":
    main()