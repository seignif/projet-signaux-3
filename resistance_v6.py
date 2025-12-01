#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION V6 (AVEC CORRECTION MANUELLE)
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

NOUVEAUTÉ: Possibilité de corriger manuellement les couleurs détectées !

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
→ Lecture: Chiffre1-Chiffre2-Multiplicateur-Tolérance

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

FOND_BLANC = True
FOND_S_MAX = 50
FOND_V_MIN = 190

# =============================================================================
# DÉFINITION DES COULEURS
# =============================================================================

COLORS = {
    'noir': {'hsv': [(0, 180, 0, 255, 0, 50)], 'value': 0, 'mult': 1, 'tol': None},
    'marron': {'hsv': [(0, 18, 60, 200, 30, 130)], 'value': 1, 'mult': 10, 'tol': '±1%'},
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

# Liste ordonnée pour l'affichage
COLOR_LIST = ['noir', 'marron', 'rouge', 'orange', 'jaune', 'vert', 'bleu', 'violet', 'gris', 'blanc', 'or', 'argent']

# Raccourcis clavier
COLOR_SHORTCUTS = {
    'n': 'noir', '0': 'noir',
    'b': 'marron', '1': 'marron',  # b = brown
    'r': 'rouge', '2': 'rouge',
    'o': 'orange', '3': 'orange',
    'j': 'jaune', '4': 'jaune',
    'v': 'vert', '5': 'vert',
    'l': 'bleu', '6': 'bleu',  # l = blue
    'p': 'violet', '7': 'violet',  # p = purple
    'g': 'gris', '8': 'gris',
    'w': 'blanc', '9': 'blanc',  # w = white
    'd': 'or',  # d = doré
    's': 'argent',  # s = silver
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
# FONCTIONS DE DÉTECTION
# =============================================================================

def preprocess(image: np.ndarray) -> np.ndarray:
    enhanced = cv2.convertScaleAbs(image, alpha=1.4, beta=15)
    return cv2.bilateralFilter(enhanced, 9, 75, 75)


def create_mask(hsv: np.ndarray, color_info: dict) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_min, h_max, s_min, s_max, v_min, v_max in color_info['hsv']:
        m = cv2.inRange(hsv, np.array([h_min, s_min, v_min]),
                        np.array([h_max, s_max, v_max]))
        mask = cv2.bitwise_or(mask, m)
    return mask


def find_bands_for_color(hsv: np.ndarray, color_name: str, color_info: dict) -> List[Band]:
    mask = create_mask(hsv, color_info)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 80:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                h, w = hsv.shape[:2]
                pos = int(M['m01'] / M['m00']) if h > w else int(M['m10'] / M['m00'])
                bands.append(Band(color=color_name, value=color_info['value'],
                                  position=pos, area=area))
    return bands


def detect_all_bands(hsv: np.ndarray) -> List[Band]:
    all_bands = []
    for color_name, color_info in COLORS.items():
        bands = find_bands_for_color(hsv, color_name, color_info)
        all_bands.extend(bands)

    all_bands.sort(key=lambda b: b.position)

    merged = []
    for band in all_bands:
        if not merged or abs(band.position - merged[-1].position) > 15:
            merged.append(band)
        elif band.area > merged[-1].area:
            merged[-1] = band

    return merged


def filter_tolerance_bands(bands: List[Band]) -> List[Band]:
    tolerance_bands = [b for b in bands if b.value < 0]
    other_bands = [b for b in bands if b.value >= 0]

    if len(tolerance_bands) > 1:
        best = max(tolerance_bands, key=lambda b: b.area)
        tolerance_bands = [best]

    result = other_bands + tolerance_bands
    result.sort(key=lambda b: b.position)
    return result


def determine_orientation(bands: List[Band]) -> List[Band]:
    if len(bands) < 2:
        return bands
    if bands[0].value < 0 and bands[-1].value >= 0:
        return list(reversed(bands))
    return bands


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLORS[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    if any(b.value < 0 for b in working[:3]):
        return None

    d1 = working[0].value
    d2 = working[1].value
    mult = working[2].value

    base = d1 * 10 + d2
    multiplier = 10 ** mult if mult >= 0 else (0.1 if mult == -1 else 0.01)
    ohms = base * multiplier

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
# INTERFACE AVEC CORRECTION MANUELLE
# =============================================================================

def print_color_menu():
    """Affiche le menu des couleurs avec raccourcis."""
    print("\n  ┌────────────────────────────────────────────┐")
    print("  │         COULEURS DISPONIBLES               │")
    print("  ├────────────────────────────────────────────┤")
    print("  │  [N/0] Noir      [B/1] Marron  [R/2] Rouge │")
    print("  │  [O/3] Orange    [J/4] Jaune   [V/5] Vert  │")
    print("  │  [L/6] Bleu      [P/7] Violet  [G/8] Gris  │")
    print("  │  [W/9] Blanc     [D]   Or      [S]   Argent│")
    print("  └────────────────────────────────────────────┘")


def get_color_from_input(user_input: str) -> Optional[str]:
    """Convertit l'entrée utilisateur en nom de couleur."""
    user_input = user_input.strip().lower()

    # Vérifier raccourci clavier
    if user_input in COLOR_SHORTCUTS:
        return COLOR_SHORTCUTS[user_input]

    # Vérifier nom complet exact d'abord
    if user_input in COLOR_LIST:
        return user_input

    # Alias courants
    aliases = {
        'gold': 'or',
        'silver': 'argent',
        'brown': 'marron',
        'red': 'rouge',
        'blue': 'bleu',
        'green': 'vert',
        'yellow': 'jaune',
        'purple': 'violet',
        'gray': 'gris',
        'grey': 'gris',
        'white': 'blanc',
        'black': 'noir',
    }
    if user_input in aliases:
        return aliases[user_input]

    # Vérifier nom partiel (mais pas pour "or" qui est ambigu)
    # On cherche une correspondance exacte au début, mais on évite les conflits
    matches = [c for c in COLOR_LIST if c.startswith(user_input)]

    # Si une seule correspondance, c'est bon
    if len(matches) == 1:
        return matches[0]

    # Si plusieurs correspondances, chercher une correspondance exacte
    if user_input in matches:
        return user_input

    return None


def manual_input_bands() -> List[Band]:
    """Permet à l'utilisateur d'entrer manuellement les couleurs."""
    print("\n" + "=" * 55)
    print("  ENTRÉE MANUELLE DES BANDES")
    print("=" * 55)
    print_color_menu()
    print("\n  Entrez les couleurs dans l'ordre (gauche → droite)")
    print("  Tapez 'fin' quand terminé, 'annuler' pour quitter")
    print()

    bands = []
    position = 0

    while True:
        prompt = f"  Bande {len(bands) + 1}: "
        user_input = input(prompt).strip()

        if user_input.lower() in ['fin', 'f', 'done', 'd', '']:
            if len(bands) >= 3:
                break
            else:
                print("  ⚠️  Minimum 3 bandes requises!")
                continue

        if user_input.lower() in ['annuler', 'a', 'cancel', 'q']:
            return []

        color = get_color_from_input(user_input)
        if color:
            bands.append(Band(
                color=color,
                value=COLORS[color]['value'],
                position=position,
                area=1000
            ))
            position += 20
            print(f"       → {color.upper()} ✓")
        else:
            print(f"  ❌ Couleur non reconnue: '{user_input}'")

    return bands


def correct_bands(bands: List[Band]) -> List[Band]:
    """Permet de corriger les bandes détectées."""
    print("\n" + "=" * 55)
    print("  CORRECTION DES BANDES")
    print("=" * 55)
    print_color_menu()

    while True:
        print("\n  Bandes actuelles:")
        for i, b in enumerate(bands):
            print(f"    {i + 1}. {b.color.upper()}")

        print("\n  Options:")
        print("    [numéro] Modifier une bande (ex: 1)")
        print("    [+]      Ajouter une bande")
        print("    [-]      Supprimer une bande")
        print("    [OK]     Valider et calculer")
        print("    [M]      Tout réentrer manuellement")

        choice = input("\n  Votre choix: ").strip().lower()

        if choice in ['ok', 'o', '']:
            break

        if choice in ['m', 'manuel']:
            new_bands = manual_input_bands()
            if new_bands:
                bands = new_bands
            continue

        if choice == '+':
            print("  Position (1 = début, {} = fin): ".format(len(bands) + 1), end='')
            try:
                pos = int(input().strip())
                pos = max(1, min(pos, len(bands) + 1))
            except:
                pos = len(bands) + 1

            print("  Couleur: ", end='')
            color_input = input().strip()
            color = get_color_from_input(color_input)
            if color:
                new_band = Band(color=color, value=COLORS[color]['value'],
                                position=0, area=1000)
                bands.insert(pos - 1, new_band)
                # Recalculer les positions
                for i, b in enumerate(bands):
                    b.position = i * 20
                print(f"  ✓ {color.upper()} ajouté en position {pos}")
            else:
                print(f"  ❌ Couleur non reconnue")
            continue

        if choice == '-':
            print("  Numéro de la bande à supprimer: ", end='')
            try:
                num = int(input().strip())
                if 1 <= num <= len(bands):
                    removed = bands.pop(num - 1)
                    print(f"  ✓ {removed.color.upper()} supprimé")
                else:
                    print("  ❌ Numéro invalide")
            except:
                print("  ❌ Entrée invalide")
            continue

        # Modification d'une bande
        try:
            num = int(choice)
            if 1 <= num <= len(bands):
                print(f"  Nouvelle couleur pour bande {num}: ", end='')
                color_input = input().strip()
                color = get_color_from_input(color_input)
                if color:
                    bands[num - 1].color = color
                    bands[num - 1].value = COLORS[color]['value']
                    print(f"  ✓ Bande {num} → {color.upper()}")
                else:
                    print(f"  ❌ Couleur non reconnue")
            else:
                print("  ❌ Numéro invalide")
        except ValueError:
            print("  ❌ Option non reconnue")

    return bands


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

    win = 'Selectionnez la resistance [ENTREE=OK | R=Reset | M=Manuel | ESC=Quitter]'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)

    print("\n" + "=" * 55)
    print("  SÉLECTION DE LA ZONE")
    print("=" * 55)
    print("  → Dessinez un rectangle SERRÉ autour du corps")
    print("  → ENTRÉE pour valider")
    print("  → R pour recommencer")
    print("  → M pour saisie manuelle (sans détection)")
    print("  → ESC pour quitter")
    print("=" * 55)

    while True:
        disp = image.copy()
        if drawing or done:
            cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and done:  # ENTRÉE
            cv2.destroyWindow(win)
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            return (x1, y1, x2 - x1, y2 - y1) if x2 - x1 > 20 and y2 - y1 > 10 else None
        elif key == ord('r'):
            done = False
        elif key == ord('m'):  # Manuel
            cv2.destroyWindow(win)
            return "MANUAL"
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

    return None


def show_result(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]], result: Result):
    """Affiche le résultat."""
    out = image.copy() if image is not None else None

    if out is not None and roi is not None:
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

    # Explication du calcul
    bands = result.bands
    print(f"\n  📝 CALCUL:")
    if len(bands) >= 3:
        has_tolerance = bands[-1].upper() in ['OR', 'ARGENT']
        working = bands[:-1] if has_tolerance else bands

        if len(working) >= 3:
            c1, c2, c3 = working[0], working[1], working[2]
            v1 = COLORS.get(c1.lower(), {}).get('value', '?')
            v2 = COLORS.get(c2.lower(), {}).get('value', '?')
            v3 = COLORS.get(c3.lower(), {}).get('value', '?')

            print(f"     {c1} ({v1}) + {c2} ({v2}) = {v1}{v2}")
            print(f"     × Multiplicateur {c3} (10^{v3})")
            print(f"     = {result.ohms:.0f} Ω = {result.formatted}")

    if out is not None:
        cv2.imshow('RESULTAT - Appuyez sur une touche', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================================================================
# ANALYSE PRINCIPALE
# =============================================================================

def analyze(image_path: str) -> Optional[Result]:
    """Analyse une image de résistance avec possibilité de correction."""
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
    roi = select_roi(processed)

    if roi is None:
        return None

    # Mode manuel complet
    if roi == "MANUAL":
        bands = manual_input_bands()
        if not bands:
            return None
        bands = determine_orientation(bands)
        result = calculate_resistance(bands)
        if result:
            result.time_ms = (time.time() - start) * 1000
            show_result(None, None, result)
        return result

    rx, ry, rw, rh = roi
    roi_hsv = hsv[ry:ry + rh, rx:rx + rw]

    # Détection automatique
    bands = detect_all_bands(roi_hsv)
    print(f"\n📊 Bandes détectées: {len(bands)}")

    bands = filter_tolerance_bands(bands)
    bands = determine_orientation(bands)

    for i, b in enumerate(bands):
        print(f"   {i + 1}. {b.color.upper()}")

    # Demander si correction nécessaire
    print("\n  Voulez-vous corriger les couleurs?")
    print("  [ENTRÉE] Non, calculer directement")
    print("  [C]      Oui, corriger")
    print("  [M]      Entrer manuellement")

    choice = input("\n  Choix: ").strip().lower()

    if choice in ['c', 'corriger']:
        bands = correct_bands(bands)
        bands = determine_orientation(bands)
    elif choice in ['m', 'manuel']:
        bands = manual_input_bands()
        if not bands:
            return None
        bands = determine_orientation(bands)

    # Calcul
    result = calculate_resistance(bands)

    if result:
        result.time_ms = (time.time() - start) * 1000
        show_result(img, roi, result)
        return result

    print(f"\n❌ Impossible de calculer")
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 55)
    print("  ANALYSEUR DE RÉSISTANCES - V6")
    print("  Avec Correction Manuelle")
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
            # Mode manuel sans image
            print("\n📝 Aucune image trouvée - Mode manuel")
            bands = manual_input_bands()
            if bands:
                bands = determine_orientation(bands)
                result = calculate_resistance(bands)
                if result:
                    result.time_ms = 0
                    show_result(None, None, result)
            return

    result = analyze(path)
    print(f"\n{'✅' if result else '❌'} Analyse {'réussie' if result else 'échouée'}")


if __name__ == "__main__":
    main()