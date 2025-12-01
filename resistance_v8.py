#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION V8 (FINALE)
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

TECHNIQUES DE TRAITEMENT DE SIGNAL:
1. CLAHE - Amélioration du contraste adaptatif
2. Filtre Bilatéral - Réduction du bruit préservant les contours
3. Segmentation HSV - Séparation fond/objet
4. Morphologie Mathématique - Nettoyage des masques
5. Détection par masques de couleur - Identification des bandes

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

# Plages HSV pour chaque couleur (calibrées pour photos réelles)
# Format: [(H_min, H_max, S_min, S_max, V_min, V_max), ...]

COLOR_CONFIG = {
    'noir': {
        'ranges': [(0, 180, 0, 255, 0, 50)],
        'value': 0,
        'mult': 1,
        'tol': None,
        'priority': 1
    },
    'marron': {
        'ranges': [(0, 20, 30, 180, 30, 140)],
        'value': 1,
        'mult': 10,
        'tol': '±1%',
        'priority': 5
    },
    'rouge': {
        'ranges': [(0, 12, 50, 255, 50, 255), (168, 180, 50, 255, 50, 255)],
        'value': 2,
        'mult': 100,
        'tol': '±2%',
        'priority': 10  # Haute priorité
    },
    'orange': {
        'ranges': [(10, 25, 100, 255, 150, 255)],
        'value': 3,
        'mult': 1000,
        'tol': None,
        'priority': 6
    },
    'jaune': {
        'ranges': [(22, 40, 80, 255, 150, 255)],
        'value': 4,
        'mult': 10000,
        'tol': None,
        'priority': 6
    },
    'vert': {
        'ranges': [(40, 85, 40, 255, 40, 255)],
        'value': 5,
        'mult': 100000,
        'tol': '±0.5%',
        'priority': 7
    },
    'bleu': {
        'ranges': [(85, 130, 40, 255, 40, 255)],
        'value': 6,
        'mult': 1000000,
        'tol': '±0.25%',
        'priority': 7
    },
    'violet': {
        'ranges': [(130, 165, 30, 255, 30, 255)],
        'value': 7,
        'mult': 10000000,
        'tol': '±0.1%',
        'priority': 6
    },
    'gris': {
        'ranges': [(0, 180, 0, 40, 70, 190)],
        'value': 8,
        'mult': 100000000,
        'tol': '±0.05%',
        'priority': 2
    },
    'blanc': {
        'ranges': [(0, 180, 0, 25, 220, 255)],
        'value': 9,
        'mult': 1000000000,
        'tol': None,
        'priority': 1
    },
    'or': {
        'ranges': [(15, 35, 50, 200, 80, 210)],
        'value': -1,
        'mult': 0.1,
        'tol': '±5%',
        'priority': 8
    },
    'argent': {
        'ranges': [(0, 180, 0, 50, 140, 220)],
        'value': -2,
        'mult': 0.01,
        'tol': '±10%',
        'priority': 3
    },
}

COLOR_LIST = ['noir', 'marron', 'rouge', 'orange', 'jaune', 'vert', 'bleu', 'violet', 'gris', 'blanc', 'or', 'argent']

COLOR_SHORTCUTS = {
    'n': 'noir', '0': 'noir', 'b': 'marron', '1': 'marron',
    'r': 'rouge', '2': 'rouge', 'o': 'orange', '3': 'orange',
    'j': 'jaune', '4': 'jaune', 'v': 'vert', '5': 'vert',
    'l': 'bleu', '6': 'bleu', 'p': 'violet', '7': 'violet',
    'g': 'gris', '8': 'gris', 'w': 'blanc', '9': 'blanc',
    'd': 'or', 's': 'argent',
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
    priority: int


@dataclass
class Result:
    ohms: float
    formatted: str
    tolerance: str
    bands: List[str]
    time_ms: float


# =============================================================================
# PRÉTRAITEMENT
# =============================================================================

def preprocess(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prétraitement avancé de l'image.

    Returns:
        (image filtrée, image HSV)
    """
    # 1. CLAHE sur le canal L (luminosité)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Filtre bilatéral
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 3. Conversion HSV
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    return filtered, hsv


def create_foreground_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Crée un masque excluant le fond blanc.
    """
    # Le fond blanc: S < 40 ET V > 200
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    background = (s < 40) & (v > 200)
    foreground = (~background).astype(np.uint8) * 255

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=3)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=2)

    return foreground


# =============================================================================
# DÉTECTION DES BANDES PAR MASQUES DE COULEUR
# =============================================================================

def detect_color_bands(hsv: np.ndarray, fg_mask: np.ndarray,
                       color_name: str, color_info: dict,
                       min_area: int = 80) -> List[Band]:
    """
    Détecte les bandes d'une couleur spécifique.
    """
    # Créer le masque pour cette couleur
    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for h_min, h_max, s_min, s_max, v_min, v_max in color_info['ranges']:
        mask = cv2.inRange(hsv,
                           np.array([h_min, s_min, v_min]),
                           np.array([h_max, s_max, v_max]))
        color_mask = cv2.bitwise_or(color_mask, mask)

    # Appliquer le masque foreground
    color_mask = cv2.bitwise_and(color_mask, fg_mask)

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Trouver les contours
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Position selon orientation
                h, w = hsv.shape[:2]
                pos = cy if h > w else cx

                bands.append(Band(
                    color=color_name,
                    value=color_info['value'],
                    position=pos,
                    area=area,
                    priority=color_info['priority']
                ))

    return bands


def detect_all_bands(hsv: np.ndarray, fg_mask: np.ndarray) -> List[Band]:
    """
    Détecte toutes les bandes de couleur.
    """
    all_bands = []

    for color_name, color_info in COLOR_CONFIG.items():
        bands = detect_color_bands(hsv, fg_mask, color_name, color_info)
        all_bands.extend(bands)

    return all_bands


def filter_and_merge_bands(bands: List[Band], min_distance: int = 20) -> List[Band]:
    """
    Filtre les faux positifs et fusionne les bandes proches.

    Pour les bandes proches, on garde celle avec la plus haute priorité.
    """
    if not bands:
        return []

    # Trier par position
    bands = sorted(bands, key=lambda b: b.position)

    # Fusionner les bandes proches
    merged = []
    for band in bands:
        if not merged:
            merged.append(band)
            continue

        last = merged[-1]
        if abs(band.position - last.position) < min_distance:
            # Bandes proches: garder celle avec priorité plus haute ou aire plus grande
            if band.priority > last.priority or \
                    (band.priority == last.priority and band.area > last.area):
                merged[-1] = band
        else:
            merged.append(band)

    # Filtrer: garder max 1 bande or/argent (tolérance)
    tolerance_bands = [b for b in merged if b.value < 0]
    other_bands = [b for b in merged if b.value >= 0]

    if len(tolerance_bands) > 1:
        # Garder celle avec la plus grande aire (probablement la vraie)
        tolerance_bands = [max(tolerance_bands, key=lambda b: b.area)]

    result = other_bands + tolerance_bands
    result.sort(key=lambda b: b.position)

    return result


def orient_bands(bands: List[Band]) -> List[Band]:
    """
    Oriente les bandes (tolérance à la fin).
    """
    if len(bands) < 2:
        return bands

    # Si or/argent au début mais pas à la fin -> inverser
    if bands[0].value < 0 and bands[-1].value >= 0:
        return list(reversed(bands))

    return bands


# =============================================================================
# CALCUL
# =============================================================================

def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    """Calcule la valeur de la résistance."""
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    # Retirer la tolérance si présente à la fin
    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLOR_CONFIG[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    # Vérifier que les 3 premières valeurs sont valides
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
# INTERFACE UTILISATEUR
# =============================================================================

def get_color_from_input(user_input: str) -> Optional[str]:
    user_input = user_input.strip().lower()
    if user_input in COLOR_SHORTCUTS:
        return COLOR_SHORTCUTS[user_input]
    if user_input in COLOR_LIST:
        return user_input
    aliases = {'gold': 'or', 'silver': 'argent', 'brown': 'marron', 'red': 'rouge',
               'blue': 'bleu', 'green': 'vert', 'yellow': 'jaune', 'purple': 'violet',
               'gray': 'gris', 'grey': 'gris', 'white': 'blanc', 'black': 'noir'}
    if user_input in aliases:
        return aliases[user_input]
    matches = [c for c in COLOR_LIST if c.startswith(user_input)]
    return matches[0] if len(matches) == 1 else None


def print_color_menu():
    print("\n  ┌────────────────────────────────────────────┐")
    print("  │         COULEURS DISPONIBLES               │")
    print("  ├────────────────────────────────────────────┤")
    print("  │  [N/0] Noir      [B/1] Marron  [R/2] Rouge │")
    print("  │  [O/3] Orange    [J/4] Jaune   [V/5] Vert  │")
    print("  │  [L/6] Bleu      [P/7] Violet  [G/8] Gris  │")
    print("  │  [W/9] Blanc     [D]   Or      [S]   Argent│")
    print("  └────────────────────────────────────────────┘")


def manual_input_bands() -> List[Band]:
    print("\n" + "=" * 55)
    print("  SAISIE MANUELLE DES BANDES")
    print("=" * 55)
    print_color_menu()
    print("\n  Entrez les couleurs (gauche → droite)")
    print("  'fin' pour terminer")

    bands = []
    pos = 0

    while True:
        user = input(f"\n  Bande {len(bands) + 1}: ").strip()

        if user.lower() in ['fin', 'f', 'done', '']:
            if len(bands) >= 3:
                break
            print("  ⚠️  Minimum 3 bandes!")
            continue

        if user.lower() in ['annuler', 'a', 'q']:
            return []

        color = get_color_from_input(user)
        if color:
            bands.append(Band(color=color, value=COLOR_CONFIG[color]['value'],
                              position=pos, area=1000, priority=10))
            pos += 20
            print(f"       → {color.upper()} ✓")
        else:
            print(f"  ❌ Non reconnu: '{user}'")

    return bands


def correct_bands(bands: List[Band]) -> List[Band]:
    print("\n" + "=" * 55)
    print("  CORRECTION DES BANDES")
    print("=" * 55)
    print_color_menu()

    while True:
        print("\n  Bandes actuelles:")
        for i, b in enumerate(bands):
            print(f"    {i + 1}. {b.color.upper()}")

        print("\n  [numéro] Modifier  [+] Ajouter  [-] Supprimer")
        print("  [OK/Entrée] Valider  [M] Tout réentrer")

        choice = input("\n  Choix: ").strip().lower()

        if choice in ['ok', 'o', '']:
            break

        if choice in ['m', 'manuel']:
            new = manual_input_bands()
            if new:
                bands = new
            continue

        if choice == '+':
            try:
                pos = int(input("  Position: "))
            except:
                pos = len(bands) + 1
            color = get_color_from_input(input("  Couleur: "))
            if color:
                bands.insert(max(0, min(pos - 1, len(bands))),
                             Band(color=color, value=COLOR_CONFIG[color]['value'],
                                  position=0, area=1000, priority=10))
                print(f"  ✓ {color.upper()} ajouté")
            continue

        if choice == '-':
            try:
                num = int(input("  Numéro: "))
                if 1 <= num <= len(bands):
                    bands.pop(num - 1)
                    print("  ✓ Supprimé")
            except:
                pass
            continue

        try:
            num = int(choice)
            if 1 <= num <= len(bands):
                color = get_color_from_input(input(f"  Nouvelle couleur: "))
                if color:
                    bands[num - 1].color = color
                    bands[num - 1].value = COLOR_CONFIG[color]['value']
                    print(f"  ✓ → {color.upper()}")
        except:
            pass

    return bands


def select_roi(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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

    win = 'Selection [ENTREE=OK | A=Auto | M=Manuel | R=Reset | ESC=Quitter]'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)

    print("\n" + "=" * 55)
    print("  SÉLECTION DE LA ZONE")
    print("=" * 55)
    print("  → Dessinez un rectangle autour de la résistance")
    print("  → ENTRÉE: valider")
    print("  → A: détection automatique (toute l'image)")
    print("  → M: saisie manuelle des couleurs")
    print("=" * 55)

    while True:
        disp = image.copy()
        if drawing or done:
            cv2.rectangle(disp, p1, p2, (0, 255, 0), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and done:
            cv2.destroyWindow(win)
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            return (x1, y1, x2 - x1, y2 - y1) if x2 - x1 > 20 and y2 - y1 > 10 else None
        elif key == ord('r'):
            done = False
        elif key == ord('a'):
            cv2.destroyWindow(win)
            return "AUTO"
        elif key == ord('m'):
            cv2.destroyWindow(win)
            return "MANUAL"
        elif key == 27:
            cv2.destroyAllWindows()
            return None


def show_result(image: Optional[np.ndarray], roi: Optional[Tuple],
                result: Result, bands: List[Band]):
    print("\n" + "=" * 55)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 55)
    print(f"  Bandes: {' > '.join(result.bands)}")
    print(f"  Temps:  {result.time_ms:.1f} ms")

    # Calcul
    working = [b for b in bands if b.value >= 0][:3]
    if len(working) >= 3:
        d1, d2, m = working[0].value, working[1].value, working[2].value
        print(f"\n  📝 Calcul: ({d1}×10 + {d2}) × 10^{m} = {result.ohms:.0f} Ω")
    print("=" * 55)

    if image is not None and roi is not None:
        out = image.copy()
        rx, ry, rw, rh = roi
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.rectangle(out, (10, 10), (350, 60), (0, 0, 0), -1)
        cv2.putText(out, f"{result.formatted} {result.tolerance}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('RESULTAT', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================================================================
# ANALYSE PRINCIPALE
# =============================================================================

def analyze(image_path: str) -> Optional[Result]:
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
    filtered, hsv = preprocess(img)

    # Sélection ROI
    roi_result = select_roi(filtered)

    if roi_result is None:
        return None

    if roi_result == "MANUAL":
        bands = manual_input_bands()
        if not bands:
            return None
        bands = orient_bands(bands)
        result = calculate_resistance(bands)
        if result:
            result.time_ms = (time.time() - start) * 1000
            show_result(None, None, result, bands)
        return result

    if roi_result == "AUTO":
        roi = (0, 0, img.shape[1], img.shape[0])
    else:
        roi = roi_result

    rx, ry, rw, rh = roi
    roi_hsv = hsv[ry:ry + rh, rx:rx + rw]

    # Masque foreground
    fg_mask = create_foreground_mask(roi_hsv)

    # Détection des bandes
    bands = detect_all_bands(roi_hsv, fg_mask)
    print(f"\n📊 Bandes brutes: {len(bands)}")

    # Filtrer et fusionner
    bands = filter_and_merge_bands(bands)
    print(f"📊 Après filtrage: {len(bands)}")

    # Orienter
    bands = orient_bands(bands)

    for i, b in enumerate(bands):
        print(f"   {i + 1}. {b.color.upper()}")

    # Demander correction si nécessaire
    if len(bands) < 3 or len(bands) > 6:
        print("\n  ⚠️  Détection incertaine!")

    print("\n  [ENTRÉE] Calculer  [C] Corriger  [M] Manuel")
    choice = input("  Choix: ").strip().lower()

    if choice in ['c', 'corriger']:
        bands = correct_bands(bands)
    elif choice in ['m', 'manuel']:
        bands = manual_input_bands()
        if not bands:
            return None

    bands = orient_bands(bands)
    result = calculate_resistance(bands)

    if result:
        result.time_ms = (time.time() - start) * 1000
        show_result(img, roi, result, bands)
        return result

    print("\n❌ Calcul impossible (pas assez de bandes valides)")
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 55)
    print("  ANALYSEUR DE RÉSISTANCES - V8 FINALE")
    print("  Projet Traitement de Signal - EPHEC 2025")
    print("=" * 55)

    print("\n📖 TECHNIQUES UTILISÉES:")
    print("   • CLAHE (contraste adaptatif)")
    print("   • Filtre bilatéral")
    print("   • Segmentation HSV")
    print("   • Morphologie mathématique")
    print("   • Détection par masques de couleur")

    print("\n📖 SENS DE LECTURE:")
    print("   → Tolérance (or/argent) à DROITE")

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
            print("\n📝 Aucune image - Mode manuel")
            bands = manual_input_bands()
            if bands:
                bands = orient_bands(bands)
                result = calculate_resistance(bands)
                if result:
                    result.time_ms = 0
                    show_result(None, None, result, bands)
            return

    result = analyze(path)
    print(f"\n{'✅' if result else '❌'} {'Réussi' if result else 'Échoué'}")


if __name__ == "__main__":
    main()