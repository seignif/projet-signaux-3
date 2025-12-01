#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION V7
    Avec Techniques de Traitement de Signal Avancées
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

TECHNIQUES DE TRAITEMENT DE SIGNAL UTILISÉES:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Améliore le contraste local de l'image
   - Évite la sur-amplification du bruit

2. Filtre Bilatéral
   - Réduit le bruit tout en préservant les contours
   - Combine filtre spatial et filtre de plage

3. Seuillage d'Otsu
   - Calcul automatique du seuil optimal
   - Minimise la variance intra-classe

4. Filtre Médian
   - Élimine le bruit impulsionnel (sel et poivre)
   - Préserve les transitions nettes (bords des bandes)

5. Détection de Pics (find_peaks)
   - Trouve les maxima locaux dans le profil de saturation
   - Paramètres: hauteur, distance, prominence

6. Morphologie Mathématique
   - Ouverture/Fermeture pour nettoyer les masques
   - Élimination des petits artefacts

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
# FONCTIONS DE TRAITEMENT DE SIGNAL (remplace scipy)
# =============================================================================

def medfilt(signal: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """
    Filtre médian 1D.

    Le filtre médian remplace chaque valeur par la médiane de ses voisins.
    Avantage: élimine le bruit impulsionnel tout en préservant les transitions.

    Args:
        signal: Signal 1D à filtrer
        kernel_size: Taille de la fenêtre (doit être impair)

    Returns:
        Signal filtré
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(signal, pad, mode='edge')
    result = np.zeros_like(signal)

    for i in range(len(signal)):
        window = padded[i:i + kernel_size]
        result[i] = np.median(window)

    return result


def find_peaks(signal: np.ndarray, height: float = 0, distance: int = 1,
               prominence: float = 0) -> List[int]:
    """
    Détection des pics (maxima locaux) dans un signal 1D.

    Un pic est un point qui est plus grand que ses voisins immédiats.

    Args:
        signal: Signal 1D
        height: Hauteur minimale du pic
        distance: Distance minimale entre deux pics
        prominence: Prominence minimale (différence avec les voisins)

    Returns:
        Liste des indices des pics
    """
    peaks = []
    n = len(signal)

    # Trouver tous les maxima locaux
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            # Vérifier la hauteur minimale
            if signal[i] >= height:
                # Calculer la prominence (différence avec le min des voisins)
                left_min = np.min(signal[max(0, i - distance):i])
                right_min = np.min(signal[i + 1:min(n, i + distance + 1)])
                prom = signal[i] - max(left_min, right_min)

                if prom >= prominence:
                    peaks.append(i)

    # Filtrer par distance minimale
    if distance > 1 and len(peaks) > 1:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= distance:
                filtered.append(p)
            elif signal[p] > signal[filtered[-1]]:
                # Garder le pic le plus haut
                filtered[-1] = p
        peaks = filtered

    return peaks


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paramètres de segmentation du fond blanc
FOND_SAT_MAX = 40  # Saturation max pour être considéré comme fond
FOND_VAL_MIN = 200  # Luminosité min pour être considéré comme fond

# Paramètres de détection des bandes
BAND_SAT_MIN = 60  # Saturation min pour être une bande colorée
PEAK_DISTANCE = 15  # Distance min entre deux pics (pixels)
PEAK_PROMINENCE = 8  # Prominence min d'un pic

# =============================================================================
# COULEURS
# =============================================================================

COLORS = {
    'noir': {'hsv': [(0, 180, 0, 255, 0, 50)], 'value': 0, 'mult': 1, 'tol': None},
    'marron': {'hsv': [(0, 20, 50, 200, 30, 140)], 'value': 1, 'mult': 10, 'tol': '±1%'},
    'rouge': {'hsv': [(0, 12, 80, 255, 60, 255), (168, 180, 80, 255, 60, 255)], 'value': 2, 'mult': 100, 'tol': '±2%'},
    'orange': {'hsv': [(10, 25, 120, 255, 150, 255)], 'value': 3, 'mult': 1000, 'tol': None},
    'jaune': {'hsv': [(22, 40, 100, 255, 170, 255)], 'value': 4, 'mult': 10000, 'tol': None},
    'vert': {'hsv': [(40, 85, 50, 255, 50, 255)], 'value': 5, 'mult': 100000, 'tol': '±0.5%'},
    'bleu': {'hsv': [(85, 130, 50, 255, 50, 255)], 'value': 6, 'mult': 1000000, 'tol': '±0.25%'},
    'violet': {'hsv': [(130, 165, 40, 255, 40, 255)], 'value': 7, 'mult': 10000000, 'tol': '±0.1%'},
    'gris': {'hsv': [(0, 180, 0, 50, 70, 190)], 'value': 8, 'mult': 100000000, 'tol': '±0.05%'},
    'blanc': {'hsv': [(0, 180, 0, 30, 220, 255)], 'value': 9, 'mult': 1000000000, 'tol': None},
    'or': {'hsv': [(15, 30, 70, 200, 80, 200)], 'value': -1, 'mult': 0.1, 'tol': '±5%'},
    'argent': {'hsv': [(0, 180, 0, 50, 140, 220)], 'value': -2, 'mult': 0.01, 'tol': '±10%'},
}

COLOR_LIST = ['noir', 'marron', 'rouge', 'orange', 'jaune', 'vert', 'bleu', 'violet', 'gris', 'blanc', 'or', 'argent']

COLOR_SHORTCUTS = {
    'n': 'noir', '0': 'noir',
    'b': 'marron', '1': 'marron',
    'r': 'rouge', '2': 'rouge',
    'o': 'orange', '3': 'orange',
    'j': 'jaune', '4': 'jaune',
    'v': 'vert', '5': 'vert',
    'l': 'bleu', '6': 'bleu',
    'p': 'violet', '7': 'violet',
    'g': 'gris', '8': 'gris',
    'w': 'blanc', '9': 'blanc',
    'd': 'or',
    's': 'argent',
}


# =============================================================================
# STRUCTURES
# =============================================================================

@dataclass
class Band:
    color: str
    value: int
    position: int
    h: float
    s: float
    v: float


@dataclass
class Result:
    ohms: float
    formatted: str
    tolerance: str
    bands: List[str]
    time_ms: float


# =============================================================================
# PRÉTRAITEMENT AVANCÉ
# =============================================================================

def preprocess_advanced(image: np.ndarray) -> np.ndarray:
    """
    Prétraitement avancé avec CLAHE et filtre bilatéral.

    CLAHE: Améliore le contraste local sans sur-amplifier le bruit.
    Filtre Bilatéral: Lisse le bruit tout en préservant les contours.
    """
    # 1. CLAHE sur le canal L (luminosité) en espace LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Filtre bilatéral
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return filtered


def segment_background(hsv: np.ndarray) -> np.ndarray:
    """
    Segmente le fond blanc de l'image.

    Utilise la saturation et la luminosité pour identifier le fond.
    Le fond blanc a: S faible ET V élevé.

    Returns:
        Masque binaire où 255 = objet (pas le fond)
    """
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Le fond est: faible saturation ET haute luminosité
    background = (s_channel < FOND_SAT_MAX) & (v_channel > FOND_VAL_MIN)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    background = background.astype(np.uint8) * 255
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel, iterations=3)

    foreground = cv2.bitwise_not(background)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=2)

    return foreground


def find_resistor_body(foreground: np.ndarray) -> Optional[Tuple[int, int, int, int, str]]:
    """
    Trouve le corps de la résistance dans le masque.

    Returns:
        (x, y, w, h, orientation) ou None
    """
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Filtrer par aire et forme
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(h / w, w / h)
            if aspect > 1.5:  # Forme allongée
                valid.append((cnt, area, x, y, w, h))

    if not valid:
        # Prendre le plus grand contour même s'il n'est pas allongé
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        valid = [(largest, cv2.contourArea(largest), x, y, w, h)]

    # Prendre le plus grand
    _, _, x, y, w, h = max(valid, key=lambda x: x[1])

    orientation = "vertical" if h > w else "horizontal"

    return (x, y, w, h, orientation)


# =============================================================================
# DÉTECTION DES BANDES PAR PROFIL DE SATURATION
# =============================================================================

def compute_saturation_profile(hsv: np.ndarray, mask: np.ndarray, orientation: str) -> np.ndarray:
    """
    Calcule le profil de saturation le long de l'axe principal.

    Pour une résistance verticale: moyenne de S pour chaque ligne Y.
    Pour une résistance horizontale: moyenne de S pour chaque colonne X.
    """
    h, w = hsv.shape[:2]

    if orientation == "vertical":
        profile = []
        for y in range(h):
            row_s = hsv[y, :, 1]
            row_mask = mask[y, :]
            valid = row_s[row_mask > 0]
            profile.append(np.mean(valid) if len(valid) > 3 else 0)
    else:
        profile = []
        for x in range(w):
            col_s = hsv[:, x, 1]
            col_mask = mask[:, x]
            valid = col_s[col_mask > 0]
            profile.append(np.mean(valid) if len(valid) > 3 else 0)

    return np.array(profile)


def detect_bands_from_profile(hsv: np.ndarray, mask: np.ndarray,
                              profile: np.ndarray, orientation: str) -> List[Band]:
    """
    Détecte les bandes en trouvant les pics dans le profil de saturation.

    Utilise:
    - Filtre médian pour lisser le bruit
    - Détection de pics avec scipy.signal.find_peaks
    """
    # 1. Filtrage médian (réduit le bruit impulsionnel)
    profile_filtered = medfilt(profile, kernel_size=11)

    # 2. Calculer le seuil adaptatif
    valid_values = profile_filtered[profile_filtered > 10]
    if len(valid_values) < 10:
        return []

    baseline = np.percentile(valid_values, 35)
    threshold = max(baseline + 15, BAND_SAT_MIN)

    # 3. Détection des pics
    peaks = find_peaks(
        profile_filtered,
        height=threshold,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE
    )

    # 4. Identifier la couleur à chaque pic
    bands = []

    for peak_pos in peaks:
        # Zone autour du pic
        start = max(0, peak_pos - 7)
        end = min(len(profile), peak_pos + 8)

        if orientation == "vertical":
            zone_hsv = hsv[start:end, :, :]
            zone_mask = mask[start:end, :]
        else:
            zone_hsv = hsv[:, start:end, :]
            zone_mask = mask[:, start:end]

        # Pixels valides
        valid_mask = (zone_mask > 0) & (zone_hsv[:, :, 1] > threshold * 0.8)
        valid_pixels = zone_hsv[valid_mask]

        if len(valid_pixels) < 10:
            continue

        h_med = np.median(valid_pixels[:, 0])
        s_med = np.median(valid_pixels[:, 1])
        v_med = np.median(valid_pixels[:, 2])

        # Identifier la couleur
        color, value = identify_color(h_med, s_med, v_med)

        if color:
            bands.append(Band(
                color=color,
                value=value,
                position=peak_pos,
                h=h_med,
                s=s_med,
                v=v_med
            ))

    return bands


def identify_color(h: float, s: float, v: float) -> Tuple[Optional[str], int]:
    """
    Identifie la couleur à partir des valeurs HSV.
    """
    # Rouge (H proche de 0 ou 180)
    if (h <= 12 or h >= 168) and s > 70:
        return ("rouge", 2)

    # Marron (H rouge-orange, V bas)
    if 0 <= h <= 20 and s > 50 and v < 150:
        return ("marron", 1)

    # Or (H jaune-orange, S moyenne)
    if 15 <= h <= 32 and 60 <= s <= 200 and 80 <= v <= 210:
        return ("or", -1)

    # Orange (H orange, V élevé)
    if 10 < h <= 25 and s > 100 and v > 150:
        return ("orange", 3)

    # Jaune
    if 25 < h <= 40 and s > 80:
        return ("jaune", 4)

    # Vert
    if 40 < h <= 85 and s > 40:
        return ("vert", 5)

    # Bleu
    if 85 < h <= 130 and s > 40:
        return ("bleu", 6)

    # Violet
    if 130 < h < 168 and s > 30:
        return ("violet", 7)

    # Noir (V très bas)
    if v < 50:
        return ("noir", 0)

    # Gris (S très bas, V moyen)
    if s < 40 and 50 < v < 200:
        return ("gris", 8)

    # Blanc (S très bas, V élevé)
    if s < 30 and v > 200:
        return ("blanc", 9)

    # Argent
    if s < 50 and 140 < v < 220:
        return ("argent", -2)

    return (None, -99)


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def orient_bands(bands: List[Band]) -> List[Band]:
    """Oriente les bandes (tolérance à la fin)."""
    if len(bands) < 2:
        return bands

    # Si or/argent au début mais pas à la fin -> inverser
    if bands[0].value < 0 and bands[-1].value >= 0:
        return list(reversed(bands))

    return bands


def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    """Calcule la valeur de la résistance."""
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    # Retirer la tolérance
    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLORS[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    # Vérifier que les valeurs sont valides
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
    """Convertit l'entrée utilisateur en nom de couleur."""
    user_input = user_input.strip().lower()

    if user_input in COLOR_SHORTCUTS:
        return COLOR_SHORTCUTS[user_input]

    if user_input in COLOR_LIST:
        return user_input

    aliases = {
        'gold': 'or', 'silver': 'argent', 'brown': 'marron',
        'red': 'rouge', 'blue': 'bleu', 'green': 'vert',
        'yellow': 'jaune', 'purple': 'violet', 'gray': 'gris',
        'grey': 'gris', 'white': 'blanc', 'black': 'noir',
    }
    if user_input in aliases:
        return aliases[user_input]

    matches = [c for c in COLOR_LIST if c.startswith(user_input)]
    if len(matches) == 1:
        return matches[0]

    return None


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
    """Saisie manuelle des bandes."""
    print("\n" + "=" * 55)
    print("  SAISIE MANUELLE DES BANDES")
    print("=" * 55)
    print_color_menu()
    print("\n  Entrez les couleurs (gauche → droite)")
    print("  'fin' pour terminer, 'annuler' pour quitter")

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
            bands.append(Band(color=color, value=COLORS[color]['value'],
                              position=pos, h=0, s=0, v=0))
            pos += 20
            print(f"       → {color.upper()} ✓")
        else:
            print(f"  ❌ Non reconnu: '{user}'")

    return bands


def correct_bands(bands: List[Band]) -> List[Band]:
    """Correction interactive des bandes."""
    print("\n" + "=" * 55)
    print("  CORRECTION DES BANDES")
    print("=" * 55)
    print_color_menu()

    while True:
        print("\n  Bandes actuelles:")
        for i, b in enumerate(bands):
            print(f"    {i + 1}. {b.color.upper()}")

        print("\n  [numéro] Modifier  [+] Ajouter  [-] Supprimer")
        print("  [OK] Valider       [M] Tout réentrer")

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
                pos = int(input("  Position (1-{}): ".format(len(bands) + 1)))
            except:
                pos = len(bands) + 1
            pos = max(1, min(pos, len(bands) + 1))

            color = get_color_from_input(input("  Couleur: "))
            if color:
                bands.insert(pos - 1, Band(color=color, value=COLORS[color]['value'],
                                           position=0, h=0, s=0, v=0))
                print(f"  ✓ {color.upper()} ajouté")
            continue

        if choice == '-':
            try:
                num = int(input("  Numéro à supprimer: "))
                if 1 <= num <= len(bands):
                    removed = bands.pop(num - 1)
                    print(f"  ✓ {removed.color.upper()} supprimé")
            except:
                pass
            continue

        try:
            num = int(choice)
            if 1 <= num <= len(bands):
                color = get_color_from_input(input(f"  Nouvelle couleur pour bande {num}: "))
                if color:
                    bands[num - 1].color = color
                    bands[num - 1].value = COLORS[color]['value']
                    print(f"  ✓ Bande {num} → {color.upper()}")
        except:
            pass

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

    win = 'Selection [ENTREE=OK | R=Reset | A=Auto | M=Manuel | ESC=Quitter]'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse)

    print("\n" + "=" * 55)
    print("  SÉLECTION DE LA ZONE")
    print("=" * 55)
    print("  → Dessinez un rectangle autour de la résistance")
    print("  → ENTRÉE: valider")
    print("  → A: détection automatique")
    print("  → M: saisie manuelle")
    print("  → R: recommencer")
    print("  → ESC: quitter")
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
        elif key == ord('a'):  # Auto
            cv2.destroyWindow(win)
            return "AUTO"
        elif key == ord('m'):  # Manuel
            cv2.destroyWindow(win)
            return "MANUAL"
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None


def show_result(image: Optional[np.ndarray], roi: Optional[Tuple], result: Result, bands: List[Band]):
    """Affiche le résultat."""
    print("\n" + "=" * 55)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 55)
    print(f"  Bandes: {' > '.join(result.bands)}")
    print(f"  Temps:  {result.time_ms:.1f} ms")
    print("=" * 55)

    # Détails du calcul
    working = [b for b in bands if b.value >= 0][:3]
    if len(working) >= 3:
        d1, d2, mult = working[0].value, working[1].value, working[2].value
        print(f"\n  📝 CALCUL:")
        print(f"     {working[0].color.upper()} ({d1}) + {working[1].color.upper()} ({d2}) = {d1}{d2}")
        print(f"     × Multiplicateur {working[2].color.upper()} (10^{mult})")
        print(f"     = {result.ohms:.0f} Ω = {result.formatted}")

    if image is not None and roi is not None:
        out = image.copy()
        rx, ry, rw, rh = roi
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.rectangle(out, (rx, ry - 45), (rx + rw + 100, ry - 5), (0, 0, 0), -1)
        cv2.putText(out, f"{result.formatted} {result.tolerance}",
                    (rx + 5, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('RESULTAT - Appuyez sur une touche', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================================================================
# ANALYSE PRINCIPALE
# =============================================================================

def analyze(image_path: str) -> Optional[Result]:
    """Analyse complète d'une image de résistance."""
    start = time.time()

    # Charger l'image
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
    processed = preprocess_advanced(img)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

    # Sélection ROI
    roi_result = select_roi(processed)

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
        # Détection automatique du corps
        foreground = segment_background(hsv)
        body = find_resistor_body(foreground)
        if body:
            rx, ry, rw, rh, orientation = body
            roi = (rx, ry, rw, rh)
            print(f"   🔍 Corps détecté automatiquement: {rw}x{rh} ({orientation})")
        else:
            print("   ❌ Corps non détecté, passage en mode manuel")
            bands = manual_input_bands()
            if not bands:
                return None
            bands = orient_bands(bands)
            result = calculate_resistance(bands)
            if result:
                result.time_ms = (time.time() - start) * 1000
                show_result(None, None, result, bands)
            return result
    else:
        roi = roi_result
        rx, ry, rw, rh = roi

    # Extraire la ROI
    roi_hsv = hsv[ry:ry + rh, rx:rx + rw]
    roi_fg = segment_background(roi_hsv)

    # Déterminer l'orientation
    orientation = "vertical" if rh > rw else "horizontal"

    # Calculer le profil de saturation
    profile = compute_saturation_profile(roi_hsv, roi_fg, orientation)

    # Détecter les bandes
    bands = detect_bands_from_profile(roi_hsv, roi_fg, profile, orientation)

    print(f"\n📊 Bandes détectées: {len(bands)}")
    for i, b in enumerate(bands):
        print(f"   {i + 1}. {b.color.upper():8s} (H={b.h:.0f}, S={b.s:.0f}, V={b.v:.0f})")

    # Demander correction
    print("\n  [ENTRÉE] Calculer  [C] Corriger  [M] Manuel")
    choice = input("  Choix: ").strip().lower()

    if choice in ['c', 'corriger']:
        bands = correct_bands(bands)
    elif choice in ['m', 'manuel']:
        bands = manual_input_bands()
        if not bands:
            return None

    # Orienter et calculer
    bands = orient_bands(bands)
    result = calculate_resistance(bands)

    if result:
        result.time_ms = (time.time() - start) * 1000
        show_result(img, roi, result, bands)
        return result

    print("\n❌ Calcul impossible")
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 55)
    print("  ANALYSEUR DE RÉSISTANCES - V7")
    print("  Techniques de Traitement de Signal Avancées")
    print("  Projet EPHEC 2025")
    print("=" * 55)

    print("\n📖 TECHNIQUES UTILISÉES:")
    print("   • CLAHE (amélioration du contraste)")
    print("   • Filtre bilatéral (réduction du bruit)")
    print("   • Seuillage adaptatif (segmentation)")
    print("   • Filtre médian (lissage du profil)")
    print("   • Détection de pics (scipy.signal)")

    print("\n📖 SENS DE LECTURE:")
    print("   → Tolérance (or/argent) à DROITE")

    # Trouver l'image
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
    print(f"\n{'✅' if result else '❌'} Analyse {'réussie' if result else 'échouée'}")


if __name__ == "__main__":
    main()