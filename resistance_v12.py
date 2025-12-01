#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION V11
    Avec Fenêtre de Réglage des Couleurs (Sliders)
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

NOUVEAUTÉS:
1. Fenêtre de COMPARAISON: Original vs Amélioré côte à côte
2. SLIDERS pour ajuster en temps réel:
   - Saturation
   - Contraste
   - Luminosité (Gamma)
   - CLAHE (contraste local)
3. Analyse RGB et HSV des pixels cliqués
4. Correction manuelle des couleurs par raccourcis clavier

TECHNIQUES DE TRAITEMENT DE SIGNAL:
- Balance des blancs (Gray World Algorithm)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Filtre bilatéral (réduction du bruit)
- Correction gamma
- Ajustement de saturation
- Ajustement de contraste/luminosité

=====================================================================
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import glob

# =============================================================================
# CONFIGURATION DES COULEURS
# =============================================================================

COLOR_CONFIG = {
    'noir': {'value': 0, 'mult': 1, 'tol': None},
    'marron': {'value': 1, 'mult': 10, 'tol': '±1%'},
    'rouge': {'value': 2, 'mult': 100, 'tol': '±2%'},
    'orange': {'value': 3, 'mult': 1000, 'tol': None},
    'jaune': {'value': 4, 'mult': 10000, 'tol': None},
    'vert': {'value': 5, 'mult': 100000, 'tol': '±0.5%'},
    'bleu': {'value': 6, 'mult': 1000000, 'tol': '±0.25%'},
    'violet': {'value': 7, 'mult': 10000000, 'tol': '±0.1%'},
    'gris': {'value': 8, 'mult': 100000000, 'tol': '±0.05%'},
    'blanc': {'value': 9, 'mult': 1000000000, 'tol': None},
    'or': {'value': -1, 'mult': 0.1, 'tol': '±5%'},
    'argent': {'value': -2, 'mult': 0.01, 'tol': '±10%'},
}

COLOR_LIST = ['noir', 'marron', 'rouge', 'orange', 'jaune', 'vert',
              'bleu', 'violet', 'gris', 'blanc', 'or', 'argent']

KEY_TO_COLOR = {
    ord('n'): 'noir', ord('0'): 'noir',
    ord('b'): 'marron', ord('1'): 'marron',
    ord('r'): 'rouge', ord('2'): 'rouge',
    ord('o'): 'orange', ord('3'): 'orange',
    ord('j'): 'jaune', ord('4'): 'jaune',
    ord('v'): 'vert', ord('5'): 'vert',
    ord('l'): 'bleu', ord('6'): 'bleu',
    ord('p'): 'violet', ord('7'): 'violet',
    ord('g'): 'gris', ord('8'): 'gris',
    ord('w'): 'blanc', ord('9'): 'blanc',
    ord('d'): 'or',
    ord('s'): 'argent',
}


# =============================================================================
# STRUCTURES
# =============================================================================

@dataclass
class Band:
    color: str
    value: int
    h: float
    s: float
    v: float
    r: int
    g: int
    b_val: int
    x: int
    y: int


@dataclass
class Result:
    ohms: float
    formatted: str
    tolerance: str
    bands: List[str]


# =============================================================================
# AMÉLIORATION DE L'IMAGE
# =============================================================================

def white_balance(img: np.ndarray) -> np.ndarray:
    """
    Balance des blancs - Algorithme Gray World.
    Corrige les dominantes de couleur (ex: photo trop jaune/bleue).
    """
    result = img.copy().astype(np.float32)

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization.
    Améliore le contraste local sans sur-exposer.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Correction Gamma - Ajuste la luminosité.
    gamma < 1: plus lumineux
    gamma > 1: plus sombre
    """
    if gamma == 1.0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def adjust_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Ajuste la saturation des couleurs.
    factor > 1: couleurs plus vives
    factor < 1: couleurs plus ternes
    """
    if factor == 1.0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast_brightness(img: np.ndarray, contrast: float = 1.0,
                               brightness: int = 0) -> np.ndarray:
    """
    Ajuste le contraste et la luminosité.
    contrast: facteur multiplicatif (1.0 = normal)
    brightness: valeur à ajouter (-100 à +100)
    """
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def enhance_image(img: np.ndarray, saturation: float = 1.5,
                  contrast: float = 1.2, gamma: float = 0.85,
                  clahe_clip: float = 3.0, brightness: int = 10) -> np.ndarray:
    """
    Pipeline complet d'amélioration avec paramètres ajustables.
    """
    result = img.copy()

    # 1. Balance des blancs
    result = white_balance(result)

    # 2. Filtre bilatéral (réduction du bruit)
    result = cv2.bilateralFilter(result, 9, 75, 75)

    # 3. CLAHE
    if clahe_clip > 0:
        result = apply_clahe(result, clahe_clip)

    # 4. Gamma
    result = adjust_gamma(result, gamma)

    # 5. Saturation
    result = adjust_saturation(result, saturation)

    # 6. Contraste et luminosité
    result = adjust_contrast_brightness(result, contrast, brightness)

    return result


# =============================================================================
# IDENTIFICATION DES COULEURS
# =============================================================================

def identify_color_from_hsv(h: float, s: float, v: float) -> Tuple[str, float]:
    """
    Identifie la couleur à partir des valeurs HSV.

    Seuils calibrés pour les résistances:
    - NOIR: V < 90 (les noirs sur résistances ne sont pas purs)
    - MARRON: V entre 90-150, H ~ 10-25
    - OR: H ~ 15-35, S moyen, V moyen-haut
    """
    candidates = []

    # NOIR - Seuil augmenté car le noir sur résistance n'est pas pur
    # Le noir a une luminosité basse (V < 90)
    if v < 90:
        # Plus V est bas, plus c'est probablement noir
        score = 100 - v
        # Si S est aussi bas, c'est encore plus probablement noir
        if s < 80:
            score += 10
        candidates.append(('noir', score))

    # BLANC
    if s < 35 and v > 200:
        candidates.append(('blanc', 50 + (v - 200) - s))

    # GRIS
    if s < 50 and 60 <= v <= 200:
        candidates.append(('gris', 80 - s))

    # ARGENT
    if s < 60 and 130 <= v <= 230:
        candidates.append(('argent', 70 - s))

    # OR - H entre 15-35, S moyen, V pas trop bas
    if 12 <= h <= 38 and 40 <= s <= 180 and 90 <= v <= 230:
        score = 85 - abs(h - 22) * 1.5
        candidates.append(('or', score))

    if s > 40:
        # ROUGE - H proche de 0 ou 180, V haut
        if h <= 10 or h >= 168:
            red_h = h if h <= 10 else 180 - h
            score = 90 - red_h * 2
            if v < 110:
                # V bas + H rouge = marron
                candidates.append(('marron', score - 5))
            else:
                candidates.append(('rouge', score))

        # MARRON - H rouge-orange (10-25), V moyen (90-150)
        if 8 <= h <= 28 and 90 <= v <= 160:
            score = 80 - abs(h - 15) * 2
            # Plus S est haut, plus c'est marron
            if s > 100:
                score += 10
            candidates.append(('marron', score))

        # ORANGE - H ~ 10-25, S très haut, V haut
        if 8 <= h <= 28 and s > 150 and v > 150:
            candidates.append(('orange', 85 - abs(h - 15) * 2))

        # JAUNE
        if 22 <= h <= 50 and s > 60 and v > 120:
            candidates.append(('jaune', 85 - abs(h - 32) * 1.5))

        # VERT
        if 35 <= h <= 95:
            candidates.append(('vert', 90 - abs(h - 60)))

        # BLEU
        if 85 <= h <= 135:
            candidates.append(('bleu', 90 - abs(h - 110)))

        # VIOLET
        if 125 <= h <= 170:
            candidates.append(('violet', 85 - abs(h - 145) * 1.5))

    if candidates:
        return max(candidates, key=lambda x: x[1])
    return ('inconnu', 0)


def analyze_pixel_area(img_bgr: np.ndarray, x: int, y: int, radius: int = 3) -> dict:
    """
    Analyse une zone RESTREINTE de pixels pour plus de précision.

    Rayon par défaut: 3 pixels (zone 7x7)
    Avant c'était 10 pixels (zone 21x21) - trop large!
    """
    h_img, w_img = img_bgr.shape[:2]

    x1, x2 = max(0, x - radius), min(w_img, x + radius + 1)
    y1, y2 = max(0, y - radius), min(h_img, y + radius + 1)

    zone_bgr = img_bgr[y1:y2, x1:x2]
    zone_hsv = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2HSV)

    return {
        'h': np.mean(zone_hsv[:, :, 0]),
        's': np.mean(zone_hsv[:, :, 1]),
        'v': np.mean(zone_hsv[:, :, 2]),
        'r': int(np.mean(zone_bgr[:, :, 2])),
        'g': int(np.mean(zone_bgr[:, :, 1])),
        'b': int(np.mean(zone_bgr[:, :, 0]))
    }


# =============================================================================
# FENÊTRE DE RÉGLAGE DES PARAMÈTRES
# =============================================================================

class ImageEnhancer:
    """Fenêtre avec sliders pour ajuster les paramètres d'amélioration."""

    def __init__(self, image: np.ndarray):
        self.original = image.copy()
        self.enhanced = image.copy()

        # Paramètres par défaut
        self.saturation = 150  # /100 = 1.5
        self.contrast = 120  # /100 = 1.2
        self.brightness = 60  # -50 = +10 (offset de 50)
        self.gamma = 85  # /100 = 0.85
        self.clahe = 30  # /10 = 3.0

        self.window_name = "Reglage des couleurs - Ajustez les sliders puis appuyez sur ENTREE"

    def update(self, _=None):
        """Met à jour l'image avec les paramètres actuels."""
        sat = self.saturation / 100.0
        cont = self.contrast / 100.0
        bright = self.brightness - 50  # Offset pour avoir -50 à +50
        gam = self.gamma / 100.0
        clahe_clip = self.clahe / 10.0

        self.enhanced = enhance_image(
            self.original,
            saturation=sat,
            contrast=cont,
            gamma=gam,
            clahe_clip=clahe_clip,
            brightness=bright
        )

        # Créer l'image de comparaison
        h, w = self.original.shape[:2]
        comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        comparison[:, :w] = self.original
        comparison[:, w + 10:] = self.enhanced

        # Ligne de séparation
        comparison[:, w:w + 10] = (100, 100, 100)

        # Labels avec instructions de touches
        # Fond noir pour meilleure lisibilité
        cv2.rectangle(comparison, (5, 5), (220, 60), (0, 0, 0), -1)
        cv2.rectangle(comparison, (w + 15, 5), (w + 250, 60), (0, 0, 0), -1)

        cv2.putText(comparison, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "[O] pour utiliser", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)

        cv2.putText(comparison, "AMELIORE", (w + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison, "[ENTREE] pour utiliser", (w + 20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

        # Afficher les valeurs actuelles en bas à droite
        info_y = h - 80
        cv2.rectangle(comparison, (w + 15, info_y - 5), (w * 2 + 5, h - 5), (0, 0, 0), -1)
        cv2.putText(comparison, f"Saturation: {sat:.1f}x", (w + 20, info_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(comparison, f"Contraste: {cont:.1f}x  Lum: {bright:+d}", (w + 20, info_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(comparison, f"Gamma: {gam:.2f}  CLAHE: {clahe_clip:.1f}", (w + 20, info_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self.window_name, comparison)

    def run(self) -> Tuple[np.ndarray, bool]:
        """Affiche la fenêtre et retourne (image, use_enhanced)."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Créer les sliders
        cv2.createTrackbar("Saturation", self.window_name, self.saturation, 300, self.update)
        cv2.createTrackbar("Contraste", self.window_name, self.contrast, 200, self.update)
        cv2.createTrackbar("Luminosite", self.window_name, self.brightness, 100, self.update)
        cv2.createTrackbar("Gamma", self.window_name, self.gamma, 200, self.update)
        cv2.createTrackbar("CLAHE", self.window_name, self.clahe, 80, self.update)

        print("\n" + "=" * 65)
        print("  RÉGLAGE DES COULEURS")
        print("=" * 65)
        print("  Ajustez les sliders pour faire ressortir les couleurs:")
        print()
        print("  • SATURATION: Intensité des couleurs (↑ = plus vif)")
        print("  • CONTRASTE:  Différence clair/sombre (↑ = plus marqué)")
        print("  • LUMINOSITÉ: Clarté globale")
        print("  • GAMMA:      Luminosité des tons moyens (↓ = plus clair)")
        print("  • CLAHE:      Contraste local adaptatif")
        print()
        print("  TOUCHES:")
        print("  ├─ ENTRÉE .... Utiliser l'image AMÉLIORÉE (droite)")
        print("  ├─ O ......... Utiliser l'image ORIGINALE (gauche)")
        print("  ├─ R ......... Réinitialiser les paramètres")
        print("  └─ ESC ....... Annuler")
        print("=" * 65)

        self.update()

        use_enhanced = True

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Lire les valeurs des sliders
            self.saturation = cv2.getTrackbarPos("Saturation", self.window_name)
            self.contrast = cv2.getTrackbarPos("Contraste", self.window_name)
            self.brightness = cv2.getTrackbarPos("Luminosite", self.window_name)
            self.gamma = cv2.getTrackbarPos("Gamma", self.window_name)
            self.clahe = cv2.getTrackbarPos("CLAHE", self.window_name)

            if key == 13:  # ENTRÉE - utiliser améliorée
                use_enhanced = True
                print("  → Image AMÉLIORÉE sélectionnée")
                break
            elif key == ord('o'):  # O - utiliser originale
                use_enhanced = False
                print("  → Image ORIGINALE sélectionnée")
                break
            elif key == ord('r'):  # Reset
                cv2.setTrackbarPos("Saturation", self.window_name, 150)
                cv2.setTrackbarPos("Contraste", self.window_name, 120)
                cv2.setTrackbarPos("Luminosite", self.window_name, 60)
                cv2.setTrackbarPos("Gamma", self.window_name, 85)
                cv2.setTrackbarPos("CLAHE", self.window_name, 30)
                self.update()
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return self.original, False

        cv2.destroyWindow(self.window_name)

        if use_enhanced:
            return self.enhanced, True
        else:
            return self.original, False


# =============================================================================
# SÉLECTION DES BANDES
# =============================================================================

class BandSelector:
    def __init__(self, image: np.ndarray):
        self.image = image.copy()
        self.display = image.copy()
        self.bands: List[Band] = []
        self.current_info = None
        self.radius = 3  # Rayon d'analyse par défaut (zone 7x7)
        self.window_name = "Cliquez sur les bandes | ESPACE=ajouter | Lettre=forcer | ENTREE=fin"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            info = analyze_pixel_area(self.image, x, y, radius=self.radius)
            color, confidence = identify_color_from_hsv(info['h'], info['s'], info['v'])

            self.current_info = {
                'color': color, 'confidence': confidence,
                'x': x, 'y': y, **info
            }
            self.update_display()

    def update_display(self):
        self.display = self.image.copy()
        h_img, w_img = self.display.shape[:2]

        # Bandes validées
        for i, band in enumerate(self.bands):
            cv2.circle(self.display, (band.x, band.y), 10, (0, 255, 0), 2)
            cv2.putText(self.display, str(i + 1), (band.x - 5, band.y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Liste des bandes en haut à gauche
        for i, band in enumerate(self.bands):
            y_pos = 25 + i * 25
            cv2.rectangle(self.display, (5, y_pos - 18), (160, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(self.display, f"{i + 1}: {band.color.upper()}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Afficher le rayon actuel en haut à droite
        cv2.rectangle(self.display, (w_img - 180, 5), (w_img - 5, 55), (0, 0, 0), -1)
        cv2.putText(self.display, f"Rayon: {self.radius} px",
                    (w_img - 170, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(self.display, f"Zone: {2 * self.radius + 1}x{2 * self.radius + 1}",
                    (w_img - 170, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Sélection courante
        if self.current_info:
            x, y = self.current_info['x'], self.current_info['y']

            # Cercle extérieur (guide visuel)
            cv2.circle(self.display, (x, y), 15, (0, 255, 255), 1)
            # Cercle de la zone analysée (rayon réel)
            cv2.circle(self.display, (x, y), self.radius, (0, 0, 255), 2)
            # Point central
            cv2.circle(self.display, (x, y), 1, (255, 255, 255), -1)

            # Box d'info en bas
            cv2.rectangle(self.display, (0, h_img - 130), (450, h_img), (30, 30, 30), -1)

            color = self.current_info['color']
            conf = self.current_info['confidence']
            h, s, v = self.current_info['h'], self.current_info['s'], self.current_info['v']
            r, g, b = self.current_info['r'], self.current_info['g'], self.current_info['b']

            cv2.putText(self.display, f"Couleur detectee: {color.upper()} ({conf:.0f}%)",
                        (10, h_img - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(self.display, f"HSV: H={h:.0f}  S={s:.0f}  V={v:.0f}",
                        (10, h_img - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(self.display, f"RGB: R={r}  G={g}  B={b}",
                        (10, h_img - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(self.display,
                        f"[+/-] Rayon: {self.radius}px -> zone {2 * self.radius + 1}x{2 * self.radius + 1}",
                        (10, h_img - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Carré de couleur
            cv2.rectangle(self.display, (360, h_img - 120), (440, h_img - 40), (b, g, r), -1)
            cv2.rectangle(self.display, (360, h_img - 120), (440, h_img - 40), (255, 255, 255), 2)

            cv2.putText(self.display, "ESPACE=OK | Lettre=forcer",
                        (10, h_img - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def add_band(self, force_color: str = None):
        if not self.current_info:
            return

        color = force_color if force_color else self.current_info['color']

        if color not in COLOR_CONFIG:
            print(f"  ❌ Couleur inconnue: {color}")
            return

        band = Band(
            color=color,
            value=COLOR_CONFIG[color]['value'],
            h=self.current_info['h'],
            s=self.current_info['s'],
            v=self.current_info['v'],
            r=self.current_info['r'],
            g=self.current_info['g'],
            b_val=self.current_info['b'],
            x=self.current_info['x'],
            y=self.current_info['y']
        )
        self.bands.append(band)

        marker = " (forcé)" if force_color else ""
        print(f"  ✓ Bande {len(self.bands)}: {color.upper()}{marker}")
        print(f"      HSV=({band.h:.0f}, {band.s:.0f}, {band.v:.0f})  RGB=({band.r}, {band.g}, {band.b_val})")

        self.current_info = None
        self.update_display()

    def run(self) -> List[Band]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 65)
        print("  SÉLECTION DES BANDES")
        print("=" * 65)
        print("  CLIQUEZ sur chaque bande (gauche → droite)")
        print()
        print("  RACCOURCIS:")
        print("  ├─ ESPACE .... Confirmer la couleur détectée")
        print("  ├─ R ......... Forcer ROUGE")
        print("  ├─ B ......... Forcer MARRON (Brown)")
        print("  ├─ O ......... Forcer ORANGE")
        print("  ├─ J ......... Forcer JAUNE")
        print("  ├─ V ......... Forcer VERT")
        print("  ├─ L ......... Forcer BLEU")
        print("  ├─ D ......... Forcer OR (Doré)")
        print("  ├─ S ......... Forcer ARGENT")
        print("  ├─ N ......... Forcer NOIR")
        print("  ├─ + ......... Augmenter rayon d'analyse")
        print("  ├─ - ......... Réduire rayon d'analyse")
        print("  ├─ Z ......... Annuler dernière bande")
        print("  └─ ENTRÉE .... Terminer")
        print("=" * 65)

        self.update_display()

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and self.current_info:
                self.add_band()
            elif key in KEY_TO_COLOR and self.current_info:
                self.add_band(force_color=KEY_TO_COLOR[key])
            elif key == ord('z') and self.bands:
                removed = self.bands.pop()
                print(f"  ✗ Annulé: {removed.color.upper()}")
                self.update_display()
            elif key == ord('+') or key == ord('='):
                self.radius = min(20, self.radius + 1)
                print(f"  → Rayon: {self.radius} px (zone {2 * self.radius + 1}x{2 * self.radius + 1})")
                if self.current_info:
                    # Recalculer avec le nouveau rayon
                    x, y = self.current_info['x'], self.current_info['y']
                    info = analyze_pixel_area(self.image, x, y, radius=self.radius)
                    color, conf = identify_color_from_hsv(info['h'], info['s'], info['v'])
                    self.current_info = {'color': color, 'confidence': conf, 'x': x, 'y': y, **info}
                self.update_display()
            elif key == ord('-') or key == ord('_'):
                self.radius = max(1, self.radius - 1)
                print(f"  → Rayon: {self.radius} px (zone {2 * self.radius + 1}x{2 * self.radius + 1})")
                if self.current_info:
                    x, y = self.current_info['x'], self.current_info['y']
                    info = analyze_pixel_area(self.image, x, y, radius=self.radius)
                    color, conf = identify_color_from_hsv(info['h'], info['s'], info['v'])
                    self.current_info = {'color': color, 'confidence': conf, 'x': x, 'y': y, **info}
                self.update_display()
            elif key == 13:
                if len(self.bands) >= 3:
                    break
                print("  ⚠️  Minimum 3 bandes!")
            elif key == 27:
                self.bands = []
                break

        cv2.destroyAllWindows()
        return self.bands


# =============================================================================
# CALCUL
# =============================================================================

def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLOR_CONFIG[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    for b in working[:3]:
        if b.value < 0 or b.value > 9:
            return None

    d1, d2, mult = working[0].value, working[1].value, working[2].value
    ohms = (d1 * 10 + d2) * (10 ** mult)

    if ohms >= 1e6:
        fmt = f"{ohms / 1e6:.2f} MΩ"
    elif ohms >= 1e3:
        fmt = f"{ohms / 1e3:.2f} kΩ"
    else:
        fmt = f"{ohms:.2f} Ω"

    return Result(ohms=ohms, formatted=fmt, tolerance=tolerance,
                  bands=[b.color.upper() for b in bands])


def show_result(result: Result, bands: List[Band]):
    print("\n" + "=" * 65)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 65)
    print(f"  Bandes: {' → '.join(result.bands)}")

    working = [b for b in bands if 0 <= b.value <= 9][:3]
    if len(working) >= 3:
        d1, d2, m = working[0].value, working[1].value, working[2].value
        print(f"\n  📝 CALCUL:")
        print(f"     {working[0].color.upper()} = {d1}")
        print(f"     {working[1].color.upper()} = {d2}")
        print(f"     {working[2].color.upper()} = ×10^{m}")
        print(f"     → ({d1}×10 + {d2}) × 10^{m} = {result.ohms:,.0f} Ω")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  ANALYSEUR DE RÉSISTANCES - V11")
    print("  Avec Réglage Interactif des Couleurs")
    print("  Projet Traitement de Signal - EPHEC 2025")
    print("=" * 65)

    print("\n📖 ÉTAPES:")
    print("   1. Ajustez les sliders pour améliorer les couleurs")
    print("   2. Cliquez sur chaque bande")
    print("   3. Confirmez ou forcez la couleur détectée")

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
            print("\n❌ Aucune image trouvée!")
            return

    # Charger
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Impossible de charger: {path}")
        return

    # Redimensionner
    h, w = img.shape[:2]
    max_dim = 600  # Plus petit pour voir la comparaison côte à côte
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    print(f"\n📷 Image: {path} ({img.shape[1]}x{img.shape[0]})")

    # Étape 1: Réglage des couleurs
    enhancer = ImageEnhancer(img)
    selected_image, is_enhanced = enhancer.run()

    if is_enhanced:
        print("   ✓ Utilisation de l'image améliorée")
    else:
        print("   ✓ Utilisation de l'image originale")

    # Étape 2: Sélection des bandes
    selector = BandSelector(selected_image)
    bands = selector.run()

    if not bands:
        print("\n❌ Aucune bande sélectionnée")
        return

    # Étape 3: Calcul
    result = calculate_resistance(bands)

    if result:
        show_result(result, bands)
        print("\n✅ Analyse réussie!")
    else:
        print("\n❌ Calcul impossible")


if __name__ == "__main__":
    main()