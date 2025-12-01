#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION V10
    Sélection Interactive + Correction Manuelle + Amélioration Image
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

FONCTIONNALITÉS:
1. Cliquez sur chaque bande pour analyser la couleur
2. Possibilité de CORRIGER manuellement si la détection est fausse
3. Amélioration avancée de l'image (CLAHE, balance des blancs, etc.)
4. Affichage des valeurs HSV ET RGB pour diagnostic

TECHNIQUES DE TRAITEMENT DE SIGNAL:
1. Balance des blancs automatique
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Filtre bilatéral (réduction du bruit)
4. Correction gamma (amélioration de la luminosité)
5. Augmentation de la saturation

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
    'noir': {'value': 0, 'mult': 1, 'tol': None, 'key': 'n'},
    'marron': {'value': 1, 'mult': 10, 'tol': '±1%', 'key': 'b'},
    'rouge': {'value': 2, 'mult': 100, 'tol': '±2%', 'key': 'r'},
    'orange': {'value': 3, 'mult': 1000, 'tol': None, 'key': 'o'},
    'jaune': {'value': 4, 'mult': 10000, 'tol': None, 'key': 'j'},
    'vert': {'value': 5, 'mult': 100000, 'tol': '±0.5%', 'key': 'v'},
    'bleu': {'value': 6, 'mult': 1000000, 'tol': '±0.25%', 'key': 'l'},
    'violet': {'value': 7, 'mult': 10000000, 'tol': '±0.1%', 'key': 'p'},
    'gris': {'value': 8, 'mult': 100000000, 'tol': '±0.05%', 'key': 'g'},
    'blanc': {'value': 9, 'mult': 1000000000, 'tol': None, 'key': 'w'},
    'or': {'value': -1, 'mult': 0.1, 'tol': '±5%', 'key': 'd'},
    'argent': {'value': -2, 'mult': 0.01, 'tol': '±10%', 'key': 's'},
}

COLOR_LIST = ['noir', 'marron', 'rouge', 'orange', 'jaune', 'vert',
              'bleu', 'violet', 'gris', 'blanc', 'or', 'argent']

# Raccourcis clavier
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
    b: int
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
    Balance des blancs automatique (algorithme Gray World).

    Principe: On suppose que la moyenne des couleurs d'une image
    devrait être grise. On ajuste chaque canal pour atteindre cet équilibre.
    """
    result = img.copy().astype(np.float32)

    # Calculer la moyenne de chaque canal
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    # Moyenne globale
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Ajuster chaque canal
    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r

    # Clipper les valeurs
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Correction gamma pour améliorer la luminosité.

    gamma < 1: image plus claire
    gamma > 1: image plus sombre (mais meilleur contraste)
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def increase_saturation(img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    """
    Augmente la saturation des couleurs.

    Convertit en HSV, multiplie S, reconvertit en BGR.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Pipeline complet d'amélioration de l'image.

    Applique plusieurs techniques de traitement de signal.
    """
    result = img.copy()

    # 1. Balance des blancs
    result = white_balance(result)

    # 2. Réduction du bruit (filtre bilatéral)
    result = cv2.bilateralFilter(result, 9, 75, 75)

    # 3. CLAHE sur le canal L (luminosité)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Correction gamma (légèrement plus lumineux)
    result = gamma_correction(result, gamma=0.9)

    # 5. Augmentation de la saturation
    result = increase_saturation(result, factor=1.4)

    # 6. Amélioration du contraste
    result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)

    return result


# =============================================================================
# IDENTIFICATION DES COULEURS
# =============================================================================

def identify_color_from_hsv(h: float, s: float, v: float) -> Tuple[str, float]:
    """
    Identifie la couleur à partir des valeurs HSV.
    Retourne (nom_couleur, score_de_confiance).
    """
    candidates = []

    # NOIR: V très bas
    if v < 60:
        candidates.append(('noir', 100 - v))

    # BLANC: S très bas, V très haut
    if s < 35 and v > 200:
        candidates.append(('blanc', (255 - s) / 2 + (v - 200)))

    # GRIS: S bas, V moyen
    if s < 50 and 50 <= v <= 200:
        candidates.append(('gris', 80 - s))

    # ARGENT: S bas, V moyen-haut
    if s < 60 and 130 <= v <= 230:
        candidates.append(('argent', 70 - s + (v - 130) / 3))

    # OR: H jaune-orange (12-38), S moyenne-haute
    if 10 <= h <= 40 and s > 50 and 70 <= v <= 230:
        score = 80 - abs(h - 22) * 1.5
        candidates.append(('or', score))

    # Couleurs saturées
    if s > 40:
        # ROUGE: H proche de 0 ou 180
        if h <= 12 or h >= 165:
            red_h = h if h <= 12 else 180 - h
            score = 90 - red_h * 2
            # Distinguer rouge/marron par V
            if v < 100:
                candidates.append(('marron', score - 5))
            else:
                candidates.append(('rouge', score))

        # MARRON: H rouge-orange, V bas-moyen
        if 0 <= h <= 25 and v < 130:
            score = 75 - abs(h - 10) * 2
            candidates.append(('marron', score))

        # ORANGE: H = 8-25, S et V hauts
        if 8 <= h <= 28 and s > 100 and v > 140:
            score = 85 - abs(h - 15) * 2
            candidates.append(('orange', score))

        # JAUNE: H = 22-45
        if 20 <= h <= 50 and s > 60:
            score = 85 - abs(h - 30) * 1.5
            candidates.append(('jaune', score))

        # VERT: H = 35-90
        if 30 <= h <= 95:
            score = 90 - abs(h - 60)
            candidates.append(('vert', score))

        # BLEU: H = 85-135
        if 80 <= h <= 140:
            score = 90 - abs(h - 110)
            candidates.append(('bleu', score))

        # VIOLET: H = 125-170
        if 120 <= h <= 175:
            score = 85 - abs(h - 145) * 1.5
            candidates.append(('violet', score))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        return best

    return ('inconnu', 0)


def analyze_pixel_area(img_bgr: np.ndarray, hsv_img: np.ndarray,
                       x: int, y: int, radius: int = 8) -> dict:
    """
    Analyse une zone de pixels autour d'un point.
    Retourne les valeurs H, S, V et R, G, B moyennes.
    """
    h_img, w_img = hsv_img.shape[:2]

    # Zone d'analyse
    x1 = max(0, x - radius)
    x2 = min(w_img, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h_img, y + radius + 1)

    # Extraire les zones
    zone_hsv = hsv_img[y1:y2, x1:x2]
    zone_bgr = img_bgr[y1:y2, x1:x2]

    # Calculer les moyennes HSV
    h_mean = np.mean(zone_hsv[:, :, 0])
    s_mean = np.mean(zone_hsv[:, :, 1])
    v_mean = np.mean(zone_hsv[:, :, 2])

    # Calculer les moyennes RGB
    b_mean = int(np.mean(zone_bgr[:, :, 0]))
    g_mean = int(np.mean(zone_bgr[:, :, 1]))
    r_mean = int(np.mean(zone_bgr[:, :, 2]))

    return {
        'h': h_mean, 's': s_mean, 'v': v_mean,
        'r': r_mean, 'g': g_mean, 'b': b_mean
    }


# =============================================================================
# INTERFACE INTERACTIVE
# =============================================================================

class BandSelector:
    def __init__(self, image_original: np.ndarray, image_enhanced: np.ndarray, hsv: np.ndarray):
        self.original = image_original.copy()
        self.enhanced = image_enhanced.copy()
        self.display = image_enhanced.copy()
        self.hsv = hsv
        self.bands: List[Band] = []
        self.current_info = None
        self.show_enhanced = True
        self.window_name = "Cliquez sur les bandes | ESPACE=ajouter | Lettre=forcer couleur | ENTREE=fin"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_to_use = self.enhanced if self.show_enhanced else self.original
            hsv_to_use = cv2.cvtColor(img_to_use, cv2.COLOR_BGR2HSV)

            info = analyze_pixel_area(img_to_use, hsv_to_use, x, y, radius=10)
            color, confidence = identify_color_from_hsv(info['h'], info['s'], info['v'])

            self.current_info = {
                'color': color,
                'confidence': confidence,
                'x': x, 'y': y,
                **info
            }
            self.update_display()

    def update_display(self):
        base = self.enhanced if self.show_enhanced else self.original
        self.display = base.copy()

        # Bandes déjà validées
        for i, band in enumerate(self.bands):
            cv2.circle(self.display, (band.x, band.y), 12, (0, 255, 0), 2)
            cv2.putText(self.display, f"{i + 1}",
                        (band.x - 5, band.y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Info box des bandes validées
        y_offset = 10
        for i, band in enumerate(self.bands):
            cv2.rectangle(self.display, (10, y_offset), (180, y_offset + 20), (0, 80, 0), -1)
            cv2.putText(self.display, f"{i + 1}: {band.color.upper()}",
                        (15, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # Sélection courante
        if self.current_info:
            x, y = self.current_info['x'], self.current_info['y']

            # Cercle jaune
            cv2.circle(self.display, (x, y), 15, (0, 255, 255), 3)

            # Info box en bas
            h_img = self.display.shape[0]
            cv2.rectangle(self.display, (0, h_img - 100), (400, h_img), (40, 40, 40), -1)

            color = self.current_info['color']
            conf = self.current_info['confidence']
            h, s, v = self.current_info['h'], self.current_info['s'], self.current_info['v']
            r, g, b = self.current_info['r'], self.current_info['g'], self.current_info['b']

            cv2.putText(self.display, f"Detecte: {color.upper()} ({conf:.0f}%)",
                        (10, h_img - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(self.display, f"HSV: H={h:.0f} S={s:.0f} V={v:.0f}",
                        (10, h_img - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(self.display, f"RGB: R={r} G={g} B={b}",
                        (10, h_img - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(self.display, "ESPACE=OK | Lettre=forcer | Z=annuler",
                        (10, h_img - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Carré de couleur
            cv2.rectangle(self.display, (320, h_img - 90), (390, h_img - 20), (b, g, r), -1)
            cv2.rectangle(self.display, (320, h_img - 90), (390, h_img - 20), (255, 255, 255), 2)

        # Instructions en haut à droite
        w_img = self.display.shape[1]
        mode = "AMELIOREE" if self.show_enhanced else "ORIGINALE"
        cv2.putText(self.display, f"[T] Image: {mode}",
                    (w_img - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def add_current_band(self, force_color: str = None):
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
            b=self.current_info['b'],
            x=self.current_info['x'],
            y=self.current_info['y']
        )
        self.bands.append(band)

        forced = " (forcé)" if force_color else ""
        print(f"  ✓ Bande {len(self.bands)}: {color.upper()}{forced}")
        print(f"      HSV=({band.h:.0f},{band.s:.0f},{band.v:.0f}) RGB=({band.r},{band.g},{band.b})")

        self.current_info = None
        self.update_display()

    def run(self) -> List[Band]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 65)
        print("  SÉLECTION INTERACTIVE DES BANDES")
        print("=" * 65)
        print("  CLIQUEZ sur chaque bande de couleur (gauche → droite)")
        print()
        print("  RACCOURCIS CLAVIER:")
        print("  ├─ ESPACE ....... Confirmer la couleur détectée")
        print("  ├─ N/0 .......... Forcer NOIR")
        print("  ├─ B/1 .......... Forcer MARRON (Brown)")
        print("  ├─ R/2 .......... Forcer ROUGE")
        print("  ├─ O/3 .......... Forcer ORANGE")
        print("  ├─ J/4 .......... Forcer JAUNE")
        print("  ├─ V/5 .......... Forcer VERT")
        print("  ├─ L/6 .......... Forcer BLEU (bLue)")
        print("  ├─ P/7 .......... Forcer VIOLET (Purple)")
        print("  ├─ G/8 .......... Forcer GRIS (Gray)")
        print("  ├─ W/9 .......... Forcer BLANC (White)")
        print("  ├─ D ............ Forcer OR (Doré)")
        print("  ├─ S ............ Forcer ARGENT (Silver)")
        print("  ├─ Z ............ Annuler dernière bande")
        print("  ├─ T ............ Basculer image originale/améliorée")
        print("  └─ ENTRÉE ....... Terminer et calculer")
        print("=" * 65)

        self.update_display()

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            # ESPACE: confirmer
            if key == ord(' ') and self.current_info:
                self.add_current_band()

            # Lettres/chiffres: forcer une couleur
            elif key in KEY_TO_COLOR and self.current_info:
                self.add_current_band(force_color=KEY_TO_COLOR[key])

            # Z: annuler
            elif key == ord('z') and self.bands:
                removed = self.bands.pop()
                print(f"  ✗ Annulé: {removed.color.upper()}")
                self.update_display()

            # T: basculer image
            elif key == ord('t'):
                self.show_enhanced = not self.show_enhanced
                mode = "améliorée" if self.show_enhanced else "originale"
                print(f"  → Image {mode}")
                if self.current_info:
                    # Recalculer avec la nouvelle image
                    x, y = self.current_info['x'], self.current_info['y']
                    img_to_use = self.enhanced if self.show_enhanced else self.original
                    hsv_to_use = cv2.cvtColor(img_to_use, cv2.COLOR_BGR2HSV)
                    info = analyze_pixel_area(img_to_use, hsv_to_use, x, y, radius=10)
                    color, confidence = identify_color_from_hsv(info['h'], info['s'], info['v'])
                    self.current_info = {'color': color, 'confidence': confidence, 'x': x, 'y': y, **info}
                self.update_display()

            # ENTRÉE: terminer
            elif key == 13:
                if len(self.bands) >= 3:
                    break
                else:
                    print("  ⚠️  Minimum 3 bandes requises!")

            # ESC: quitter
            elif key == 27:
                self.bands = []
                break

        cv2.destroyAllWindows()
        return self.bands


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    # Tolérance à la fin?
    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLOR_CONFIG[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    # Vérifier les 3 premières valeurs
    for i in range(3):
        if working[i].value < 0 or working[i].value > 9:
            return None

    d1 = working[0].value
    d2 = working[1].value
    mult = working[2].value

    base = d1 * 10 + d2
    multiplier = 10 ** mult
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
        bands=[b.color.upper() for b in bands]
    )


def show_result(result: Result, bands: List[Band]):
    print("\n" + "=" * 65)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 65)
    print(f"  Bandes: {' → '.join(result.bands)}")

    working = [b for b in bands if 0 <= b.value <= 9][:3]
    if len(working) >= 3:
        d1, d2, m = working[0].value, working[1].value, working[2].value
        print(f"\n  📝 CALCUL:")
        print(f"     Chiffre 1: {working[0].color.upper()} = {d1}")
        print(f"     Chiffre 2: {working[1].color.upper()} = {d2}")
        print(f"     Multipli.: {working[2].color.upper()} = ×10^{m}")
        print(f"     ─────────────────────────────")
        print(f"     ({d1} × 10 + {d2}) × 10^{m}")
        print(f"     = {d1}{d2} × {10 ** m:,}")
        print(f"     = {result.ohms:,.0f} Ω")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  ANALYSEUR DE RÉSISTANCES - V10")
    print("  Sélection Interactive + Correction Manuelle")
    print("  Projet Traitement de Signal - EPHEC 2025")
    print("=" * 65)

    print("\n📖 TECHNIQUES D'AMÉLIORATION D'IMAGE:")
    print("   • Balance des blancs (Gray World)")
    print("   • CLAHE (contraste adaptatif)")
    print("   • Filtre bilatéral (réduction du bruit)")
    print("   • Correction gamma")
    print("   • Augmentation de la saturation")

    print("\n📖 UTILISATION:")
    print("   1. Cliquez sur chaque bande")
    print("   2. ESPACE pour confirmer OU lettre pour forcer la couleur")
    print("   3. ENTRÉE quand terminé")

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
            print("   Usage: python resistance_v10.py image.jpg")
            return

    # Charger
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Impossible de charger: {path}")
        return

    # Redimensionner
    h, w = img.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    print(f"\n📷 Image: {path}")
    print(f"   Dimensions: {img.shape[1]}x{img.shape[0]}")

    # Améliorer l'image
    print("   Amélioration en cours...")
    enhanced = enhance_image(img)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    print("   ✓ Image améliorée")

    # Sélection interactive
    selector = BandSelector(img, enhanced, hsv)
    bands = selector.run()

    if not bands:
        print("\n❌ Aucune bande sélectionnée")
        return

    # Calculer
    result = calculate_resistance(bands)

    if result:
        show_result(result, bands)
        print("\n✅ Analyse réussie!")
    else:
        print("\n❌ Calcul impossible")
        print("   Vérifiez les bandes sélectionnées")


if __name__ == "__main__":
    main()