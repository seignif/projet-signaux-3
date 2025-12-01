#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES - VERSION V9
    Avec Sélection Interactive des Bandes (Clic sur l'image)
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

NOUVELLE APPROCHE:
Au lieu de détecter automatiquement les couleurs (souvent imprécis),
l'utilisateur CLIQUE sur chaque bande et le programme analyse la
couleur à cet endroit précis.

TECHNIQUES UTILISÉES:
1. Analyse des pixels dans une zone autour du clic
2. Conversion RGB → HSV pour identification de couleur
3. Moyenne des valeurs pour robustesse au bruit
4. Affichage des valeurs HSV pour diagnostic

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
    'noir': {'value': 0, 'mult': 1, 'tol': None, 'rgb': (0, 0, 0)},
    'marron': {'value': 1, 'mult': 10, 'tol': '±1%', 'rgb': (139, 69, 19)},
    'rouge': {'value': 2, 'mult': 100, 'tol': '±2%', 'rgb': (255, 0, 0)},
    'orange': {'value': 3, 'mult': 1000, 'tol': None, 'rgb': (255, 165, 0)},
    'jaune': {'value': 4, 'mult': 10000, 'tol': None, 'rgb': (255, 255, 0)},
    'vert': {'value': 5, 'mult': 100000, 'tol': '±0.5%', 'rgb': (0, 255, 0)},
    'bleu': {'value': 6, 'mult': 1000000, 'tol': '±0.25%', 'rgb': (0, 0, 255)},
    'violet': {'value': 7, 'mult': 10000000, 'tol': '±0.1%', 'rgb': (148, 0, 211)},
    'gris': {'value': 8, 'mult': 100000000, 'tol': '±0.05%', 'rgb': (128, 128, 128)},
    'blanc': {'value': 9, 'mult': 1000000000, 'tol': None, 'rgb': (255, 255, 255)},
    'or': {'value': -1, 'mult': 0.1, 'tol': '±5%', 'rgb': (255, 215, 0)},
    'argent': {'value': -2, 'mult': 0.01, 'tol': '±10%', 'rgb': (192, 192, 192)},
}

COLOR_LIST = ['noir', 'marron', 'rouge', 'orange', 'jaune', 'vert',
              'bleu', 'violet', 'gris', 'blanc', 'or', 'argent']


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
    x: int
    y: int


@dataclass
class Result:
    ohms: float
    formatted: str
    tolerance: str
    bands: List[str]


# =============================================================================
# IDENTIFICATION DES COULEURS
# =============================================================================

def identify_color_from_hsv(h: float, s: float, v: float) -> Tuple[str, float]:
    """
    Identifie la couleur à partir des valeurs HSV.
    Retourne (nom_couleur, score_de_confiance).
    """
    # Règles d'identification basées sur HSV
    # H = Teinte (0-180 en OpenCV)
    # S = Saturation (0-255)
    # V = Valeur/Luminosité (0-255)

    candidates = []

    # NOIR: V très bas (sombre)
    if v < 50:
        candidates.append(('noir', 90 - v))

    # BLANC: S très bas, V très haut
    if s < 30 and v > 200:
        candidates.append(('blanc', (255 - s) + (v - 200)))

    # GRIS: S très bas, V moyen
    if s < 40 and 50 <= v <= 200:
        candidates.append(('gris', 100 - s))

    # ARGENT: S bas, V moyen-haut (gris brillant)
    if s < 50 and 140 <= v <= 230:
        candidates.append(('argent', 80 - s + (v - 140) / 2))

    # OR: H jaune-orange (15-35), S moyenne
    if 12 <= h <= 40 and 40 <= s <= 200 and 80 <= v <= 220:
        score = 100 - abs(h - 25) * 2 - abs(s - 120) / 3
        candidates.append(('or', score))

    # Pour les couleurs saturées (S > 30)
    if s > 30:
        # ROUGE: H proche de 0 ou 180
        if h <= 10 or h >= 170:
            score = 100 - min(h, 180 - h) * 3
            if v < 130 and s < 150:
                # Pourrait être marron
                candidates.append(('marron', score - 10))
            else:
                candidates.append(('rouge', score))

        # MARRON: H rouge-orange, V bas
        if 0 <= h <= 25 and v < 150:
            score = 80 - abs(h - 12) * 2 - (150 - v) / 5
            candidates.append(('marron', score))

        # ORANGE: H = 10-25, V haut
        if 8 <= h <= 25 and v > 150:
            score = 90 - abs(h - 15) * 3
            candidates.append(('orange', score))

        # JAUNE: H = 22-40
        if 20 <= h <= 45:
            score = 90 - abs(h - 30) * 2
            candidates.append(('jaune', score))

        # VERT: H = 40-85
        if 35 <= h <= 90:
            score = 90 - abs(h - 60) * 1.5
            candidates.append(('vert', score))

        # BLEU: H = 85-130
        if 80 <= h <= 135:
            score = 90 - abs(h - 110) * 1.5
            candidates.append(('bleu', score))

        # VIOLET: H = 130-170
        if 125 <= h <= 175:
            score = 90 - abs(h - 145) * 2
            candidates.append(('violet', score))

    # Retourner la meilleure correspondance
    if candidates:
        best = max(candidates, key=lambda x: x[1])
        return best

    return ('inconnu', 0)


def analyze_pixel_area(hsv_img: np.ndarray, x: int, y: int,
                       radius: int = 5) -> Tuple[float, float, float]:
    """
    Analyse une zone de pixels autour d'un point.
    Retourne les valeurs H, S, V moyennes.
    """
    h, w = hsv_img.shape[:2]

    # Définir la zone
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    # Extraire la zone
    zone = hsv_img[y1:y2, x1:x2]

    # Calculer les moyennes
    h_mean = np.mean(zone[:, :, 0])
    s_mean = np.mean(zone[:, :, 1])
    v_mean = np.mean(zone[:, :, 2])

    return h_mean, s_mean, v_mean


# =============================================================================
# INTERFACE INTERACTIVE
# =============================================================================

class BandSelector:
    """Interface pour sélectionner les bandes en cliquant."""

    def __init__(self, image: np.ndarray, hsv: np.ndarray):
        self.original = image.copy()
        self.display = image.copy()
        self.hsv = hsv
        self.bands: List[Band] = []
        self.current_color = None
        self.window_name = "Cliquez sur les bandes (ESPACE=suivant | Z=annuler | ENTREE=fin)"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Analyser la zone cliquée
            h, s, v = analyze_pixel_area(self.hsv, x, y, radius=8)
            color, confidence = identify_color_from_hsv(h, s, v)

            self.current_color = {
                'color': color,
                'confidence': confidence,
                'h': h, 's': s, 'v': v,
                'x': x, 'y': y
            }

            # Mise à jour de l'affichage
            self.update_display()

    def update_display(self):
        self.display = self.original.copy()

        # Dessiner les bandes déjà sélectionnées
        for i, band in enumerate(self.bands):
            cv2.circle(self.display, (band.x, band.y), 15, (0, 255, 0), 2)
            cv2.putText(self.display, f"{i + 1}:{band.color[:3].upper()}",
                        (band.x - 20, band.y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dessiner la sélection courante
        if self.current_color:
            x, y = self.current_color['x'], self.current_color['y']
            color = self.current_color['color']
            conf = self.current_color['confidence']
            h, s, v = self.current_color['h'], self.current_color['s'], self.current_color['v']

            # Cercle de sélection
            cv2.circle(self.display, (x, y), 15, (0, 255, 255), 3)

            # Info box
            cv2.rectangle(self.display, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(self.display, (10, 10), (300, 120), (255, 255, 255), 2)

            cv2.putText(self.display, f"Couleur: {color.upper()}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(self.display, f"Confiance: {conf:.0f}%",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(self.display, f"H={h:.0f} S={s:.0f} V={v:.0f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(self.display, "ESPACE=ajouter Z=annuler ENTREE=fin",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Instructions
        h = self.display.shape[0]
        cv2.rectangle(self.display, (10, h - 40), (400, h - 10), (0, 0, 0), -1)
        cv2.putText(self.display, f"Bandes: {len(self.bands)} | Cliquez sur une bande",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self) -> List[Band]:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("  SÉLECTION INTERACTIVE DES BANDES")
        print("=" * 60)
        print("  1. CLIQUEZ sur chaque bande de couleur (gauche → droite)")
        print("  2. ESPACE pour confirmer la couleur détectée")
        print("  3. Z pour annuler la dernière bande")
        print("  4. ENTRÉE quand toutes les bandes sont sélectionnées")
        print("=" * 60)

        self.update_display()

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            # ESPACE: confirmer la couleur
            if key == ord(' ') and self.current_color:
                band = Band(
                    color=self.current_color['color'],
                    value=COLOR_CONFIG.get(self.current_color['color'], {}).get('value', -99),
                    h=self.current_color['h'],
                    s=self.current_color['s'],
                    v=self.current_color['v'],
                    x=self.current_color['x'],
                    y=self.current_color['y']
                )
                self.bands.append(band)
                print(
                    f"  ✓ Bande {len(self.bands)}: {band.color.upper()} (H={band.h:.0f}, S={band.s:.0f}, V={band.v:.0f})")
                self.current_color = None
                self.update_display()

            # Z: annuler dernière bande
            elif key == ord('z') and self.bands:
                removed = self.bands.pop()
                print(f"  ✗ Annulé: {removed.color.upper()}")
                self.update_display()

            # C: corriger la couleur détectée
            elif key == ord('c') and self.current_color:
                self.show_color_menu()

            # ENTRÉE: terminer
            elif key == 13:  # Enter
                if len(self.bands) >= 3:
                    break
                else:
                    print("  ⚠️ Minimum 3 bandes requises!")

            # ESC: annuler
            elif key == 27:
                self.bands = []
                break

        cv2.destroyAllWindows()
        return self.bands

    def show_color_menu(self):
        """Affiche un menu pour corriger manuellement la couleur."""
        print("\n  Correction manuelle - entrez la couleur:")
        print("  " + ", ".join(COLOR_LIST))

        user_input = input("  > ").strip().lower()

        if user_input in COLOR_LIST:
            self.current_color['color'] = user_input
            self.current_color['confidence'] = 100
            self.update_display()
            print(f"  → Corrigé en {user_input.upper()}")
        else:
            print("  ❌ Couleur non reconnue")


# =============================================================================
# CALCUL DE LA RÉSISTANCE
# =============================================================================

def calculate_resistance(bands: List[Band]) -> Optional[Result]:
    if len(bands) < 3:
        return None

    tolerance = "±20%"
    working = list(bands)

    # Vérifier si la dernière bande est une tolérance
    if working[-1].value < 0:
        tol_color = working[-1].color
        tolerance = COLOR_CONFIG[tol_color]['tol'] or "±20%"
        working = working[:-1]

    if len(working) < 3:
        return None

    # Vérifier que les 3 premières valeurs sont valides
    if any(b.value < 0 or b.value > 9 for b in working[:3]):
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
    print("\n" + "=" * 60)
    print(f"  🎯 RÉSULTAT: {result.formatted} {result.tolerance}")
    print("=" * 60)
    print(f"  Bandes: {' → '.join(result.bands)}")

    # Détail du calcul
    working = [b for b in bands if b.value >= 0 and b.value <= 9][:3]
    if len(working) >= 3:
        d1, d2, m = working[0].value, working[1].value, working[2].value
        print(f"\n  📝 CALCUL:")
        print(f"     {working[0].color.upper()} ({d1})")
        print(f"     {working[1].color.upper()} ({d2})")
        print(f"     {working[2].color.upper()} (×10^{m})")
        print(f"     → ({d1}×10 + {d2}) × 10^{m}")
        print(f"     → {d1}{d2} × {10 ** m}")
        print(f"     → {result.ohms:.0f} Ω")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  ANALYSEUR DE RÉSISTANCES - V9")
    print("  Sélection Interactive par Clic")
    print("  Projet Traitement de Signal - EPHEC 2025")
    print("=" * 60)

    print("\n📖 PRINCIPE:")
    print("   → Vous CLIQUEZ sur chaque bande de couleur")
    print("   → Le programme analyse les pixels cliqués")
    print("   → Plus besoin de détection automatique!")

    print("\n📖 SENS DE LECTURE:")
    print("   → Tolérance (or/argent) à DROITE")
    print("   → Cliquez de GAUCHE à DROITE")

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
            print("\n❌ Aucune image trouvée!")
            print("   Usage: python resistance_v9.py image.jpg")
            return

    # Charger l'image
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Impossible de charger: {path}")
        return

    # Redimensionner si nécessaire
    h, w = img.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    print(f"\n📷 Image: {path}")
    print(f"   Dimensions: {img.shape[1]}x{img.shape[0]}")

    # Prétraitement
    # CLAHE pour améliorer le contraste
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Filtre bilatéral
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Conversion HSV
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    # Sélection interactive
    selector = BandSelector(filtered, hsv)
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
        print("   Vérifiez que vous avez sélectionné au moins 3 bandes valides")


if __name__ == "__main__":
    main()