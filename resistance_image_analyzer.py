#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES PAR TRAITEMENT D'IMAGE
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

Objectif: Identifier la valeur d'une résistance à partir d'une image
          en utilisant des techniques de traitement de signal (filtrage,
          analyse spectrale, morphologie mathématique).

Temps de traitement: < 1 seconde (exigence MVP)

Auteur: Noah
Date: 2025
=====================================================================
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import time
import sys
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# SECTION 1: STRUCTURES DE DONNÉES ET CONSTANTES
# =============================================================================

class ColorBand(Enum):
    """Énumération des couleurs de bandes de résistance avec leurs valeurs"""
    BLACK = 0
    BROWN = 1
    RED = 2
    ORANGE = 3
    YELLOW = 4
    GREEN = 5
    BLUE = 6
    VIOLET = 7
    GRAY = 8
    WHITE = 9
    GOLD = -1  # Multiplicateur 0.1 ou Tolérance ±5%
    SILVER = -2  # Multiplicateur 0.01 ou Tolérance ±10%


@dataclass
class DetectedBand:
    """Structure pour stocker les informations d'une bande détectée"""
    color_name: str
    value: int
    x_position: int
    confidence: float
    width: int
    height: int


@dataclass
class AnalysisResult:
    """Résultat de l'analyse d'une résistance"""
    resistance_value: float
    unit: str
    tolerance: str
    bands: List[DetectedBand]
    processing_time: float
    confidence: float
    formatted_value: str


# =============================================================================
# SECTION 2: DÉFINITION DES PLAGES DE COULEURS HSV
# =============================================================================

def get_color_definitions() -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Définit les plages HSV optimisées pour chaque couleur de résistance.

    Justification technique:
    - L'espace HSV (Hue, Saturation, Value) est préféré à RGB car il sépare
      la teinte (couleur) de la luminosité, ce qui le rend plus robuste
      aux variations d'éclairage.
    - Les plages ont été calibrées expérimentalement sur un fond blanc/uni.

    Returns:
        Dict avec (lower_bound, upper_bound, numeric_value)
    """
    return {
        # Noir: Très faible luminosité (V < 60)
        # Note: Le noir est caractérisé principalement par une valeur très basse
        'black': (np.array([0, 0, 0]), np.array([180, 255, 60]), 0),

        # Marron: Teinte rouge-orange foncé
        # H: 0-20 (teintes rouges-orangées), S: moyenne, V: assez foncé
        # Distingué du noir par saturation, de l'orange par luminosité plus basse
        'brown': (np.array([0, 60, 40]), np.array([20, 200, 140]), 1),

        # Rouge: Deux plages car le rouge "wrappe" autour de H=0/180
        # Rouge vif avec haute saturation et bonne luminosité
        'red_low': (np.array([0, 140, 100]), np.array([8, 255, 255]), 2),
        'red_high': (np.array([165, 140, 100]), np.array([180, 255, 255]), 2),

        # Orange: Teinte entre rouge et jaune
        # H: 8-22, haute saturation, très lumineux (distingué du marron)
        'orange': (np.array([8, 150, 150]), np.array([22, 255, 255]), 3),

        # Jaune: Teinte bien définie, haute saturation
        # H: 22-38, très saturé et très lumineux
        'yellow': (np.array([22, 120, 150]), np.array([38, 255, 255]), 4),

        # Vert: Large plage de teinte verte
        'green': (np.array([38, 60, 60]), np.array([85, 255, 255]), 5),

        # Bleu: Teinte bleue distincte
        'blue': (np.array([85, 60, 60]), np.array([130, 255, 255]), 6),

        # Violet: Entre bleu et rouge/magenta
        'violet': (np.array([125, 40, 40]), np.array([165, 255, 255]), 7),

        # Gris: Faible saturation, luminosité moyenne
        # Important: ne pas confondre avec argent (S plus élevée pour gris)
        'gray': (np.array([0, 0, 80]), np.array([180, 60, 200]), 8),

        # Blanc: Très faible saturation, haute luminosité
        'white': (np.array([0, 0, 210]), np.array([180, 35, 255]), 9),

        # Or: Jaune-orange métallique
        # ATTENTION: L'or a une teinte jaune mais moins saturé et moins lumineux
        # que le jaune pur. Calibration critique pour éviter confusion avec orange.
        'gold': (np.array([18, 80, 100]), np.array([35, 200, 220]), -1),

        # Argent: Gris très clair/brillant
        # Faible saturation mais haute luminosité
        'silver': (np.array([0, 0, 150]), np.array([180, 40, 230]), -2),
    }


# =============================================================================
# SECTION 3: PRÉTRAITEMENT DE L'IMAGE (FILTRAGE)
# =============================================================================

class ImagePreprocessor:
    """
    Classe de prétraitement d'image utilisant des techniques de traitement
    de signal pour améliorer la qualité de détection.
    """

    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, d: int = 9,
                               sigma_color: float = 75,
                               sigma_space: float = 75) -> np.ndarray:
        """
        Applique un filtre bilatéral pour réduire le bruit tout en
        préservant les bords des bandes de couleur.

        Justification technique:
        - Le filtre bilatéral combine un filtre gaussien spatial avec
          un filtre gaussien dans le domaine des intensités.
        - Avantage: Lisse les zones homogènes (bruit) tout en conservant
          les transitions nettes entre les bandes de couleur.

        Args:
            image: Image BGR d'entrée
            d: Diamètre du voisinage (9 est un bon compromis vitesse/qualité)
            sigma_color: Filtre sigma dans l'espace couleur
            sigma_space: Filtre sigma dans l'espace des coordonnées

        Returns:
            Image filtrée
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                    tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applique l'égalisation adaptative d'histogramme avec limitation
        de contraste (CLAHE) pour améliorer le contraste local.

        Justification technique:
        - CLAHE divise l'image en tuiles et applique l'égalisation
          d'histogramme localement.
        - clip_limit empêche l'amplification excessive du bruit.
        - Améliore la distinction des couleurs dans des conditions
          d'éclairage non uniformes.

        Args:
            image: Image BGR d'entrée
            clip_limit: Limite de contraste pour éviter l'amplification du bruit
            tile_size: Taille des tuiles pour l'égalisation locale

        Returns:
            Image avec contraste amélioré
        """
        # Convertir en LAB (L = luminance, A et B = chrominance)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Appliquer CLAHE uniquement sur le canal de luminance
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)

        # Recombiner les canaux
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    @staticmethod
    def reduce_noise_gaussian(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Applique un filtre gaussien pour réduire le bruit haute fréquence.

        Justification technique:
        - Le filtre gaussien est un filtre passe-bas qui atténue les
          hautes fréquences (bruit) tout en préservant les basses
          fréquences (structure générale de l'image).
        - Kernel 3x3: bon compromis entre réduction du bruit et
          préservation des détails.

        Args:
            image: Image d'entrée
            kernel_size: Taille du noyau gaussien (doit être impair)

        Returns:
            Image filtrée
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline complet de prétraitement.

        Ordre des opérations:
        1. Filtre gaussien léger pour réduire le bruit initial
        2. CLAHE pour améliorer le contraste
        3. Filtre bilatéral pour lisser tout en préservant les bords

        Args:
            image: Image BGR brute

        Returns:
            Image prétraitée optimisée pour la détection des couleurs
        """
        # Étape 1: Réduction du bruit initial
        denoised = self.reduce_noise_gaussian(image, kernel_size=3)

        # Étape 2: Amélioration du contraste
        enhanced = self.apply_clahe(denoised, clip_limit=2.0)

        # Étape 3: Filtre bilatéral pour préserver les bords
        filtered = self.apply_bilateral_filter(enhanced, d=9)

        return filtered


# =============================================================================
# SECTION 4: DÉTECTION DU CORPS DE LA RÉSISTANCE (SEGMENTATION)
# =============================================================================

class ResistorBodyDetector:
    """
    Détecte et isole le corps de la résistance du fond de l'image.
    Utilise des opérations morphologiques et l'analyse de contours.
    """

    @staticmethod
    def detect_resistor_body(image_hsv: np.ndarray,
                             image_bgr: np.ndarray) -> np.ndarray:
        """
        Crée un masque binaire isolant le corps de la résistance.

        Justification technique:
        - Le corps des résistances est généralement de couleur beige/crème.
        - On détecte d'abord les zones non-blanches (fond blanc supposé).
        - Les opérations morphologiques nettoient le masque.

        Args:
            image_hsv: Image en espace HSV
            image_bgr: Image originale en BGR

        Returns:
            Masque binaire (255 = résistance, 0 = fond)
        """
        h, s, v = cv2.split(image_hsv)

        # Méthode 1: Exclure le fond blanc (haute luminosité + faible saturation)
        white_background = cv2.inRange(image_hsv,
                                       np.array([0, 0, 200]),
                                       np.array([180, 40, 255]))

        # Méthode 2: Détecter les zones beiges (corps de résistance)
        beige_mask1 = cv2.inRange(image_hsv,
                                  np.array([10, 20, 100]),
                                  np.array([30, 150, 240]))
        beige_mask2 = cv2.inRange(image_hsv,
                                  np.array([0, 10, 120]),
                                  np.array([25, 100, 230]))
        beige_mask = cv2.bitwise_or(beige_mask1, beige_mask2)

        # Combiner: tout ce qui n'est pas blanc OU est beige
        resistor_mask = cv2.bitwise_or(cv2.bitwise_not(white_background), beige_mask)

        # Opérations morphologiques pour nettoyer le masque
        # CLOSE: Ferme les petits trous dans la résistance
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        resistor_mask = cv2.morphologyEx(resistor_mask, cv2.MORPH_CLOSE,
                                         kernel_close, iterations=2)

        # OPEN: Supprime les petits points de bruit
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        resistor_mask = cv2.morphologyEx(resistor_mask, cv2.MORPH_OPEN,
                                         kernel_open, iterations=1)

        # Garder uniquement le plus grand contour (la résistance)
        contours, _ = cv2.findContours(resistor_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros(resistor_mask.shape, dtype=np.uint8)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
            return final_mask

        return resistor_mask


# =============================================================================
# SECTION 5: DÉTECTION DES BANDES DE COULEUR
# =============================================================================

class ColorBandDetector:
    """
    Détecte les bandes de couleur sur le corps de la résistance.
    Utilise la segmentation par couleur en espace HSV et l'analyse
    de profil horizontal.
    """

    def __init__(self):
        self.color_definitions = get_color_definitions()

    def detect_color_in_roi(self, hsv_roi: np.ndarray,
                            resistor_mask: np.ndarray,
                            color_name: str,
                            lower: np.ndarray,
                            upper: np.ndarray,
                            min_area: int = 30) -> List[DetectedBand]:
        """
        Détecte une couleur spécifique dans la région d'intérêt.

        Args:
            hsv_roi: ROI en espace HSV
            resistor_mask: Masque du corps de la résistance
            color_name: Nom de la couleur
            lower: Borne inférieure HSV
            upper: Borne supérieure HSV
            min_area: Aire minimale pour valider une détection

        Returns:
            Liste des bandes détectées pour cette couleur
        """
        # Créer le masque pour cette couleur
        color_mask = cv2.inRange(hsv_roi, lower, upper)

        # Appliquer le masque de la résistance
        color_mask = cv2.bitwise_and(color_mask, resistor_mask)

        # Morphologie pour nettoyer les détections
        # Élément structurant vertical pour favoriser la détection de bandes verticales
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE,
                                      kernel_vertical, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                      kernel_vertical, iterations=1)

        # Trouver les contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        roi_height = hsv_roi.shape[0]
        roi_width = hsv_roi.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Critères de validation pour une bande:
            # - Pas trop large (< 40% de la largeur de la ROI)
            # - Suffisamment haute (> 15% de la hauteur de la ROI)
            # - Forme verticale (h > w/2 ou aire significative)

            is_not_too_wide = w < roi_width * 0.40
            is_tall_enough = h > roi_height * 0.15
            is_vertical_shape = h > w * 0.5 or area > 50

            if is_not_too_wide and is_tall_enough and is_vertical_shape:
                # Calculer le centre X de la bande
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                else:
                    cx = x + w // 2

                # Calculer un score de confiance
                # Basé sur: taille, position, forme
                area_score = min(1.0, area / (roi_width * roi_height * 0.05))
                shape_score = min(1.0, h / (w + 1))
                confidence = (area_score + shape_score) / 2

                value = self.get_color_value(color_name)

                detected.append(DetectedBand(
                    color_name=color_name.replace('_low', '').replace('_high', ''),
                    value=value,
                    x_position=cx,
                    confidence=confidence,
                    width=w,
                    height=h
                ))

        return detected

    def get_color_value(self, color_name: str) -> int:
        """Retourne la valeur numérique associée à une couleur."""
        color_name_clean = color_name.replace('_low', '').replace('_high', '')
        _, _, value = self.color_definitions.get(color_name, (None, None, -99))
        if value == -99:
            # Chercher sans suffixes
            for name, (_, _, val) in self.color_definitions.items():
                if name.startswith(color_name_clean):
                    return val
        return value

    def detect_all_bands(self, hsv_roi: np.ndarray,
                         resistor_mask: np.ndarray) -> List[DetectedBand]:
        """
        Détecte toutes les bandes de couleur dans la ROI.

        Args:
            hsv_roi: ROI en espace HSV
            resistor_mask: Masque du corps de la résistance

        Returns:
            Liste de toutes les bandes détectées, triées par position X
        """
        all_bands = []

        for color_name, (lower, upper, _) in self.color_definitions.items():
            bands = self.detect_color_in_roi(hsv_roi, resistor_mask,
                                             color_name, lower, upper)
            all_bands.extend(bands)

        # Supprimer les doublons (même couleur au même endroit)
        unique_bands = self._remove_duplicates(all_bands)

        # Trier par position X (gauche à droite)
        unique_bands.sort(key=lambda b: b.x_position)

        return unique_bands

    def _remove_duplicates(self, bands: List[DetectedBand],
                           min_distance: int = 15) -> List[DetectedBand]:
        """
        Supprime les détections en double basées sur la proximité.
        Garde la détection avec la meilleure confiance.
        """
        if not bands:
            return []

        # Trier par confiance décroissante
        bands.sort(key=lambda b: b.confidence, reverse=True)

        unique = []
        for band in bands:
            is_duplicate = False
            for existing in unique:
                if abs(band.x_position - existing.x_position) < min_distance:
                    # Même position: garder celui qui existe déjà (meilleure confiance)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(band)

        return unique


# =============================================================================
# SECTION 6: CALCUL DE LA VALEUR DE RÉSISTANCE
# =============================================================================

class ResistanceCalculator:
    """
    Calcule la valeur de résistance à partir des bandes détectées.
    Gère les résistances à 4 et 5 bandes.
    """

    TOLERANCE_MAP = {
        'brown': '±1%',
        'red': '±2%',
        'green': '±0.5%',
        'blue': '±0.25%',
        'violet': '±0.10%',
        'gray': '±0.05%',
        'gold': '±5%',
        'silver': '±10%',
        'none': '±20%'
    }

    def correct_orientation(self, bands: List[DetectedBand]) -> List[DetectedBand]:
        """
        Corrige l'orientation si nécessaire (or/argent doit être à la fin).

        Justification technique:
        - Les bandes or/argent représentent la tolérance et sont toujours
          positionnées à la fin de la résistance.
        - Si on détecte or/argent au début, on inverse l'ordre.
        """
        if len(bands) < 3:
            return bands

        first_value = bands[0].value
        last_value = bands[-1].value

        # Si or/argent au début et pas à la fin -> inverser
        if first_value < 0 and last_value >= 0:
            return list(reversed(bands))

        # Si pas de or/argent à la fin, vérifier l'espacement
        # (la bande de tolérance est généralement plus espacée)
        if len(bands) >= 4 and last_value >= 0:
            positions = [b.x_position for b in bands]
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

            if gaps[0] > max(gaps[1:]) * 1.3:
                return list(reversed(bands))

        return bands

    def calculate(self, bands: List[DetectedBand]) -> Optional[AnalysisResult]:
        """
        Calcule la valeur de résistance à partir des bandes.

        Args:
            bands: Liste des bandes détectées (triées par position)

        Returns:
            AnalysisResult ou None si calcul impossible
        """
        if len(bands) < 3:
            return None

        # Corriger l'orientation
        bands = self.correct_orientation(bands)

        # Résistance à 4 bandes: Chiffre1, Chiffre2, Multiplicateur, Tolérance
        # Résistance à 5 bandes: Chiffre1, Chiffre2, Chiffre3, Multiplicateur, Tolérance

        try:
            tolerance = 'none'

            if len(bands) >= 4:
                # Vérifier si la dernière bande est une tolérance
                if bands[-1].value < 0 or bands[-1].color_name in self.TOLERANCE_MAP:
                    tolerance = bands[-1].color_name
                    significant_bands = bands[:-1]
                else:
                    significant_bands = bands[:4]
            else:
                significant_bands = bands[:3]

            # Extraction des valeurs
            if len(significant_bands) >= 3:
                digit1 = significant_bands[0].value
                digit2 = significant_bands[1].value

                if len(significant_bands) >= 4:
                    # 5 bandes: 3 chiffres significatifs
                    digit3 = significant_bands[2].value
                    multiplier_idx = significant_bands[3].value
                    base_value = digit1 * 100 + digit2 * 10 + digit3
                else:
                    # 4 bandes: 2 chiffres significatifs
                    multiplier_idx = significant_bands[2].value
                    base_value = digit1 * 10 + digit2

                # Calcul du multiplicateur
                if multiplier_idx >= 0:
                    multiplier = 10 ** multiplier_idx
                elif multiplier_idx == -1:  # Gold
                    multiplier = 0.1
                elif multiplier_idx == -2:  # Silver
                    multiplier = 0.01
                else:
                    return None

                resistance = base_value * multiplier

                # Formatage du résultat
                if resistance >= 1_000_000:
                    value = resistance / 1_000_000
                    unit = 'MΩ'
                elif resistance >= 1_000:
                    value = resistance / 1_000
                    unit = 'kΩ'
                else:
                    value = resistance
                    unit = 'Ω'

                # Formater la valeur
                if value == int(value):
                    formatted = f"{int(value)} {unit}"
                else:
                    formatted = f"{value:.2f} {unit}"

                tolerance_str = self.TOLERANCE_MAP.get(tolerance, '±20%')

                # Calculer la confiance moyenne
                avg_confidence = sum(b.confidence for b in bands) / len(bands)

                return AnalysisResult(
                    resistance_value=resistance,
                    unit=unit,
                    tolerance=tolerance_str,
                    bands=bands,
                    processing_time=0,  # Sera mis à jour
                    confidence=avg_confidence,
                    formatted_value=f"{formatted} {tolerance_str}"
                )

        except (IndexError, ValueError, KeyError) as e:
            print(f"Erreur de calcul: {e}")
            return None

        return None


# =============================================================================
# SECTION 7: INTERFACE UTILISATEUR ET VISUALISATION
# =============================================================================

class ResistanceAnalyzer:
    """
    Classe principale orchestrant l'analyse complète d'une résistance.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.body_detector = ResistorBodyDetector()
        self.band_detector = ColorBandDetector()
        self.calculator = ResistanceCalculator()

    def analyze_image(self, image_path: str,
                      interactive: bool = True) -> Optional[AnalysisResult]:
        """
        Analyse une image de résistance.

        Args:
            image_path: Chemin vers l'image
            interactive: Si True, affiche une interface de sélection

        Returns:
            AnalysisResult ou None
        """
        start_time = time.time()

        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Erreur: Impossible de charger '{image_path}'")
            return None

        # Redimensionner pour standardiser le traitement
        height, width = image.shape[:2]
        target_width = 800
        scale = target_width / width
        image_resized = cv2.resize(image, (target_width, int(height * scale)))

        if interactive:
            roi, roi_coords = self._interactive_roi_selection(image_resized)
            if roi is None:
                return None
        else:
            # Mode automatique: utiliser toute l'image
            roi = image_resized
            roi_coords = (0, 0, image_resized.shape[1], image_resized.shape[0])

        # Pipeline d'analyse
        result = self._analyze_roi(roi)

        if result:
            result.processing_time = time.time() - start_time

            # Afficher les résultats
            self._display_results(image_resized, roi_coords, result)

        return result

    def _analyze_roi(self, roi: np.ndarray) -> Optional[AnalysisResult]:
        """
        Analyse une région d'intérêt contenant la résistance.

        Pipeline:
        1. Prétraitement (filtrage, amélioration contraste)
        2. Détection du corps de la résistance
        3. Détection des bandes de couleur
        4. Calcul de la valeur
        """
        # 1. Prétraitement
        preprocessed = self.preprocessor.preprocess(roi)

        # 2. Conversion en HSV
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)

        # 3. Détection du corps de la résistance
        resistor_mask = self.body_detector.detect_resistor_body(hsv, preprocessed)

        # 4. Détection des bandes
        bands = self.band_detector.detect_all_bands(hsv, resistor_mask)

        print(f"\n📊 Bandes détectées: {len(bands)}")
        for i, band in enumerate(bands):
            print(f"   {i + 1}. {band.color_name.upper():8s} (val={band.value:2d}, "
                  f"x={band.x_position:3d}, conf={band.confidence:.2f})")

        # 5. Calcul de la valeur
        if len(bands) >= 3:
            result = self.calculator.calculate(bands)
            return result
        else:
            print(f"❌ Pas assez de bandes détectées (minimum: 3)")
            return None

    def _interactive_roi_selection(self, image: np.ndarray) -> Tuple[Optional[np.ndarray],
    Optional[Tuple]]:
        """
        Interface de sélection manuelle de la région d'intérêt.
        """
        roi_selected = False
        drawing = False
        roi_start = (0, 0)
        roi_end = (0, 0)

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, roi_start, roi_end, roi_selected

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                roi_start = (x, y)
                roi_end = (x, y)
                roi_selected = False
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                roi_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                roi_end = (x, y)
                roi_selected = True

        window_name = 'Selectionnez la resistance (ENTREE=valider, R=recommencer, ESC=quitter)'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "=" * 60)
        print("    ANALYSEUR DE RÉSISTANCES - MODE INTERACTIF")
        print("=" * 60)
        print("\nInstructions:")
        print("  1. Tracez un rectangle serré autour de la résistance")
        print("  2. Appuyez sur ENTRÉE pour valider")
        print("  3. Appuyez sur R pour recommencer")
        print("  4. Appuyez sur ESC pour quitter")
        print("=" * 60)

        while True:
            display = image.copy()

            if drawing or roi_selected:
                cv2.rectangle(display, roi_start, roi_end, (0, 255, 0), 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == 13 and roi_selected:  # ENTRÉE
                break
            elif key == ord('r'):  # R
                roi_selected = False
                drawing = False
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None, None

        cv2.destroyWindow(window_name)

        # Extraire la ROI
        x1 = min(roi_start[0], roi_end[0])
        y1 = min(roi_start[1], roi_end[1])
        x2 = max(roi_start[0], roi_end[0])
        y2 = max(roi_start[1], roi_end[1])

        if x2 - x1 < 30 or y2 - y1 < 15:
            print("❌ Zone sélectionnée trop petite")
            return None, None

        roi = image[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)

    def _display_results(self, image: np.ndarray,
                         roi_coords: Tuple[int, int, int, int],
                         result: AnalysisResult):
        """
        Affiche les résultats de l'analyse.
        """
        output = image.copy()
        x1, y1, x2, y2 = roi_coords

        # Dessiner le rectangle de la ROI
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Afficher la valeur
        cv2.rectangle(output, (x1, y1 - 40), (x2, y1), (0, 0, 0), -1)
        cv2.putText(output, result.formatted_value,
                    (x1 + 5, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Afficher les informations
        print("\n" + "=" * 60)
        print(f"  🎯 RÉSULTAT: {result.formatted_value}")
        print("=" * 60)
        print(f"\n  ⏱️  Temps de traitement: {result.processing_time * 1000:.1f} ms")
        print(f"  📊 Confiance moyenne: {result.confidence * 100:.1f}%")
        print(f"  🔢 Bandes détectées: {len(result.bands)}")

        colors = [b.color_name for b in result.bands]
        print(f"  🌈 Couleurs: {' → '.join(colors)}")
        print("=" * 60)

        # Afficher l'image résultat
        cv2.imshow('RESULTAT', output)

        print("\n💡 Appuyez sur une touche pour fermer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================================================================
# SECTION 8: POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def find_image_file() -> Optional[str]:
    """
    Recherche automatiquement un fichier image de résistance dans le dossier courant.
    Gère les problèmes d'encodage Windows avec les accents.
    """
    import glob
    import os

    # Liste des patterns à rechercher (avec et sans accents)
    patterns = [
        "resistance*.jpg", "resistance*.png", "resistance*.jpeg",
        "résistance*.jpg", "résistance*.png", "résistance*.jpeg",
        "Resistance*.jpg", "Resistance*.png", "Resistance*.jpeg",
        "Résistance*.jpg", "Résistance*.png", "Résistance*.jpeg",
        "*.jpg", "*.png", "*.jpeg"  # En dernier recours, prendre n'importe quelle image
    ]

    for pattern in patterns:
        try:
            files = glob.glob(pattern)
            if files:
                # Retourner le premier fichier trouvé qui contient "resist" dans le nom
                for f in files:
                    if "resist" in f.lower() or "rési" in f.lower():
                        return f
                # Si aucun fichier "resistance", retourner le premier trouvé
                if pattern in ["*.jpg", "*.png", "*.jpeg"]:
                    continue  # Ne pas prendre n'importe quelle image automatiquement
                return files[0]
        except Exception:
            continue

    return None


def main():
    """Point d'entrée principal du programme."""

    print("\n" + "=" * 60)
    print("    ANALYSEUR DE RÉSISTANCES PAR TRAITEMENT D'IMAGE")
    print("    Projet de Traitement de Signal - EPHEC 2025")
    print("=" * 60)

    # Chemin de l'image (peut être passé en argument)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Recherche automatique du fichier
        image_path = find_image_file()
        if image_path is None:
            print("\n❌ Aucun fichier image trouvé dans le dossier courant.")
            print("   Utilisation: python resistance_image_analyzer.py [chemin_image.jpg]")
            print("\n   Astuce: Renommez votre fichier en 'resistance.jpg' (sans accent)")
            return None

    print(f"\n📁 Image: {image_path}")

    # Créer l'analyseur et lancer l'analyse
    analyzer = ResistanceAnalyzer()
    result = analyzer.analyze_image(image_path, interactive=True)

    if result:
        print(f"\n✅ Analyse terminée avec succès!")
        print(f"   Valeur: {result.formatted_value}")
        print(f"   Temps: {result.processing_time * 1000:.1f} ms")

        # Vérifier le critère de temps < 1 seconde
        if result.processing_time < 1.0:
            print(f"   ✓ Critère de temps respecté (< 1s)")
        else:
            print(f"   ⚠️ Critère de temps dépassé (> 1s)")
    else:
        print("\n❌ Échec de l'analyse")

    return result


if __name__ == "__main__":
    main()