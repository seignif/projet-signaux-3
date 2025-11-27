#!/usr/bin/env python3
"""
=====================================================================
    ANALYSEUR DE RÉSISTANCES EN TEMPS RÉEL
    Projet de Traitement de Signal - EPHEC 2025
=====================================================================

Objectif: Analyser en temps réel la valeur d'une résistance via webcam
          ou application smartphone (iVCam, DroidCam, etc.)

Temps de traitement: < 1 seconde par frame (exigence MVP)

Techniques de traitement de signal utilisées:
- Filtrage bilatéral (préservation des bords)
- Égalisation adaptative d'histogramme (CLAHE)
- Morphologie mathématique (ouverture, fermeture)
- Moyenne temporelle pour stabiliser les détections

Auteur: Noah
Date: 2025
=====================================================================
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque
from dataclasses import dataclass
import time
from enum import Enum


# =============================================================================
# SECTION 1: STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class ColorBandInfo:
    """Information sur une bande de couleur détectée"""
    color_name: str
    value: int
    x_position: int
    area: int
    height: int
    width: int


@dataclass
class ResistanceReading:
    """Lecture de résistance avec métadonnées"""
    value_ohms: float
    formatted_value: str
    tolerance: str
    bands: List[str]
    confidence: float
    timestamp: float


# =============================================================================
# SECTION 2: CONFIGURATION DES COULEURS HSV
# =============================================================================

class HSVColorConfig:
    """
    Configuration des plages HSV pour la détection des couleurs.

    Justification des choix:
    - L'espace HSV sépare teinte/saturation/luminosité
    - Plus robuste aux variations d'éclairage que RGB
    - Les plages ont été calibrées pour un fond blanc/uni
    """

    # Format: (H_min, S_min, V_min), (H_max, S_max, V_max), valeur_numérique
    COLORS = {
        # Noir: très faible luminosité
        'noir': ([0, 0, 0], [180, 255, 45], 0),

        # Marron: teinte rouge-orange foncé
        'marron': ([0, 50, 25], [18, 200, 130], 1),

        # Rouge (partie basse du spectre H)
        'rouge1': ([0, 100, 80], [8, 255, 255], 2),
        # Rouge (partie haute du spectre H - wrap around)
        'rouge2': ([165, 100, 80], [180, 255, 255], 2),

        # Orange
        'orange': ([8, 130, 100], [22, 255, 255], 3),

        # Jaune
        'jaune': ([20, 100, 100], [35, 255, 255], 4),

        # Vert
        'vert': ([35, 50, 50], [80, 255, 255], 5),

        # Bleu
        'bleu': ([85, 50, 50], [125, 255, 255], 6),

        # Violet
        'violet': ([125, 30, 30], [155, 255, 255], 7),

        # Gris: faible saturation
        'gris': ([0, 0, 70], [180, 40, 180], 8),

        # Blanc: très haute luminosité, faible saturation
        'blanc': ([0, 0, 200], [180, 25, 255], 9),

        # Or: jaune-orange métallique sombre
        'or': ([15, 80, 80], [30, 200, 180], -1),

        # Argent: gris brillant
        'argent': ([0, 0, 140], [180, 25, 210], -2),
    }

    @classmethod
    def get_colors(cls) -> Dict:
        """Retourne le dictionnaire des couleurs"""
        return cls.COLORS


# =============================================================================
# SECTION 3: FILTRES DE TRAITEMENT DE SIGNAL
# =============================================================================

class SignalFilters:
    """
    Collection de filtres de traitement de signal pour l'amélioration d'image.
    """

    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9,
                         sigma_color: float = 75,
                         sigma_space: float = 75) -> np.ndarray:
        """
        Filtre bilatéral: lisse le bruit tout en préservant les bords.

        Principe mathématique:
        - Combine pondération spatiale (distance) et pondération de similarité (intensité)
        - I_filtered(x) = Σ w_s(x,y) * w_r(I(x),I(y)) * I(y) / normalisation
        - w_s = exp(-|x-y|²/2σ_s²) : poids spatial gaussien
        - w_r = exp(-|I(x)-I(y)|²/2σ_r²) : poids de similarité d'intensité

        Avantage pour notre application:
        - Préserve les bords nets entre les bandes de couleur
        - Lisse le bruit dans les zones homogènes (fond, corps de résistance)
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def clahe_enhancement(image: np.ndarray,
                          clip_limit: float = 2.0,
                          tile_size: int = 8) -> np.ndarray:
        """
        CLAHE: Contrast Limited Adaptive Histogram Equalization

        Principe:
        1. Divise l'image en tuiles de taille tile_size x tile_size
        2. Calcule l'histogramme de chaque tuile
        3. Limite le contraste (clip_limit) pour éviter l'amplification du bruit
        4. Redistribue les valeurs en excès
        5. Applique l'égalisation localement

        Avantage pour notre application:
        - Améliore le contraste local même avec éclairage non uniforme
        - Fait ressortir les différences de couleur subtiles
        """
        # Convertir en espace LAB (L=luminance, A/B=chrominance)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Appliquer CLAHE sur le canal L uniquement
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_size, tile_size))
        l_enhanced = clahe.apply(l)

        # Reconstruire l'image
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Filtre gaussien: filtre passe-bas classique.

        Fonction de transfert (dans le domaine fréquentiel):
        H(u,v) = exp(-2π²σ²(u²+v²))

        Effet: atténue les hautes fréquences (bruit) proportionnellement
        à leur distance du centre fréquentiel.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# =============================================================================
# SECTION 4: OPÉRATIONS MORPHOLOGIQUES
# =============================================================================

class MorphologyOperations:
    """
    Opérations de morphologie mathématique pour le nettoyage des masques.
    """

    @staticmethod
    def create_structuring_element(shape: int, size: Tuple[int, int]) -> np.ndarray:
        """
        Crée un élément structurant pour les opérations morphologiques.

        Types:
        - cv2.MORPH_RECT: rectangle (efficace pour bandes verticales)
        - cv2.MORPH_ELLIPSE: ellipse (préserve mieux les formes arrondies)
        - cv2.MORPH_CROSS: croix
        """
        return cv2.getStructuringElement(shape, size)

    @staticmethod
    def opening(mask: np.ndarray, kernel: np.ndarray,
                iterations: int = 1) -> np.ndarray:
        """
        Ouverture morphologique: Érosion suivie de Dilatation

        Effet: supprime les petits objets (bruit) tout en préservant
        la forme générale des grands objets.

        Utilisation: nettoyer les fausses détections de couleur
        """
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    @staticmethod
    def closing(mask: np.ndarray, kernel: np.ndarray,
                iterations: int = 1) -> np.ndarray:
        """
        Fermeture morphologique: Dilatation suivie d'Érosion

        Effet: remplit les petits trous et connecte les régions proches.

        Utilisation: unifier les bandes de couleur fragmentées
        """
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    @staticmethod
    def clean_mask(mask: np.ndarray) -> np.ndarray:
        """
        Pipeline de nettoyage complet d'un masque binaire.

        Étapes:
        1. Fermeture (combler les trous)
        2. Ouverture (supprimer le bruit)
        """
        # Élément structurant vertical (favorise les bandes verticales)
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))

        # Fermeture pour combler les trous
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical, iterations=2)

        # Ouverture pour supprimer le bruit
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_vertical, iterations=1)

        return opened


# =============================================================================
# SECTION 5: STABILISATION TEMPORELLE
# =============================================================================

class TemporalStabilizer:
    """
    Stabilise les lectures en faisant une moyenne temporelle.

    Justification technique:
    - Les détections frame par frame peuvent être bruitées
    - Une moyenne sur plusieurs frames donne un résultat plus fiable
    - Implémenté comme un filtre moyenneur (moving average filter)
    """

    def __init__(self, history_size: int = 10):
        """
        Args:
            history_size: Nombre de lectures à conserver pour la moyenne
        """
        self.history: deque = deque(maxlen=history_size)
        self.history_size = history_size

    def add_reading(self, reading: ResistanceReading):
        """Ajoute une nouvelle lecture à l'historique"""
        self.history.append(reading)

    def get_stable_reading(self) -> Optional[ResistanceReading]:
        """
        Retourne la lecture la plus fréquente dans l'historique.

        Algorithme:
        1. Compter les occurrences de chaque valeur formatée
        2. Retourner celle qui apparaît le plus souvent
        3. Minimum 3 lectures identiques pour validation
        """
        if len(self.history) < 3:
            return None

        # Compter les occurrences de chaque valeur
        value_counts: Dict[str, int] = {}
        value_readings: Dict[str, ResistanceReading] = {}

        for reading in self.history:
            key = reading.formatted_value
            value_counts[key] = value_counts.get(key, 0) + 1
            value_readings[key] = reading

        # Trouver la valeur la plus fréquente
        if value_counts:
            most_common = max(value_counts, key=value_counts.get)
            count = value_counts[most_common]

            # Minimum 3 lectures identiques pour validation
            if count >= 3:
                reading = value_readings[most_common]
                # Mettre à jour la confiance basée sur la stabilité
                stability = count / len(self.history)
                reading.confidence = min(1.0, reading.confidence * (0.5 + 0.5 * stability))
                return reading

        return None

    def clear(self):
        """Vide l'historique"""
        self.history.clear()


# =============================================================================
# SECTION 6: DÉTECTEUR DE BANDES
# =============================================================================

class BandDetector:
    """
    Détecte et analyse les bandes de couleur dans une région d'intérêt.
    """

    def __init__(self):
        self.colors = HSVColorConfig.get_colors()
        self.morphology = MorphologyOperations()

    def detect_bands(self, roi_hsv: np.ndarray,
                     roi_bgr: np.ndarray) -> List[ColorBandInfo]:
        """
        Détecte toutes les bandes de couleur dans la ROI.

        Args:
            roi_hsv: ROI en espace HSV
            roi_bgr: ROI en espace BGR (pour debug)

        Returns:
            Liste des bandes détectées, triées par position X
        """
        detected_bands: List[ColorBandInfo] = []
        roi_h, roi_w = roi_hsv.shape[:2]

        # Créer un masque pour isoler le corps de la résistance
        # (exclure le fond blanc)
        white_mask = cv2.inRange(roi_hsv,
                                 np.array([0, 0, 210]),
                                 np.array([180, 35, 255]))
        resistor_mask = cv2.bitwise_not(white_mask)

        # Nettoyer le masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        resistor_mask = cv2.morphologyEx(resistor_mask, cv2.MORPH_CLOSE, kernel)
        resistor_mask = cv2.morphologyEx(resistor_mask, cv2.MORPH_OPEN, kernel)

        for color_name, (lower, upper, value) in self.colors.items():
            # Créer le masque pour cette couleur
            lower_arr = np.array(lower)
            upper_arr = np.array(upper)
            color_mask = cv2.inRange(roi_hsv, lower_arr, upper_arr)

            # Appliquer le masque de la résistance
            color_mask = cv2.bitwise_and(color_mask, resistor_mask)

            # Nettoyer le masque
            color_mask = self.morphology.clean_mask(color_mask)

            # Trouver les contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                # Filtre minimum d'aire
                if area < 20:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Critères de forme pour une bande:
                # - Pas trop large (< 35% de la ROI)
                # - Assez haute (> 15% de la ROI)
                is_valid_width = w < roi_w * 0.35
                is_valid_height = h > roi_h * 0.15

                if is_valid_width and is_valid_height:
                    # Calculer le centre X
                    M = cv2.moments(cnt)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2

                    # Vérifier que ce n'est pas un doublon
                    is_duplicate = any(abs(cx - b.x_position) < 12
                                       for b in detected_bands)

                    if not is_duplicate:
                        # Nettoyer le nom de la couleur
                        clean_name = color_name.replace('1', '').replace('2', '')

                        detected_bands.append(ColorBandInfo(
                            color_name=clean_name,
                            value=value,
                            x_position=cx,
                            area=area,
                            height=h,
                            width=w
                        ))

        # Trier par position X (gauche à droite)
        detected_bands.sort(key=lambda b: b.x_position)

        return detected_bands


# =============================================================================
# SECTION 7: CALCULATEUR DE RÉSISTANCE
# =============================================================================

class ResistanceCalculator:
    """
    Calcule la valeur de résistance à partir des bandes détectées.
    """

    TOLERANCE_MAP = {
        'marron': '±1%',
        'rouge': '±2%',
        'vert': '±0.5%',
        'bleu': '±0.25%',
        'violet': '±0.10%',
        'gris': '±0.05%',
        'or': '±5%',
        'argent': '±10%',
    }

    def correct_orientation(self, bands: List[ColorBandInfo]) -> List[ColorBandInfo]:
        """
        Corrige l'orientation si or/argent est au début.

        Convention: la bande de tolérance (or/argent) est toujours à la fin.
        """
        if len(bands) < 3:
            return bands

        first_val = bands[0].value
        last_val = bands[-1].value

        # Si or/argent au début mais pas à la fin -> inverser
        if first_val < 0 and last_val >= 0:
            return list(reversed(bands))

        return bands

    def calculate(self, bands: List[ColorBandInfo]) -> Optional[ResistanceReading]:
        """
        Calcule la valeur de résistance.

        Format 4 bandes: Chiffre1, Chiffre2, Multiplicateur, Tolérance
        Format 5 bandes: Chiffre1, Chiffre2, Chiffre3, Multiplicateur, Tolérance
        """
        if len(bands) < 3:
            return None

        # Corriger l'orientation
        bands = self.correct_orientation(bands)

        try:
            tolerance = '±20%'

            # Déterminer si on a une bande de tolérance
            if len(bands) >= 4 and bands[-1].value < 0:
                tolerance = self.TOLERANCE_MAP.get(bands[-1].color_name, '±5%')
                working_bands = bands[:-1]
            elif len(bands) >= 4 and bands[-1].color_name in self.TOLERANCE_MAP:
                tolerance = self.TOLERANCE_MAP[bands[-1].color_name]
                working_bands = bands[:-1]
            else:
                working_bands = bands[:3]

            # Valider que les premiers chiffres sont >= 0
            if working_bands[0].value < 0 or working_bands[1].value < 0:
                return None

            # Calculer la valeur
            if len(working_bands) >= 4:
                # 5 bandes: 3 chiffres significatifs
                if working_bands[2].value < 0:
                    return None
                digit1 = working_bands[0].value
                digit2 = working_bands[1].value
                digit3 = working_bands[2].value
                multiplier_val = working_bands[3].value
                base = digit1 * 100 + digit2 * 10 + digit3
            else:
                # 4 bandes: 2 chiffres significatifs
                digit1 = working_bands[0].value
                digit2 = working_bands[1].value
                multiplier_val = working_bands[2].value
                base = digit1 * 10 + digit2

            # Calculer le multiplicateur
            if multiplier_val >= 0:
                multiplier = 10 ** multiplier_val
            elif multiplier_val == -1:  # Or
                multiplier = 0.1
            elif multiplier_val == -2:  # Argent
                multiplier = 0.01
            else:
                return None

            resistance = base * multiplier

            # Formater la valeur
            if resistance >= 1_000_000:
                formatted = f"{resistance / 1_000_000:.2f} MΩ"
            elif resistance >= 1_000:
                formatted = f"{resistance / 1_000:.2f} kΩ"
            else:
                formatted = f"{resistance:.1f} Ω"

            # Calculer la confiance
            confidence = min(1.0, len(bands) / 4.0 * 0.8)

            band_names = [b.color_name for b in bands]

            return ResistanceReading(
                value_ohms=resistance,
                formatted_value=formatted,
                tolerance=tolerance,
                bands=band_names,
                confidence=confidence,
                timestamp=time.time()
            )

        except (IndexError, ValueError, TypeError) as e:
            return None


# =============================================================================
# SECTION 8: ANALYSEUR TEMPS RÉEL
# =============================================================================

class RealtimeResistanceAnalyzer:
    """
    Analyseur temps réel de résistances via webcam.

    Pipeline:
    1. Capture frame
    2. Prétraitement (filtrage bilatéral, CLAHE)
    3. Extraction ROI
    4. Détection des bandes
    5. Calcul de la valeur
    6. Stabilisation temporelle
    7. Affichage
    """

    def __init__(self, camera_id: int = 0,
                 roi_width: int = 400, roi_height: int = 120):
        """
        Args:
            camera_id: ID de la caméra (0 = webcam par défaut, ou iVCam)
            roi_width: Largeur de la zone de scan
            roi_height: Hauteur de la zone de scan
        """
        self.camera_id = camera_id
        self.roi_size = (roi_width, roi_height)

        self.filters = SignalFilters()
        self.band_detector = BandDetector()
        self.calculator = ResistanceCalculator()
        self.stabilizer = TemporalStabilizer(history_size=15)

        self.cap = None
        self.is_running = False

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applique le pipeline de prétraitement.

        Ordre des opérations:
        1. Filtre gaussien léger (réduction du bruit HF)
        2. CLAHE (amélioration du contraste)
        3. Filtre bilatéral (lissage + préservation des bords)
        """
        # Étape 1: Filtre gaussien léger
        denoised = self.filters.gaussian_blur(frame, kernel_size=3)

        # Étape 2: CLAHE pour améliorer le contraste
        enhanced = self.filters.clahe_enhancement(denoised, clip_limit=2.5)

        # Étape 3: Filtre bilatéral
        filtered = self.filters.bilateral_filter(enhanced, d=7,
                                                 sigma_color=50,
                                                 sigma_space=50)

        return filtered

    def extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extrait la région d'intérêt au centre de l'image.

        Returns:
            (roi, (x1, y1, x2, y2))
        """
        h, w = frame.shape[:2]
        roi_w, roi_h = self.roi_size

        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        roi = frame[y1:y2, x1:x2].copy()

        return roi, (x1, y1, x2, y2)

    def analyze_frame(self, frame: np.ndarray) -> Optional[ResistanceReading]:
        """
        Analyse une frame et retourne la lecture de résistance.
        """
        # Prétraitement
        preprocessed = self.preprocess_frame(frame)

        # Extraire la ROI
        roi, roi_coords = self.extract_roi(preprocessed)

        # Convertir en HSV
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Détecter les bandes
        bands = self.band_detector.detect_bands(roi_hsv, roi)

        if len(bands) >= 3:
            # Calculer la valeur
            reading = self.calculator.calculate(bands)

            if reading:
                # Ajouter à la stabilisation
                self.stabilizer.add_reading(reading)
                return reading

        return None

    def draw_ui(self, frame: np.ndarray,
                roi_coords: Tuple[int, int, int, int],
                reading: Optional[ResistanceReading],
                stable_reading: Optional[ResistanceReading],
                fps: float):
        """
        Dessine l'interface utilisateur sur la frame.
        """
        x1, y1, x2, y2 = roi_coords

        # Dessiner le rectangle de la ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Instruction
        cv2.putText(frame, "Placez la resistance ici (fond uni)",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)

        # Zone d'affichage du résultat
        result_y = y2 + 15
        cv2.rectangle(frame, (x1, result_y), (x2, result_y + 55), (0, 0, 0), -1)

        # Afficher le résultat
        if stable_reading:
            color = (0, 255, 0)  # Vert = stable
            text = f"Valeur: {stable_reading.formatted_value} {stable_reading.tolerance}"
            bands_text = " → ".join(stable_reading.bands)
        elif reading:
            color = (0, 255, 255)  # Jaune = en cours
            text = f"Detection: {reading.formatted_value}"
            bands_text = " → ".join(reading.bands)
        else:
            color = (0, 0, 255)  # Rouge = pas de détection
            text = "Valeur: En attente..."
            bands_text = ""

        cv2.putText(frame, text, (x1 + 5, result_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if bands_text:
            cv2.putText(frame, f"Bandes: {bands_text}", (x1 + 5, result_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Afficher les FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "Q: Quitter | R: Reset | C: Calibrer",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """
        Lance l'analyseur en temps réel.
        """
        print("\n" + "=" * 60)
        print("    ANALYSEUR DE RÉSISTANCES EN TEMPS RÉEL")
        print("    Projet de Traitement de Signal - EPHEC 2025")
        print("=" * 60)
        print("\nDémarrage de la caméra...")

        # Ouvrir la caméra
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"❌ Erreur: Impossible d'ouvrir la caméra {self.camera_id}")
            print("   Essayez avec un autre ID (1, 2, ...)")
            return

        print(f"✓ Caméra {self.camera_id} ouverte")
        print("\nCommandes:")
        print("  Q ou ESC: Quitter")
        print("  R: Réinitialiser la stabilisation")
        print("  C: Afficher les infos de calibration")
        print("=" * 60)

        self.is_running = True
        frame_times = deque(maxlen=30)

        try:
            while self.is_running:
                start_time = time.time()

                # Capturer une frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Erreur de lecture de la caméra")
                    break

                # Analyser la frame
                reading = self.analyze_frame(frame)
                stable_reading = self.stabilizer.get_stable_reading()

                # Calculer les FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

                # Extraire les coordonnées de la ROI pour l'affichage
                _, roi_coords = self.extract_roi(frame)

                # Dessiner l'interface
                display_frame = self.draw_ui(frame, roi_coords,
                                             reading, stable_reading, fps)

                # Afficher
                cv2.imshow('Analyseur de Resistances', display_frame)

                # Gérer les touches
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q ou ESC
                    break
                elif key == ord('r'):  # R = Reset
                    self.stabilizer.clear()
                    print("🔄 Stabilisation réinitialisée")
                elif key == ord('c'):  # C = Calibration info
                    self._print_calibration_info()

                # Vérifier le critère de temps < 1s
                if frame_time > 1.0:
                    print(f"⚠️ Temps de traitement élevé: {frame_time * 1000:.0f}ms")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Analyse terminée")

    def _print_calibration_info(self):
        """Affiche les informations de calibration des couleurs."""
        print("\n📊 Plages HSV des couleurs:")
        print("-" * 50)
        for name, (lower, upper, val) in HSVColorConfig.get_colors().items():
            print(f"  {name:10s}: H=[{lower[0]:3d}-{upper[0]:3d}], "
                  f"S=[{lower[1]:3d}-{upper[1]:3d}], "
                  f"V=[{lower[2]:3d}-{upper[2]:3d}] -> {val}")
        print("-" * 50)


def find_available_cameras(max_cameras: int = 10) -> List[Tuple[int, str]]:
    """
    Recherche toutes les caméras disponibles sur le système.
    Utile pour trouver iVCam, DroidCam, etc.

    Returns:
        Liste de tuples (id, description)
    """
    available = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Essayer de lire une frame pour vérifier
            ret, _ = cap.read()
            if ret:
                # Obtenir la résolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, f"Camera {i} ({width}x{height})"))
            cap.release()

    return available


def select_camera() -> int:
    """
    Affiche les caméras disponibles et permet à l'utilisateur de choisir.

    Returns:
        ID de la caméra sélectionnée
    """
    print("\n🔍 Recherche des caméras disponibles...")
    cameras = find_available_cameras()

    if not cameras:
        print("❌ Aucune caméra trouvée!")
        return 0

    print(f"\n📷 {len(cameras)} caméra(s) trouvée(s):")
    for cam_id, desc in cameras:
        print(f"   [{cam_id}] {desc}")

    if len(cameras) == 1:
        print(f"\n→ Utilisation de la caméra {cameras[0][0]}")
        return cameras[0][0]

    # Si plusieurs caméras, demander à l'utilisateur
    print("\n💡 Conseil: Si iVCam ne fonctionne pas avec l'ID 0,")
    print("   essayez un autre numéro (souvent 1 ou 2 pour les caméras virtuelles)")

    while True:
        try:
            choice = input(f"\nEntrez le numéro de caméra (0-{len(cameras) - 1}, ou 'a' pour auto): ").strip()
            if choice.lower() == 'a':
                # Auto: prendre la caméra avec la plus haute résolution (souvent iVCam)
                best = max(cameras, key=lambda x: int(x[1].split('(')[1].split('x')[0]))
                print(f"→ Sélection automatique: {best[1]}")
                return best[0]
            else:
                cam_id = int(choice)
                if any(c[0] == cam_id for c in cameras):
                    return cam_id
                print("⚠️ Numéro invalide, réessayez.")
        except ValueError:
            print("⚠️ Entrez un numéro valide.")
        except KeyboardInterrupt:
            return cameras[0][0]


# =============================================================================
# SECTION 9: POINT D'ENTRÉE
# =============================================================================

def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyseur de résistances en temps réel'
    )
    parser.add_argument('-c', '--camera', type=int, default=-1,
                        help='ID de la caméra (-1 = sélection interactive)')
    parser.add_argument('-w', '--width', type=int, default=400,
                        help='Largeur de la zone de scan (défaut: 400)')
    parser.add_argument('-H', '--height', type=int, default=120,
                        help='Hauteur de la zone de scan (défaut: 120)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='Lister les caméras disponibles et quitter')

    args = parser.parse_args()

    # Si demande de liste des caméras
    if args.list:
        print("\n🔍 Recherche des caméras disponibles...")
        cameras = find_available_cameras()
        if cameras:
            print(f"\n📷 {len(cameras)} caméra(s) trouvée(s):")
            for cam_id, desc in cameras:
                print(f"   [{cam_id}] {desc}")
            print("\n💡 Pour utiliser iVCam, lancez l'app sur votre téléphone")
            print("   puis relancez ce script. iVCam aura souvent l'ID 1 ou 2.")
        else:
            print("❌ Aucune caméra trouvée!")
        return

    # Sélection de la caméra
    if args.camera == -1:
        camera_id = select_camera()
    else:
        camera_id = args.camera

    # Créer et lancer l'analyseur
    analyzer = RealtimeResistanceAnalyzer(
        camera_id=camera_id,
        roi_width=args.width,
        roi_height=args.height
    )

    analyzer.run()


if __name__ == "__main__":
    main()