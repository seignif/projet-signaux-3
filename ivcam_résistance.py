import cv2
import numpy as np

def get_color_ranges():
    """
    Définit les plages HSV pour les couleurs des résistances.
    NOTE : Ces valeurs sont sensibles à l'éclairage.
    À calibrer selon ta lampe !
    """
    return {
        'black':  ([0, 0, 0], [180, 255, 30], 0),     # Noir (très sombre)
        'brown':  ([0, 60, 30], [20, 255, 200], 1),   # Marron
        'red':    ([0, 100, 100], [10, 255, 255], 2), # Rouge (bas du spectre)
        'red2':   ([160, 100, 100], [180, 255, 255], 2), # Rouge (haut du spectre)
        'orange': ([11, 100, 100], [25, 255, 255], 3),
        'yellow': ([26, 80, 100], [35, 255, 255], 4),
        'green':  ([36, 50, 50], [85, 255, 255], 5),
        'blue':   ([86, 50, 50], [125, 255, 255], 6),
        'violet': ([126, 50, 50], [145, 255, 255], 7),
        'gray':   ([0, 0, 50], [180, 50, 200], 8),    # Gris (faible saturation)
        'white':  ([0, 0, 200], [180, 30, 255], 9),   # Blanc (haute luminosité, faible sat)
        'gold':   ([20, 100, 100], [30, 255, 150], -1), # Or (souvent sombre/jaune sale)
        'silver': ([0, 0, 100], [180, 20, 200], -2)   # Argent (gris brillant)
    }

def calculate_resistance(bands):
    """Calcule la valeur en Ohms basée sur la liste des bandes détectées"""
    if len(bands) < 3:
        return "Attente...", (0, 0, 255)

    # Filtrer les valeurs inconnues ou doublons trop proches
    # Pour ce MVP, on prend les 3 ou 4 premières bandes triées par position X

    vals = [b['val'] for b in bands]

    # Logique simple pour 4 bandes (la plus courante)
    # Chiffre 1, Chiffre 2, Multiplicateur, Tolérance

    # Si la dernière bande est Or ou Argent, c'est la tolérance (donc la fin)
    if vals[-1] in [-1, -2] and len(vals) >= 4:
         # Orientation Correcte
         pass
    elif vals[0] in [-1, -2]:
         # Inversé (Or/Argent au début)
         vals.reverse()

    try:
        digit1 = vals[0]
        digit2 = vals[1]
        multiplier_idx = vals[2]

        if digit1 < 0 or digit2 < 0: return "Erreur Lecture", (0,0,255)

        multiplier = 0
        if multiplier_idx >= 0:
            multiplier = 10 ** multiplier_idx
        elif multiplier_idx == -1: # Or
            multiplier = 0.1
        elif multiplier_idx == -2: # Argent
            multiplier = 0.01

        ohms = (digit1 * 10 + digit2) * multiplier

        # Formatage
        if ohms >= 1_000_000:
            label = f"{ohms/1_000_000:.2f} M Ohms"
        elif ohms >= 1_000:
            label = f"{ohms/1_000:.2f} k Ohms"
        else:
            label = f"{ohms:.1f} Ohms"

        return label, (0, 255, 0)

    except IndexError:
        return "Detection Instable", (0, 255, 255)

def main():
    # 0 est généralement la webcam par défaut (ou iVCam)
    # Si ça ne marche pas, essaye 1 ou 2
    cap = cv2.VideoCapture(0)

    # Paramètres de la fenêtre de scan (ROI)
    scan_w, scan_h = 400, 100 # Zone rectangulaire

    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h_img, w_img, _ = frame.shape

        # Définir le rectangle central (ROI)
        x1 = (w_img - scan_w) // 2
        y1 = (h_img - scan_h) // 2
        x2 = x1 + scan_w
        y2 = y1 + scan_h

        # Extraire la ROI et travailler uniquement dessus (Gain de temps calcul < 1s)
        roi = frame[y1:y2, x1:x2].copy()

        # 1. Prétraitement : Filtre Bilatéral (Garde les bords, lisse le fond)
        roi_blurred = cv2.bilateralFilter(roi, 9, 75, 75)
        roi_hsv = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2HSV)

        # 2. Détection des bandes
        color_ranges = get_color_ranges()
        detected_bands = []

        # Debug: créer une image noire pour voir ce que l'algo détecte
        debug_mask_sum = np.zeros(roi.shape[:2], dtype=np.uint8)

        for color_name, (lower, upper, val) in color_ranges.items():
            mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))

            # Morphologie pour nettoyer le bruit (points isolés)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)

                # Filtres géométriques : Une bande est une ligne verticale
                # Elle doit être assez haute (relativement à la ROI) et pas trop large
                if area > 50 and h > 20 and w < 40:
                    # Éviter les doublons (ex: rouge détecté deux fois au même endroit)
                    duplicate = False
                    for b in detected_bands:
                        if abs(b['x'] - x) < 10: # Trop proche d'une autre bande
                            duplicate = True
                            break

                    if not duplicate:
                        detected_bands.append({'color': color_name, 'val': val, 'x': x})
                        # Dessiner sur la ROI
                        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(roi, color_name[0].upper(), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Pour la visu debug
            debug_mask_sum = cv2.bitwise_or(debug_mask_sum, mask)

        # 3. Trier les bandes de gauche à droite
        detected_bands.sort(key=lambda b: b['x'])

        # 4. Calculer la valeur
        result_text, text_color = calculate_resistance(detected_bands)

        # --- Affichage ---

        # Dessiner le viseur sur l'image principale
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "Placez la resistance ici (Fond Uni)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Afficher le résultat
        cv2.rectangle(frame, (x1, y2 + 10), (x2, y2 + 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Valeur: {result_text}", (x1 + 10, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # Afficher les bandes trouvées (texte)
        bands_str = " ".join([b['color'] for b in detected_bands])
        cv2.putText(frame, bands_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Scanner Resistance (iVCam)', frame)
        cv2.imshow('Debug Masques', debug_mask_sum) # Utile pour voir si les couleurs sont détectées

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()