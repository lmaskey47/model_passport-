import numpy as np
import tensorflow as tf
import cv2
import json
import keras_ocr
import os
import re
from difflib import SequenceMatcher
from object_detection.utils import label_map_util
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# === Paramètres ===
PATH_TO_MODEL = './exported-models/passport_exported/saved_model'
PATH_TO_LABELS = './annotations/label_map.pbtxt'
PATH_TO_IMAGES = './images/test'
OUTPUT_FOLDER = './results'

# === Préparation du modèle ===
detect_fn = tf.saved_model.load(PATH_TO_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
ocr_pipeline = keras_ocr.pipeline.Pipeline()

# === Dictionnaires de référence ===
COUNTRIES = {
    'united states of america', 'united states', 'usa', 'america',
    'france', 'french republic', 'republic of france',
    'canada', 'united kingdom', 'uk', 'great britain',
    'germany', 'deutschland', 'federal republic of germany',
    'spain', 'reino de espana', 'kingdom of spain',
    'italy', 'italia', 'italian republic',
    'netherlands', 'kingdom of the netherlands',
    'belgium', 'kingdom of belgium',
    'switzerland', 'swiss confederation',
    'austria', 'republic of austria',
    'portugal', 'portuguese republic',
    'sweden', 'kingdom of sweden',
    'norway', 'kingdom of norway',
    'denmark', 'kingdom of denmark',
    'finland', 'republic of finland',
    'poland', 'republic of poland',
    'czech republic', 'czechia',
    'hungary', 'republic of hungary',
    'greece', 'hellenic republic',
    'turkey', 'republic of turkey',
    'russia', 'russian federation',
    'china', 'peoples republic of china',
    'japan', 'japan',
    'south korea', 'republic of korea',
    'australia', 'commonwealth of australia',
    'brazil', 'federative republic of brazil',
    'argentina', 'argentine republic',
    'mexico', 'united mexican states',
    'india', 'republic of india',
    'south africa', 'republic of south africa'
}

COMMON_NAMES = {
    # Prénoms masculins
    'aaron', 'adam', 'alan', 'albert', 'alex', 'alexander', 'andrew', 'anthony', 'antonio',
    'arthur', 'benjamin', 'bernard', 'brian', 'bruce', 'carl', 'charles', 'christopher',
    'daniel', 'david', 'dennis', 'donald', 'douglas', 'edward', 'eric', 'eugene',
    'frank', 'gary', 'george', 'gerald', 'gregory', 'harold', 'henry', 'jack',
    'james', 'jason', 'jeffrey', 'jerry', 'jesse', 'john', 'jonathan', 'joseph',
    'joshua', 'juan', 'kenneth', 'kevin', 'larry', 'lawrence', 'louis', 'mark',
    'martin', 'matthew', 'michael', 'paul', 'peter', 'philip', 'raymond', 'richard',
    'robert', 'ronald', 'ryan', 'samuel', 'scott', 'stephen', 'steven', 'thomas',
    'timothy', 'walter', 'wayne', 'william',
    
    # Prénoms féminins
    'amy', 'angela', 'anna', 'barbara', 'betty', 'brenda', 'carol', 'carolyn',
    'catherine', 'christine', 'cynthia', 'deborah', 'diane', 'donna', 'dorothy',
    'elizabeth', 'emily', 'frances', 'helen', 'janet', 'jennifer', 'jessica',
    'joan', 'judith', 'judy', 'julia', 'julie', 'karen', 'kathleen', 'kimberly',
    'laura', 'linda', 'lisa', 'margaret', 'maria', 'marie', 'martha', 'mary',
    'michelle', 'nancy', 'patricia', 'rebecca', 'ruth', 'sandra', 'sarah',
    'sharon', 'stephanie', 'susan', 'teresa', 'virginia',
    
    # Noms de famille courants
    'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
    'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson',
    'thomas', 'taylor', 'moore', 'jackson', 'martin', 'lee', 'perez', 'thompson',
    'white', 'harris', 'sanchez', 'clark', 'ramirez', 'lewis', 'robinson', 'walker',
    'young', 'allen', 'king', 'wright', 'scott', 'torres', 'nguyen', 'hill',
    'flores', 'green', 'adams', 'nelson', 'baker', 'hall', 'rivera', 'campbell',
    'mitchell', 'carter', 'roberts', 'gomez', 'phillips', 'evans', 'turner',
    'diaz', 'parker', 'cruz', 'edwards', 'collins', 'reyes', 'stewart', 'morris',
    'morales', 'murphy', 'cook', 'rogers', 'gutierrez', 'ortiz', 'morgan',
    'cooper', 'peterson', 'bailey', 'reed', 'kelly', 'howard', 'ramos', 'kim',
    'cox', 'ward', 'richardson', 'watson', 'brooks', 'chavez', 'wood', 'james',
    'bennett', 'gray', 'mendoza', 'ruiz', 'hughes', 'price', 'alvarez', 'castillo',
    'sanders', 'patel', 'myers', 'long', 'ross', 'foster', 'jimenez', 'powell',
    'jenkins', 'perry', 'russell', 'sullivan', 'bell', 'coleman', 'butler',
    'henderson', 'barnes', 'gonzales', 'fisher', 'vasquez', 'simmons', 'romero',
    'jordan', 'patterson', 'alexander', 'hamilton', 'graham', 'reynolds', 'griffin',
    'wallace', 'moreno', 'west', 'cole', 'hayes', 'bryant', 'herrera', 'gibson',
    'ellis', 'tran', 'medina', 'aguilar', 'stevens', 'murray', 'ford', 'castro',
    'marshall', 'owen', 'harrison', 'burton', 'kennedy', 'rivera', 'warren',
    'dixon', 'ramos', 'reyna', 'reid', 'fleming', 'nielsen', 'stone', 'andrews',
    'webster', 'crawford', 'oliver', 'andrews', 'warner', 'baldwin'
}

MONTHS = {
    'january': 'jan', 'february': 'feb', 'march': 'mar', 'april': 'apr',
    'may': 'may', 'june': 'jun', 'july': 'jul', 'august': 'aug',
    'september': 'sep', 'october': 'oct', 'november': 'nov', 'december': 'dec',
    'jan': 'jan', 'feb': 'feb', 'mar': 'mar', 'apr': 'apr',
    'may': 'may', 'jun': 'jun', 'jul': 'jul', 'aug': 'aug',
    'sep': 'sep', 'oct': 'oct', 'nov': 'nov', 'dec': 'dec'
}

# === Stop words (sans M et F pour le sexe) ===
stop_words = {
    't', 'e', 'l', 'd', 'i', 'of', 'de', 'du', 'le', 'la', 'no', 'and',
    'nacimiento', 'caducidad', 'vationality', 'inacionalidad', 'surname', 'isexe', 'asspont',
    'to', 'nom', 'apellidos', 'given', 'names', 'prenoms', 'nombres', 'date', 'birth', 'naissance',
    'fecha', 'sex', 'sexe', 'sexo', 'nationality', 'nationalite', 'nacionalidad', 'document',
    'passeport', 'pasaporie', 'expiration', 'expiracion', 'expirationt', 'place', 'issue', 'autorite',
    'authority', 'state', 'department', 'usa', 'type', 'code', 'passport',
    'ate', 'jate', 'lationality', 'asspon', 'caducidac', 'passepot', 'ino', 'pasapore', 'lsexo', 'isexo'
}
def process_multiple_images(image_paths, progress_callback=None):
    """Traite plusieurs images en parallèle"""
    results = {}
    total_images = len(image_paths)
    
    def process_single_with_callback(args):
        idx, image_path = args
        try:
            image_rgb, detections = process_image(image_path)
            if progress_callback:
                progress_callback(idx + 1, total_images, os.path.basename(image_path))
            return image_path, (image_rgb, detections)
        except Exception as e:
            if progress_callback:
                progress_callback(idx + 1, total_images, os.path.basename(image_path), error=str(e))
            return image_path, None
    
    # Traitement en parallèle avec 2 threads max pour éviter la surcharge mémoire
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = list(executor.map(process_single_with_callback, enumerate(image_paths)))
        
    for image_path, result in futures:
        results[image_path] = result
    
    return results
def similarity(a, b):
    """Calcule la similarité entre deux chaînes"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(text, reference_set, min_similarity=0.6):
    """Trouve la meilleure correspondance dans un ensemble de référence"""
    if not text or len(text) < 2:
        return None
        
    text_clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    best_match = None
    best_score = 0
    
    for ref in reference_set:
        score = similarity(text_clean, ref)
        if score > best_score and score >= min_similarity:
            best_score = score
            best_match = ref
    
    return best_match if best_score >= min_similarity else None

def correct_country(text):
    """Corrige les erreurs d'OCR pour les pays"""
    # D'abord essayer une correspondance exacte
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    if cleaned in COUNTRIES:
        return cleaned
    
    # Ensuite chercher la meilleure correspondance
    best_match = find_best_match(text, COUNTRIES, min_similarity=0.7)
    if best_match:
        return best_match
    
    # Traitement spécial pour "United States"
    if any(word in cleaned for word in ['unit', 'unite', 'unitle', 'united']):
        if any(word in cleaned for word in ['stat', 'state', 'statds']):
            if any(word in cleaned for word in ['amer', 'ameri', 'america']):
                return 'united states of america'
            else:
                return 'united states'
    
    return text

def correct_name(text):
    """Corrige les erreurs d'OCR pour les noms"""
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    # Correspondance exacte
    if cleaned in COMMON_NAMES:
        return cleaned
    
    # Correspondance approximative
    best_match = find_best_match(text, COMMON_NAMES, min_similarity=0.75)
    return best_match if best_match else cleaned

def correct_month(text):
    """Corrige les erreurs d'OCR pour les mois"""
    cleaned = re.sub(r'[^\w]', '', text.lower()).strip()
    
    # Correspondance exacte
    if cleaned in MONTHS:
        return MONTHS[cleaned]
    
    # Correspondance approximative
    best_match = find_best_match(text, MONTHS.keys(), min_similarity=0.7)
    return MONTHS[best_match] if best_match else text

def clean_ocr_field(preds, field_type):
    """
    Nettoie et corrige le texte OCR selon le type de champ
    """
    # Extraire tous les mots significatifs
    all_words = []
    for txt, _ in preds:
        cleaned = re.sub(r'[^\w\s]', '', txt.lower()).strip()
        if len(cleaned) > 1 and cleaned not in stop_words:
            all_words.append(cleaned)
    
    if not all_words:
        return ""
    
    # Traitement spécifique par type de champ
    if field_type in ['surname', 'names']:
        # Pour les noms, garder seulement les mots alphabétiques
        name_words = [word for word in all_words if word.isalpha() and len(word) >= 3]
        corrected_names = [correct_name(word) for word in name_words]
        return ' '.join(filter(None, corrected_names))
    
    elif field_type == 'document_number':
        # Pour les numéros, garder le plus long contenant des chiffres
        number_candidates = [word for word in all_words if any(c.isdigit() for c in word) and len(word) >= 5]
        return max(number_candidates, key=len) if number_candidates else ""
    
    elif field_type in ['data_of_birth', 'date_of_expiration']:
        # Pour les dates, identifier et corriger les composants
        date_parts = []
        for word in all_words:
            if word.isdigit() and len(word) in [1, 2, 4]:
                date_parts.append(word)
            elif word.isalpha():
                corrected_month = correct_month(word)
                if corrected_month != word:  # Si correction trouvée
                    date_parts.append(corrected_month)
                elif word in MONTHS:
                    date_parts.append(word)
        return ' '.join(date_parts)
    
    elif field_type == 'nationality':
        # Pour la nationalité, corriger les pays
        country_text = ' '.join(all_words)
        return correct_country(country_text)
    
    elif field_type == 'sexe':
        # Pour le sexe, chercher M ou F dans TOUS les mots (même d'une lettre)
        for txt, _ in preds:
            # Nettoyer mais garder les lettres seules
            cleaned = re.sub(r'[^\w\s]', '', txt).strip()
            for word in cleaned.split():
                if word.upper() in ['M', 'F', 'MALE', 'FEMALE']:
                    return 'M' if word.upper() in ['M', 'MALE'] else 'F'
        return ""
    
    return ' '.join(all_words)

def process_image(image_path):
    """Traite une seule image et retourne les résultats"""
    image_np = cv2.imread(image_path)
    if image_np is None:
        return None, None
    
    image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([image_np_rgb], dtype=tf.uint8)
    
    detections = detect_fn(input_tensor)
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()
    
    h, w, _ = image_np.shape
    results = []
    
    for j in range(min(50, boxes.shape[0])):
        if scores[j] < 0.5:
            continue
        
        ymin, xmin, ymax, xmax = boxes[j]
        left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
        roi = image_np_rgb[top:bottom, left:right]
        if roi.size == 0:
            continue
        
        preds = ocr_pipeline.recognize([roi])[0]
        full_text = ', '.join([txt for txt, _ in preds])
        
        # Obtenir le nom du champ
        field_name = category_index[classes[j]]['name'] if classes[j] in category_index else "unknown"
        
        # Nettoyer et corriger selon le type de champ
        clean_text = clean_ocr_field(preds, field_name)
        
        results.append({
            "class_id": int(classes[j]),
            "class_name": field_name,
            "score": float(scores[j]),
            "bbox_normalized": {
                "ymin": float(ymin),
                "xmin": float(xmin),
                "ymax": float(ymax),
                "xmax": float(xmax)
            },
            "bbox_pixels": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom
            },
            "ocr_text": full_text,
            "ocr_text_clean": clean_text
        })
    
    return image_np_rgb, results

# === Interface graphique ===
class PassportDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualiseur de Détection de Passeport")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')
        
        # Variables
        self.current_image = None
        self.current_results = None
        self.photo_image = None
        self.current_batch_results = None
        self.current_image_index = 0
        self.image_list = []
        
        # Couleurs pour les différents champs
        self.field_colors = {
            'surname': '#E74C3C',
            'names': '#3498DB',
            'document_number': '#E67E22',
            'nationality': '#27AE60',
            'data_of_birth': '#9B59B6',
            'date_of_expiration': '#F39C12',
            'sexe': '#E91E63',
            'unknown': '#95A5A6'
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre
        title_label = tk.Label(main_frame, text="Visualiseur de Détection de Passeport", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2C3E50')
        title_label.pack(pady=(0, 20))
        
        # Frame des contrôles
        control_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Boutons de contrôle
        btn_frame = tk.Frame(control_frame, bg='#34495E')
        btn_frame.pack(pady=10)
                # Ajouter après btn_frame.pack(pady=10) :
        # Frame de navigation pour les images multiples
        nav_frame = tk.Frame(control_frame, bg='#34495E')
        nav_frame.pack(pady=5)

        self.nav_label = tk.Label(nav_frame, text="", fg='white', bg='#34495E', font=('Arial', 10))
        self.nav_label.pack(side=tk.LEFT, padx=10)

        tk.Button(nav_frame, text="◀ Précédent", command=self.previous_image,
                bg='#95A5A6', fg='white', font=('Arial', 9),
                padx=10, pady=2).pack(side=tk.LEFT, padx=2)

        tk.Button(nav_frame, text="Suivant ▶", command=self.next_image,
                bg='#95A5A6', fg='white', font=('Arial', 9),
                padx=10, pady=2).pack(side=tk.LEFT, padx=2)

        # Barre de progression
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        self.progress.pack_forget()  # Cacher initialement
        tk.Button(btn_frame, text="Charger Image", command=self.load_image,
                bg='#3498DB', fg='white', font=('Arial', 10, 'bold'),
                padx=20, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Charger Plusieurs Images", command=self.load_multiple_images,
                bg='#9B59B6', fg='white', font=('Arial', 10, 'bold'),
                padx=20, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Traiter Image(s)", command=self.process_current_image,
                bg='#27AE60', fg='white', font=('Arial', 10, 'bold'),
                padx=20, pady=5).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="Sauvegarder Résultats", command=self.save_results,
                bg='#E67E22', fg='white', font=('Arial', 10, 'bold'),
                padx=20, pady=5).pack(side=tk.LEFT, padx=5)
                
        # Frame principal de contenu
        content_frame = tk.Frame(main_frame, bg='#2C3E50')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame image (gauche)
        image_frame = tk.Frame(content_frame, bg='#34495E', relief=tk.RAISED, bd=2)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(image_frame, text="Image avec Détections", font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E').pack(pady=5)
        
        # Canvas pour l'image
        self.image_canvas = tk.Canvas(image_frame, bg='white')
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame résultats (droite)
        results_frame = tk.Frame(content_frame, bg='#34495E', relief=tk.RAISED, bd=2, width=400)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        results_frame.pack_propagate(False)
        
        tk.Label(results_frame, text="Champs Détectés", font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E').pack(pady=5)
        
        # Treeview pour les résultats
        tree_frame = tk.Frame(results_frame, bg='#34495E')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar pour le treeview
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        self.results_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set,
                                        selectmode="extended")
        self.results_tree['columns'] = ("Champ", "Score", "Texte Nettoyé")
        self.results_tree.column("#0", width=0, stretch=tk.NO)
        self.results_tree.column("Champ", anchor=tk.W, width=100)
        self.results_tree.column("Score", anchor=tk.CENTER, width=60)
        
        self.results_tree.column("Texte Nettoyé", anchor=tk.W, width=120)
        
        self.results_tree.heading("#0", text="", anchor=tk.W)
        self.results_tree.heading("Champ", text="Champ", anchor=tk.W)
        self.results_tree.heading("Score", text="Score", anchor=tk.CENTER)
        
        self.results_tree.heading("Texte Nettoyé", text="Texte Nettoyé", anchor=tk.W)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.results_tree.yview)
        
        # Binding pour surligner les détections
        self.results_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        # Frame de statut
        status_frame = tk.Frame(main_frame, bg='#34495E', height=30)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Prêt - Chargez une image pour commencer",
                                   fg='white', bg='#34495E', font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    
    
    
    
    
    def load_multiple_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Sélectionner plusieurs images de passeport",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
    
        if file_paths:
            self.image_list = list(file_paths)
            self.current_image_index = 0
            self.current_batch_results = None
            
            # Charger la première image pour prévisualisation
            self.load_single_image_from_list(0)
            self.update_navigation_display()
            
            self.status_label.config(text=f"{len(file_paths)} images chargées - Prêt pour traitement")

    def load_single_image_from_list(self, index):
        """Charge une image spécifique de la liste"""
        if 0 <= index < len(self.image_list):
            file_path = self.image_list[index]
            try:
                pil_image = Image.open(file_path)
                self.current_image_path = file_path
                
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
                
                self.photo_image = ImageTk.PhotoImage(pil_image)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    self.image_canvas.winfo_width()//2,
                    self.image_canvas.winfo_height()//2,
                    image=self.photo_image,
                    anchor=tk.CENTER
                )
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger l'image: {str(e)}")

    def previous_image(self):
        """Image précédente"""
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_single_image_from_list(self.current_image_index)
            self.display_current_results()
            self.update_navigation_display()

    def next_image(self):
        """Image suivante"""
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_single_image_from_list(self.current_image_index)
            self.display_current_results()
            self.update_navigation_display()

    def update_navigation_display(self):
        """Met à jour l'affichage de navigation"""
        if self.image_list:
            current_name = os.path.basename(self.image_list[self.current_image_index])
            self.nav_label.config(text=f"Image {self.current_image_index + 1}/{len(self.image_list)}: {current_name}")
        else:
            self.nav_label.config(text="")

    def display_current_results(self):
        """Affiche les résultats de l'image courante"""
        if (self.current_batch_results and 
            self.current_image_index < len(self.image_list)):
            
            current_path = self.image_list[self.current_image_index]
            if current_path in self.current_batch_results:
                result_data = self.current_batch_results[current_path]
                if result_data:
                    self.current_image, self.current_results = result_data
                    self.update_display()

    def progress_callback(self, current, total, filename, error=None):
        """Callback pour la barre de progression"""
        def update_ui():
            progress_percent = (current / total) * 100
            self.progress['value'] = progress_percent
            
            if error:
                status_text = f"Erreur avec {filename}: {error[:50]}..."
            else:
                status_text = f"Traitement: {current}/{total} - {filename}"
            
            self.status_label.config(text=status_text)
            self.root.update()
        
        self.root.after(0, update_ui)
        
        
        

        
        
        
        
        
        
        
    
    
    def load_image(self):
        """Charge une image"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image de passeport",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Charger l'image avec PIL
                pil_image = Image.open(file_path)
                self.current_image_path = file_path
                
                # Redimensionner pour l'affichage si nécessaire
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
                
                # Convertir pour tkinter
                self.photo_image = ImageTk.PhotoImage(pil_image)
                
                # Afficher dans le canvas
                self.image_canvas.delete("all")
                self.image_canvas.create_image(
                    self.image_canvas.winfo_width()//2,
                    self.image_canvas.winfo_height()//2,
                    image=self.photo_image,
                    anchor=tk.CENTER
                )
                
                self.status_label.config(text=f"Image chargée: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger l'image: {str(e)}")
   
   
   
   
    
    def process_current_image(self):
        if not hasattr(self, 'current_image_path') and not self.image_list:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une ou plusieurs images")
            return
    
        # Si plusieurs images sont chargées
        if self.image_list:
            self.process_multiple_images_gui()
        # Si une seule image
        elif hasattr(self, 'current_image_path'):
            self.process_single_image_gui()

    def process_single_image_gui(self):
        """Traite une seule image (méthode existante renommée)"""
        self.status_label.config(text="Traitement en cours...")
        self.root.update()
        
        def process_thread():
            try:
                image_rgb, results = process_image(self.current_image_path)
                
                if image_rgb is not None and results is not None:
                    self.current_image = image_rgb
                    self.current_results = results
                    self.root.after(0, self.update_display)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Erreur", "Impossible de traiter l'image"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur de traitement: {str(e)}"))
                self.root.after(0, lambda: self.status_label.config(text="Erreur de traitement"))
        
        threading.Thread(target=process_thread, daemon=True).start()

    def process_multiple_images_gui(self):
        """Traite plusieurs images"""
        self.progress.pack(fill=tk.X, padx=10, pady=5)  # Afficher la barre de progression
        self.progress['value'] = 0
        
        def process_thread():
            try:
                start_time = time.time()
                self.current_batch_results = process_multiple_images(
                    self.image_list, 
                    progress_callback=self.progress_callback
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Afficher les résultats de la première image
                self.root.after(0, lambda: self.display_current_results())
                
                successful = sum(1 for result in self.current_batch_results.values() if result is not None)
                failed = len(self.image_list) - successful
                
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Traitement terminé: {successful} réussies, {failed} échouées en {processing_time:.1f}s"
                ))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur de traitement: {str(e)}"))
                self.root.after(0, lambda: self.status_label.config(text="Erreur de traitement"))
            finally:
                self.root.after(0, lambda: self.progress.pack_forget())  # Cacher la barre de progression
        
        threading.Thread(target=process_thread, daemon=True).start()
 
 
 
 
 
 
 
 
 
    
    def update_display(self):
        """Met à jour l'affichage avec les résultats"""
        if self.current_image is None or self.current_results is None:
            return
        
        # Créer l'image avec les boîtes de détection
        pil_image = Image.fromarray(self.current_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Dessiner les boîtes de détection
        for result in self.current_results:
            bbox = result['bbox_pixels']
            field_name = result['class_name']
            color = self.field_colors.get(field_name, self.field_colors['unknown'])
            
            # Dessiner le rectangle
            draw.rectangle([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']], 
                         outline=color, width=3)
            
            # Ajouter le label
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            draw.text((bbox['left'], bbox['top']-20), 
                     f"{field_name} ({result['score']:.2f})", 
                     fill=color, font=font)
        
        # Redimensionner pour l'affichage
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        
        # Convertir pour tkinter et afficher
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.photo_image, anchor=tk.CENTER
        )
        
        # Mettre à jour le treeview
        self.results_tree.delete(*self.results_tree.get_children())
        
        for i, result in enumerate(self.current_results):
            self.results_tree.insert("", "end", iid=i, values=(
                result['class_name'],
                f"{result['score']:.3f}",
                result['ocr_text_clean'][:30] + "..." if len(result['ocr_text_clean']) > 30 else result['ocr_text_clean']
            ))
        
        self.status_label.config(text=f"Traitement terminé - {len(self.current_results)} champs détectés")
    
    def on_tree_select(self, event):
        selection = self.results_tree.selection()
        if not selection or not self.current_results:
            return

        selected_idx = int(selection[0])
        result = self.current_results[selected_idx]
        pil_image = Image.fromarray(self.current_image)
        draw = ImageDraw.Draw(pil_image)

        bbox = result['bbox_pixels']
        color = self.field_colors.get(result['class_name'], self.field_colors['unknown'])
        draw.rectangle([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']], 
                       outline=color, width=4)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        draw.text((bbox['left'], bbox['top'] - 20),
                  f"{result['class_name']} ({result['score']:.2f})",
                  fill=color, font=font)

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            pil_image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo_image, anchor=tk.CENTER
        )
    
    def save_results(self):
        if not self.current_results and not self.current_batch_results:
            messagebox.showwarning("Aucun résultat", "Aucun résultat à sauvegarder.")
            return

        # Si on a des résultats multiples
        if self.current_batch_results:
            # Proposer de sauvegarder tous les résultats ou seulement l'image courante
            choice = messagebox.askyesnocancel(
                "Type de sauvegarde", 
                "Voulez-vous sauvegarder tous les résultats ?\n\n"
                "Oui = Tous les résultats\n"
                "Non = Seulement l'image courante\n"
                "Annuler = Annuler"
            )
            
            if choice is None:  # Annuler
                return
            elif choice:  # Tous les résultats
                self.save_all_results()
            else:  # Image courante seulement
                self.save_current_result()
        else:
            # Résultat unique
            self.save_current_result()

    def save_current_result(self):
        """Sauvegarde le résultat de l'image courante"""
        if not self.current_results:
            messagebox.showwarning("Aucun résultat", "Aucun résultat pour l'image courante.")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json", 
            filetypes=[("JSON files", "*.json")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Succès", f"Résultats sauvegardés dans : {save_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder : {str(e)}")

    def save_all_results(self):
        """Sauvegarde tous les résultats dans un dossier"""
        save_dir = filedialog.askdirectory(title="Sélectionner le dossier de sauvegarde")
        if save_dir:
            try:
                successful_saves = 0
                for image_path, result_data in self.current_batch_results.items():
                    if result_data and result_data[1]:  # Si on a des résultats
                        image_name = os.path.splitext(os.path.basename(image_path))[0]
                        save_path = os.path.join(save_dir, f"{image_name}_results.json")
                        
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data[1], f, ensure_ascii=False, indent=2)
                        successful_saves += 1
                
                messagebox.showinfo("Succès", 
                    f"{successful_saves} fichiers de résultats sauvegardés dans : {save_dir}")
                    
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder : {str(e)}")
    
if __name__ == '__main__':
    root = tk.Tk()
    app = PassportDetectionGUI(root)
    root.mainloop()
