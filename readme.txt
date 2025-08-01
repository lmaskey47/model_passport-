🛂 Projet OCR Passeport — Guide d’utilisation complet
Ce projet combine TensorFlow Object Detection et Keras-OCR pour détecter, extraire et corriger automatiquement les informations présentes sur un passeport (nom, prénom, sexe, pays, date de naissance, etc.) via une interface utilisateur simple.

🚧 Étape 1 – Tester l’installation de TensorFlow Object Detection (à faire une seule fois)

cd models\research
python object_detection/builders/model_builder_tf2_test.py
✅ Pourquoi ?
Ce test vérifie que l’installation du framework TensorFlow Object Detection est fonctionnelle :
dépendances installées
structure des répertoires correcte
compilation des fichiers .proto bien effectuée

❗ À faire uniquement après la première installation ou si des erreurs apparaissent plus tard.

🧪 Étape 2 – Activer l’environnement virtuel (à faire à chaque session)
cd Documents\projettensorflow
tfod310\Scripts\activate.
🔁 Pourquoi ?
L’environnement virtuel contient toutes les librairies spécifiques nécessaires au projet (TensorFlow, keras_ocr, etc.).

⚠️ Obligatoire à chaque redémarrage du terminal.
🔧 Étape 3 – Définir les variables d’environnement PythonPath (à faire à chaque session)
cd tfod310\models\research
set PYTHONPATH=%cd%;%cd%\slim
🔁 Pourquoi ?
Cela permet à Python de localiser les modules internes de TensorFlow Object Detection.
Sans cette variable, les imports comme from object_detection import ... échouent.

⚠️ À faire après chaque activation de l’environnement si non automatisé.


🧠 Étape 4 – Entraîner le modèle (à faire que si nouveau dataset ou reconfiguration)
python object_detection/model_main_tf2.py ^
  --model_dir=../../../workspace/models/passport_model ^
  --pipeline_config_path=../../../workspace/models/passport_model/pipeline.config ^
  --alsologtostderr
❓ Pourquoi pas besoin de relancer à chaque fois ?
Le modèle est entraîné une fois puis sauvegardé dans passport_model sous forme de checkpoints.
Tant que :

tu ne modifies pas le dataset

tu ne changes pas les labels

tu ne touches pas aux hyperparamètres (pipeline.config)

👉 Aucun besoin de relancer un entraînement.

Le modèle peut ensuite être exporté pour prédiction (étape suivante).

📤 Étape 5 – Exporter le modèle (à faire après chaque nouvel entraînement)



python object_detection/exporter_main_v2.py ^
  --input_type image_tensor ^
  --pipeline_config_path ../../../workspace/models/passport_model/pipeline.config ^
  --trained_checkpoint_dir ../../../workspace/models/passport_model ^
  --output_directory ../../../workspace/exported-models/passport_exported
📦 Pourquoi ?
Cette étape génère le modèle au format SavedModel, compatible avec TensorFlow Serve ou une interface de prédiction.
Le dossier passport_exported contient :

le graphe prêt à l’inférence

les poids

les signatures pour prédiction

À faire uniquement si tu as lancé un nouvel entraînement.

🖥️ Étape 6 – Lancer l’interface utilisateur (à faire à chaque fois que tu veux détecter un passeport)


ce placer dans le wokspace et taper : 
python predict_passport.py
Interface avec 4 boutons :
Bouton	Fonction
📂 Charger Image	Sélectionne une image unique à traiter
📂 Charger Plusieurs Images	Charge un lot d’images (traitement par ch)
▶️ Traiter Image(s)	Applique détection des zones, OCR, nettoyage et correction
💾 Sauvegarder Résultats	Exporte les résultats JSON (image sélectionnée ou toutes)

🧠 Fonctionnement interne du script predict_passport.py
Détection par modèle TensorFlow
Le modèle repère les zones contenant les champs clés (ex: name, dob, sex, etc.)

OCR avec keras_ocr
Chaque zone est découpée et le texte est lu automatiquement.

Nettoyage et corrections :

Mise en majuscules

Suppression des symboles parasites

Correction des mois, pays, prénoms via dictionnaires (MONTHS, COUNTRIES, etc.)

Fuzzy matching (SequenceMatcher) pour corriger les fautes OCR

Affichage des résultats :

Sur l’image (boîtes colorées avec texte)

Dans un tableau latéral pour relecture

Score de confiance pour chaque champ (de 0 à 1)

Sauvegarde JSON structurée :
Format par image avec nom de fichier, champs détectés, score, texte brut, texte corrigé.

📊 Score de détection
Chaque champ a un score ∈ [0, 1], correspondant à la confiance du modèle de détection.
Exemple :

json


{
  "field": "birth_date",
  "value": "14 JUN 2001",
  "score": 0.92
}
Tu peux ignorer ou signaler les champs avec un score trop faible (< 0.5) si besoin.

✅ Résumé : que dois-tu faire et quand ?
Action	À faire quand ?
Test d’installation	Une seule fois après setup initial
Activation env virtuel	À chaque session
Configuration PYTHONPATH	À chaque session
Entraînement du modèle	Seulement si dataset changé
Export du modèle	Après un nouvel entraînement
Lancer l’interface	À chaque utilisation
