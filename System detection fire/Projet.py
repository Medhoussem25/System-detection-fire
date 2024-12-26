from ultralytics import YOLO
import winsound
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image  # Importation de PIL pour l'icône
import os

# Configuration du son d'alerte
FREQUENCE = 500
DUREE = 1000

# Récupérer le chemin du dossier du projet
chemin_dossier_projet = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le modèle 'fire.pt' dans le même dossier que le script
chemin_modele = os.path.join(chemin_dossier_projet, 'fire.pt')
print("Chemin du modèle :", chemin_modele)

# Chargement automatique du modèle YOLO
try:
    MODELE = YOLO(chemin_modele)  # Chargement du modèle avec le chemin automatique
except Exception as e:
    MODELE = None
    print(f"Erreur lors du chargement du modèle : {e}")

CHEMIN_VIDEO = None

# Fonction pour sélectionner une vidéo
def charger_video():
    global CHEMIN_VIDEO
    chemin = filedialog.askopenfilename(
        title="Sélectionner une vidéo",
        filetypes=[("Fichiers vidéo", "*.mp4 *.avi *.mkv")],
    )
    if chemin:
        CHEMIN_VIDEO = chemin
        messagebox.showinfo("Succès", "Vidéo chargée avec succès !")

# Fonction principale pour exécuter la détection
def executer_detection():
    if not MODELE:
        messagebox.showerror("Erreur", "Le modèle n'a pas pu être chargé. Vérifiez le fichier 'fire.pt'.")
        return
    if not CHEMIN_VIDEO:
        messagebox.showwarning("Attention", "Veuillez charger une vidéo avant de lancer la détection.")
        return

    cap = cv2.VideoCapture(CHEMIN_VIDEO)
    while cap.isOpened():
        succes, frame = cap.read()

        if succes:
            resultats = MODELE.predict(frame, conf=0.6, stream_buffer=False)
            noms = MODELE.names

            for c in resultats[0].boxes.cls:
                if noms[int(c)] == 'fire':  # Modifier selon les noms des classes du modèle
                    print("Alerte : Feu détecté !")
                    winsound.Beep(FREQUENCE, DUREE)
                    break

            # Afficher la vidéo annotée
            frame_annoté = resultats[0].plot()
            frame_annoté = cv2.resize(frame_annoté, (640, 640))
            cv2.imshow("Détection YOLOv8", frame_annoté)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Fonction pour afficher les informations du développeur
def afficher_info_dev():
    messagebox.showinfo("Information Développeur", "Nom: Houssam Bouagal\nEmail: mouhamedhoussem813@gmail.com")

# Création de l'interface graphique avec ttk pour un meilleur style
fenetre = tk.Tk()
fenetre.title("Détection fire")
fenetre.geometry("500x350")  # Agrandissement pour ajouter un bouton supplémentaire
fenetre.configure(bg="#f4f4f4")

# Ajouter une icône à la fenêtre
icon_path = os.path.join(chemin_dossier_projet, "fire-alarm.png")
try:
    icon_image = ImageTk.PhotoImage(file=icon_path)
    fenetre.iconphoto(False, icon_image)
except Exception as e:
    print(f"Erreur lors du chargement de l'icône : {e}")

# Style ttk pour améliorer l'interface
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 14), background="#f4f4f4", foreground="#333")

# Étiquette de titre
titre = ttk.Label(fenetre, text="Systeme Détection fire")
titre.pack(pady=20)

# Boutons pour charger la vidéo et lancer la détection
btn_charger_video = ttk.Button(fenetre, text="Charger une vidéo", command=charger_video)
btn_charger_video.pack(pady=10)

btn_lancer_detection = ttk.Button(fenetre, text="Lancer la détection", command=executer_detection)
btn_lancer_detection.pack(pady=10)

# Nouveau bouton pour afficher les informations du développeur
btn_info_dev = ttk.Button(fenetre, text="Info Développeur", command=afficher_info_dev)
btn_info_dev.pack(pady=10)

# Étiquette de footer
footer = ttk.Label(fenetre, text="Appuyez sur 'q' pour quitter la détection.")
footer.pack(pady=20)

# Lancer l'application
fenetre.mainloop()
