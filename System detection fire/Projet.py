from ultralytics import YOLO
import winsound
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image
import os

# Configuration du son d'alerte
FREQUENCE = 500
DUREE = 1000

# Récupérer le chemin du dossier du projet
chemin_dossier_projet = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le modèle 'fire.pt'
chemin_modele = os.path.join(chemin_dossier_projet, 'fire.pt')
print("Chemin du modèle :", chemin_modele)

# Chargement du modèle YOLO
try:
    MODELE = YOLO(chemin_modele)
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

# Fonction pour exécuter la détection
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
                if noms[int(c)] == 'fire':
                    print("Alerte : Feu détecté !")
                    winsound.Beep(FREQUENCE, DUREE)
                    break
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

# Création de l'interface graphique
fenetre = tk.Tk()
fenetre.title("Fire detection system")
fenetre.geometry("500x400")
fenetre.configure(bg="#e3f2fd")  # Couleur de fond douce

# Ajouter une icône
icon_path = os.path.join(chemin_dossier_projet, "fire-alarm.png")
try:
    icon_image = ImageTk.PhotoImage(file=icon_path)
    fenetre.iconphoto(False, icon_image)
except Exception as e:
    print(f"Erreur lors du chargement de l'icône : {e}")

# Style ttk
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Arial", 12), padding=10, relief="raised")
style.configure("TLabel", font=("Arial", 14), background="#e3f2fd", foreground="#1a237e")

# Étiquette de titre
titre = ttk.Label(fenetre, text="Fire detection system", font=("Arial", 16, "bold"))
titre.pack(pady=15)

# Boutons avec des couleurs spécifiques
btn_charger_video = tk.Button(fenetre, text="📂 Upload a video", command=charger_video, bg="#0288d1", fg="white", font=("Arial", 12), padx=10, pady=5)
btn_charger_video.pack(pady=10)

btn_lancer_detection = tk.Button(fenetre, text="🔥 Start detection", command=executer_detection, bg="#d32f2f", fg="white", font=("Arial", 12), padx=10, pady=5)
btn_lancer_detection.pack(pady=10)

btn_info_dev = tk.Button(fenetre, text="ℹ Developer Info", command=afficher_info_dev, bg="#388e3c", fg="white", font=("Arial", 12), padx=10, pady=5)
btn_info_dev.pack(pady=10)

# Message de footer
footer = ttk.Label(fenetre, text="Press 'q' to exit detection.")
footer.pack(pady=20)

# Lancer l'application
fenetre.mainloop()
