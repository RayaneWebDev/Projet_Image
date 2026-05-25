"""
Interface graphique pour la reconnaissance de pièces euro.

Permet de charger une image, de choisir entre le pipeline k-NN classique
et le pipeline K-moyennes, d'afficher les pièces détectées avec leur
valeur et d'exporter les résultats en CSV ou JSON.

Stack technique : tkinter + customtkinter, thème clair pastel.

Utilisation :
    python app.py

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import csv
import json
import cv2
import numpy as np
from PIL import Image, ImageTk

import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.segmentation import segment_piece
from core.features import extract_features
from core.classification import classify_all
from core.classification_kmeans import classify_all_kmeans
from core.utils import COIN_VALUES_EUR

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Palette pastel clair ───────────────────────────────────────────────────────
BG_MAIN     = "#FAFAFA"
BG_PANEL    = "#FFFFFF"
BG_CARD     = "#F1F5F9"
BG_CARD2    = "#EEF2FF"
ACCENT      = "#93C5FD"
ACCENT_D    = "#3B82F6"
SECONDARY   = "#C4B5FD"
SECONDARY_D = "#7C3AED"
SUCCESS     = "#86EFAC"
SUCCESS_D   = "#16A34A"
WARNING     = "#FCD34D"
DANGER      = "#FCA5A5"
TEXT_MAIN   = "#1E293B"
TEXT_SUB    = "#64748B"
TEXT_MUTED  = "#94A3B8"
BORDER      = "#E2E8F0"

# Couleur des cercles de détection tracés sur l'image résultat (vert vif, BGR)
CIRCLE_GREEN = (50, 205, 50)

# Couleur de fond des badges de couleur dans la liste de résultats
BADGE_HEX = {
    "bronze":  "#FCD9A0",
    "gold":    "#FDE68A",
    "silver":  "#E2E8F0",
    "unknown": "#E2E8F0",
}

# Correspondance label de pièce → groupe de couleur pour l'affichage
LABEL_TO_COLOR = {
    "1 cent":  "bronze", "2 cent": "bronze", "5 cent":  "bronze",
    "10 cent": "gold",   "20 cent": "gold",  "50 cent": "gold",
    "1 Euro":  "silver", "2 Euro":  "silver",
}


def _hex_to_rgb(h):
    """
    Convertit une couleur hexadécimale en triplet RGB.

    Args:
        h (str): couleur au format '#RRGGBB'.

    Returns:
        tuple[int, int, int]: valeurs (R, G, B) entre 0 et 255.
    """
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class CoinRow(ctk.CTkFrame):
    """
    Ligne de la liste de résultats représentant une pièce détectée.

    Affiche un badge de couleur, le label, la confiance et la valeur
    en euros dans une grille horizontale.
    """

    def __init__(self, master, label, conf, value, color_group, **kw):
        """
        Args:
            master      : widget parent (CTkScrollableFrame).
            label (str) : label de la pièce (ex. '50 cent').
            conf (float): confiance de classification entre 0 et 1.
            value (float): valeur faciale en euros.
            color_group (str): 'bronze', 'gold', 'silver' ou 'unknown'.
        """
        super().__init__(master, fg_color=BG_PANEL, corner_radius=8, **kw)

        badge_color = BADGE_HEX.get(color_group, "#E2E8F0")
        r, g, b     = _hex_to_rgb(badge_color)
        # Choix de la couleur de texte du badge selon la luminance du fond
        lum      = 0.299 * r + 0.587 * g + 0.114 * b
        badge_fg = "#1E293B" if lum > 180 else "#FFFFFF"

        ctk.CTkLabel(
            self, text="", width=14, height=14,
            fg_color=badge_color, corner_radius=3,
        ).grid(row=0, column=0, padx=(10, 6), pady=10)

        ctk.CTkLabel(
            self, text=label,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=TEXT_MAIN,
        ).grid(row=0, column=1, sticky="w")

        conf_txt   = f"{conf * 100:.0f}%"
        conf_color = SUCCESS_D if conf > 0.7 else (WARNING if conf > 0.45 else "#DC2626")
        ctk.CTkLabel(
            self, text=conf_txt,
            font=ctk.CTkFont("Segoe UI", 11),
            text_color=conf_color,
        ).grid(row=0, column=2, padx=(8, 4), sticky="e")

        ctk.CTkLabel(
            self, text=f"{value:.2f} €",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=ACCENT_D,
        ).grid(row=0, column=3, padx=(4, 12), sticky="e")

        self.columnconfigure(1, weight=1)


class EuroApp(ctk.CTk):
    """
    Application principale de reconnaissance de pièces euro.

    Gère le chargement d'image, le lancement du pipeline (k-NN ou K-moyennes)
    dans un thread séparé, l'affichage des résultats et l'export CSV/JSON.
    """

    def __init__(self):
        super().__init__()
        self.title("Euro Coin Recognizer")
        self.geometry("1300x750")
        self.minsize(1200, 700)
        self.configure(fg_color=BG_MAIN)

        self._result_image    = None   # PIL.Image de l'image annotée
        self._photo           = None   # ImageTk pour éviter le garbage collect
        self._last_results    = []     # [(label, conf, value, color_group), ...]
        self._last_image_path = None
        self._processing      = False  # verrou : empêche un double lancement

        self._build_ui()
        self.bind("<Configure>", self._on_resize)

    # ── Construction de l'interface ────────────────────────────────────────────

    def _build_ui(self):
        """Construit la structure principale : en-tête, trois panneaux, barre de statut."""
        self._build_header()

        body = ctk.CTkFrame(self, fg_color=BG_MAIN)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        body.columnconfigure(0, weight=0, minsize=210)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=290)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_center_panel(body)
        self._build_right_panel(body)
        self._build_status_bar()

    def _build_header(self):
        """Crée la barre de titre fixe en haut de la fenêtre."""
        hdr = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=58)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        inner = ctk.CTkFrame(hdr, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=24, pady=0)

        ctk.CTkLabel(
            inner,
            text="Euro Coin Recognizer",
            font=ctk.CTkFont("Segoe UI", 20, "bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left", pady=14)

        ctk.CTkLabel(
            inner,
            text="  —  Reconnaissance de pièces",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=TEXT_SUB,
        ).pack(side="left", pady=14)

        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(fill="x")

    def _build_left_panel(self, parent):
        """
        Crée le panneau gauche contenant les contrôles.

        Comprend le sélecteur d'algorithme, les boutons charger/exporter/reset
        et la légende des couleurs de pièces.
        """
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=12,
                             border_color=BORDER, border_width=1)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=8)

        ctk.CTkLabel(
            panel, text="Contrôles",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=18, pady=(18, 4))
        ctk.CTkFrame(panel, fg_color=BORDER, height=1).pack(fill="x", padx=18, pady=(0, 14))

        ctk.CTkLabel(
            panel, text="Algorithme",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=TEXT_SUB,
        ).pack(anchor="w", padx=18)

        self._algo_var = tk.StringVar(value="k-NN")
        seg = ctk.CTkSegmentedButton(
            panel,
            values=["k-NN", "K-Means"],
            variable=self._algo_var,
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            selected_color=ACCENT_D,
            selected_hover_color=SECONDARY_D,
            fg_color=BG_CARD,
            unselected_color=BG_CARD,
            unselected_hover_color=ACCENT,
            text_color=TEXT_MAIN,
        )
        seg.pack(fill="x", padx=18, pady=(6, 14))

        self._algo_desc = ctk.CTkLabel(
            panel,
            text="k-NN classique\n(scale factor + votes)",
            font=ctk.CTkFont("Segoe UI", 10),
            text_color=TEXT_MUTED,
            justify="left",
            wraplength=170,
        )
        self._algo_desc.pack(anchor="w", padx=18, pady=(0, 18))
        self._algo_var.trace_add("write", self._on_algo_change)

        ctk.CTkFrame(panel, fg_color=BORDER, height=1).pack(fill="x", padx=18, pady=(0, 14))

        self.btn_load = ctk.CTkButton(
            panel,
            text="Charger une image",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            fg_color=ACCENT_D,
            hover_color=SECONDARY_D,
            corner_radius=8,
            height=40,
            command=self._load_image,
        )
        self.btn_load.pack(fill="x", padx=18, pady=(0, 10))

        self.btn_export = ctk.CTkButton(
            panel,
            text="Exporter résultats",
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color=BG_CARD,
            hover_color=SUCCESS,
            text_color=TEXT_MAIN,
            border_color=BORDER,
            border_width=1,
            corner_radius=8,
            height=36,
            state="disabled",
            command=self._export,
        )
        self.btn_export.pack(fill="x", padx=18, pady=(0, 10))

        self.btn_reset = ctk.CTkButton(
            panel,
            text="Réinitialiser",
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color="transparent",
            hover_color=DANGER,
            text_color=TEXT_SUB,
            corner_radius=8,
            height=32,
            command=self._reset,
        )
        self.btn_reset.pack(fill="x", padx=18, pady=(0, 18))

        ctk.CTkFrame(panel, fg_color=BORDER, height=1).pack(fill="x", padx=18, pady=(0, 14))

        ctk.CTkLabel(
            panel, text="Légende",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=TEXT_SUB,
        ).pack(anchor="w", padx=18)

        for group, color in [("Bronze", "#FCD9A0"), ("Gold", "#FDE68A"), ("Silver", "#E2E8F0")]:
            row = ctk.CTkFrame(panel, fg_color="transparent")
            row.pack(anchor="w", padx=18, pady=2)
            ctk.CTkLabel(row, text="", width=12, height=12,
                         fg_color=color, corner_radius=2).pack(side="left")
            ctk.CTkLabel(row, text=f"  {group}",
                         font=ctk.CTkFont("Segoe UI", 10),
                         text_color=TEXT_SUB).pack(side="left")

    def _build_center_panel(self, parent):
        """
        Crée le panneau central contenant le canvas d'affichage de l'image
        et la barre de progression pendant l'analyse.
        """
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=12,
                             border_color=BORDER, border_width=1)
        panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=8)

        self.canvas = tk.Canvas(panel, bg=BG_PANEL, bd=0,
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Button-1>", lambda e: self._load_image()
                         if not self._processing else None)

        self.progress = ctk.CTkProgressBar(
            panel, fg_color=BG_CARD, progress_color=ACCENT_D,
            height=4, corner_radius=0,
        )
        self.progress.pack(fill="x", side="bottom")
        self.progress.set(0)
        self.progress.configure(mode="indeterminate")

        self._draw_placeholder()

    def _draw_placeholder(self):
        """Dessine l'état vide du canvas (invitation à charger une image)."""
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        w  = self.canvas.winfo_width() or 700
        h  = self.canvas.winfo_height() or 500
        cx, cy = w // 2, h // 2
        r  = min(w, h) // 5

        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                outline=BORDER, width=2)
        self.canvas.create_oval(cx - r + 14, cy - r + 14, cx + r - 14, cy + r - 14,
                                outline=ACCENT, width=1, dash=(6, 5))
        self.canvas.create_text(cx, cy - 16,
                                text="Cliquez pour charger une image",
                                font=("Segoe UI", 14), fill=TEXT_SUB)
        self.canvas.create_text(cx, cy + 14,
                                text="JPG · PNG · BMP",
                                font=("Segoe UI", 10), fill=TEXT_MUTED)

    def _build_right_panel(self, parent):
        """
        Crée le panneau droit contenant le total, les statistiques rapides
        et la liste défilante des pièces détectées.
        """
        panel = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=12,
                             border_color=BORDER, border_width=1)
        panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0), pady=8)

        ctk.CTkLabel(
            panel, text="Résultats",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=20, pady=(18, 4))
        ctk.CTkFrame(panel, fg_color=BORDER, height=1).pack(fill="x", padx=20, pady=(0, 14))

        total_card = ctk.CTkFrame(panel, fg_color=BG_CARD2, corner_radius=10)
        total_card.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            total_card, text="TOTAL",
            font=ctk.CTkFont("Segoe UI", 9, "bold"),
            text_color=TEXT_SUB,
        ).pack(anchor="w", padx=16, pady=(12, 0))

        self.lbl_total = ctk.CTkLabel(
            total_card, text="—",
            font=ctk.CTkFont("Segoe UI", 36, "bold"),
            text_color=ACCENT_D,
        )
        self.lbl_total.pack(anchor="w", padx=16, pady=(0, 4))

        self.lbl_total_sub = ctk.CTkLabel(
            total_card, text="",
            font=ctk.CTkFont("Segoe UI", 10),
            text_color=TEXT_MUTED,
        )
        self.lbl_total_sub.pack(anchor="w", padx=16, pady=(0, 12))

        stats = ctk.CTkFrame(panel, fg_color="transparent")
        stats.pack(fill="x", padx=16, pady=(0, 10))
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)

        self.lbl_count = self._stat_card(stats, "Pièces", "—", 0, ACCENT)
        self.lbl_time  = self._stat_card(stats, "Temps",  "—", 1, SECONDARY)

        ctk.CTkFrame(panel, fg_color=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 10))

        ctk.CTkLabel(
            panel, text="Détail",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=TEXT_SUB,
        ).pack(anchor="w", padx=20, pady=(0, 8))

        self.coin_list_frame = ctk.CTkScrollableFrame(
            panel, fg_color=BG_MAIN, corner_radius=8,
            scrollbar_button_color=ACCENT,
            scrollbar_button_hover_color=ACCENT_D,
        )
        self.coin_list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        ctk.CTkLabel(
            self.coin_list_frame,
            text="Aucun résultat",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color=TEXT_MUTED,
        ).pack(pady=30)

    def _stat_card(self, parent, label, value, col, color):
        """
        Crée une mini-carte de statistique (Pièces ou Temps).

        Args:
            parent : frame conteneur (grille 2 colonnes).
            label  (str)  : titre de la carte.
            value  (str)  : valeur initiale affichée.
            col    (int)  : colonne dans la grille (0 ou 1).
            color  (str)  : couleur hex de la valeur.

        Returns:
            CTkLabel : référence au label de valeur pour mise à jour ultérieure.
        """
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        card.grid(row=0, column=col,
                  sticky="ew",
                  padx=(0, 4) if col == 0 else (4, 0))
        ctk.CTkLabel(card, text=label,
                     font=ctk.CTkFont("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED).pack(anchor="w", padx=12, pady=(8, 0))
        lbl = ctk.CTkLabel(card, text=value,
                           font=ctk.CTkFont("Segoe UI", 18, "bold"),
                           text_color=color)
        lbl.pack(anchor="w", padx=12, pady=(0, 8))
        return lbl

    def _build_status_bar(self):
        """Crée la barre de statut en bas de la fenêtre."""
        bar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=34)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(
            fill="x", side="bottom")

        self.lbl_status = ctk.CTkLabel(
            bar, text="Prêt",
            font=ctk.CTkFont("Segoe UI", 10),
            text_color=TEXT_MUTED,
        )
        self.lbl_status.pack(side="left", padx=20, pady=8)

        self.lbl_algo_badge = ctk.CTkLabel(
            bar, text="k-NN",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
            text_color=ACCENT_D,
        )
        self.lbl_algo_badge.pack(side="right", padx=20, pady=8)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_algo_change(self, *_):
        """Met à jour la description et le badge lorsque l'algorithme change."""
        algo = self._algo_var.get()
        if algo == "K-Means":
            self._algo_desc.configure(text="K-moyennes\n(centroïdes par classe)")
            self.lbl_algo_badge.configure(text="K-Means", text_color=SECONDARY_D)
        else:
            self._algo_desc.configure(text="k-NN classique\n(scale factor + votes)")
            self.lbl_algo_badge.configure(text="k-NN", text_color=ACCENT_D)

    def _on_resize(self, event):
        """Rafraîchit l'image affichée lorsque la fenêtre est redimensionnée."""
        if self._result_image is not None:
            self._show_image(self._result_image)

    def _load_image(self):
        """Ouvre un sélecteur de fichier et lance l'analyse dans un thread."""
        if self._processing:
            return
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("Tous", "*.*")],
        )
        if not path:
            return
        self._last_image_path = path
        self._processing = True
        self.btn_load.configure(state="disabled")
        self.btn_export.configure(state="disabled")
        self.lbl_status.configure(text="Analyse en cours…", text_color=ACCENT_D)
        self.progress.start()
        threading.Thread(target=self._process, args=(path,), daemon=True).start()

    def _process(self, path):
        """
        Pipeline complet exécuté dans un thread secondaire.

        Charge l'image, détecte et classifie les pièces, annote l'image
        résultat, puis programme la mise à jour de l'interface via after().

        Args:
            path (str): chemin absolu de l'image à analyser.
        """
        t0 = time.time()
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Impossible de lire l'image")

            h, w = img.shape[:2]
            if max(h, w) > 1200:
                img = cv2.resize(img, None,
                                 fx=1200 / max(h, w), fy=1200 / max(h, w))

            circles       = segment_piece(img)
            features_list, image_out = extract_features(circles, img)

            algo    = self._algo_var.get()
            results = classify_all_kmeans(features_list) if algo == "K-Means" else classify_all(features_list)

            total   = 0.0
            details = []

            for i, (label, d_mm, conf) in enumerate(results):
                value       = COIN_VALUES_EUR.get(label, 0.0)
                total      += value
                color_group = LABEL_TO_COLOR.get(label, "unknown")

                feat   = features_list[i]
                cx, cy = feat["center"]
                r      = feat["radius"]
                x, y_  = feat["box"][:2]

                cv2.circle(image_out, (cx, cy), r, CIRCLE_GREEN, 2)

                conf_pct = f"{conf * 100:.0f}%"
                # Contour noir pour lisibilité, puis texte vert
                cv2.putText(image_out, f"{label}  {conf_pct}",
                            (x, max(y_ - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image_out, f"{label}  {conf_pct}",
                            (x, max(y_ - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, CIRCLE_GREEN, 2, cv2.LINE_AA)

                details.append((label, conf, value, color_group))

            elapsed = time.time() - t0
            img_rgb = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            self._last_results = details
            self._result_image = pil_img
            self.after(0, lambda: self._update_ui(total, details, elapsed, len(circles)))

        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda msg=msg: self._on_error(msg))

    def _update_ui(self, total, details, elapsed, n_pieces):
        """
        Met à jour tous les widgets après une analyse réussie.

        Appelé depuis le thread principal via after().

        Args:
            total    (float)      : somme des valeurs des pièces détectées.
            details  (list)       : [(label, conf, value, color_group), ...].
            elapsed  (float)      : durée de l'analyse en secondes.
            n_pieces (int)        : nombre de pièces détectées.
        """
        self.progress.stop()
        self.progress.set(0)
        self._show_image(self._result_image)

        self.lbl_total.configure(text=f"{total:.2f} €")
        sub = f"{n_pieces} pièce{'s' if n_pieces > 1 else ''} détectée{'s' if n_pieces > 1 else ''}"
        self.lbl_total_sub.configure(text=sub)
        self.lbl_count.configure(text=str(n_pieces))
        self.lbl_time.configure(text=f"{elapsed:.1f}s")

        for widget in self.coin_list_frame.winfo_children():
            widget.destroy()

        if not details:
            ctk.CTkLabel(
                self.coin_list_frame, text="Aucune pièce détectée",
                font=ctk.CTkFont("Segoe UI", 11), text_color=TEXT_MUTED,
            ).pack(pady=30)
        else:
            # Tri par valeur décroissante pour mettre les pièces importantes en premier
            for label, conf, value, color_group in sorted(details, key=lambda x: -x[2]):
                CoinRow(self.coin_list_frame, label, conf, value, color_group).pack(
                    fill="x", pady=3, padx=4)

        self.lbl_status.configure(
            text=f"Analyse terminée  ·  {n_pieces} pièce(s)  ·  {elapsed:.1f}s",
            text_color=SUCCESS_D,
        )
        self.btn_load.configure(state="normal")
        self.btn_export.configure(state="normal")
        self._processing = False

    def _on_error(self, msg):
        """
        Affiche un message d'erreur dans la barre de statut.

        Args:
            msg (str): message d'erreur à afficher.
        """
        self.progress.stop()
        self.progress.set(0)
        self.lbl_status.configure(text=f"Erreur : {msg}", text_color="#DC2626")
        self.btn_load.configure(state="normal")
        self._processing = False

    def _show_image(self, pil_img):
        """
        Affiche une PIL.Image dans le canvas en la redimensionnant pour remplir
        l'espace disponible sans déformer les proportions.

        Args:
            pil_img (PIL.Image): image à afficher.
        """
        if pil_img is None:
            return
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width() or 700
        ch = self.canvas.winfo_height() or 500
        if cw < 10 or ch < 10:
            return
        img = pil_img.copy()
        img.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._photo)

    def _reset(self):
        """Remet l'interface dans son état initial (aucun résultat)."""
        self._result_image    = None
        self._last_results    = []
        self._last_image_path = None
        self._draw_placeholder()
        self.lbl_total.configure(text="—")
        self.lbl_total_sub.configure(text="")
        self.lbl_count.configure(text="—")
        self.lbl_time.configure(text="—")
        self.progress.stop()
        self.progress.set(0)
        for widget in self.coin_list_frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(
            self.coin_list_frame, text="Aucun résultat",
            font=ctk.CTkFont("Segoe UI", 11), text_color=TEXT_MUTED,
        ).pack(pady=30)
        self.lbl_status.configure(text="Prêt", text_color=TEXT_MUTED)
        self.btn_export.configure(state="disabled")

    def _export(self):
        """
        Exporte les résultats de la dernière analyse en CSV ou JSON.

        Ouvre un sélecteur de fichier et écrit les données selon l'extension choisie.
        """
        if not self._last_results:
            return
        path = filedialog.asksaveasfilename(
            title="Exporter les résultats",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not path:
            return

        algo = self._algo_var.get()
        rows = [
            {"label": label, "confidence": round(conf, 4),
             "value_eur": value, "color": color_group, "algorithm": algo}
            for label, conf, value, color_group in self._last_results
        ]
        total = sum(r["value_eur"] for r in rows)

        try:
            if path.endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"algorithm": algo, "total_eur": round(total, 2),
                               "coins": rows}, f, ensure_ascii=False, indent=2)
            else:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            self.lbl_status.configure(
                text=f"Exporté : {os.path.basename(path)}", text_color=SUCCESS_D)
        except Exception as exc:
            messagebox.showerror("Erreur export", str(exc))


if __name__ == "__main__":
    app = EuroApp()
    app.mainloop()
