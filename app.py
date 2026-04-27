"""
Interface graphique pour la reconnaissance de pieces euro.
Design moderne sombre avec tkinter.
"""
import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.segmentation import segment_piece
from core.features import extract_features
from core.classification import classify_piece, get_coin_value
from core.utils import draw_label

# ── Palette de couleurs ──────────────────────────────────────────
BG_DARK    = "#0f0f13"
BG_CARD    = "#1a1a24"
BG_HOVER   = "#22222f"
ACCENT     = "#f5c518"
ACCENT2    = "#e8a010"
TEXT_MAIN  = "#f0f0f0"
TEXT_SUB   = "#888899"
SUCCESS    = "#4caf82"
WARNING    = "#f5a623"
DANGER     = "#e05c5c"
BORDER     = "#2a2a3a"


class EuroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Euro Coin Recognizer")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self._result_image = None
        self._photo        = None

        self._build_ui()

    # ── Construction UI ─────────────────────────────────────────
    def _build_ui(self):
        # Titre
        header = tk.Frame(self, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(24, 0))

        tk.Label(header, text="🪙", font=("Segoe UI Emoji", 28),
                 bg=BG_DARK, fg=ACCENT).pack(side="left")
        tk.Label(header, text=" Euro Coin Recognizer",
                 font=("Segoe UI", 22, "bold"),
                 bg=BG_DARK, fg=TEXT_MAIN).pack(side="left")
        tk.Label(header, text="  —  Traitement d'image classique",
                 font=("Segoe UI", 11),
                 bg=BG_DARK, fg=TEXT_SUB).pack(side="left", pady=6)

        # Séparateur
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=30, pady=16)

        # Contenu principal
        content = tk.Frame(self, bg=BG_DARK)
        content.pack(fill="both", expand=True, padx=30, pady=0)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # Panneau gauche : image
        self._build_image_panel(content)

        # Panneau droit : résultats
        self._build_results_panel(content)

        # Barre du bas
        self._build_bottom_bar()

    def _build_image_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0,
                         highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=0)

        # Zone d'image cliquable
        self.canvas = tk.Canvas(frame, bg=BG_CARD, bd=0,
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Button-1>", lambda e: self._load_image())

        # Placeholder
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 600
        h = self.canvas.winfo_height() or 500
        cx, cy = w // 2, h // 2

        # Cercle décoratif
        r = min(w, h) // 5
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                outline=BORDER, width=2)
        self.canvas.create_oval(cx-r+10, cy-r+10, cx+r-10, cy+r-10,
                                outline=ACCENT, width=1, dash=(4, 4))

        self.canvas.create_text(cx, cy - 15, text="🪙",
                                font=("Segoe UI Emoji", 32),
                                fill=ACCENT)
        self.canvas.create_text(cx, cy + 35,
                                text="Cliquez pour charger une image",
                                font=("Segoe UI", 12),
                                fill=TEXT_SUB)
        self.canvas.create_text(cx, cy + 58,
                                text="JPG · PNG · BMP",
                                font=("Segoe UI", 9),
                                fill=BORDER)

    def _build_results_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0,
                         highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=1, sticky="nsew", pady=0)

        # Titre panneau
        tk.Label(frame, text="Résultats",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(0, 12))

        # Total
        total_frame = tk.Frame(frame, bg=BG_DARK, bd=0)
        total_frame.pack(fill="x", padx=20, pady=(0, 16))

        tk.Label(total_frame, text="TOTAL",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_DARK, fg=TEXT_SUB).pack(anchor="w", padx=16, pady=(12, 0))
        self.lbl_total = tk.Label(total_frame, text="— EUR",
                                  font=("Segoe UI", 32, "bold"),
                                  bg=BG_DARK, fg=ACCENT)
        self.lbl_total.pack(anchor="w", padx=16, pady=(0, 12))

        # Stats rapides
        stats_frame = tk.Frame(frame, bg=BG_CARD)
        stats_frame.pack(fill="x", padx=20, pady=(0, 12))

        self.lbl_count = self._stat_widget(stats_frame, "Pièces", "—", 0)
        self.lbl_time  = self._stat_widget(stats_frame, "Temps", "—", 1)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(0, 12))

        # Liste des pièces
        tk.Label(frame, text="Détail",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_CARD, fg=TEXT_MAIN).pack(anchor="w", padx=20, pady=(0, 8))

        list_container = tk.Frame(frame, bg=BG_CARD)
        list_container.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        scrollbar = tk.Scrollbar(list_container, bg=BG_DARK,
                                 troughcolor=BG_DARK,
                                 activebackground=ACCENT)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(list_container,
                                  bg=BG_CARD, fg=TEXT_MAIN,
                                  font=("Segoe UI", 10),
                                  selectbackground=ACCENT,
                                  selectforeground=BG_DARK,
                                  bd=0, highlightthickness=0,
                                  activestyle="none",
                                  yscrollcommand=scrollbar.set)
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

    def _stat_widget(self, parent, label, value, col):
        f = tk.Frame(parent, bg=BG_DARK)
        f.grid(row=0, column=col, sticky="ew",
               padx=(0, 8) if col == 0 else (8, 0), pady=4)
        tk.Label(f, text=label, font=("Segoe UI", 8, "bold"),
                 bg=BG_DARK, fg=TEXT_SUB).pack(anchor="w", padx=12, pady=(8, 0))
        lbl = tk.Label(f, text=value, font=("Segoe UI", 16, "bold"),
                       bg=BG_DARK, fg=TEXT_MAIN)
        lbl.pack(anchor="w", padx=12, pady=(0, 8))
        return lbl

    def _build_bottom_bar(self):
        bar = tk.Frame(self, bg=BG_DARK)
        bar.pack(fill="x", padx=30, pady=(12, 20))

        # Bouton charger
        self.btn_load = tk.Button(
            bar, text="📂  Charger une image",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg=BG_DARK,
            activebackground=ACCENT2, activeforeground=BG_DARK,
            bd=0, padx=24, pady=10,
            cursor="hand2", relief="flat",
            command=self._load_image
        )
        self.btn_load.pack(side="left")

        # Bouton reset
        self.btn_reset = tk.Button(
            bar, text="↺  Réinitialiser",
            font=("Segoe UI", 10),
            bg=BG_CARD, fg=TEXT_SUB,
            activebackground=BG_HOVER, activeforeground=TEXT_MAIN,
            bd=0, padx=18, pady=10,
            cursor="hand2", relief="flat",
            command=self._reset
        )
        self.btn_reset.pack(side="left", padx=(12, 0))

        # Status
        self.lbl_status = tk.Label(bar, text="Prêt",
                                   font=("Segoe UI", 9),
                                   bg=BG_DARK, fg=TEXT_SUB)
        self.lbl_status.pack(side="right")

    # ── Logique ─────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("Tous", "*.*")]
        )
        if not path:
            return
        self._set_status("Analyse en cours...", WARNING)
        self.btn_load.config(state="disabled")
        threading.Thread(target=self._process, args=(path,), daemon=True).start()

    def _process(self, path):
        import time
        t0 = time.time()

        img = cv2.imread(path)
        if img is None:
            self.after(0, lambda: self._set_status("Impossible de lire l'image", DANGER))
            self.after(0, lambda: self.btn_load.config(state="normal"))
            return

        h, w = img.shape[:2]
        if max(h, w) > 1200:
            img = cv2.resize(img, None, fx=1200/max(h, w), fy=1200/max(h, w))

        circles       = segment_piece(img)
        features_list, image_out = extract_features(circles, img)

        total   = 0.0
        details = []

        for feat in features_list:
            label, d, conf = classify_piece(feat, features_list)
            value = get_coin_value(label)
            total += value
            details.append((label, conf, value))

            cx, cy = feat["center"]
            r      = feat["radius"]
            x, y_  = feat["box"][:2]

            if conf > 0.7:
                color = (76, 175, 130)
            elif conf > 0.4:
                color = (90, 165, 245)
            else:
                color = (80, 92, 224)

            overlay = image_out.copy()
            cv2.circle(overlay, (cx, cy), r, color, -1)
            cv2.addWeighted(overlay, 0.2, image_out, 0.8, 0, image_out)
            cv2.circle(image_out, (cx, cy), r, color, 2)
            draw_label(image_out, f"{label}  {conf*100:.0f}%",
                       (x, max(y_ - 12, 10)), color)

        elapsed = time.time() - t0

        # Convertir pour tkinter
        img_rgb = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        self._result_image = Image.fromarray(img_rgb)

        self.after(0, lambda: self._update_ui(total, details, elapsed, len(circles)))

    def _update_ui(self, total, details, elapsed, n_pieces):
        # Image
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        img = self._result_image.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        x = cw // 2
        y = ch // 2
        self.canvas.create_image(x, y, anchor="center", image=self._photo)

        # Total
        self.lbl_total.config(text=f"{total:.2f} EUR")

        # Stats
        self.lbl_count.config(text=str(n_pieces))
        self.lbl_time.config(text=f"{elapsed:.1f}s")

        # Liste
        self.listbox.delete(0, "end")
        coin_counts = {}
        for label, conf, value in details:
            coin_counts[label] = coin_counts.get(label, {"count": 0, "value": value})
            coin_counts[label]["count"] += 1

        for label, info in sorted(coin_counts.items(),
                                  key=lambda x: -x[1]["value"]):
            c = info["count"]
            v = info["value"]
            line = f"  {'×'+str(c):<4}  {label:<12}  {v:.2f} EUR"
            self.listbox.insert("end", line)

        self.lbl_status.config(text=f"Analyse terminee  ·  {n_pieces} piece(s)", fg=SUCCESS)
        self.btn_load.config(state="normal")

    def _reset(self):
        self.canvas.delete("all")
        self._draw_placeholder()
        self.lbl_total.config(text="— EUR")
        self.lbl_count.config(text="—")
        self.lbl_time.config(text="—")
        self.listbox.delete(0, "end")
        self._set_status("Pret", TEXT_SUB)

    def _set_status(self, msg, color=TEXT_SUB):
        self.lbl_status.config(text=msg, fg=color)


if __name__ == "__main__":
    app = EuroApp()
    app.mainloop()