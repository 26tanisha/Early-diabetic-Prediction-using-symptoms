"""

HOW TO RUN:
    1. First run:  python train_model.py   (creates diabetes_model.pkl)
    2. Then run:   python diabetes.app.py

FILES NEEDED IN SAME FOLDER:
    diabetes_model.pkl
    diabetes_features.json
    diabetes_data_upload.csv
    train_model.py

INSTALL PACKAGES:
    pip install pandas scikit-learn imbalanced-learn xgboost joblib
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import urllib.request
import json
import os
import sys

try:
    import pandas as pd
    import numpy as np
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    MISSING_LIB = str(e)

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0d1b2a",
    "panel":     "#1b2838",
    "card":      "#1f3044",
    "accent":    "#00c2a8",
    "danger":    "#ff5252",
    "warn":      "#ffb300",
    "success":   "#69f0ae",
    "text":      "#e8f4f8",
    "muted":     "#7ea8b8",
    "border":    "#2a4060",
    "btn_hover": "#00a090",
    "entry_bg":  "#253545",
    "entry_fg":  "#e8f4f8",
}

# ─────────────────────────────────────────────────────────────────
# SYMPTOM DEFINITIONS
# ─────────────────────────────────────────────────────────────────
SYMPTOM_DEFS = [
    ("Age",                         "age",                "spin",  (16, 90)),
    ("Gender",                      "gender",             "radio", ["Male", "Female"]),
    ("Polyuria\n(Excessive Urine)",  "polyuria",           "radio", ["Yes", "No"]),
    ("Polydipsia\n(Excess Thirst)",  "polydipsia",         "radio", ["Yes", "No"]),
    ("Sudden Weight Loss",          "sudden weight loss", "radio", ["Yes", "No"]),
    ("Weakness",                    "weakness",           "radio", ["Yes", "No"]),
    ("Polyphagia\n(Excess Hunger)",  "polyphagia",         "radio", ["Yes", "No"]),
    ("Genital Thrush",              "genital thrush",     "radio", ["Yes", "No"]),
    ("Visual Blurring",             "visual blurring",    "radio", ["Yes", "No"]),
    ("Itching",                     "itching",            "radio", ["Yes", "No"]),
    ("Irritability",                "irritability",       "radio", ["Yes", "No"]),
    ("Delayed Healing",             "delayed healing",    "radio", ["Yes", "No"]),
    ("Partial Paresis",             "partial paresis",    "radio", ["Yes", "No"]),
    ("Muscle Stiffness",            "muscle stiffness",   "radio", ["Yes", "No"]),
    ("Alopecia\n(Hair Loss)",        "alopecia",           "radio", ["Yes", "No"]),
]

STEPS_PER_PAGE = 3

# ─────────────────────────────────────────────────────────────────
# FOOD & LIFESTYLE ADVICE
# ─────────────────────────────────────────────────────────────────
FOOD_ADVICE = {
    "eat": [
        "1:  Non-starchy vegetables (broccoli, spinach)",
        "2:  Berries — low glycemic index",
        "3:  Fatty fish (salmon, mackerel)",
        "4:  Eggs — great protein, low carbs",
        "5:  Legumes (lentils, chickpeas, rajma)",
        "6:  Whole grains (brown rice, oats)",
        "7:  Avocado — healthy fats & fiber",
        "8:  Citrus fruits (in moderation)",
        "9:  Almonds, walnuts, chia seeds",
        "10:  Green tea / fenugreek water",
    ],
    "avoid": [
        "1:  Sugary drinks, sodas, juices",
        "2:  White bread, maida, refined flour",
        "3:  White rice in large quantities",
        "4:  Deep-fried fast food",
        "5:  Sweets, mithai, cakes, biscuits",
        "6:  Full-fat dairy in excess",
        "7:  Alcohol",
        "8:  Packaged fruit juices",
        "9:  Ultra-processed snacks",
    ],
    "lifestyle": [
        "1:  Walk 30 min daily (after meals)",
        "2:  Take medication on time",
        "3:  Monitor blood sugar regularly",
        "4:  Yoga or meditation daily",
        "5:  Drink 8-10 glasses of water",
        "6:  Get 7-8 hours of sleep",
        "7:  Doctor check-up every 3 months",
        "8:  Quit smoking",
        "️9:  Maintain healthy body weight",
        "10:  Eat small meals every 3-4 hours",
    ],
}

# ─────────────────────────────────────────────────────────────────
# OPENROUTER AI CHATBOT (Free — openrouter.ai)
# ─────────────────────────────────────────────────────────────────
GROQ_SYSTEM = (
    "You are DiabetesBot, a diabetes health assistant. "
    "Give short, direct, and natural answers. "
    "Never explain your reasoning process. "
    "Never describe what the user asked. "
    "Do not act like ChatGPT thinking step-by-step. "
    "Answer in maximum 2-3 sentences. "
    "Respond in the same language as the user."
)


def ask_groq(api_key: str, conversation: list) -> str:
    """Send conversation to OpenRouter (free) and return reply text."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    messages = [{"role": "system", "content": GROQ_SYSTEM}]
    for msg in conversation:
        messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": "openrouter/free",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.2,
    }

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type",  "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("HTTP-Referer",  "https://diabetes-detection-app")
    req.add_header("X-Title",       "Diabetes Detection System")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:300]
        if "401" in str(e.code):
            return "❌ Invalid API key. Please check your key at openrouter.ai"
        if "429" in str(e.code):
            return "⚠️ Rate limit hit. Please wait a moment and try again."
        return f"[API Error {e.code}] {err}"
    except Exception as e:
        return f"[Connection Error] {e}"


# ─────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────
class DiabetesApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Early Diabetes Detection System")
        self.geometry("980x720")
        self.minsize(860, 620)
        self.configure(bg=C["bg"])

        self.model        = None
        self.feature_cols = None
        self.model_ready  = False
        self.answers      = {}
        self.wizard_page  = 0
        self.chat_history = []
        import os
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")

        self.api_key = tk.StringVar(value=api_key)

        self._build_styles()
        self._load_model()
        self._build_ui()
        self.after(100, self._force_render)

    def _force_render(self):
        self.geometry("981x720")
        self.update_idletasks()
        self.geometry("980x720")

    # ── auto-load saved model ─────────────────────────────────────
    def _load_model(self):
        base = os.path.dirname(os.path.abspath(__file__))
        pkl  = os.path.join(base, "final_diabetes_model.pkl")
        feat = os.path.join(base, "final_diabetes_features.json")

        if not os.path.exists(pkl):
            messagebox.showerror(
                "File Missing",
                f"final_diabetes_model.pkl not found!\n\n"
                f"Please run train_model.py first:\n"
                f"   python train_model.py\n\n"
                f"Expected location:\n{base}"
            )
            return

        if not os.path.exists(feat):
            messagebox.showerror(
                "File Missing",
                f"final_diabetes_features.json not found!\n"
                f"Please run train_model.py first."
            )
            return

        try:
            import joblib
            self.model        = joblib.load(pkl)
            with open(feat, "r") as f:
                self.feature_cols = json.load(f)
            self.model_ready  = True
        except Exception as e:
            messagebox.showerror(
                "Load Error",
                f"Could not load model:\n{e}\n\n"
                f"Please re-run train_model.py to recreate the model."
            )

    # ── styles ────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",      background=C["bg"])
        s.configure("TLabel",      background=C["bg"], foreground=C["text"],
                    font=("Courier", 10))
        s.configure("TNotebook",   background=C["bg"], borderwidth=0)
        s.configure("TNotebook.Tab",
                    background=C["panel"], foreground=C["muted"],
                    padding=[14, 6], font=("Courier", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", C["accent"])],
              foreground=[("selected", C["bg"])])

    # ── root UI ───────────────────────────────────────────────────
    def _build_ui(self):
        # header bar
        hdr = tk.Frame(self, bg=C["panel"], height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🩺  Early Diabetes Detection System",
                 bg=C["panel"], fg=C["accent"],
                 font=("Courier", 15, "bold")).pack(side="left", padx=20, pady=16)
        badge = "✅  Model Loaded" if self.model_ready else "❌  Model Not Loaded"
        bc    = C["success"]      if self.model_ready else C["danger"]
        tk.Label(hdr, text=badge, bg=C["panel"], fg=bc,
                 font=("Courier", 10, "bold")).pack(side="right", padx=20)

        # tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_predict = ttk.Frame(self.nb, style="TFrame")
        self.tab_result  = ttk.Frame(self.nb, style="TFrame")
        self.tab_chat    = ttk.Frame(self.nb, style="TFrame")
        self.tab_about   = ttk.Frame(self.nb, style="TFrame")

        self.nb.add(self.tab_predict, text="    Symptoms  ")
        self.nb.add(self.tab_result,  text="    Result    ")
        self.nb.add(self.tab_chat,    text="    AI Chat   ")
        self.nb.add(self.tab_about,   text="  ️  About     ")

        self._build_predict_tab()
        self._build_result_tab()
        self._build_chat_tab()
        self._build_about_tab()

    # ══════════════════════════════════════════════════════════════
    # TAB 1 – SYMPTOM WIZARD
    # ══════════════════════════════════════════════════════════════
    def _build_predict_tab(self):
        banner = tk.Frame(self.tab_predict, bg=C["panel"], padx=16, pady=10)
        banner.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(banner, text="  Fill in the patient's symptoms step by step",
                 bg=C["panel"], fg=C["accent"],
                 font=("Courier", 11, "bold")).pack(side="left")
        tk.Label(banner, text="Answer every question honestly for accurate results",
                 bg=C["panel"], fg=C["muted"],
                 font=("Courier", 9)).pack(side="right")

        self.wizard_frame = tk.Frame(self.tab_predict, bg=C["bg"])
        self.wizard_frame.pack(fill="both", expand=True, padx=16, pady=4)
        self._render_wizard_page()

    def _render_wizard_page(self):
        for w in self.wizard_frame.winfo_children():
            w.destroy()

        total_pages = (len(SYMPTOM_DEFS) + STEPS_PER_PAGE - 1) // STEPS_PER_PAGE
        start       = self.wizard_page * STEPS_PER_PAGE
        end         = min(start + STEPS_PER_PAGE, len(SYMPTOM_DEFS))
        page_items  = SYMPTOM_DEFS[start:end]

        # progress bar
        prog_frame = tk.Frame(self.wizard_frame, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(0, 8))
        tk.Label(prog_frame, text=f"Step {self.wizard_page+1} of {total_pages}",
                 bg=C["bg"], fg=C["muted"],
                 font=("Courier", 9)).pack(side="right")
        pb = ttk.Progressbar(prog_frame, length=400, mode="determinate",
                             maximum=total_pages, value=self.wizard_page+1)
        pb.pack(side="left", fill="x", expand=True)

        self.current_vars = {}

        for (label, field, kind, opts) in page_items:
            card = tk.Frame(self.wizard_frame, bg=C["card"], bd=0, padx=20, pady=16)
            card.pack(fill="x", pady=6)
            card._radios = []

            tk.Label(card, text=label,
                     bg=C["card"], fg=C["accent"],
                     font=("Courier", 12, "bold"),
                     justify="left").pack(anchor="w")

            if kind == "radio":
                # Default: "No" for symptoms, first option for Gender
                default_val = self.answers.get(field, "No" if opts == ["Yes", "No"] else opts[0])
                var = tk.StringVar(value=default_val)
                self.current_vars[field] = var
                row = tk.Frame(card, bg=C["card"])
                row.pack(anchor="w", pady=8)
                for opt in opts:
                    is_sel = (var.get() == opt)
                    rb = tk.Button(
                        row, text=f"   {opt}   ",
                        bg=C["accent"] if is_sel else C["panel"],
                        fg=C["bg"]     if is_sel else C["text"],
                        font=("Courier", 11, "bold"),
                        bd=0, relief="flat", padx=16, pady=8, cursor="hand2",
                    )
                    rb.configure(
                        command=lambda v=var, o=opt, c=card: self._select_option(v, o, c)
                    )
                    rb.pack(side="left", padx=6)
                    card._radios.append((opt, rb))

            elif kind == "spin":
                lo, hi = opts
                var = tk.IntVar(value=self.answers.get(field, 30))
                self.current_vars[field] = var
                inner = tk.Frame(card, bg=C["card"])
                inner.pack(anchor="w", pady=8)
                tk.Button(inner, text="  −  ",
                          bg=C["panel"], fg=C["accent"],
                          font=("Courier", 16, "bold"),
                          bd=0, relief="flat", cursor="hand2",
                          command=lambda v=var, l=lo: v.set(max(l, v.get()-1))
                          ).pack(side="left")
                tk.Label(inner, textvariable=var,
                         bg=C["accent"], fg=C["bg"],
                         font=("Courier", 20, "bold"),
                         width=4, anchor="center").pack(side="left", padx=10)
                tk.Button(inner, text="  +  ",
                          bg=C["panel"], fg=C["accent"],
                          font=("Courier", 16, "bold"),
                          bd=0, relief="flat", cursor="hand2",
                          command=lambda v=var, h=hi: v.set(min(h, v.get()+1))
                          ).pack(side="left")
                tk.Label(card, text=f"Valid range: {lo} to {hi} years",
                         bg=C["card"], fg=C["muted"],
                         font=("Courier", 9)).pack(anchor="w")

        # navigation
        nav = tk.Frame(self.wizard_frame, bg=C["bg"])
        nav.pack(fill="x", pady=12)

        if self.wizard_page > 0:
            tk.Button(nav, text="  ← Back  ",
                      bg=C["panel"], fg=C["text"],
                      font=("Courier", 10), bd=0, relief="flat",
                      padx=14, pady=8, cursor="hand2",
                      command=self._prev_page).pack(side="left", padx=8)

        tk.Button(nav, text="  ↺ Reset  ",
                  bg=C["panel"], fg=C["muted"],
                  font=("Courier", 9), bd=0, relief="flat",
                  padx=10, pady=8, cursor="hand2",
                  command=self._reset_form).pack(side="left", padx=4)

        is_last  = (end >= len(SYMPTOM_DEFS))
        btn_text = "  🔬  Predict Now  " if is_last else "  Next →  "
        btn_col  = C["danger"] if is_last else C["accent"]
        tk.Button(nav, text=btn_text,
                  bg=btn_col, fg=C["bg"],
                  font=("Courier", 11, "bold"),
                  bd=0, relief="flat", padx=18, pady=10, cursor="hand2",
                  command=self._next_or_predict).pack(side="right", padx=8)

    def _select_option(self, var, opt, card):
        var.set(opt)
        for o, rb in card._radios:
            rb.configure(
                bg=C["accent"] if o == opt else C["panel"],
                fg=C["bg"]     if o == opt else C["text"],
            )

    def _save_current_page(self):
        for field, var in self.current_vars.items():
            self.answers[field] = var.get()

    def _prev_page(self):
        self._save_current_page()
        self.wizard_page -= 1
        self._render_wizard_page()

    def _next_or_predict(self):
        self._save_current_page()
        total = (len(SYMPTOM_DEFS) + STEPS_PER_PAGE - 1) // STEPS_PER_PAGE
        if self.wizard_page < total - 1:
            self.wizard_page += 1
            self._render_wizard_page()
        else:
            self._run_prediction()

    def _reset_form(self):
        self.answers     = {}
        self.wizard_page = 0
        self._render_wizard_page()

    def _run_prediction(self):
        if not self.model_ready:
            messagebox.showwarning(
                "Model Not Ready",
                "Model not loaded.\n\n"
                "Please run train_model.py first:\n"
                "   python train_model.py"
            )
            return
        try:
            label, prob = self._predict(self.answers)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return
        self._render_result(label, prob)
        self.nb.select(1)

    def _predict(self, answers: dict):
        """
        Encode exactly the same way as new_train_model.py:
          age     : integer as-is
          gender  : Male=1, Female=0
          Yes/No  : Yes=1, No=0
        Pipeline only has MinMaxScaler + SMOTEENN + SVM (no OneHotEncoder).
        """
        row = {}
        for col in self.feature_cols:
            val = answers.get(col, "No")
            if col == "age":
                row[col] = [int(val)]
            elif col == "gender":
                row[col] = [1 if str(val) == "Male" else 0]
            else:
                row[col] = [1 if str(val) == "Yes" else 0]

        df_in = pd.DataFrame(row)
        print(f"DEBUG input:\n{df_in.to_string()}")  # remove after testing
        proba = self.model.predict_proba(df_in)[0]
        prob  = float(proba[1])
        label = "Diabetic" if prob >= 0.5 else "Non-Diabetic"
        return label, prob

    # ══════════════════════════════════════════════════════════════
    # TAB 2 – RESULT
    # ══════════════════════════════════════════════════════════════
    def _build_result_tab(self):
        self.result_inner = tk.Frame(self.tab_result, bg=C["bg"])
        self.result_inner.pack(fill="both", expand=True)
        tk.Label(self.result_inner,
                 text="Complete the symptom form to see your result here.",
                 bg=C["bg"], fg=C["muted"],
                 font=("Courier", 11)).pack(pady=60)

    def _render_result(self, label: str, prob: float):
        for w in self.result_inner.winfo_children():
            w.destroy()

        is_diabetic = (label == "Diabetic")
        color = C["danger"] if is_diabetic else C["success"]
        pct   = int(prob * 100)

        # hero card
        hero = tk.Frame(self.result_inner, bg=C["card"], padx=30, pady=24)
        hero.pack(fill="x", padx=20, pady=16)

        emoji = "⚠️  " if is_diabetic else "✅  "
        tk.Label(hero, text=f"{emoji}{label}",
                 bg=C["card"], fg=color,
                 font=("Courier", 26, "bold")).pack()
        tk.Label(hero, text=f"Diabetes Probability: {pct}%",
                 bg=C["card"], fg=C["text"],
                 font=("Courier", 13)).pack(pady=4)

        # gauge bar
        canvas = tk.Canvas(hero, height=22, bg=C["border"], highlightthickness=0)
        canvas.pack(fill="x", pady=8)
        canvas.update_idletasks()
        cw = canvas.winfo_width() or 700
        canvas.create_rectangle(0, 0, cw * prob, 22, fill=color, outline="")
        canvas.create_text(cw // 2, 11, text=f"{pct}%",
                           fill="white", font=("Courier", 9, "bold"))

        # risk badge
        if pct < 30:   risk, rc = "LOW RISK",      C["success"]
        elif pct < 60: risk, rc = "MODERATE RISK",  C["warn"]
        else:          risk, rc = "HIGH RISK",       C["danger"]
        tk.Label(hero, text=f"Risk Level: {risk}",
                 bg=C["card"], fg=rc,
                 font=("Courier", 11, "bold")).pack(pady=4)

        if not is_diabetic:
            tk.Label(self.result_inner,
                     text="✅  Great news! You appear to be at low risk.\n"
                          "Keep maintaining a healthy lifestyle!",
                     bg=C["bg"], fg=C["success"],
                     font=("Courier", 12), justify="center").pack(pady=20)
            self._add_retest_btn()
            return

        # advice columns
        tk.Label(self.result_inner,
                 text="📌  Recommended Actions for Diabetic Patients",
                 bg=C["bg"], fg=C["accent"],
                 font=("Courier", 12, "bold")).pack(pady=(0, 6))

        adv = tk.Frame(self.result_inner, bg=C["bg"])
        adv.pack(fill="both", expand=True, padx=20, pady=4)
        adv.columnconfigure(0, weight=1)
        adv.columnconfigure(1, weight=1)
        adv.columnconfigure(2, weight=1)

        for col, (title, items, hc) in enumerate([
            ("  Foods to EAT",   FOOD_ADVICE["eat"],       C["success"]),
            ("  Foods to AVOID", FOOD_ADVICE["avoid"],     C["danger"]),
            ("  Lifestyle Tips", FOOD_ADVICE["lifestyle"], C["accent"]),
        ]):
            card = tk.Frame(adv, bg=C["card"], padx=12, pady=12)
            card.grid(row=0, column=col, padx=5, pady=4, sticky="nsew")
            tk.Label(card, text=title, bg=C["card"], fg=hc,
                     font=("Courier", 10, "bold")).pack(anchor="w", pady=(0, 6))
            for item in items:
                tk.Label(card, text=item, bg=C["card"], fg=C["text"],
                         font=("Courier", 9), justify="left",
                         wraplength=210, anchor="w").pack(anchor="w", pady=1)

        tk.Label(self.result_inner,
                 text="⚠️  This is an AI screening tool — NOT a medical diagnosis. "
                      "Please consult a qualified doctor.",
                 bg=C["bg"], fg=C["warn"],
                 font=("Courier", 9)).pack(pady=8)
        self._add_retest_btn()

    def _add_retest_btn(self):
        tk.Button(self.result_inner,
                  text="  ↺  Test Another Patient  ",
                  bg=C["panel"], fg=C["accent"],
                  font=("Courier", 10, "bold"),
                  bd=0, relief="flat", padx=16, pady=8, cursor="hand2",
                  command=self._new_patient).pack(pady=6)

    def _new_patient(self):
        self.answers     = {}
        self.wizard_page = 0
        self._render_wizard_page()
        self.nb.select(0)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 – AI CHATBOT (Groq)
    # ══════════════════════════════════════════════════════════════
    def _build_chat_tab(self):
        f = self.tab_chat

        # API key row
        kf = tk.Frame(f, bg=C["panel"], padx=14, pady=8)
        kf.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(kf, text="AI API Key:    ",
                 bg=C["panel"], fg=C["muted"],
                 font=("Courier", 9)).pack(side="left")
        tk.Entry(kf, textvariable=self.api_key, width=50, show="•",
                 bg=C["entry_bg"], fg=C["entry_fg"],
                 insertbackground=C["accent"],
                 font=("Courier", 10), bd=0, relief="flat"
                 ).pack(side="left", padx=8, ipady=4)
        tk.Label(kf, text="Free key → openrouter.ai/keys",
                 bg=C["panel"], fg=C["muted"],
                 font=("Courier", 8)).pack(side="right")

        # chat display
        self.chat_display = scrolledtext.ScrolledText(
            f, height=18, bg=C["panel"], fg=C["text"],
            font=("Courier", 10), bd=0, relief="flat",
            wrap="word", state="disabled",
            insertbackground=C["accent"])
        self.chat_display.pack(fill="both", expand=True, padx=16, pady=8)
        self.chat_display.tag_configure("you", foreground=C["accent"],
                                               font=("Courier", 10, "bold"))
        self.chat_display.tag_configure("bot", foreground=C["text"])
        self.chat_display.tag_configure("sys", foreground=C["muted"],
                                               font=("Courier", 9, "italic"))
        self.chat_display.tag_configure("err", foreground=C["danger"])
        self._chat_append("sys",
            "  DiabetesBot is ready!\n"
            "     Ask anything about diabetes, diet, symptoms, or your results.\n\n")

        # quick questions
        qq = tk.Frame(f, bg=C["bg"])
        qq.pack(fill="x", padx=16, pady=(0, 6))
        tk.Label(qq, text="Quick:", bg=C["bg"], fg=C["muted"],
                 font=("Courier", 9)).pack(side="left", padx=(0, 6))
        for q in ["What is diabetes?", "What should I eat?",
                  "How to control sugar?", "What is Polyuria?",
                  "Is diabetes curable?"]:
            tk.Button(qq, text=q, bg=C["panel"], fg=C["accent"],
                      font=("Courier", 9), bd=0, relief="flat",
                      padx=8, pady=4, cursor="hand2",
                      command=lambda qv=q: self._send_chat(qv)
                      ).pack(side="left", padx=3)

        # input row
        inp = tk.Frame(f, bg=C["bg"])
        inp.pack(fill="x", padx=16, pady=(0, 14))
        self.chat_entry = tk.Entry(
            inp, bg=C["entry_bg"], fg=C["entry_fg"],
            insertbackground=C["accent"],
            font=("Courier", 11), bd=0, relief="flat")
        self.chat_entry.pack(side="left", fill="x", expand=True, ipady=9, padx=(0, 8))
        self.chat_entry.bind("<Return>", lambda e: self._send_chat())
        tk.Button(inp, text=" Send ➤ ",
                  bg=C["accent"], fg=C["bg"],
                  font=("Courier", 10, "bold"),
                  bd=0, relief="flat", padx=14, pady=8, cursor="hand2",
                  command=self._send_chat).pack(side="right")

    def _chat_append(self, tag: str, text: str):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text, tag)
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _send_chat(self, preset: str = None):
        text = preset or self.chat_entry.get().strip()
        if not text:
            return
        self.chat_entry.delete(0, "end")
        self._chat_append("you", f"You:  {text}\n")

        key = self.api_key.get().strip()
        if not key:
            self._chat_append("err",
                "⚠️  Please enter your OpenRouter API key above.\n"
                "     Get a free key at: openrouter.ai/keys\n\n")
            return

        self.chat_history.append({"role": "user", "content": text})
        self._chat_append("sys", "     …thinking…\n")

        def worker():
            reply = ask_groq(key, self.chat_history)
            self.chat_history.append({"role": "assistant", "content": reply})
            self.after(0, lambda: (
                self._erase_thinking(),
                self._chat_append("bot", f"Bot:  {reply}\n\n"),
            ))
        threading.Thread(target=worker, daemon=True).start()

    def _erase_thinking(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("end-2l", "end-1c")
        self.chat_display.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════
    # TAB 4 – ABOUT
    # ══════════════════════════════════════════════════════════════
    def _build_about_tab(self):
        card = tk.Frame(self.tab_about, bg=C["card"], padx=30, pady=30)
        card.pack(fill="both", expand=True, padx=40, pady=30)

        tk.Label(card, text="  Early Diabetes Detection System",
                 bg=C["card"], fg=C["accent"],
                 font=("Courier", 14, "bold")).pack(anchor="w", pady=(0, 16))

        for key, val in [
            ("Model",        "SVM (SVC) + SMOTEENN Pipeline"),
            ("Dataset",      "UCI Early Stage Diabetes Risk Prediction"),
            ("Features",     "15 clinical symptoms + Age + Gender"),
            ("Evaluation",   "Stratified K-Fold Cross Validation"),
            ("UI Framework", "Python Tkinter"),
            ("AI Chatbot",   "OpenRouter AI (Mistral-7B)"),
        ]:
            row = tk.Frame(card, bg=C["card"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{key}:", bg=C["card"], fg=C["muted"],
                     font=("Courier", 10), width=16, anchor="w").pack(side="left")
            tk.Label(row, text=val, bg=C["card"], fg=C["text"],
                     font=("Courier", 10), anchor="w").pack(side="left")

        tk.Label(card,
                 text="\n⚠️  DISCLAIMER\n"
                      "This tool is for educational purposes only.\n"
                      "It is NOT a substitute for professional medical advice.\n"
                      "Always consult a qualified doctor for diagnosis and treatment.",
                 bg=C["card"], fg=C["warn"],
                 font=("Courier", 10), justify="left").pack(anchor="w", pady=20)


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not ML_AVAILABLE:
        print(f"❌ Missing library: {MISSING_LIB}")
        print("Run:  pip install pandas scikit-learn imbalanced-learn xgboost joblib")
        sys.exit(1)
    app = DiabetesApp()
    app.mainloop()
