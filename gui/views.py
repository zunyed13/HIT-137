import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from typing import Optional, Dict, Tuple

from core.adapters import GPT2TextAdapter, ViTGPT2CaptionAdapter

BG = "#0F1115"
FG = "#E6E6E6"
FIELD_BG = "#1B1F27"
MUTED = "#9AA4B2"

# One dropdown with both options: label -> (kind, hf_model_id)
# kind is "text" or "image"
MODEL_OPTIONS: Dict[str, Tuple[str, str]] = {
    "GPT-2 (openai-community/gpt2)": ("text", "openai-community/gpt2"),
    "ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)": ("image", "nlpconnect/vit-gpt2-image-captioning"),
}

# Brief per-model info
MODEL_BRIEFS: Dict[str, str] = {
    "openai-community/gpt2": (
        "GPT-2 is a causal language model trained to predict the next token. "
        "Good for short-form generation and quick ideation."
    ),
    "nlpconnect/vit-gpt2-image-captioning": (
        "ViT-GPT2 couples a Vision Transformer encoder with a GPT-2 decoder to caption images. "
        "Useful for quick, general-purpose descriptions."
    ),
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # Title bar requirement
        self.title("tkinter AI GUI")
        self.geometry("1180x760")
        self.configure(bg=BG)

        # Adapters (created lazily for the selected model)
        self.txt_adapter: Optional[GPT2TextAdapter] = None
        self.cap_adapter: Optional[ViTGPT2CaptionAdapter] = None

        # UI state
        default_label = list(MODEL_OPTIONS.keys())[0]
        self.var_selected_label = tk.StringVar(value=default_label)
        self.var_model_info = tk.StringVar(value=self._model_info_for_current())

        # Image state
        self._img_path = None
        self._thumb_ref = None

        self._build_ui()
        self._show_section_for_current()  # ensure only one section is visible

    # UI
    def _build_ui(self):
        # Top banner
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(
            header,
            text="Hugging Face — Text (GPT-2) + Image Captioning (ViT-GPT2)",
            bg=BG, fg=FG, font=("Segoe UI", 14, "bold")
        ).pack(side="left")

        # Main split pane: left (work area) / right (info)
        paned = tk.PanedWindow(self, orient="horizontal", sashwidth=4, bg=BG, bd=0, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=14, pady=10)

        # LEFT COLUMN
        left = tk.Frame(paned, bg=BG)
        paned.add(left, stretch="always")

        # Unified model selector
        selector_row = tk.Frame(left, bg=BG)
        selector_row.pack(fill="x", pady=(0, 10))
        tk.Label(selector_row, text="Model", bg=BG, fg=FG).pack(side="left")
        cmb = ttk.Combobox(
            selector_row,
            values=list(MODEL_OPTIONS.keys()),
            textvariable=self.var_selected_label,
            state="readonly",
            width=52,
        )
        cmb.pack(side="left", padx=8)
        cmb.bind("<<ComboboxSelected>>", self._on_model_changed)

        # Text Generation section (hidden/shown based on dropdown)
        self.text_card = tk.LabelFrame(left, text=" Text Generation ", fg=FG, bg=BG)
        self.text_card.configure(labelanchor="nw")

        tk.Label(self.text_card, text="Prompt", bg=BG, fg=FG).pack(anchor="w", padx=8, pady=(8, 0))
        self.txt_prompt = tk.Text(self.text_card, height=8, bg=FIELD_BG, fg=FG, insertbackground=FG, wrap="word")
        self.txt_prompt.pack(fill="x", padx=8, pady=(4, 8))

        controls = tk.Frame(self.text_card, bg=BG)
        controls.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(controls, text="Generate", command=self._on_generate).pack(side="left")
        ttk.Button(controls, text="Clear", command=self._on_clear_text_inputs).pack(side="left", padx=(6, 0))
        self.status_text = tk.Label(controls, text="Ready", bg=BG, fg=MUTED)
        self.status_text.pack(side="right")

        tk.Label(self.text_card, text="Output", bg=BG, fg=FG).pack(anchor="w", padx=8, pady=(6, 0))
        self.txt_out = tk.Text(self.text_card, height=12, bg=FIELD_BG, fg=FG, insertbackground=FG, wrap="word")
        self.txt_out.pack(fill="x", padx=8, pady=(4, 10))

        # --- Image Captioning section (hidden/shown based on dropdown) ---
        self.img_card = tk.LabelFrame(left, text=" Image Captioning ", fg=FG, bg=BG)
        self.img_card.configure(labelanchor="nw")

        topi = tk.Frame(self.img_card, bg=BG)
        topi.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(topi, text="Browse Image", command=self._on_browse).pack(side="left")
        self.lbl_path = tk.Label(topi, text="No image selected", bg=BG, fg=MUTED)
        self.lbl_path.pack(side="left", padx=8)

        midi = tk.Frame(self.img_card, bg=BG)
        midi.pack(fill="x", padx=8, pady=(10, 8))
        self.thumb = tk.Label(midi, bg=BG)
        self.thumb.pack(side="left", padx=(0, 12))
        ttk.Button(midi, text="Generate Caption", command=self._on_caption).pack(side="left")
        ttk.Button(midi, text="Clear", command=self._on_clear_image_inputs).pack(side="left", padx=(6, 0))

        tk.Label(self.img_card, text="Caption", bg=BG, fg=FG).pack(anchor="w", padx=8, pady=(6, 0))
        self.txt_cap = tk.Text(self.img_card, height=10, bg=FIELD_BG, fg=FG, insertbackground=FG, wrap="word")
        self.txt_cap.pack(fill="x", padx=8, pady=(4, 10))

        # RIGHT SIDEBAR: Model Info + Tips + OOP Concepts
        right = tk.Frame(paned, bg=BG)
        paned.add(right, width=360)

        tk.Label(right, text="Selected Model Info", bg=BG, fg=FG, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.lbl_model_info = tk.Message(right, textvariable=self.var_model_info, width=330, bg=BG, fg=MUTED, justify="left")
        self.lbl_model_info.pack(fill="x")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(right, text="Tips", bg=BG, fg=FG, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tips = (
            "• First run may download model weights.\n"
            "• Clear buttons reset inputs quickly.\n"
            "• Switch model with the dropdown; only the relevant controls are shown.\n"
            "• Loading is lazy: the model loads the first time you use it."
        )
        tk.Message(right, text=tips, width=330, bg=BG, fg=MUTED, justify="left").pack(fill="x", pady=(2, 10))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

        tk.Label(right, text="OOP Concepts in this App", bg=BG, fg=FG, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2, 6))
        oop_text = (
            "• Multiple Inheritance: Adapters inherit a Base plus Mixins. The Base holds shared state; "
            "Mixins add logging and file validation.\n\n"
            "• Multiple Decorators: @requires_input validates inputs and @timed measures runtime—stacked for clarity.\n\n"
            "• Encapsulation: Hugging Face pipelines are hidden behind load()/run(). The GUI never touches internals.\n\n"
            "• Polymorphism: Text and Image adapters share method names (load/run) but do different work; "
            "the GUI can treat them uniformly.\n\n"
            "• Method Overriding: Each adapter overrides load()/run() with model-specific logic."
        )
        tk.Message(right, text=oop_text, width=330, bg=BG, fg=MUTED, justify="left").pack(fill="x")

    # Visibility control
    def _show_section_for_current(self):
        """Show only the section that matches the selected model kind."""
        label = self.var_selected_label.get()
        kind, _ = MODEL_OPTIONS[label]

        # Hide both
        self.text_card.pack_forget()
        self.img_card.pack_forget()

        # Show one
        if kind == "text":
            self.text_card.pack(fill="both", expand=True, pady=(0, 8))
        else:
            self.img_card.pack(fill="both", expand=True, pady=(0, 8))

    # Handlers
    def _on_model_changed(self, _event=None):
        # Reset adapters so new selection loads lazily when used
        self.txt_adapter = None
        self.cap_adapter = None
        self._show_section_for_current()
        self.var_model_info.set(self._model_info_for_current())

    def _on_generate(self):
        prompt = self.txt_prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Missing", "Please enter a prompt.")
            return
        try:
            label = self.var_selected_label.get()
            kind, model_id = MODEL_OPTIONS[label]
            if kind != "text":
                messagebox.showinfo("Wrong model", "Switch to GPT-2 in the model dropdown to use text generation.")
                return
            if self.txt_adapter is None:
                self.txt_adapter = GPT2TextAdapter(model_name=model_id)
                self.txt_adapter.load()
            out = self.txt_adapter.run(prompt, max_new_tokens=80)
            text = out[0].get("generated_text", "") if isinstance(out, list) and out else str(out)
            self._set_text(self.txt_out, text)  # output stays read-only
            self.status_text.config(text=f"Generated with {model_id}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif"), ("All", "*.*")]
        )
        if not path:
            return
        self._img_path = path
        self.lbl_path.config(text=path)
        try:
            im = Image.open(path)
            im.thumbnail((360, 360))
            tkimg = ImageTk.PhotoImage(im)
            self.thumb.configure(image=tkimg)
            self._thumb_ref = tkimg
        except Exception as e:
            messagebox.showerror("Preview failed", str(e))

    def _on_caption(self):
        if not self._img_path:
            messagebox.showwarning("Missing", "Please select an image first.")
            return
        try:
            label = self.var_selected_label.get()
            kind, model_id = MODEL_OPTIONS[label]
            if kind != "image":
                messagebox.showinfo("Wrong model", "Switch to ViT-GPT2 in the model dropdown to caption images.")
                return
            if self.cap_adapter is None:
                self.cap_adapter = ViTGPT2CaptionAdapter(model_name=model_id)
                self.cap_adapter.load()
            out = self.cap_adapter.run(self._img_path, max_new_tokens=30)
            caption = out[0].get("generated_text", "") if isinstance(out, list) and out else str(out)
            self._set_text(self.txt_cap, caption)  # caption box is read-only
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_clear_text_inputs(self):
        # Prompt should remain editable after clearing
        self._set_text(self.txt_prompt, "", disable=False)  # <-- keep editable
        self._set_text(self.txt_out, "", disable=True)
        self.status_text.config(text="Cleared")

    def _on_clear_image_inputs(self):
        self._img_path = None
        self.lbl_path.config(text="No image selected")
        self.thumb.configure(image="")
        self._thumb_ref = None
        self._set_text(self.txt_cap, "", disable=True)

    # helpers
    def _set_text(self, widget: tk.Text, content: str, disable: bool = True):
        """
        Write text to a tk.Text widget.
        If disable=True, leave it read-only afterward (for outputs).
        If disable=False, leave it editable (for prompts).
        """
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled" if disable else "normal")

    def _model_info_for_current(self) -> str:
        label = self.var_selected_label.get()
        kind, model_id = MODEL_OPTIONS[label]
        info = MODEL_BRIEFS.get(model_id, "No info available.")
        title = "Text model" if kind == "text" else "Image model"
        return f"{title}: {model_id}\n{info}"


if __name__ == "__main__":
    App().mainloop()
