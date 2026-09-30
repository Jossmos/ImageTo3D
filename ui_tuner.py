import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESETS_FILE = os.path.join(ROOT, "blender_presets.json")

DEFAULT_PRESETS = {
    "high_relief": {
        "max_height": "2.0",
        "smoothing_iters": "2",
        "smoothing_factor": "0.2",
        "subsurf_levels": "2",
        "subsurf_render_levels": "2",
        "subsurf_type": "SIMPLE",
        "displace_strength": "1.5",
        "displace_mid_level": "0.0",
        "corrective_iterations": "8",
        "corrective_factor": "0.6",
        "use_modifiers": "true",
        "apply_modifiers_for_stl": "false"
    },
    "soft_smooth": {
        "max_height": "1.0",
        "smoothing_iters": "4",
        "smoothing_factor": "0.45",
        "subsurf_levels": "3",
        "subsurf_render_levels": "3",
        "subsurf_type": "CATMULL_CLARK",
        "displace_strength": "0.9",
        "displace_mid_level": "0.0",
        "corrective_iterations": "12",
        "corrective_factor": "0.4",
        "use_modifiers": "true",
        "apply_modifiers_for_stl": "false"
    },
    "preserve_edges": {
        "max_height": "0.9",
        "smoothing_iters": "1",
        "smoothing_factor": "0.12",
        "subsurf_levels": "1",
        "subsurf_render_levels": "1",
        "subsurf_type": "SIMPLE",
        "displace_strength": "0.8",
        "displace_mid_level": "0.0",
        "corrective_iterations": "4",
        "corrective_factor": "0.3",
        "use_modifiers": "true",
        "apply_modifiers_for_stl": "false"
    }
}


def load_presets():
    if os.path.isfile(PRESETS_FILE):
        try:
            return json.load(open(PRESETS_FILE, "r", encoding="utf-8"))
        except Exception:
            return DEFAULT_PRESETS.copy()
    return DEFAULT_PRESETS.copy()


def save_presets(presets):
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
    except Exception as e:
        messagebox.showerror("Error", f"Could not save presets: {e}")


class UITuner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ImageTo3D - Blender Modifier Tuner")
        self.geometry("720x520")
        self.presets = load_presets()

        self.image_path_var = tk.StringVar()
        self.preset_var = tk.StringVar()

        # Parameter vars
        self.params = {}
        keys = [
            "max_height",
            "max_samples",
            "alpha_threshold",
            "smoothing_iters",
            "smoothing_factor",
            "bottom_z",
            "plane_scale_x",
            "subsurf_levels",
            "subsurf_render_levels",
            "subsurf_type",
            "subsurf_use_limit_surface",
            "displace_strength",
            "displace_mid_level",
            "corrective_iterations",
            "corrective_factor",
            "use_modifiers",
            "apply_modifiers_for_stl",
        ]

        defaults = self.presets.get("soft_smooth", {})

        for k in keys:
            self.params[k] = tk.StringVar(value=str(defaults.get(k, "")))

        self._build_ui()

    def _build_ui(self):
        frm_top = tk.Frame(self)
        frm_top.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(frm_top, text="Input image:").pack(side=tk.LEFT)
        tk.Entry(frm_top, textvariable=self.image_path_var, width=60).pack(side=tk.LEFT, padx=6)
        tk.Button(frm_top, text="Browse", command=self.browse_image).pack(side=tk.LEFT)

        frm_mid = tk.Frame(self)
        frm_mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: numeric params
        left = tk.Frame(frm_mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def mkrow(label_text, var):
            r = tk.Frame(left)
            r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=label_text, width=20, anchor=tk.W).pack(side=tk.LEFT)
            tk.Entry(r, textvariable=var, width=18).pack(side=tk.LEFT)

        mkrow("max_height", self.params["max_height"])
        mkrow("max_samples", self.params["max_samples"])
        mkrow("alpha_threshold", self.params["alpha_threshold"])
        mkrow("smoothing_iters", self.params["smoothing_iters"])
        mkrow("smoothing_factor", self.params["smoothing_factor"])
        mkrow("bottom_z", self.params["bottom_z"])
        mkrow("plane_scale_x", self.params["plane_scale_x"])

        # Right: subdivision / displace
        right = tk.Frame(frm_mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12)

        mkrow("subsurf_levels", self.params["subsurf_levels"])
        mkrow("subsurf_render_levels", self.params["subsurf_render_levels"])
        mkrow("subsurf_type", self.params["subsurf_type"])
        mkrow("subsurf_use_limit_surface", self.params["subsurf_use_limit_surface"])
        mkrow("displace_strength", self.params["displace_strength"])
        mkrow("displace_mid_level", self.params["displace_mid_level"])
        mkrow("corrective_iterations", self.params["corrective_iterations"])
        mkrow("corrective_factor", self.params["corrective_factor"])
        mkrow("use_modifiers", self.params["use_modifiers"])
        mkrow("apply_modifiers_for_stl", self.params["apply_modifiers_for_stl"])

        frm_bot = tk.Frame(self)
        frm_bot.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(frm_bot, text="Load Preset", command=self.load_preset_menu).pack(side=tk.LEFT)
        tk.Button(frm_bot, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=6)
        tk.Button(frm_bot, text="Run Pipeline", bg="#4CAF50", fg="white", command=self.run_pipeline).pack(side=tk.RIGHT)
        tk.Button(frm_bot, text="Run Blender Only", command=self.run_blender_only).pack(side=tk.RIGHT, padx=6)

    def browse_image(self):
        p = filedialog.askopenfilename(title="Select input image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*")])
        if p:
            self.image_path_var.set(p)

    def load_preset_menu(self):
        presets = list(self.presets.keys())
        if not presets:
            messagebox.showinfo("No presets", "No presets available")
            return
        def on_select():
            name = lb.get(tk.ACTIVE)
            if not name:
                return
            data = self.presets.get(name, {})
            for k, v in data.items():
                if k in self.params:
                    self.params[k].set(str(v))
            top.destroy()

        top = tk.Toplevel(self)
        top.title("Select preset")
        lb = tk.Listbox(top)
        for p in presets:
            lb.insert(tk.END, p)
        lb.pack(fill=tk.BOTH, expand=True)
        tk.Button(top, text="Load", command=on_select).pack()

    def save_preset(self):
        name = filedialog.asksaveasfilename(title="Preset name (will be used as key)", defaultextension=".json", initialfile="my_preset.json")
        if not name:
            return
        key = os.path.splitext(os.path.basename(name))[0]
        data = {k: self.params[k].get() for k in self.params}
        self.presets[key] = data
        save_presets(self.presets)
        messagebox.showinfo("Saved", f"Preset '{key}' saved")

    def _build_blender_opts(self):
        parts = []
        for k, v in self.params.items():
            val = v.get().strip()
            if val == "":
                continue
            parts.append(f"{k}={val}")
        return " ".join(parts)

    def _get_venv_python(self):
        v = os.path.join(ROOT, ".venv311", "Scripts", "python.exe")
        if os.path.isfile(v):
            return v
        # fallback to system python
        return sys.executable

    def run_pipeline(self):
        image_path = self.image_path_var.get().strip()
        if not image_path or not os.path.isfile(image_path):
            messagebox.showerror("Error", "Please choose a valid input image first")
            return

        blender_opts = self._build_blender_opts()
        venv_py = self._get_venv_python()

        code = (
            "import os,sys\n"
            "os.environ['BLENDER_OPTS']=%s\n"
            "import image_pipeline as p\n"
            "p.INPUT_IMAGE_PATH=%s\n"
            "p.OUTPUT_FOLDER_PATH=''\n"
            "p.USE_OUTPUT_SUBFOLDER=False\n"
            "p.main()\n"
        ) % (json.dumps(blender_opts), json.dumps(image_path))

        try:
            subprocess.run([venv_py, "-c", code], check=True)
            messagebox.showinfo("Done", "Pipeline finished (check console/output folder)")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Pipeline failed: {e}")

    def run_blender_only(self):
        # Same as run_pipeline but set SKIP_API=1 and expect depth/alpha in same folder
        image_path = self.image_path_var.get().strip()
        if not image_path or not os.path.isfile(image_path):
            messagebox.showerror("Error", "Please choose a valid input image first")
            return

        # We will run pipeline but with SKIP_API and require user to supply depth/alpha names
        blender_opts = self._build_blender_opts()
        venv_py = self._get_venv_python()

        code = (
            "import os,sys\n"
            "os.environ['BLENDER_OPTS']=%s\n"
            "os.environ['SKIP_API']='1'\n"
            "import image_pipeline as p\n"
            "p.INPUT_IMAGE_PATH=%s\n"
            "p.OUTPUT_FOLDER_PATH=''\n"
            "p.USE_OUTPUT_SUBFOLDER=False\n"
            "p.main()\n"
        ) % (json.dumps(blender_opts), json.dumps(image_path))

        try:
            subprocess.run([venv_py, "-c", code], check=True)
            messagebox.showinfo("Done", "Blender-only run finished (check console/output folder)")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Run failed: {e}")


if __name__ == "__main__":
    app = UITuner()
    app.mainloop()
