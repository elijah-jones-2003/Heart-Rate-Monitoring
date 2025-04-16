import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np

def load_model(path="model.pth"):
    model = torch.load(path)
    model.eval()
    return model

# Function to load data from a .dat file
def load_data(dat_file):
    # Adjust this depending on your .dat file format
    return np.loadtxt(dat_file)

def run_model_on_data(dat_file):
    try:
        data = load_data(dat_file)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        model = load_model()
        with torch.no_grad():
            output = model(data_tensor)
        
        messagebox.showinfo("Model Output", f"Output: {output}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
    if file_path:
        run_model_on_data(file_path)


root = tk.Tk()
root.title("Run Model on .dat File")
root.geometry("300x150")

label = tk.Label(root, text="Click the button to select a .dat file:")
label.pack(pady=20)

button = tk.Button(root, text="Select .dat File", command=select_file)
button.pack(pady=10)

root.mainloop()