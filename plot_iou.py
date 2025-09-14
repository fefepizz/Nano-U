import matplotlib.pyplot as plt
import os

# --- Dati del Grafico ---
models = ['BU_Net', 'Nano_U', 'Nano_U_microflow']
ious = [0.8795, 0.8785, 0.8789]
bar_colors = ['#1f77b4', '#ff7f0e', (0.5, 0, 0.5)]

# --- Creazione del Grafico (Stile benchmark.py) ---
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('IoU Comparison for Models', fontsize=16, fontweight='bold')

bars = ax.bar(models, ious, color=bar_colors)
ax.set_ylabel('IoU')

# --- ANNOTAZIONE MIGLIORATA ---
# Il ciclo 'for' è stato sostituito con ax.bar_label per un codice più pulito e moderno
ax.bar_label(bars, fmt='%.4f', fontsize=12, fontweight='bold', padding=3)

# --- Y-LIMIT CORRETTO ---
# Imposta i limiti per focalizzare sui risultati e dare spazio alle etichette
ax.set_ylim(bottom=0.87, top=0.885)

# Assicura una corretta spaziatura
plt.tight_layout()
plt.subplots_adjust(top=0.92) # Aggiusta per far spazio al suptitle

# --- Salvataggio del Grafico ---
os.makedirs("exp", exist_ok=True)
output_path = os.path.join("exp", "iou_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f'Plot saved to {output_path}')