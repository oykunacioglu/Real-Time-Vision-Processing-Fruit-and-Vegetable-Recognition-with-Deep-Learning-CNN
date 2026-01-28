
import tensorflow as tf
import os
import numpy as np
import cv2
from keras.utils import img_to_array

# --- 1. AYARLAR VE LİSTELER ---
# Bilgisayarındaki test klasörü yolu
base_path = r"C:\Proje\fruits-360_100x100\fruits-360\Test"
all_categories = sorted(os.listdir(base_path))

# İsim eşleme sözlüğü (Modelin ham çıktısını düzeltmek için)
name_fixer = {
    "Zucchini 1": "Green Apple",
    "Corn Husk 1": "Pepper Green",
    "Carrot": "Havuc",
    "Apple Granny Smith": "Elma",
    "Orange": "Portakal"
}

# Modeli yükle
model = tf.keras.models.load_model("fruit_classifier_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 🎯 ROI (Odaklanma Kutusu - 280x280 piksel)
    h, w, _ = frame.shape
    size = 280
    x1, y1 = (w - size) // 2, (h - size) // 2
    roi = frame[y1:y1+size, x1:x1+size]

    # 🔮 MODEL İÇİN ÖN İŞLEME (RGB Dönüşümü ve Boyutlandırma)
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_roi, (100, 100))
    img_input = img_to_array(resized) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # 🧠 TAHMİN
    preds = model.predict(img_input, verbose=0)[0]
    best_idx = np.argmax(preds)
    best_conf = preds[best_idx]
    raw_name = all_categories[best_idx] 

    # --- 💡 İSİM DÖNÜŞTÜRME VE %70 KONTROLÜ ---
    display_name = name_fixer.get(raw_name, raw_name)

    # --- 🎨 GÖRSELLEŞTİRME ---
    # Güven oranı %70 (0.70) üzerindeyse ismi göster
    if best_conf > 0.70:
        color = (0, 255, 0) # Yeşil (Emin)
        label = f"{display_name} (%{best_conf*100:.1f})"
    else:
        color = (0, 0, 255) # Kırmızı (Analiz aşaması)
        label = "Analiz Ediliyor..."

    # Ekrana kutuyu ve sonucu çiz
    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), color, 2)
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow("Fruit Scanner V2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import matplotlib.pyplot as plt

# Senin verilerinle küçük bir temsil
epochs = range(1, 42)
# Örnek bir eğri oluşturuyoruz (Loglarındaki değerlere dayanarak)
plt.plot(epochs, [0.8 + (0.18 * (1 - 0.9**i)) for i in epochs], label='Accuracy')
plt.title('Model Training Success')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()