import cv2
import numpy as np
import os

#Pastikan folder output ada
os.makedirs("output", exist_ok=True)

#Membuat kanvas kosong hitam (300x300 pixel)
canvas = np.zeros((300, 300, 3), dtype=np.uint8)

#Membuat karakter (Robot)
#Kepala robot
cv2.circle(canvas, (150, 90), 40, (0, 100, 0), -1)  # BGR: hijau tua

#Antena kepala
cv2.line(canvas, (150, 50), (150, 30), (0, 150, 0), 2)
cv2.circle(canvas, (150, 25), 5, (0, 0, 255), -1)  # ujung merah

#Mata 
cv2.circle(canvas, (135, 90), 8, (0, 0, 255), -1)
cv2.circle(canvas, (165, 90), 8, (0, 0, 255), -1)


#Mulut
cv2.line(canvas, (135, 110), (165, 110), (0, 0, 0), 2)

#Badan
cv2.rectangle(canvas, (120, 130), (180, 230), (50, 50, 50), -1)

#Sensor dada 
sensor_pts = np.array([[150, 160], [140, 190], [160, 190]], np.int32)
cv2.fillPoly(canvas, [sensor_pts], (0, 0, 255))  # merah terang

#Tangan
cv2.line(canvas, (100, 150), (120, 180), (0, 100, 0), 6)
cv2.line(canvas, (180, 180), (200, 150), (0, 100, 0), 6)

#Kaki
cv2.line(canvas, (135, 230), (135, 270), (0, 100, 0), 6)
cv2.line(canvas, (165, 230), (165, 270), (0, 100, 0), 6)

#Simpan karakter asli
cv2.imwrite("output/karakter.png", canvas)


#Transformasi
#Translasi
M_translate = np.float32([[1, 0, 30], [0, 1, 20]])
translated = cv2.warpAffine(canvas, M_translate, (300, 300))

#Rotasi
M_rotate = cv2.getRotationMatrix2D((150, 150), 25, 1)
rotated = cv2.warpAffine(canvas, M_rotate, (300, 300))
cv2.imwrite("output/rotate.png", rotated)

#Resize (ubah ukuran jadi 150x150)
resized = cv2.resize(canvas, (150, 150))

#Crop (potong bagian tengah)
crop = canvas[70:230, 90:210]
cv2.imwrite("output/crop.png", crop)

#Bitwise
#Background default apabila tidak ada img/background.jpg
bg = np.full((300, 300, 3), (80, 80, 80), dtype=np.uint8)

#Jika ada img/background.jpg 
bg_path = "img/background.jpg"
if os.path.exists(bg_path):
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (300, 300))
    print("✅ Background ditemukan dan digunakan.")
else:
    print("⚠️ Background tidak ditemukan, pakai default.")

#bitwise_and antara karakter dan background
bitwise = cv2.bitwise_and(canvas, bg)
cv2.imwrite("output/bitwise.png", bitwise)

#bitwise_or untuk efek gabungan
final = cv2.bitwise_or(rotated, bg)
cv2.imwrite("output/final.png", final)

#Menempelkan karakter ke background
#Buat mask berdasarkan warna hitam
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

#Balik mask (untuk hapus area hitam)
mask_inv = cv2.bitwise_not(mask)

#Area background yang akan diisi karakter
bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
fg_part = cv2.bitwise_and(canvas, canvas, mask=mask)

#Gabungkan keduanya
combined = cv2.add(bg_part, fg_part)

#Simpan hasil akhir
cv2.imwrite("output/final.png", combined)

#Tampilkan semua hasil
cv2.imshow("Karakter - Robot Hijau Gelap", canvas)
cv2.imshow("Rotate", rotated)
cv2.imshow("Crop", crop)
cv2.imshow("Bitwise", bitwise)
cv2.imshow("Final", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
