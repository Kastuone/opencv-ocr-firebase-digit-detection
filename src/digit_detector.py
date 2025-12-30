import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from paddleocr import PaddleOCR
import time
import re

# Firebase Yapılandırması
# Service Account Key JSON dosyası kullanarak bağlanma
# Windows için dosya yolunda çift backslash veya raw string kullanın
cred = credentials.Certificate("config/firebase_config.json")


# Firebase'i başlat
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ika-db-eb609-default-rtdb.europe-west1.firebasedatabase.app'  # Database URL'niz
})

# Kamera ID'si (birden fazla kamera kullanacaksanız değiştirebilirsiniz)
DEVICE_ID = 'CAMERA_01'

# Firebase database referansı - Sabit path kullanıyoruz
device_ref = db.reference(f'detected_digits/{DEVICE_ID}')

class PaddleOCRDigitDetector:
    def __init__(self):
        """PaddleOCR'ı başlat"""
        print("PaddleOCR yükleniyor...")
        
        # PaddleOCR'ı yapılandır
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # Açı düzeltme
            lang='en',  # İngilizce/Rakamlar için
            use_gpu=True,  # GPU kullanımı (varsa True yapın)
            show_log=False,  # Log gösterme
            det_db_thresh=0.3,  # Algılama eşiği
            rec_batch_num=1,
            max_text_length=25,
            use_space_char=False,
            drop_score=0.5  # Düşük skorlu sonuçları ele
        )
        
        print("PaddleOCR başarıyla yüklendi!")
    
    def extract_digits_from_text(self, text):
        """Metinden sadece rakamları çıkar"""
        # Sadece rakamları al
        digits = re.findall(r'\d', text)
        return digits
    
    def detect_digit(self, img):
        """Görüntüden rakam algıla"""
        try:
            # PaddleOCR ile OCR işlemi yap
            result = self.ocr.ocr(img, cls=True)
            
            if result is None or len(result) == 0:
                return None, 0, None
            
            detected_digits = []
            
            # Sonuçları işle
            for line in result:
                if line is None:
                    continue
                    
                for word_info in line:
                    if word_info is None:
                        continue
                    
                    # Metin ve güven skorunu al
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    
                    # Metinden rakamları çıkar
                    digits = self.extract_digits_from_text(text)
                    
                    for digit_str in digits:
                        digit = int(digit_str)
                        if 0 <= digit <= 9:  # 0-9 arası tüm rakamlar
                            detected_digits.append({
                                'digit': digit,
                                'confidence': confidence,
                                'bbox': word_info[0]  # Sınırlayıcı kutu koordinatları
                            })
            
            # En yüksek güven skoruna sahip rakamı döndür
            if detected_digits:
                best_detection = max(detected_digits, key=lambda x: x['confidence'])
                return best_detection['digit'], best_detection['confidence'], best_detection['bbox']
            
            return None, 0, None
            
        except Exception as e:
            print(f"OCR hatası: {e}")
            return None, 0, None
    
    def draw_detection(self, img, bbox, digit):
        """Algılanan rakamı görüntü üzerinde göster"""
        if bbox is None:
            return img
        
        # Kopyasını al
        result_img = img.copy()
        
        # Bbox noktalarını al
        points = np.array(bbox, dtype=np.int32)
        
        # Dikdörtgen çiz
        cv2.polylines(result_img, [points], True, (0, 255, 0), 2)
        
        # Metin ekle
        x, y = points[0]
        text = f"Rakam: {digit}"
        cv2.putText(result_img, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result_img

def update_firebase(digit):
    """Firebase'deki mevcut path'i güncelle - sadece digit değeri"""
    try:
        # Firebase'deki sabit path'i güncelle - sadece digit değeri
        # Direkt olarak digit path'ine yazıyoruz
        digit_ref = db.reference(f'detected_digits/{DEVICE_ID}/digit')
        digit_ref.set(int(digit))
        
        print(f"Firebase güncellendi: {DEVICE_ID}/digit -> {digit}")
        return True
        
    except Exception as e:
        print(f"Firebase hatası: {e}")
        return False

def main():
    """Ana program döngüsü"""
    
    # PaddleOCR algılayıcıyı başlat
    detector = PaddleOCRDigitDetector()
    
    # Kamerayı başlat (0: varsayılan kamera, 1,2.. diğer kameralar)
    cap = cv2.VideoCapture(0)  # Kamera numarasını değiştirebilirsiniz
    
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    # Kamera ayarları (opsiyonel)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*50)
    print("KAMERA BAŞLATILDI - KONTROLLER:")
    print("="*50)
    print("'s' tuşu: Rakam algıla ve Firebase'e gönder")
    print("'a' tuşu: Otomatik algılama modunu aç/kapat")
    print("'r' tuşu: ROI (ilgi alanı) seçimi yap")
    print("'q' tuşu: Programı kapat")
    print("="*50)
    print(f"Firebase Path: /detected_digits/{DEVICE_ID}/digit")
    print("NOT: Sadece rakam değeri gönderilecek!")
    print("="*50 + "\n")
    
    # Son gönderilen rakam ve zaman kontrolü
    last_sent_digit = None
    cooldown_time = 2  # Saniye cinsinden bekleme süresi
    last_send_time = 0
    
    # Otomatik algılama modu
    auto_detect = False
    
    # ROI (Region of Interest) değişkenleri
    roi_selected = False
    roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
    
    # FPS hesaplama
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Görüntü alınamadı!")
            break
        
        # FPS hesapla
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Görüntüyü kopyala (işlemler için)
        display_frame = frame.copy()
        
        # ROI seçilmişse göster
        if roi_selected:
            cv2.rectangle(display_frame, (roi_x, roi_y), 
                         (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
            cv2.putText(display_frame, "ROI", (roi_x, roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Durum bilgilerini ekle
        status_text = f"FPS: {fps} | Auto: {'ON' if auto_detect else 'OFF'}"
        if last_sent_digit is not None:
            status_text += f" | Son: {last_sent_digit}"
        
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Firebase path bilgisini göster
        path_text = f"Firebase: /detected_digits/{DEVICE_ID}/digit"
        cv2.putText(display_frame, path_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Görüntüyü göster
        cv2.imshow('PaddleOCR - Rakam Algılama', display_frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' tuşuna basılırsa çık
        if key == ord('q'):
            break
        
        # 'a' tuşuna basılırsa otomatik algılamayı aç/kapat
        elif key == ord('a'):
            auto_detect = not auto_detect
            print(f"Otomatik algılama: {'AÇIK' if auto_detect else 'KAPALI'}")
        
        # 'r' tuşuna basılırsa ROI seç
        elif key == ord('r'):
            print("ROI seçimi için alan belirleyin...")
            roi = cv2.selectROI("ROI Seçimi", frame, False)
            cv2.destroyWindow("ROI Seçimi")
            
            if roi[2] > 0 and roi[3] > 0:
                roi_x, roi_y, roi_w, roi_h = roi
                roi_selected = True
                print(f"ROI seçildi: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
            else:
                roi_selected = False
                print("ROI seçimi iptal edildi")
        
        # Manuel veya otomatik algılama
        should_detect = False
        
        if key == ord('s'):
            should_detect = True
        elif auto_detect:
            current_time = time.time()
            if current_time - last_send_time > cooldown_time:
                should_detect = True
        
        if should_detect:
            current_time = time.time()
            
            # Cooldown kontrolü
            if current_time - last_send_time < cooldown_time:
                remaining = cooldown_time - (current_time - last_send_time)
                print(f"Lütfen {remaining:.1f} saniye bekleyin...")
                continue
            
            # İşlenecek görüntüyü belirle
            if roi_selected:
                # ROI alanını kullan
                process_img = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            else:
                # Tüm görüntüyü kullan
                process_img = frame
            
            # Görüntü ön işleme (opsiyonel)
            # Gri tonlama
            gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
            
            # Kontrast artırma (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Gürültü azaltma
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # BGR'ye geri dönüştür (PaddleOCR için)
            process_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            
            # Rakam algıla
            result = detector.detect_digit(process_img)
            
            if result[0] is not None:
                digit, confidence, bbox = result
                
                # Firebase'i güncelle (sadece digit değeri)
                if update_firebase(digit):
                    print(f"Algılanan: {digit} (Güven: %{confidence*100:.1f})")
                    print(f"Path güncellendi: /detected_digits/{DEVICE_ID}/digit = {digit}")
                    last_sent_digit = digit
                    last_send_time = current_time
                    
                    # Algılanan rakamı göster
                    if roi_selected and bbox is not None:
                        # ROI içindeki koordinatları ana frame'e dönüştür
                        adjusted_bbox = []
                        for point in bbox:
                            adjusted_point = [point[0] + roi_x, point[1] + roi_y]
                            adjusted_bbox.append(adjusted_point)
                        detection_frame = detector.draw_detection(frame, adjusted_bbox, digit)
                    else:
                        detection_frame = detector.draw_detection(frame, bbox, digit)
                    
                    cv2.imshow('Algılanan Rakam', detection_frame)
            else:
                print("Rakam algılanamadı")
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram sonlandırıldı.")

if __name__ == "__main__":
    # Gerekli kütüphaneleri kontrol et
    required_packages = ['paddlepaddle', 'paddleocr', 'opencv-python', 'firebase-admin', 'numpy']
    
    print("=" * 60)
    print("PADDLEOCR İLE RAKAM ALGILAMA VE FIREBASE SİSTEMİ")
    print("SADECE RAKAM (DIGIT) VERSİYONU")
    print("=" * 60)
    print("\nGEREKLİ KÜTÜPHANELER:")
    for package in required_packages:
        print(f"- {package}")
    
    print("\nKURULUM:")
    print("pip install paddlepaddle")
    print("pip install paddleocr")
    print("pip install opencv-python firebase-admin numpy")
    
    print("\nNOT: GPU desteği için:")
    print("pip install paddlepaddle-gpu")
    
    print("\n" + "="*60)
    print("ÖNEMLİ DEĞİŞİKLİK:")
    print("="*60)
    print(f"✓ Firebase'de tek path kullanılıyor: /detected_digits/{DEVICE_ID}/digit")
    print("✓ Sadece rakam (digit) değeri gönderiliyor")
    print("✓ Harf dönüşümü KALDIRILDI")
    print("✓ Timestamp, device_id vb. alanlar KALDIRILDI")
    print("✓ Model takibi için optimize edildi")
    
    print("\n" + "="*60)
    print("DÜZENLEMENİZ GEREKEN YERLER:")
    print("="*60)
    print("1. Firebase Admin SDK dosya yolunu düzenleyin (satır 12)")
    print("2. Firebase database URL'nizi yazın (satır 16)")
    print("3. DEVICE_ID'yi değiştirmek isterseniz (satır 20)")
    print("4. Kamera numarasını ayarlayın (satır 136)")
    print("5. GPU kullanımı için satır 34'te use_gpu=True yapın")
    
    print("\n" + "="*60)
    print("ÖZELLİKLER:")
    print("="*60)
    print("✓ PaddleOCR ile güçlü rakam algılama")
    print("✓ Otomatik algılama modu")
    print("✓ ROI (ilgi alanı) seçimi")
    print("✓ Görüntü ön işleme (kontrast, gürültü azaltma)")
    print("✓ FPS göstergesi")
    print("✓ Firebase'de sadece digit değeri güncelleme")
    print("=" * 60 + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nHata oluştu: {e}")
        print("Lütfen Firebase yapılandırmanızı ve kamera bağlantınızı kontrol edin.")
        print("\nPaddleOCR kurulumu için:")
        print("https://github.com/PaddlePaddle/PaddleOCR")
