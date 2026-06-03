import json
import os
import re

# DANH SÁCH TỪ VỰNG THEO ĐÚNG YÊU CẦU CỦA NGƯỜI DÙNG
POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "perfect", "awesome", "fantastic", 
    "wonderful", "outstanding", "superb", "decent", "reliable", "durable", "satisfied",
    "fast", "smooth", "responsive", "powerful", "snappy", "seamless", "efficient",
    "crisp", "clear", "sharp", "bright", "vibrant", "loud", "immersive", "beautiful",
    "lightweight", "compact", "portable", "sturdy", "premium", "sleek", "ergonomic",
    "intuitive", "user-friendly", "easy", "handy", "long-lasting", "quick-charging", 
    "stable", "strong", "worth", "budget-friendly", "affordable", "value"
]

NEGATIVE_WORDS = [
    "bad", "poor", "terrible", "awful", "horrible", "disappointed", "useless", 
    "junk", "trash", "cheap", "waste", "defective", "broken", "returned",
    "slow", "laggy", "sluggish", "freezing", "crashing", "lag", "glitchy", "buggy",
    "overheating", "hot", "warm", "short-lived", "draining", "dead",
    "blurry", "dim", "washed-out", "noisy", "muddy", "distorted", "muffled", 
    "disconnected", "dropping", "unstable", "static", "heavy", "bulky", 
    "flimsy", "plastic", "scratch", "fragile", "clunky"
]

ASPECT_KEYWORDS = [
    "battery", "battery life", "charge", "charger", "charging", "runtime", "power", "cord", "cable",
    "screen", "display", "monitor", "panel", "brightness", "resolution", "color", "refresh rate", 
    "oled", "amoled", "lcd", "pixels", "performance", "speed", "processor", "cpu", "ram", "graphics", 
    "gpu", "storage", "hard drive", "ssd", "memory", "fps", "sound", "audio", "bass", "treble", 
    "volume", "speaker", "mic", "microphone", "sound quality", "headphone", "earbud", "weight", 
    "size", "design", "build quality", "material", "dimensions", "portability", "look", "shape", "case",
    "camera", "lens", "video", "photo", "picture quality", "sensor", "zoom", "flash", "focus",
    "bluetooth", "wifi", "connection", "connectivity", "pairing", "port", "usb", "hdmi", 
    "software", "ui", "app", "firmware", "os", "system", "setup", "price", "cost", "value", 
    "warranty", "customer service", "support"
]

def phan_tich_sac_thai_khia_canh(van_ban, diem_so):
    van_ban_sach = van_ban.lower()
    khia_canh_phat_hien = []
    
    for khia_canh in ASPECT_KEYWORDS:
        if re.search(r'\b' + re.escape(khia_canh) + r'\b', van_ban_sach):
            # Quy tắc kết hợp điểm số đánh giá và từ vựng sắc thái
            so_tu_tich_cuc = sum(1 for tu in POSITIVE_WORDS if re.search(r'\b' + re.escape(tu) + r'\b', van_ban_sach))
            so_tu_tieu_cuc = sum(1 for tu in NEGATIVE_WORDS if re.search(r'\b' + re.escape(tu) + r'\b', van_ban_sach))
            
            huong_sac_thai = "chua_xac_dinh"
            
            # Nếu điểm đánh giá cao (4 hoặc 5 sao), ưu tiên phân loại tích cực
            if diem_so >= 4.0:
                huong_sac_thai = "positive" if so_tu_tieu_cuc <= so_tu_tich_cuc else "negative"
            # Nếu điểm đánh giá thấp (1 hoặc 2 sao), ưu tiên phân loại tiêu cực
            elif diem_so <= 2.0:
                huong_sac_thai = "negative" if so_tu_tich_cuc <= so_tu_tieu_cuc else "positive"
            # Nếu điểm trung bình (3 sao), phụ thuộc hoàn toàn vào số lượng từ vựng xuất hiện
            else:
                if so_tu_tich_cuc > so_tu_tieu_cuc:
                    huong_sac_thai = "positive"
                elif so_tu_tieu_cuc > so_tu_tich_cuc:
                    huong_sac_thai = "negative"
                else:
                    huong_sac_thai = "positive" # Hướng mặc định khi cân bằng
                    
            khia_canh_phat_hien.append({
                "aspect": khia_canh,
                "sentiment": huong_sac_thai
            })
            
    return khia_canh_phat_hien

def trich_xuat_toan_bo_khia_canh(duong_dan_phan_hoi, duong_dan_dich):
    print("Đang tiến hành phân tách và trích xuất khia cạnh từ văn bản phản hồi...")
    os.makedirs(os.path.dirname(duong_dan_dich), exist_ok=True)
    
    with open(duong_dan_phan_hoi, 'r', encoding='utf-8') as file_doc, \
         open(duong_dan_dich, 'w', encoding='utf-8') as file_ghi:
         
        for dong in file_doc:
            phan_hoi = json.loads(dong)
            noi_dung_gộp = phan_hoi['title'] + " " + phan_hoi['text']
            diem_so = phan_hoi['rating']
            ma_cha = phan_hoi['parent_asin']
            
            danh_sach_khia_canh = phan_tich_sac_thai_khia_canh(noi_dung_gộp, diem_so)
            
            if danh_sach_khia_canh:
                ket_qua = {
                    "parent_asin": ma_cha,
                    "aspects": danh_sach_khia_canh
                }
                file_ghi.write(json.dumps(ket_qua, ensure_ascii=False) + '\n')
                
    print("Đã hoàn thành công việc trích xuất khía cạnh!")

if __name__ == "__main__":
    duong_dan_phan_hoi = "data/interim/reviews.jsonl"
    duong_dan_dich = "data/interim/review_aspects.jsonl"
    trich_xuat_toan_bo_khia_canh(duong_dan_phan_hoi, duong_dan_dich)