import gzip
import json
import os

def xu_ly_phan_hoi_nguoi_dung(duong_dan_goc_phan_hoi, duong_dan_san_pham_trung_gian, duong_dan_dich):
    print("Đang bắt đầu lọc và phân tích phản hồi của người dùng...")
    
    # Tải danh sách mã sản phẩm đã chọn từ bước trước vào bộ nhớ để tìm kiếm nhanh
    danh_sach_ma_san_pham = set()
    with open(duong_dan_san_pham_trung_gian, 'r', encoding='utf-8') as f:
        for dong in f:
            sp = json.loads(dong)
            danh_sach_ma_san_pham.add(sp['parent_asin'])
            
    so_luong_phan_hoi_hop_le = 0
    os.makedirs(os.path.dirname(duong_dan_dich), exist_ok=True)
    
    with gzip.open(duong_dan_goc_phan_hoi, 'rt', encoding='utf-8') as file_doc, \
         open(duong_dan_dich, 'w', encoding='utf-8') as file_ghi:
         
        for dong in file_doc:
            du_lieu_tho = json.loads(dong)
            ma_cha = du_lieu_tho.get('parent_asin')
            
            # Chỉ lấy các phản hồi thuộc danh sách 5000 sản phẩm đã chọn
            if ma_cha in danh_sach_ma_san_pham:
                thong_tin_phan_hoi = {
                    "rating": float(du_lieu_tho.get('rating', 0.0)),
                    "title": du_lieu_tho.get('title', ''),
                    "text": du_lieu_tho.get('text', ''),
                    "parent_asin": ma_cha,
                    "verified_purchase": bool(du_lieu_tho.get('verified_purchase', False)),
                    "helpful_vote": int(du_lieu_tho.get('helpful_vote', 0))
                }
                file_ghi.write(json.dumps(thong_tin_phan_hoi, ensure_ascii=False) + '\n')
                so_luong_phan_hoi_hop_le += 1
                
    print(f"Đã hoàn thành! Tìm thấy và lưu trữ {so_luong_phan_hoi_hop_le} phản hồi hợp lệ.")

if __name__ == "__main__":
    duong_dan_goc_phan_hoi = "data/raw/Electronics.jsonl.gz"
    duong_dan_san_pham_trung_gian = "data/interim/products.jsonl"
    duong_dan_dich = "data/interim/reviews.jsonl"
    xu_ly_phan_hoi_nguoi_dung(duong_dan_goc_phan_hoi, duong_dan_san_pham_trung_gian, duong_dan_dich)