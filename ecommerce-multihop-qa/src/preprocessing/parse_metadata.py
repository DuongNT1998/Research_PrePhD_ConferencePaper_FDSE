import gzip
import json
import os

def xu_ly_du_lieu_mo_ta(duong_dan_goc, duong_dan_dich, gioi_han_san_pham=5000):
    print("Đang bắt đầu phân tích cú pháp dữ liệu mô tả sản phẩm (Cập nhật Categories)...")
    danh_sach_ma_san_pham = set()
    so_luong_da_luu = 0
    
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(os.path.dirname(duong_dan_dich), exist_ok=True)
    
    with gzip.open(duong_dan_goc, 'rt', encoding='utf-8') as file_doc, \
         open(duong_dan_dich, 'w', encoding='utf-8') as file_ghi:
         
        for dong in file_doc:
            if so_luong_da_luu >= gioi_han_san_pham:
                break
                
            du_lieu_tho = json.loads(dong)
            ma_cha = du_lieu_tho.get('parent_asin')
            
            if not ma_cha or ma_cha in danh_sach_ma_san_pham:
                continue
                
            # Trích xuất các trường thông tin - ĐÃ BỔ SUNG TRƯỜNG CATEGORIES THÀNH CÔNG
            thong_tin_tinh_goc = {
                "main_category": du_lieu_tho.get('main_category', ''),
                "title": du_lieu_tho.get('title', ''),
                "average_rating": float(du_lieu_tho.get('average_rating', 0.0)) if du_lieu_tho.get('average_rating') is not None else 0.0,
                "rating_number": int(du_lieu_tho.get('rating_number', 0)) if du_lieu_tho.get('rating_number') is not None else 0,
                "features": du_lieu_tho.get('features', []),
                "price": float(du_lieu_tho.get('price', 0.0)) if (du_lieu_tho.get('price') and str(du_lieu_tho.get('price')).replace('.','',1).isdigit()) else 0.0,
                "details": du_lieu_tho.get('details', {}),
                "categories": du_lieu_tho.get('categories', []),  # <--- DÒNG KHẮC PHỤC LỖI TRỐNG FILE
                "bought_together": du_lieu_tho.get('bought_together', []),
                "parent_asin": ma_cha
            }
            
            file_ghi.write(json.dumps(thong_tin_tinh_goc, ensure_ascii=False) + '\n')
            danh_sach_ma_san_pham.add(ma_cha)
            so_luong_da_luu += 1

    print(f"Đã hoàn thành! Đã trích xuất {so_luong_da_luu} sản phẩm độc nhất kèm danh mục phân cấp.")

if __name__ == "__main__":
    duong_dan_goc = "data/raw/meta_Electronics.jsonl.gz"
    duong_dan_dich = "data/interim/products.jsonl"
    xu_ly_du_lieu_mo_ta(duong_dan_goc, duong_dan_dich)