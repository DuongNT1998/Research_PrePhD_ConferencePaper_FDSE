import os
import sys

# Đảm bảo hệ thống nhận dạng được thư mục gốc của dự án khi chạy độc lập
duong_dan_tep_hien_tai = os.path.abspath(__file__)
thu_muc_kg = os.path.dirname(duong_dan_tep_hien_tai)
thu_muc_goc_du_an = os.path.dirname(os.path.dirname(thu_muc_kg))
if thu_muc_goc_du_an not in sys.path:
    sys.path.insert(0, thu_muc_goc_du_an)

import csv
import json
from src.kg.clean_product_titles import lam_sach_chuoi_van_ban

def khoi_tao_tap_tin_do_thi(duong_dan_sp, duong_dan_khia_canh, thu_muc_import):
    print("Đang tiến hành chuyển đổi và xây dựng hệ thống tệp tin bảng (.csv)...")
    os.makedirs(thu_muc_import, exist_ok=True)
    
    cac_nut_san_pham = []
    cac_nut_nhan_hieu = set()
    cac_nut_danh_muc = set()
    cac_nut_dac_tinh = set()
    cac_nut_khia_canh = set()
    cac_nut_chi_tiet_dong = [] 
    
    quan_he_sp_nhan_hieu = set()
    quan_he_sp_danh_muc = set()
    quan_he_sp_dac_tinh = set()
    quan_he_sp_chi_tiet = []
    
    thong_ke_tich_cuc = {} 
    thong_ke_tieu_cuc = {} 

    # 1. Đọc file sản phẩm (Bây giờ đã có đầy đủ trường danh mục 'categories')
    with open(duong_dan_sp, 'r', encoding='utf-8') as f:
        for dong in f:
            sp = json.loads(dong)
            ma_cha = sp['parent_asin']
            tieu_de_sach = lam_sach_chuoi_van_ban(sp['title'])
            danh_muc_chinh = lam_sach_chuoi_van_ban(sp['main_category'])
            
            cac_nut_san_pham.append([
                ma_cha, tieu_de_sach, sp['average_rating'], 
                sp['rating_number'], sp['price'], danh_muc_chinh
            ])
            
            tu_dien_chi_tiet = sp.get('details', {})
            if isinstance(tu_dien_chi_tiet, dict):
                ten_nhan_hieu = tu_dien_chi_tiet.get('Brand') or tu_dien_chi_tiet.get('brand')
                if ten_nhan_hieu:
                    ten_nhan_hieu_sach = lam_sach_chuoi_van_ban(str(ten_nhan_hieu))
                    cac_nut_nhan_hieu.add(ten_nhan_hieu_sach)
                    quan_he_sp_nhan_hieu.add((ma_cha, ten_nhan_hieu_sach))
                
                for khoa, gia_tri in tu_dien_chi_tiet.items():
                    if gia_tri:
                        chuoi_gia_tri = lam_sach_chuoi_van_ban(str(gia_tri))
                        id_chi_tiet = f"det_{ma_cha}_{lam_sach_chuoi_van_ban(khoa)}"
                        cac_nut_chi_tiet_dong.append([id_chi_tiet, lam_sach_chuoi_van_ban(khoa), chuoi_gia_tri])
                        quan_he_sp_chi_tiet.append([ma_cha, id_chi_tiet])
            
            # Khâu bóc tách danh mục (Bây giờ sẽ chạy chính xác vì dữ liệu đầu vào đã có sẵn)
            danh_sach_muc = sp.get('categories', [])
            for muc in danh_sach_muc:
                muc_sach = lam_sach_chuoi_van_ban(muc)
                if muc_sach:
                    cac_nut_danh_muc.add(muc_sach)
                    quan_he_sp_danh_muc.add((ma_cha, muc_sach))
                    
            danh_sach_tinh_nang = sp.get('features', [])
            for tn in danh_sach_tinh_nang:
                tn_sach = lam_sach_chuoi_van_ban(tn)
                if tn_sach:
                    cac_nut_dac_tinh.add(tn_sach)
                    quan_he_sp_dac_tinh.add((ma_cha, tn_sach))

    # 2. Xử lý tệp khía cạnh đánh giá
    if os.path.exists(duong_dan_khia_canh):
        with open(duong_dan_khia_canh, 'r', encoding='utf-8') as f:
            for dong in f:
                du_lieu = json.loads(dong)
                ma_cha = du_lieu['parent_asin']
                for kc_dict in du_lieu['aspects']:
                    ten_kc = lam_sach_chuoi_van_ban(kc_dict['aspect'])
                    sac_thai = kc_dict['sentiment']
                    
                    cac_nut_khia_canh.add(ten_kc)
                    cap_khoa = (ma_cha, ten_kc)
                    
                    if sac_thai == "positive":
                        thong_ke_tich_cuc[cap_khoa] = thong_ke_tich_cuc.get(cap_khoa, 0) + 1
                    elif sac_thai == "negative":
                        thong_ke_tieu_cuc[cap_khoa] = thong_ke_tieu_cuc.get(cap_khoa, 0) + 1

    # 3. Kết xuất hệ thống tệp tin CSV dữ liệu phẳng
    with open(f"{thu_muc_import}/products.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["parent_asin:ID(Product)", "title", "average_rating:float", "rating_number:int", "price:float", "main_category"])
        w.writerows(cac_nut_san_pham)
        
    with open(f"{thu_muc_import}/brands.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["brand_id:ID(Brand)", "name"])
        for b in cac_nut_nhan_hieu:
            w.writerow([b, b])
            
    # Ghi tệp nút Danh mục - BÂY GIỜ SẼ CHỨA ĐẦY ĐỦ DỮ LIỆU ĐỘC NHẤT
    with open(f"{thu_muc_import}/categories.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["category_id:ID(Category)", "name"])
        for c in cac_nut_danh_muc:
            w.writerow([c, c])
            
    with open(f"{thu_muc_import}/features.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["feature_id:ID(Feature)", "text"])
        for index, ft in enumerate(cac_nut_dac_tinh):
            w.writerow([f"feat_{index}", ft])
            
    with open(f"{thu_muc_import}/aspects.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["aspect_id:ID(Aspect)", "name"])
        for a in cac_nut_khia_canh:
            w.writerow([a, a])

    with open(f"{thu_muc_import}/details.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["detail_id:ID(Detail)", "key", "value"])
        w.writerows(cac_nut_chi_tiet_dong)

    # Ghi hệ thống liên kết đồ thị
    with open(f"{thu_muc_import}/rel_product_brand.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Product)", ":END_ID(Brand)"])
        w.writerows(quan_he_sp_nhan_hieu)
        
    with open(f"{thu_muc_import}/rel_product_category.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Product)", ":END_ID(Category)"])
        w.writerows(quan_he_sp_danh_muc)

    ban_do_dac_tinh = {ft: f"feat_{index}" for index, ft in enumerate(cac_nut_dac_tinh)}
    with open(f"{thu_muc_import}/rel_product_feature.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Product)", ":END_ID(Feature)"])
        for sp_id, ft_text in quan_he_sp_dac_tinh:
            w.writerow([sp_id, ban_do_dac_tinh[ft_text]])

    with open(f"{thu_muc_import}/rel_product_detail.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Product)", ":END_ID(Detail)"])
        w.writerows(quan_he_sp_chi_tiet)

    with open(f"{thu_muc_import}/rel_positive_aspect.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Product)", ":END_ID(Aspect)", "weight:int"])
        for (sp_id, kc_id), t_so in thong_ke_tich_cuc.items():
            w.writerow([sp_id, kc_id, t_so])
            
    with open(f"{thu_muc_import}/rel_negative_aspect.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([":START_ID(Product)", ":END_ID(Aspect)", "weight:int"])
        for (sp_id, kc_id), t_so in thong_ke_tieu_cuc.items():
            w.writerow([sp_id, kc_id, t_so])

    print("Hệ thống tệp tin CSV bao gồm danh mục phân cấp đã được sửa đổi và khởi tạo thành công!")

if __name__ == "__main__":
    duong_dan_sp = "data/interim/products.jsonl"
    duong_dan_khia_canh = "data/interim/review_aspects.jsonl"
    thu_muc_import = "neo4j_setup/import"
    khoi_tao_tap_tin_do_thi(duong_dan_sp, duong_dan_khia_canh, thu_muc_import)