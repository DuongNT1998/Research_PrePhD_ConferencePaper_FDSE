import re
from pathlib import Path
import pandas as pd

# Đường dẫn thư mục chứa file dữ liệu giống file setup ban đầu của bạn
IMPORT_DIR = Path("neo4j_setup/import")
INPUT_FILE = IMPORT_DIR / "products.csv"
OUTPUT_FILE = IMPORT_DIR / "products.csv"  # Bạn có thể đổi tên thành "products_seo.csv" nếu muốn tạo file riêng


def clean_seo_title(title):
    if not isinstance(title, str) or not title.strip():
        return title

    title = title.strip()

    # 1. Xử lý viết hoa toàn bộ: Nếu cả tiêu đề bị viết hoa, chuyển về dạng Title Case thông thường
    if title.isupper():
        title = title.title()

    # 2. Loại bỏ các ký tự đặc biệt/emoji gây spam, chỉ giữ lại chữ, số, khoảng trắng và các dấu cơ bản
    title = re.sub(r"[^\w\s\-\.,\/\(\)\[\]\+:\&\’\'\"]", "", title)

    # 3. Tách tiêu đề dựa trên các dấu phân tách thông thường mà người bán hay dùng để nhồi nhét tính năng
    # Tách theo dấu phẩy, dấu chấm phẩy, hoặc dấu gạch đứng/gạch ngang có khoảng trắng xung quanh (để tránh làm hỏng từ nối như Wi-Fi)
    parts = re.split(r"\s+[\-\|]\s+|[,;]", title)

    seo_title_parts = []
    word_counts = {}
    current_length = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Kiểm tra chống trùng lặp / spam từ khóa trong phân đoạn này
        words = part.split()
        cleaned_part_words = []
        for w in words:
            # Chuẩn hóa từ về chữ thường, bỏ dấu câu bám quanh từ để kiểm tra chính xác
            norm_w = re.sub(r"[^\w]", "", w.lower())
            if norm_w:
                # NGUYÊN TẮC: Tuyệt đối không lặp lại một từ khóa quá 2 lần trong toàn tiêu đề
                if word_counts.get(norm_w, 0) >= 2:
                    continue
                word_counts[norm_w] = word_counts.get(norm_w, 0) + 1
            cleaned_part_words.append(w)

        if not cleaned_part_words:
            continue

        cleaned_part = " ".join(cleaned_part_words)

        # Kiểm tra giới hạn độ dài lý tưởng (ưu tiên gom cụm thông tin gọn gàng dưới 100-110 ký tự trước)
        if not seo_title_parts:
            seo_title_parts.append(cleaned_part)
            current_length = len(cleaned_part)
        else:
            if current_length + len(cleaned_part) + 2 <= 110:
                seo_title_parts.append(cleaned_part)
                current_length += len(cleaned_part) + 2
            else:
                # Đã đạt độ dài lý tưởng, dừng thêm phân đoạn phụ để tiêu đề không bị loãng
                break

    # Kết nối lại các phần bằng dấu phẩy hợp lý
    seo_title = ", ".join(seo_title_parts)

    # Dọn dẹp khoảng trắng thừa cấu trúc lại chuỗi gọn gàng
    seo_title = re.sub(r"\s+", " ", seo_title).strip()

    # 4. Giới hạn cứng nghiêm ngặt: Tuyệt đối dưới 120 ký tự
    if len(seo_title) > 120:
        # Cắt tại vị trí khoảng trắng gần nhất để không bị mất chữ giữa chừng
        seo_title = seo_title[:120].rsplit(" ", 1)[0]
        # Loại bỏ các dấu phân tách thừa bám ở cuối chuỗi sau khi cắt
        seo_title = seo_title.rstrip(", -|/;:")

    return seo_title
import re

def lam_sach_chuoi_van_ban(chuoi_nhap):
    if not chuoi_nhap:
        return ""
    # Loại bỏ các thẻ định dạng văn bản siêu liên kết nếu có
    chuoi_sach = re.sub(r'<[^>]+>', '', chuoi_nhap)
    # Thay thế dấu nháy kép bằng dấu nháy đơn để tránh lỗi định dạng bảng CSV
    chuoi_sach = chuoi_sach.replace('"', "'")
    # Loại bỏ các ký tự xuống dòng gây ngắt hàng sai quy cách
    chuoi_sach = chuoi_sach.replace('\n', ' ').replace('\r', ' ')
    return chuoi_sach.strip()

def main():
    if not INPUT_FILE.exists():
        print(f"❌ Không tìm thấy file {INPUT_FILE}. Vui lòng chạy file sinh dữ liệu trước!")
        return

    print(f"⏳ Đang đọc dữ liệu từ {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    if "title" not in df.columns:
        print("❌ File csv không có cột 'title'!")
        return

    print(f"⚡ Đang tiến hành tối ưu SEO cho {len(df)} dòng sản phẩm...")

    # Lưu lại một vài ví dụ trước khi sửa để đối chiếu
    samples_before = df["title"].head(3).tolist()

    # Áp dụng hàm tối ưu hóa tiêu đề
    df["title"] = df["title"].apply(clean_seo_title)

    # In kết quả đối chiếu mẫu ra màn hình để bạn kiểm tra độ hiệu quả
    print("\n📊 BÁO CÁO ĐỐI CHIẾU MẪU TIÊU ĐỀ SAU KHI TỐI ƯU SEO:")
    print("=" * 60)
    for i, (before, after) in enumerate(zip(samples_before, df["title"].head(3))):
        print(f"Sản phẩm {i+1}:")
        print(f" -> GỐC: {before}")
        print(f" -> SEO: {after} ({len(after)} ký tự)")
        print("-" * 60)

    # Lưu kết quả xuống file CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 HOÀN THÀNH! Đã lưu file tiêu đề chuẩn SEO tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()