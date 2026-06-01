// ============================================================================
// PHẦN 1: KHỞI TẠO CÁC RÀNG BUỘC DUY NHẤT (CONSTRAINTS)
// Mục đích: Đảm bảo tính toàn vẹn dữ liệu, không trùng lặp và tăng tốc độ tìm kiếm
// ============================================================================

CREATE CONSTRAINT ma_san_pham_duy_nhat IF NOT EXISTS
FOR (p:Product) REQUIRE p.parent_asin IS UNIQUE;

CREATE CONSTRAINT ma_nhan_hieu_duy_nhat IF NOT EXISTS
FOR (b:Brand) REQUIRE b.brand_id IS UNIQUE;

CREATE CONSTRAINT ma_danh_muc_duy_nhat IF NOT EXISTS
FOR (c:Category) REQUIRE c.category_id IS UNIQUE;

CREATE CONSTRAINT ma_dac_tinh_duy_nhat IF NOT EXISTS
FOR (f:Feature) REQUIRE f.feature_id IS UNIQUE;

CREATE CONSTRAINT ma_khia_canh_duy_nhat IF NOT EXISTS
FOR (a:Aspect) REQUIRE a.aspect_id IS UNIQUE;

CREATE CONSTRAINT ma_chi_tiet_duy_nhat IF NOT EXISTS
FOR (d:Detail) REQUIRE d.detail_id IS UNIQUE;


// ============================================================================
// PHẦN 2: NẠP DỮ LIỆU CÁC NÚT THỰC THỂ (NODES)
// Thực thi theo từng khối giao dịch nhằm tối ưu hóa tài nguyên hệ thống
// ============================================================================

// 1. Nạp dữ liệu các nút Sản phẩm
LOAD CSV WITH HEADERS FROM 'file:///products.csv' AS row
CALL {
    WITH row
    MERGE (p:Product {parent_asin: row.`parent_asin:ID(Product)`})
    SET p.title = row.title,
        p.average_rating = toFloat(row.`average_rating:float`),
        p.rating_number = toInteger(row.`rating_number:int`),
        p.price = toFloat(row.`price:float`),
        p.main_category = row.main_category
} IN TRANSACTIONS OF 1000 ROWS;

// 2. Nạp dữ liệu các nút Nhãn hiệu
LOAD CSV WITH HEADERS FROM 'file:///brands.csv' AS row
CALL {
    WITH row
    MERGE (b:Brand {brand_id: row.`brand_id:ID(Brand)`})
    SET b.name = row.name
} IN TRANSACTIONS OF 1000 ROWS;

// 3. Nạp dữ liệu các nút Danh mục sản phẩm
LOAD CSV WITH HEADERS FROM 'file:///categories.csv' AS row
CALL {
    WITH row
    MERGE (c:Category {category_id: row.`category_id:ID(Category)`})
    SET c.name = row.name
} IN TRANSACTIONS OF 1000 ROWS;

// 4. Nạp dữ liệu các nút Đặc tính sản phẩm
LOAD CSV WITH HEADERS FROM 'file:///features.csv' AS row
CALL {
    WITH row
    MERGE (f:Feature {feature_id: row.`feature_id:ID(Feature)`})
    SET f.text = row.text
} IN TRANSACTIONS OF 1000 ROWS;

// 5. Nạp dữ liệu các nút Khía cạnh đánh giá
LOAD CSV WITH HEADERS FROM 'file:///aspects.csv' AS row
CALL {
    WITH row
    MERGE (a:Aspect {aspect_id: row.`aspect_id:ID(Aspect)`})
    SET a.name = row.name
} IN TRANSACTIONS OF 1000 ROWS;

// 6. Nạp dữ liệu các nút Chi tiết động (Các cặp Thuộc tính - Giá trị cấu trúc lồng nhau ban đầu)
LOAD CSV WITH HEADERS FROM 'file:///details.csv' AS row
CALL {
    WITH row
    MERGE (d:Detail {detail_id: row.`detail_id:ID(Detail)`})
    SET d.key = row.key,
        d.value = row.value
} IN TRANSACTIONS OF 1000 ROWS;


// ============================================================================
// PHẦN 3: THIẾT LẬP CÁC MỐI QUAN HỆ LIÊN KẾT (RELATIONSHIPS)
// Thực thi kết nối các nút dựa trên các mã định danh đã thiết lập ràng buộc
// ============================================================================

// 1. Liên kết giữa Sản phẩm và Nhãn hiệu tương ứng
LOAD CSV WITH HEADERS FROM 'file:///rel_product_brand.csv' AS row
CALL {
    WITH row
    MATCH (p:Product {parent_asin: row.`:START_ID(Product)`})
    MATCH (b:Brand {brand_id: row.`:END_ID(Brand)`})
    MERGE (p)-[:HAS_BRAND]->(b)
} IN TRANSACTIONS OF 1000 ROWS;

// 2. Liên kết giữa Sản phẩm và Danh mục phân cấp
LOAD CSV WITH HEADERS FROM 'file:///rel_product_category.csv' AS row
CALL {
    WITH row
    MATCH (p:Product {parent_asin: row.`:START_ID(Product)`})
    MATCH (c:Category {category_id: row.`:END_ID(Category)`})
    MERGE (p)-[:BELONGS_TO_CATEGORY]->(c)
} IN TRANSACTIONS OF 1000 ROWS;

// 3. Liên kết giữa Sản phẩm và Đặc tính nổi bật
LOAD CSV WITH HEADERS FROM 'file:///rel_product_feature.csv' AS row
CALL {
    WITH row
    MATCH (p:Product {parent_asin: row.`:START_ID(Product)`})
    MATCH (f:Feature {feature_id: row.`:END_ID(Feature)`})
    MERGE (p)-[:HAS_FEATURE]->(f)
} IN TRANSACTIONS OF 1000 ROWS;

// 4. Liên kết giữa Sản phẩm và nút Thuộc tính chi tiết lồng nhau
LOAD CSV WITH HEADERS FROM 'file:///rel_product_detail.csv' AS row
CALL {
    WITH row
    MATCH (p:Product {parent_asin: row.`:START_ID(Product)`})
    MATCH (d:Detail {detail_id: row.`:END_ID(Detail)`})
    MERGE (p)-[:HAS_DETAIL]->(d)
} IN TRANSACTIONS OF 1000 ROWS;

// 5. Liên kết Khía cạnh Tích cực kèm theo trọng số tần suất phản hồi
LOAD CSV WITH HEADERS FROM 'file:///rel_positive_aspect.csv' AS row
CALL {
    WITH row
    MATCH (p:Product {parent_asin: row.`:START_ID(Product)`})
    MATCH (a:Aspect {aspect_id: row.`:END_ID(Aspect)`})
    MERGE (p)-[r:HAS_POSITIVE_ASPECT]->(a)
    SET r.weight = toInteger(row.`weight:int`)
} IN TRANSACTIONS OF 1000 ROWS;

// 6. Liên kết Khía cạnh Tiêu cực kèm theo trọng số tần suất phản hồi
LOAD CSV WITH HEADERS FROM 'file:///rel_negative_aspect.csv' AS row
CALL {
    WITH row
    MATCH (p:Product {parent_asin: row.`:START_ID(Product)`})
    MATCH (a:Aspect {aspect_id: row.`:END_ID(Aspect)`})
    MERGE (p)-[r:HAS_NEGATIVE_ASPECT]->(a)
    SET r.weight = toInteger(row.`weight:int`)
} IN TRANSACTIONS OF 1000 ROWS;