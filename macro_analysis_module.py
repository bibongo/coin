
import pandas as pd

# Hàm mô phỏng lấy dữ liệu vĩ mô từ API (ở đây dùng dữ liệu tĩnh để test)
def get_us_macro_data():
    # Mô phỏng dữ liệu vĩ mô mới nhất
    return {
        'Fed Rate': {'value': 5.5, 'unit': '%'},
        'CPI YoY': {'value': 3.2, 'unit': '%'},
        'DXY': {'value': 105.4, 'unit': ''},
        'Non-Farm Payroll': {'value': 275000, 'unit': ''},
        'PMI': {'value': 52.5, 'unit': ''},
        'M2 YoY': {'value': 6.0, 'unit': '%'}
    }

# Hàm đánh giá từng chỉ số vĩ mô
def evaluate_macro_data(data):
    impact_score = 0
    analysis = []

    # Fed Rate
    rate = data['Fed Rate']['value']
    if rate > 5.0:
        impact_score -= 0.3
        remark = "Cao → Áp lực giảm giá"
        impact = "BÁN"
    else:
        remark = "Thấp → Tiền rẻ"
        impact = "MUA"
        impact_score += 0.1
    analysis.append(["Fed Rate", f"{rate}%", remark, impact])

    # CPI
    cpi = data['CPI YoY']['value']
    if cpi > 4.0:
        impact_score -= 0.2
        remark = "CPI cao → lo ngại lạm phát"
        impact = "BÁN"
    elif cpi < 2.0:
        impact_score -= 0.1
        remark = "CPI thấp → lo ngại giảm phát"
        impact = "BÁN"
    else:
        remark = "Ổn định"
        impact = "TRUNG TÍNH"
    analysis.append(["CPI YoY", f"{cpi}%", remark, impact])

    # DXY
    dxy = data['DXY']['value']
    if dxy > 104:
        impact_score -= 0.2
        remark = "USD mạnh → áp lực giảm crypto"
        impact = "BÁN"
    elif dxy < 98:
        impact_score += 0.2
        remark = "USD yếu → hỗ trợ crypto"
        impact = "MUA"
    else:
        remark = "Ổn định"
        impact = "TRUNG TÍNH"
    analysis.append(["DXY", f"{dxy}", remark, impact])

    # Non-Farm Payroll
    nfp = data['Non-Farm Payroll']['value']
    if nfp > 250000:
        impact_score -= 0.1
        remark = "Việc làm mạnh → FED có thể tăng lãi suất"
        impact = "BÁN"
    else:
        remark = "Việc làm yếu → kích thích tiền tệ"
        impact = "MUA"
        impact_score += 0.1
    analysis.append(["Non-Farm Payroll", f"{nfp:,}", remark, impact])

    # PMI
    pmi = data['PMI']['value']
    if pmi > 55:
        impact_score += 0.1
        remark = "Kinh tế tăng trưởng mạnh"
        impact = "MUA"
    elif pmi < 45:
        impact_score -= 0.1
        remark = "Kinh tế suy yếu"
        impact = "BÁN"
    else:
        remark = "Ổn định"
        impact = "TRUNG TÍNH"
    analysis.append(["PMI", f"{pmi}", remark, impact])

    # M2
    m2 = data['M2 YoY']['value']
    if m2 > 5:
        impact_score += 0.3
        remark = "Tiền rẻ → hỗ trợ đầu tư"
        impact = "MUA"
    else:
        impact_score -= 0.1
        remark = "Cung tiền giảm → hạn chế dòng tiền"
        impact = "BÁN"
    analysis.append(["M2 YoY", f"{m2}%", remark, impact])

    # Tổng kết
    if impact_score >= 0.3:
        summary = "Tích cực → Ưu tiên MUA"
    elif impact_score <= -0.3:
        summary = "Tiêu cực → Ưu tiên BÁN"
    else:
        summary = "Trung tính → Ưu tiên GIỮ"

    df = pd.DataFrame(analysis, columns=["Chỉ Số", "Giá Trị", "Đánh Giá", "Tác Động"])
    return df, impact_score, summary
