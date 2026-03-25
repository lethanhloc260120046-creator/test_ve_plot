import pandas as pd
import matplotlib.pyplot as plt
import ollama
import re
import numpy as np
import os 


output_folder = "ket_qua_du_bao"
os.makedirs(output_folder, exist_ok=True) # Nếu chưa có folder thì tự tạo, có rồi thì bỏ qua
df = pd.read_csv("../../data.csv", header=[0, 1], index_col=0)
aapl_prices = df['Close']['AAPL'].dropna()
predict_days = 7
lookback_list = [5, 30, 100] 

for lookback_days in lookback_list:
    input_sequence = aapl_prices.iloc[-(lookback_days + predict_days) : -predict_days].tolist()
    input_string = ", ".join([f"{x:.2f}" for x in input_sequence])
    actual_values = aapl_prices.iloc[-predict_days:].tolist()

    prompt = f""" 
Nhiệm vụ của bạn là dự đoán {predict_days} giá trị tiếp theo của chuỗi số thời gian.
Đây là chuỗi {lookback_days} giá trị gần nhất:
{input_string}

Hãy dự đoán {predict_days} giá trị tiếp theo.
CHỈ in ra đúng {predict_days} con số, phân tách nhau bằng dấu phẩy (,). 
KHÔNG giải thích, KHÔNG viết thêm bất kỳ chữ nào khác.
"""
    
    response = ollama.chat(model='gemma3:4b', messages=[{'role': 'user', 'content': prompt}])
    kq_string = response['message']['content'].strip()
    
    try:
        # Dùng Regex để ép lấy đúng các con số từ kết quả trả về
        predicted_values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.*\d+)", kq_string)]
        
        if len(predicted_values) < predict_days:
            print("Kết quả trả về không đủ 7 ngày")
            continue
            
        predicted_values = predicted_values[:predict_days]
        
        result_df = pd.DataFrame({
            'Day': [f"Day +{i+1}" for i in range(predict_days)],
            'Actual Value': actual_values,
            'Forecast Value': predicted_values
        })
        
        csv_filename = os.path.join(output_folder, f"forecast_lb_{lookback_days}.csv")
        result_df.to_csv(csv_filename, index=False)
        print(f" Đã lưu dữ liệu vào: {csv_filename}")
        plt.figure(figsize=(12, 7))
        
        x_past = np.arange(-lookback_days, 0)
        x_future = np.arange(0, predict_days) 
        
        past_vals = input_sequence
        last_past_val = past_vals[-1] 
        actual_with_context = [last_past_val] + actual_values
        forecast_with_context = [last_past_val] + predicted_values
        x_future_context = np.arange(-1, predict_days)

        plt.plot(x_past, past_vals, label='Lịch sử giá', color='#7f7f7f', alpha=0.6, linewidth=1.5)
        plt.plot(x_future_context, actual_with_context, marker='o', label='Actual Value', color='#1f77b4', linewidth=2.5)
        plt.plot(x_future_context, forecast_with_context, marker='x', label='Forecast Value', color='#ff7f0e', linestyle='dashed', linewidth=2.5)
        
        plt.title(f"Giá cổ phiếu dự đoán của AAPL - Lookback: {lookback_days} Ngày", fontsize=16, fontweight='bold')
        plt.ylabel("Stock Price (USD)", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        xticks_pos = [ -lookback_days, -1, predict_days-1]
        xticks_labels = [f"-{lookback_days} Days", "-1 Day", f"Forecast Day +{predict_days}"]
        
        if lookback_days > 20:
             mid_point = int(-lookback_days / 2)
             xticks_pos.insert(1, mid_point)
             xticks_labels.insert(1, f"{mid_point} Days")

        plt.xticks(xticks_pos, xticks_labels, fontsize=10)
        plt.axvline(x=-1, color='red', linestyle=':', alpha=0.8, linewidth=1)
        plt.text(-1, plt.ylim()[0], ' Prediction Start', color='red', rotation=90, verticalalignment='bottom')
        plt.legend(fontsize=11)
        
        img_filename = os.path.join(output_folder, f"context_plot_lb_{lookback_days}.png")
        plt.savefig(img_filename, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu biểu đồ bối cảnh vào: {img_filename}")

    except Exception as e:
        print("{lookback_days} Lỗi: {e}")

