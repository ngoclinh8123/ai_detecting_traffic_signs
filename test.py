import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Tải lại mô hình
loaded_model = load_model("traffic_sign_model.keras")

# Bảng ánh xạ giữa số lớp và tên biển báo
class_mapping_en = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

class_mapping_vie = {
    0: "Hạn chế tốc độ (20km/h)",
    1: "Hạn chế tốc độ (30km/h)",
    2: "Hạn chế tốc độ (50km/h)",
    3: "Hạn chế tốc độ (60km/h)",
    4: "Hạn chế tốc độ (70km/h)",
    5: "Hạn chế tốc độ (80km/h)",
    6: "Kết thúc hạn chế tốc độ (80km/h)",
    7: "Hạn chế tốc độ (100km/h)",
    8: "Hạn chế tốc độ (120km/h)",
    9: "Cấm vượt",
    10: "Cấm vượt (đối với xe trên 3.5 tấn)",
    11: "Ưu tiên tại giao lộ kế tiếp",
    12: "Đường ưu tiên",
    13: "Nhường đường",
    14: "Dừng lại",
    15: "Cấm xe cơ giới",
    16: "Cấm xe trên 3.5 tấn",
    17: "Cấm đi",
    18: "Cảnh báo chung",
    19: "Curve nguy hiểm bên trái",
    20: "Curve nguy hiểm bên phải",
    21: "Curve nguy hiểm kép",
    22: "Đường xấu",
    23: "Đường trơn trượt",
    24: "Đường hẹp bên phải",
    25: "Công trường",
    26: "Đèn tín hiệu giao thông",
    27: "Người đi bộ",
    28: "Băng qua đường",
    29: "Xe đạp băng qua",
    30: "Cảnh báo đường trơn đá băng/tuyết",
    31: "Cảnh báo về việc vượt qua động vật hoang dã",
    32: "Kết thúc tất cả các giới hạn tốc độ và cấm vượt",
    33: "Rẽ phải phía trước",
    34: "Rẽ trái phía trước",
    35: "Chỉ đi thẳng",
    36: "Chỉ đi thẳng hoặc rẽ phải",
    37: "Chỉ đi thẳng hoặc rẽ trái",
    38: "Đi bên phải",
    39: "Đi bên trái",
    40: "Bắt buộc đi vòng xuyến",
    41: "Kết thúc cấm vượt",
    42: "Kết thúc cấm vượt (đối với xe trên 3.5 tấn)",
}

# Chuẩn bị ảnh đầu vào
input_image = cv2.imread("00005.png")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (32, 32))
input_image = input_image / 255.0
input_image = np.expand_dims(input_image, axis=0)  # Thêm chiều batch

# Dự đoán
predictions = loaded_model.predict(input_image)
predicted_class = np.argmax(predictions)

# Lấy tỷ lệ dự đoán của lớp được chọn
confidence = predictions[0][predicted_class]

# Kiểm tra tỷ lệ dự đoán, nếu dưới 50% thì trả về "Unknown sign"
if confidence < 0.5:
    predicted_sign = "Không thể xác định"
else:
    # Lấy tên của biển báo từ bảng ánh xạ
    predicted_sign = class_mapping_vie.get(predicted_class, "Không thể xác định")

# In kết quả dự đoán
print(f"Predicted Sign: {predicted_sign}")
