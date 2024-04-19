from ultralytics.models import RTDETR
import torch
import cv2

def detect_objects(image_path, pt_path):
    print("Starting object detection...")

    model = RTDETR(pt_path)
    print("Model loaded.")

    # predict(image_path, stream=True, save=False, imgsz=640, conf=0.5)

    # 运行预测并获取结果
    results = model(image_path)
    print("Object detection completed.")
    # 获取原始图像
    orig_img = results[0].orig_img

    # 获取原始图像的形状
    height, width, _ = orig_img.shape

    # 将原始图像数据转换为OpenCV可以显示的格式
    orig_img_cv2 = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

    # 遍历所有的检测框
    for box in results[0].boxes:
        # box 是一个包含边界框信息的对象，具体属性取决于您的库版本
        # 通常，它至少包含 'xyxy' 或 'xywh' 格式的坐标
        xyxy = box.xyxy[0]  # 假设 box.xyxy 是一个包含坐标的数组
        x1, y1, x2, y2 = xyxy

        # 在图像上绘制矩形框
        cv2.rectangle(orig_img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        # 获取检测到的类别索引
        cls = box.cls

        # 获取类别名称
        class_name = results[0].names[int(cls)]

        # 显示类别和置信度（如果有的话）
        conf = box.conf if hasattr(box, 'conf') else None
        if conf is not None:
            # 假设 conf 是一个包含一个元素的 Tensor
            if isinstance(conf, torch.Tensor):
                conf_value = conf.item()  # 将 Tensor 转换为 Python 浮点数
            else:
                conf_value = conf  # 如果 conf 已经是一个浮点数，则直接使用它

            # 现在可以使用 f-string 格式化 conf_value
            text = f'{class_name} {conf_value:.2f}'

            # 接下来，将文本绘制到图像上
            cv2.putText(orig_img_cv2, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(orig_img_cv2, f'{class_name}', (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

    # 将图像从RGB转换回BGR（如果需要的话，取决于您如何使用返回的图像）
    orig_img_cv2 = cv2.cvtColor(orig_img_cv2, cv2.COLOR_RGB2BGR)

    # 返回处理后的图像
    return orig_img_cv2


if __name__ == '__main__':
    pt_path = 'best.pt'
    image_path = './assets/crazing_5.jpg'
    processed_image = detect_objects(image_path, pt_path)
    cv2.imshow('Detection Results', processed_image)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口
