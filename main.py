import cv2
from ultralytics import YOLO

# Carrega o modelo pré-treinado (YOLOv8x)
model = YOLO('yolov8x.pt')

# Função para detectar objetos em uma imagem
def detect_objects(image_path, target_class):
    # Carrega a imagem
    img = cv2.imread(image_path)
    # Verifica se a imagem foi carregada corretamente
    if img is None:
        print(f"Não foi possível carregar a imagem: {image_path}")
        return False
    # Converte a imagem para RGB (YOLOv8 também espera imagens RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Realiza a detecção
    results = model(img_rgb)

    # Processa os resultados
    detected = False
    for result in results:  # YOLOv8 retorna uma lista de resultados
        boxes = result.boxes  # Caixas delimitadoras

        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Coordenadas da caixa delimitadora
            confidence = box.conf[0].item()  # Confiança da detecção
            class_idx = int(box.cls[0].item())  # Índice da classe detectada
            class_name = model.names[class_idx]  # Nome da classe detectada

            # Filtra pelo material específico (classe)
            if class_name == target_class and confidence > 0.5:
                detected = True
                # Desenha um retângulo ao redor do objeto detectado
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, f'{class_name} {confidence:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Exibe a imagem com as detecções
    cv2.imshow('Detecções', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected

# Exemplo de uso
if __name__ == "__main__":
    image_path = './Imgs/IMG_2878_jpeg_jpeg.rf.6500bea9ebe31d9e4246479b53bd4ea0.jpg'
    target_class = 'cup'

    # Exibe as classes disponíveis no modelo
    print("Classes disponíveis no modelo:")
    print(model.names)

    material_detectado = detect_objects(image_path, target_class)
    if material_detectado:
        print(f"O material '{target_class}' foi detectado na imagem.")
    else:
        print(f"O material '{target_class}' NÃO foi detectado na imagem.")
