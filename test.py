import json
import cv2
import os
import shutil

input_path = "TACO/data"  # Caminho onde estão as pastas "batch_1", "batch_2", etc.
output_path = "./"

# Carregando o arquivo de anotações COCO
with open('TACO/data/annotations.json') as f:
    data = json.load(f)

os.makedirs(f"{output_path}images", exist_ok=True)
os.makedirs(f"{output_path}labels", exist_ok=True)

file_names = []

# Função para percorrer subpastas e carregar imagens
def load_images_from_folders(main_folder):
    count = 0
    for subdir, _, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                full_path_in_annotations = os.path.relpath(os.path.join(subdir, filename), start=main_folder)
                
                source = os.path.join(subdir, filename)
                destination = f"{output_path}images/img{count}.jpg"
                
                try:
                    shutil.copy(source, destination)
                    print(f"File {filename} copied successfully.")
                except shutil.SameFileError:
                    print(f"Source and destination represent the same file: {filename}.")
                
                file_names.append(full_path_in_annotations.replace("\\", "/"))  # Adaptar para compatibilidade com o formato do JSON
                count += 1

# Chama a função para carregar todas as imagens de todas as subpastas
load_images_from_folders(input_path)

# Função para obter anotações de uma imagem específica
def get_img_ann(image_id):
    img_ann = []
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
    return img_ann if img_ann else None

# Função para obter informações de uma imagem a partir do nome do arquivo
def get_img(filename):
    for img in data['images']:
        if img['file_name'] == filename:
            return img

# Processamento das imagens e suas anotações
count = 0
for filename in file_names:
    # Extraindo informações da imagem
    img = get_img(filename)
    if not img:
        continue

    img_id = img['id']
    img_w = img['width']
    img_h = img['height']

    # Obtendo as anotações para esta imagem
    img_ann = get_img_ann(img_id)

    if img_ann:
        # Abrindo o arquivo de anotações para a imagem atual
        with open(f"{output_path}labels/img{count}.txt", "a") as file_object:
            for ann in img_ann:
                current_category = ann['category_id'] - 1  # As classes no YOLO começam em 0
                current_bbox = ann['bbox']
                x, y, w, h = current_bbox

                # Calculando os pontos centrais
                x_centre = (x + w / 2) / img_w
                y_centre = (y + h / 2) / img_h
                w /= img_w
                h /= img_h

                # Formatando os valores para 6 casas decimais
                x_centre = format(x_centre, '.6f')
                y_centre = format(y_centre, '.6f')
                w = format(w, '.6f')
                h = format(h, '.6f')

                # Escrevendo as anotações no formato YOLO
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

        count += 1
